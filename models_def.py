"""
╔══════════════════════════════════════════════════════════════╗
║   BOXING AI — SHARED MODEL DEFINITIONS                       ║
║   Import this in: server.py, stage2, stage3, stage4          ║
╚══════════════════════════════════════════════════════════════╝

CHANGE LOG:
  v2 — Added "idle" as a real trained class (index 4).
  v3 — Replaced GRU with 2nd-order Markov predictor.
       GRU_NextMove class removed.
       infer_gru removed → use infer_markov from markov_predictor.py
       load_gru removed  → use load_markov from markov_predictor.py
"""

import torch
import torch.nn as nn
import numpy as np
import collections

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
CLASSES       = ["jab", "cross", "hook", "uppercut", "idle"]
NUM_CLASSES   = len(CLASSES)
IDLE_IDX      = CLASSES.index("idle")
IDX_TO_CLASS  = {i: c for i, c in enumerate(CLASSES)}
CLASS_TO_IDX  = {c: i for i, c in enumerate(CLASSES)}

PUNCH_CLASSES = [c for c in CLASSES if c != "idle"]
PUNCH_INDICES = [CLASS_TO_IDX[c] for c in PUNCH_CLASSES]

LANDMARKS_USED = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 23, 24]
SEQ_LEN        = 30
BASE_FEATURES  = len(LANDMARKS_USED) * 3   # 13 × 3 = 39
INPUT_SIZE     = BASE_FEATURES * 3          # 117 = pose + velocity + acceleration
CONTEXT_LEN    = 2                          # Markov looks back 2 punches (2nd-order)


# ─────────────────────────────────────────────
#  MOTION FEATURES
# ─────────────────────────────────────────────
def compute_motion_features(seq: np.ndarray) -> np.ndarray:
    """
    (T, 39) → (T, 117) = [pose | velocity | acceleration]
    Idle: near-zero vel AND accel — very distinctive signature.
    """
    T     = len(seq)
    vel   = np.zeros_like(seq)
    accel = np.zeros_like(seq)

    if T >= 2:
        vel[0]  = seq[1] - seq[0]
        vel[-1] = seq[-1] - seq[-2]
        if T >= 3:
            vel[1:-1] = (seq[2:] - seq[:-2]) / 2.0
        accel[0]  = vel[1] - vel[0]   if T >= 2 else 0
        accel[-1] = vel[-1] - vel[-2] if T >= 2 else 0
        if T >= 3:
            accel[1:-1] = (vel[2:] - vel[:-2]) / 2.0

    return np.concatenate([seq, vel, accel], axis=1).astype(np.float32)


# ─────────────────────────────────────────────
#  TCN ARCHITECTURE
# ─────────────────────────────────────────────
class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv    = nn.Conv1d(in_ch, out_ch, kernel_size,
                                 padding=self.padding, dilation=dilation)

    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :-self.padding] if self.padding else out


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(in_ch, out_ch, kernel_size, dilation),
            nn.BatchNorm1d(out_ch), nn.ReLU(), nn.Dropout(dropout),
            CausalConv1d(out_ch, out_ch, kernel_size, dilation),
            nn.BatchNorm1d(out_ch), nn.ReLU(), nn.Dropout(dropout),
        )
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.net(x) + self.skip(x))


class TCN_Boxing(nn.Module):
    """
    TCN for boxing move recognition.
    Input:  (batch, 117, 30)
    Output: (batch, 5) logits  [jab, cross, hook, uppercut, idle]

    POOL NOTE: uses both AdaptiveAvgPool (overall pattern) and
    last-timestep (snap/peak detection) concatenated → 128-dim.
    This helps jab (brief peak at end) vs cross (sustained velocity).
    """
    def __init__(self, input_size=INPUT_SIZE, num_classes=NUM_CLASSES,
                 channels=None, kernel_size=3, dropout=0.2):
        super().__init__()
        if channels is None:
            channels = [64, 128, 128, 64]
        layers, in_ch = [], input_size
        for out_ch, dil in zip(channels, [1, 2, 4, 8]):
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dil, dropout))
            in_ch = out_ch
        self.tcn      = nn.Sequential(*layers)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        final_ch      = channels[-1]
        # Concatenate avg-pool + last timestep → richer representation
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(final_ch * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        feat      = self.tcn(x)                          # (B, 64, 30)
        avg       = self.avg_pool(feat).squeeze(-1)      # (B, 64)
        last      = feat[:, :, -1]                       # (B, 64) last timestep
        combined  = torch.cat([avg, last], dim=1)        # (B, 128)
        return self.classifier(combined)


# ─────────────────────────────────────────────
#  POSE EXTRACTION
# ─────────────────────────────────────────────
def extract_keypoints_from_results(results, landmarks_used=None):
    if landmarks_used is None:
        landmarks_used = LANDMARKS_USED
    if not results.pose_landmarks:
        return None
    lm  = results.pose_landmarks.landmark
    arr = np.array([[l.x, l.y, l.visibility] for l in lm], dtype=np.float32)
    hip_center  = (arr[23, :2] + arr[24, :2]) / 2.0
    arr[:, 0]  -= hip_center[0]
    arr[:, 1]  -= hip_center[1]
    return arr[landmarks_used].flatten()


# ─────────────────────────────────────────────
#  SMOOTH PREDICTOR
# ─────────────────────────────────────────────
class SmoothPredictor:
    """
    Majority-vote smoother. idle is now a real class (index 4).
    combo only contains punches — idle never added.
    """
    def __init__(self, window=4, conf_thresh=0.60, idle_thresh=0.50,
                 idle_streak_needed=3, min_interval=0.30):
        self.window             = collections.deque(maxlen=window)
        self.conf_thresh        = conf_thresh
        self.idle_thresh        = idle_thresh
        self.idle_streak_needed = idle_streak_needed
        self.idle_streak        = 0
        self.min_interval       = min_interval
        self.last_logged_time   = 0.0
        self.current            = "idle"
        self.last_logged        = "idle"
        self.combo              = []

    def update(self, logits, frame_buffer=None):
        import time as _time
        now      = _time.time()
        probs    = torch.softmax(logits, dim=0).cpu().numpy()
        top_conf = float(probs.max())
        top_idx  = int(probs.argmax())

        is_idle_frame = (top_idx == IDLE_IDX) or (top_conf < self.idle_thresh)
        frame_vote    = -1 if is_idle_frame else top_idx

        self.window.append(frame_vote)
        if is_idle_frame:
            self.idle_streak += 1
        else:
            self.idle_streak = 0

        if len(self.window) < self.window.maxlen:
            return self.current, probs, False

        idle_votes = sum(1 for w in self.window if w == -1)

        if idle_votes > len(self.window) // 2 and self.idle_streak >= self.idle_streak_needed:
            if self.current != "idle":
                self.last_logged = "idle"
                if frame_buffer is not None and len(frame_buffer) > 5:
                    recent = list(frame_buffer)[-5:]
                    frame_buffer.clear()
                    frame_buffer.extend(recent)
            self.current = "idle"
            return "idle", probs, False

        valid = [w for w in self.window if w >= 0]
        if not valid:
            return self.current, probs, False

        counts = np.bincount(valid, minlength=NUM_CLASSES)
        best   = int(counts.argmax())

        if best == IDLE_IDX:
            return self.current, probs, False

        if counts[best] >= len(self.window) * 0.5 and probs[best] >= self.conf_thresh:
            new_move = IDX_TO_CLASS[best]
            time_ok  = (now - self.last_logged_time) >= self.min_interval
            is_new   = (new_move != self.last_logged and new_move != "idle" and time_ok)
            if is_new:
                self.combo.append(new_move)
                if len(self.combo) > 8:
                    self.combo.pop(0)
                self.last_logged      = new_move
                self.last_logged_time = now
            self.current = new_move
            return new_move, probs, is_new

        return self.current, probs, False


# ─────────────────────────────────────────────
#  TCN LOADER + INFERENCE
# ─────────────────────────────────────────────
def load_tcn(path, device):
    ckpt  = torch.load(str(path), map_location=device, weights_only=False)
    model = TCN_Boxing(
        input_size  = ckpt.get("input_size",  INPUT_SIZE),
        num_classes = ckpt.get("num_classes", NUM_CLASSES),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def infer_tcn(model, frame_buffer, device):
    raw = np.array(frame_buffer, dtype=np.float32)
    seq = compute_motion_features(raw)
    x   = torch.tensor(seq.T, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(x)[0]


# ─────────────────────────────────────────────
#  MARKOV — convenience re-exports
#  Import markov_predictor directly for full API.
#  These thin wrappers keep server.py / stage4 imports simple.
# ─────────────────────────────────────────────
def load_markov(username: str = "global"):
    from markov_predictor import load_markov as _load
    return _load(username)


def infer_markov(markov, combo_history: list):
    from markov_predictor import infer_markov as _infer
    return _infer(markov, combo_history)