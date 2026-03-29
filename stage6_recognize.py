"""
╔══════════════════════════════════════════════════════════════╗
║   BOXING AI — MOVEMENT RECOGNIZER (Test Tool)                ║
║   Just prints what punch it sees. Nothing else.              ║
╚══════════════════════════════════════════════════════════════╝

Run:
    python recognize.py

Keys:
    Q = quit
    C = clear terminal
    S = strict mode (higher confidence needed)
    L = loose mode  (lower confidence needed)
"""

import cv2
import mediapipe as mp
import numpy as np
import torch
import collections
import time
import os

from models_def import (
    LANDMARKS_USED, SEQ_LEN, INPUT_SIZE, BASE_FEATURES,
    extract_keypoints_from_results,
    load_tcn, infer_tcn,
    CLASSES, IDX_TO_CLASS, IDLE_IDX,
)

# ── Colours for terminal output ────────────────────────────────
TERM = {
    "jab":      "\033[96m",   # cyan
    "cross":    "\033[93m",   # yellow
    "hook":     "\033[92m",   # green
    "uppercut": "\033[95m",   # magenta
    "idle":     "\033[2m",    # dim
    "RESET":    "\033[0m",
    "BOLD":     "\033[1m",
    "DIM":      "\033[2m",
}

# ── OpenCV window colours (BGR) ────────────────────────────────
WIN_COLORS = {
    "jab":      (255, 200,  50),
    "cross":    ( 50, 200, 255),
    "hook":     ( 50, 255, 120),
    "uppercut": (180,  80, 255),
    "idle":     (100, 100, 100),
}


# ──────────────────────────────────────────────────────────────
#  SIMPLE RECOGNIZER
#  idle is now a REAL class (index 4) — not just low confidence.
#  A frame votes idle if EITHER:
#    (a) model explicitly predicts idle (top_idx == IDLE_IDX), OR
#    (b) top confidence is below idle_thresh (uncertain)
# ──────────────────────────────────────────────────────────────
class SimpleRecognizer:
    def __init__(self, window=4, conf_thresh=0.60,
                 idle_thresh=0.35, idle_streak=3):
        self.q              = collections.deque(maxlen=window)
        self.conf_thresh    = conf_thresh
        self.idle_thresh    = idle_thresh  # lowered to 0.35 — idle predictions
        self.idle_streak    = idle_streak  # are real class votes now, not just
        self._idle_count    = 0            # low-confidence fallback
        self.last_printed   = "idle"

    def update(self, logits):
        probs    = torch.softmax(logits, dim=0).cpu().numpy()
        top_conf = float(probs.max())
        top_idx  = int(probs.argmax())

        # Frame votes idle if model says idle OR confidence too low
        is_idle_frame = (top_idx == IDLE_IDX) or (top_conf < self.idle_thresh)
        vote = -1 if is_idle_frame else top_idx
        self.q.append(vote)

        self._idle_count = self._idle_count + 1 if is_idle_frame else 0

        if len(self.q) < self.q.maxlen:
            return "idle", probs, False

        idle_votes = sum(1 for v in self.q if v == -1)

        # Majority idle + streak → idle state
        if idle_votes > len(self.q) // 2 and self._idle_count >= self.idle_streak:
            is_new = (self.last_printed != "idle")
            if is_new:
                self.last_printed = "idle"
            return "idle", probs, is_new   # ← now fires is_new when transitioning TO idle

        # Find majority punch class
        valid = [v for v in self.q if v >= 0]
        if not valid:
            return "idle", probs, False

        counts = np.bincount(valid, minlength=len(CLASSES))
        best   = int(counts.argmax())

        # Don't confirm idle via vote path
        if best == IDLE_IDX:
            return "idle", probs, False

        if counts[best] >= len(self.q) * 0.5 and probs[best] >= self.conf_thresh:
            move   = IDX_TO_CLASS[best]
            is_new = (move != self.last_printed)
            if is_new:
                self.last_printed = move
            return move, probs, is_new

        return "idle", probs, False


# ──────────────────────────────────────────────────────────────
#  DRAW WINDOW HUD
# ──────────────────────────────────────────────────────────────
def draw(frame, move, probs, conf_thresh, idle_thresh, count):
    h, w = frame.shape[:2]

    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 60), (5, 5, 10), -1)
    cv2.addWeighted(ov, 0.85, frame, 0.15, 0, frame)

    col = WIN_COLORS.get(move, (150, 150, 150))

    # Show IDLE explicitly when idle — not "---"
    if move == "idle":
        cv2.putText(frame, "IDLE", (16, 46),
                    cv2.FONT_HERSHEY_DUPLEX, 1.4, (80, 80, 100), 2)
    else:
        cv2.putText(frame, move.upper(), (16, 46),
                    cv2.FONT_HERSHEY_DUPLEX, 1.4, col, 2)

    cv2.putText(frame, f"Detections: {count}",
                (w - 200, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1)

    # Probability bars — all 5 classes including idle
    ov2 = frame.copy()
    cv2.rectangle(ov2, (0, h - 155), (w, h), (5, 5, 10), -1)
    cv2.addWeighted(ov2, 0.82, frame, 0.18, 0, frame)

    bx, by = 14, h - 148
    for i, (cls, p) in enumerate(zip(CLASSES, probs)):
        c  = WIN_COLORS[cls]
        bw = int(p * 220)
        # Highlight the active class bar
        is_active = (cls == move)
        bar_col   = c if is_active else tuple(int(x * 0.4) for x in c)
        cv2.rectangle(frame, (bx, by + i*28), (bx + 200, by + i*28 + 20), (18,18,18), -1)
        if bw > 0:
            cv2.rectangle(frame, (bx, by + i*28), (bx + bw, by + i*28 + 20), bar_col, -1)
        lbl = f"{'IDLE' if cls == 'idle' else cls.upper():<10} {p:.0%}"
        cv2.putText(frame, lbl,
                    (bx + 208, by + i*28 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    c if is_active else (70, 70, 70), 1)

    cv2.putText(frame,
                f"conf={conf_thresh:.2f}  idle={idle_thresh:.2f}   "
                f"S=strict  L=loose  Q=quit",
                (14, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

    return frame


# ──────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────
def run():
    print("\033[2J\033[H")
    print(f"{TERM['BOLD']}╔══════════════════════════════════════════╗")
    print(f"║   BOXING AI — Movement Recognizer        ║")
    print(f"║   5 classes incl. idle                   ║")
    print(f"╚══════════════════════════════════════════╝{TERM['RESET']}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device : {device}")

    import pathlib
    tcn_path = pathlib.Path("models/tcn_boxing.pth")
    if not tcn_path.exists():
        print("  ❌ models/tcn_boxing.pth not found.")
        print("     Run: python stage2_train_tcn.py")
        return

    tcn, ckpt = load_tcn(tcn_path, device)
    n_classes = ckpt.get('num_classes', 4)
    print(f"  TCN    : loaded  val_acc={ckpt.get('val_acc', 0):.1%}  classes={n_classes}")

    if n_classes != len(CLASSES):
        print(f"\n  ⚠️  Model has {n_classes} classes but code expects {len(CLASSES)}.")
        print(f"     Retrain: python stage2_train_tcn.py")
        return

    print(f"  Classes: {CLASSES}\n")
    print(f"  {TERM['DIM']}Q=quit  C=clear  S=strict  L=loose{TERM['RESET']}\n")
    print("  " + "─" * 45)
    print(f"  {'MOVE':<12} {'CONFIDENCE':>10}  {'TIME'}")
    print("  " + "─" * 45)

    pose_model = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.55,
        min_tracking_confidence=0.55,
    )
    mp_draw     = mp.solutions.drawing_utils
    mp_pose_sol = mp.solutions.pose

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    buf        = collections.deque(maxlen=SEQ_LEN)
    recognizer = SimpleRecognizer()
    probs      = [0.0] * len(CLASSES)
    move       = "idle"
    count      = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame   = cv2.flip(frame, 1)
        results = pose_model.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            mp_draw.draw_landmarks(
                frame, results.pose_landmarks, mp_pose_sol.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 100), thickness=2, circle_radius=4),
                mp_draw.DrawingSpec(color=(255, 255, 0),  thickness=2),
            )

        kpts = extract_keypoints_from_results(results, LANDMARKS_USED)
        if kpts is not None:
            buf.append(kpts)

        if len(buf) == SEQ_LEN:
            logits              = infer_tcn(tcn, list(buf), device)
            move, probs, is_new = recognizer.update(logits)

            if is_new:
                col  = TERM.get(move, "")
                conf = float(max(probs))
                ts   = time.strftime("%H:%M:%S")
                if move == "idle":
                    # Print idle transition dimly — shows model is working
                    print(f"  {TERM['DIM']}{'idle':<12} {'—':>10}   {ts}{TERM['RESET']}")
                else:
                    count += 1
                    print(f"  {col}{TERM['BOLD']}{move.upper():<12}{TERM['RESET']}"
                          f" {conf:>9.1%}   {ts}")

        frame = draw(frame, move, probs,
                     recognizer.conf_thresh, recognizer.idle_thresh, count)
        cv2.imshow("Boxing AI — Recognizer (Q to quit)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"  {'MOVE':<12} {'CONFIDENCE':>10}  {'TIME'}")
            print("  " + "─" * 45)
            count = 0
        elif key == ord('s'):
            recognizer.conf_thresh = min(recognizer.conf_thresh + 0.05, 0.95)
            recognizer.idle_thresh = min(recognizer.idle_thresh + 0.05, 0.70)
            print(f"  {TERM['DIM']}[strict]  conf={recognizer.conf_thresh:.2f}  "
                  f"idle={recognizer.idle_thresh:.2f}{TERM['RESET']}")
        elif key == ord('l'):
            recognizer.conf_thresh = max(recognizer.conf_thresh - 0.05, 0.30)
            recognizer.idle_thresh = max(recognizer.idle_thresh - 0.05, 0.15)
            print(f"  {TERM['DIM']}[loose]   conf={recognizer.conf_thresh:.2f}  "
                  f"idle={recognizer.idle_thresh:.2f}{TERM['RESET']}")

    cap.release()
    cv2.destroyAllWindows()
    pose_model.close()

    print(f"\n  Punch detections this session: {count}")
    print(f"  Done.\n")


if __name__ == "__main__":
    run()