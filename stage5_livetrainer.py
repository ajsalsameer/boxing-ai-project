"""
╔══════════════════════════════════════════════════════════════╗
║   BOXING AI — LIVE TRAINER                                   ║
║   Record clean punch samples + idle → auto-retrain TCN       ║
╚══════════════════════════════════════════════════════════════╝

HOW IT WORKS:
  - HOLD a key while throwing your punch → release = saves sample
  - Only frames while you hold the key are recorded
  - This gives you PURE punch frames with zero contamination
  - After each session you can retrain immediately

KEYS:
  J = Jab           (hold during punch)
  C = Cross          (hold during punch)
  H = Hook           (hold during punch)
  U = Uppercut       (hold during punch)
  I = Idle/Guard     (hold while standing still in guard)
  R = Retrain TCN now
  D = Show dataset stats
  Q = Quit

IDLE RECORDING TIPS:
  Idle is the most important class to record well.
  Hold I while doing ALL of these:
    - Standing still in orthodox guard
    - Shifting weight left/right
    - Nodding/moving your head slightly
    - Defensive guard (elbows tight, chin down)
    - Moving your guard hand slightly
    - Brief pause between combos
  Record from: the same spot you throw punches from.
  Aim for 80+ idle samples — it is the most common state.

PUNCH TIPS:
  1. Stand in your normal boxing stance in front of camera
  2. Make sure your full upper body is visible
  3. Press and HOLD the key → throw the punch → release
  4. Throw at realistic speed — not slow motion
  5. Aim for 60+ samples per class (takes ~5 minutes)
  6. Mix up: from guard, after a slip, defensive position
  7. Specifically record CROSS from defensive position →
     fixes the cross/hook confusion at low confidence

Run:
  python live_trainer.py
  python live_trainer.py --min_frames 10   (for faster punches)
  python live_trainer.py --no_retrain      (skip auto-retrain)
"""

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import os
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict

from models_def import (
    CLASSES, NUM_CLASSES, CLASS_TO_IDX, IDLE_IDX,
    LANDMARKS_USED, SEQ_LEN, INPUT_SIZE, BASE_FEATURES,
    TCN_Boxing, extract_keypoints_from_results,
    compute_motion_features,
)

# ──────────────────────────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────────────────────────
DATASET_DIR  = Path("dataset")
MODEL_DIR    = Path("models")
MIN_FRAMES   = 15
MAX_FRAMES   = SEQ_LEN

# Idle needs a shorter minimum — you're just standing, not throwing
IDLE_MIN_FRAMES = 20   # must hold I for at least 20 frames of stillness

KEY_TO_CLASS = {
    ord('j'): "jab",      ord('J'): "jab",
    ord('c'): "cross",    ord('C'): "cross",
    ord('h'): "hook",     ord('H'): "hook",
    ord('u'): "uppercut", ord('U'): "uppercut",
    ord('i'): "idle",     ord('I'): "idle",
}

CLASS_COLORS = {
    "jab":      (50,  200, 255),   # cyan
    "cross":    (255, 200, 50),    # yellow
    "hook":     (50,  255, 120),   # green
    "uppercut": (255, 80,  180),   # pink
    "idle":     (140, 140, 160),   # grey-blue
}

# Idle has a higher target since it's the most common state
CLASS_TARGETS = {
    "jab": 60, "cross": 60, "hook": 60, "uppercut": 60, "idle": 80,
}


# ──────────────────────────────────────────────────────────────────
#  DATASET HELPERS
# ──────────────────────────────────────────────────────────────────
def count_samples():
    counts = {}
    for cls in CLASSES:
        d = DATASET_DIR / cls
        counts[cls] = len(list(d.glob("*.npy"))) if d.exists() else 0
    return counts


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two flat vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def check_variety(new_frames: list, class_name: str) -> tuple:
    """
    Compare mean pose of new_frames against all existing saved samples
    for class_name using cosine similarity.

    Returns (is_duplicate: bool, max_similarity: float, most_similar_idx: int)
    Duplicate threshold: similarity > 0.97
    """
    d = DATASET_DIR / class_name
    if not d.exists():
        return False, 0.0, -1

    new_arr  = np.array(new_frames, dtype=np.float32)
    # Use first SEQ_LEN frames or all if shorter
    new_arr  = new_arr[:SEQ_LEN]
    new_mean = new_arr.mean(axis=0)   # (39,) mean pose

    existing = list(d.glob("*.npy"))
    if not existing:
        return False, 0.0, -1

    max_sim   = 0.0
    max_idx   = -1
    for i, f in enumerate(existing):
        arr  = np.load(str(f))            # (30, 39)
        mean = arr.mean(axis=0)           # (39,)
        sim  = cosine_similarity(new_mean, mean)
        if sim > max_sim:
            max_sim = sim
            max_idx = i

    is_dup = max_sim > 0.97
    return is_dup, round(max_sim, 4), max_idx


def compute_variety_score(class_name: str) -> float:
    """
    Average pairwise cosine DISTANCE across all saved samples for class_name.
    distance = 1 - similarity
    0.0 = all identical  |  higher = more variety (good)
    Returns -1.0 if fewer than 2 samples.
    """
    d = DATASET_DIR / class_name
    if not d.exists():
        return -1.0
    files = list(d.glob("*.npy"))
    if len(files) < 2:
        return -1.0

    means = []
    for f in files:
        arr = np.load(str(f))
        means.append(arr.mean(axis=0))

    # Sample up to 40 pairs to keep it fast
    np.random.seed(42)
    n       = len(means)
    pairs   = [(i, j) for i in range(n) for j in range(i+1, n)]
    if len(pairs) > 40:
        pairs = [pairs[k] for k in np.random.choice(len(pairs), 40, replace=False)]

    distances = [1.0 - cosine_similarity(means[i], means[j]) for i, j in pairs]
    return round(float(np.mean(distances)), 4)


def save_sample(frames: list, class_name: str) -> str:
    """
    For punches: pad/trim to SEQ_LEN, save raw pose (30, 39).
    For idle:    take a random SEQ_LEN-frame window — idle doesn't
                 have a single "peak" moment so we just sample it.
    """
    out_dir = DATASET_DIR / class_name
    out_dir.mkdir(parents=True, exist_ok=True)

    arr = np.array(frames, dtype=np.float32)

    if class_name == "idle":
        # For long idle holds, randomly sample multiple non-overlapping windows
        # so one long hold gives several training examples
        windows = []
        if len(arr) >= SEQ_LEN:
            # Extract up to 3 windows from a long hold
            step = max(SEQ_LEN, len(arr) // 3)
            for start in range(0, len(arr) - SEQ_LEN + 1, step):
                windows.append(arr[start:start + SEQ_LEN])
                if len(windows) >= 3:
                    break
        else:
            pad = np.tile(arr[-1:], (SEQ_LEN - len(arr), 1))
            windows.append(np.vstack([arr, pad]))

        saved_paths = []
        existing = list(out_dir.glob("live_*.npy"))
        idx_start = len(existing)
        for i, window in enumerate(windows):
            path = out_dir / f"live_{idx_start + i:05d}.npy"
            np.save(str(path), window)
            saved_paths.append(str(path))
        return saved_paths[0], len(windows)   # return first path + count

    else:
        # Punch: trim/pad to capture the peak
        if len(arr) < SEQ_LEN:
            pad = np.tile(arr[-1:], (SEQ_LEN - len(arr), 1))
            arr = np.vstack([arr, pad])
        if len(arr) > SEQ_LEN:
            start = (len(arr) - SEQ_LEN) // 2
            arr = arr[start:start + SEQ_LEN]

        existing = list(out_dir.glob("live_*.npy"))
        path = out_dir / f"live_{len(existing):05d}.npy"
        np.save(str(path), arr)
        return str(path), 1


# ──────────────────────────────────────────────────────────────────
#  QUICK RETRAIN
# ──────────────────────────────────────────────────────────────────
class QuickDataset(Dataset):
    def __init__(self):
        self.samples, self.labels = [], []
        skipped = 0
        for idx, cls in enumerate(CLASSES):
            d = DATASET_DIR / cls
            if not d.exists():
                continue
            for f in d.glob("*.npy"):
                arr = np.load(str(f))
                if arr.shape == (SEQ_LEN, BASE_FEATURES):
                    motion = compute_motion_features(arr)
                    self.samples.append(motion.astype(np.float32))
                    self.labels.append(idx)
                else:
                    skipped += 1
        if skipped > 0:
            print(f"  ⚠️  Skipped {skipped} files with wrong shape")

    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        return torch.tensor(self.samples[i]).T, torch.tensor(self.labels[i], dtype=torch.long)


def retrain_tcn(epochs=60, device_str=None):
    device = device_str or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🏋️  RETRAINING TCN on {device}  ({NUM_CLASSES} classes)...")

    ds = QuickDataset()
    if len(ds) < 20:
        print("  ❌ Not enough samples. Need at least 20 total.")
        return False

    label_counts = np.bincount([ds.labels[i] for i in range(len(ds))], minlength=NUM_CLASSES)
    print(f"  Samples per class:")
    for i, cls in enumerate(CLASSES):
        target  = CLASS_TARGETS[cls]
        status  = "✅" if label_counts[i] >= target else ("⚠️ " if label_counts[i] >= target//2 else "❌")
        print(f"    {status} {cls:<10}: {label_counts[i]:4d}  (target: {target}+)")

    # Check idle has samples — it's now mandatory
    if label_counts[IDLE_IDX] < 10:
        print(f"\n  ❌ Only {label_counts[IDLE_IDX]} idle samples — need at least 10.")
        print(f"     Hold 'I' while standing still to record idle samples first.")
        return False

    weights = torch.tensor(
        [1.0 / max(label_counts[i], 1) for i in range(NUM_CLASSES)],
        dtype=torch.float32
    )
    weights = weights / weights.sum() * NUM_CLASSES

    val_sz = max(1, int(len(ds) * 0.2))
    trn_sz = len(ds) - val_sz
    trn_ds, val_ds = random_split(ds, [trn_sz, val_sz])
    trn_dl = DataLoader(trn_ds, batch_size=16, shuffle=True,  num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    model     = TCN_Boxing(input_size=INPUT_SIZE, num_classes=NUM_CLASSES)
    device_t  = torch.device(device)
    model     = model.to(device_t)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device_t))
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    MODEL_DIR.mkdir(exist_ok=True)
    best_acc = 0.0

    for ep in range(1, epochs + 1):
        model.train()
        for x, y in trn_dl:
            x, y = x.to(device_t), y.to(device_t)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_dl:
                preds    = model(x.to(device_t)).argmax(1).cpu()
                correct += (preds == y).sum().item()
                total   += len(y)
        acc = correct / total if total > 0 else 0

        if acc > best_acc:
            best_acc = acc
            torch.save({
                "model_state": model.state_dict(),
                "val_acc":     acc,
                "classes":     CLASSES,
                "input_size":  INPUT_SIZE,
                "seq_len":     SEQ_LEN,
                "num_classes": NUM_CLASSES,
                "config":      {"window_size": SEQ_LEN, "landmarks_used": LANDMARKS_USED},
            }, MODEL_DIR / "tcn_boxing.pth")

        if ep % 10 == 0 or ep == epochs:
            print(f"  Epoch {ep:3d}/{epochs}  val_acc={acc:.1%}  best={best_acc:.1%}")

    print(f"\n  ✅ Done. Best val_acc = {best_acc:.1%}")
    print(f"  💾 Saved → models/tcn_boxing.pth\n")

    if best_acc < 0.70:
        print("  ⚠️  Accuracy below 70% — collect more samples before using!")
        print("       Focus on: idle (80+ samples) and cross from defensive stance.")
    elif best_acc < 0.85:
        print("  ⚠️  Accuracy OK. More varied idle samples will help most.")
    else:
        print("  🎯 Accuracy looks good. Ready for stage4!")
    return True


# ──────────────────────────────────────────────────────────────────
#  HUD DRAWING
# ──────────────────────────────────────────────────────────────────
def draw_hud(frame, current_class, recording, rec_frames,
             sample_counts, pose_ok, status_msg):
    h, w = frame.shape[:2]

    cv2.rectangle(frame, (0, 0), (w, 55), (10, 10, 20), -1)

    if current_class:
        col = CLASS_COLORS[current_class]
        label = f"CLASS: {current_class.upper()}"
        if current_class == "idle":
            label += "  (stand still)"
        cv2.putText(frame, label, (14, 36), cv2.FONT_HERSHEY_DUPLEX, 1.0, col, 2)

    # Sample counts — right side, with target indicators
    x_off = w - 210
    for i, cls in enumerate(CLASSES):
        col    = CLASS_COLORS[cls]
        cnt    = sample_counts.get(cls, 0)
        target = CLASS_TARGETS[cls]
        pct    = min(cnt / target, 1.0)
        status = "✓" if cnt >= target else ("~" if cnt >= target // 2 else "!")
        cv2.putText(frame, f"{cls[:4].upper()} {status}{cnt:3d}/{target}",
                    (x_off, 16 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1)

    # Recording indicator
    if recording and current_class:
        col   = CLASS_COLORS[current_class]
        pulse = int((time.time() * 4) % 2) == 0
        cv2.circle(frame, (14, h - 50), 8, (0, 0, 255) if pulse else (0, 0, 180), -1)

        pct   = min(rec_frames / MAX_FRAMES, 1.0)
        bar_w = int(pct * (w - 30))
        cv2.rectangle(frame, (14, h - 35), (14 + bar_w, h - 15), col, -1)
        cv2.rectangle(frame, (14, h - 35), (w - 14,     h - 15), col, 1)
        verb = "HOLDING STILL..." if current_class == "idle" else "HOLDING..."
        cv2.putText(frame, f"{verb} {rec_frames} frames",
                    (30, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    else:
        cv2.rectangle(frame, (0, h - 45), (w, h), (10, 10, 20), -1)
        cv2.putText(frame,
                    "J=Jab  C=Cross  H=Hook  U=Uppercut  I=Idle/Guard   R=Retrain  Q=Quit",
                    (10, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)

    if not pose_ok:
        cv2.putText(frame, "NO POSE DETECTED",
                    (14, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 100, 255), 2)

    if status_msg:
        cv2.putText(frame, status_msg,
                    (14, h - 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 100), 2)

    return frame


# ──────────────────────────────────────────────────────────────────
#  MAIN LOOP
# ──────────────────────────────────────────────────────────────────
def run(min_frames=MIN_FRAMES, auto_retrain=True):
    print("╔══════════════════════════════════════════════╗")
    print("║   BOXING AI — Live Trainer  (5 classes)      ║")
    print("╚══════════════════════════════════════════════╝\n")
    print("  HOLD a key while in position, release to save.\n")
    print("  Keys:")
    print("    J = Jab       C = Cross")
    print("    H = Hook      U = Uppercut")
    print("    I = Idle/Guard  ← HOLD while standing still")
    print("    R = Retrain   D = Stats   Q = Quit\n")
    print("  ⭐ Record idle FIRST and AIM FOR 80+ samples.")
    print("     Idle quality directly fixes false positives.\n")

    pose_model = mp.solutions.pose.Pose(
        static_image_mode=False, model_complexity=1, smooth_landmarks=True,
        min_detection_confidence=0.55, min_tracking_confidence=0.55,
    )
    mp_draw     = mp.solutions.drawing_utils
    mp_pose_sol = mp.solutions.pose

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    recording      = False
    current_class  = None
    record_buffer  = []
    sample_counts  = count_samples()
    status_msg     = ""
    status_timer   = 0.0
    session_counts = defaultdict(int)

    print(f"  Current dataset: { {c: sample_counts[c] for c in CLASSES} }\n")
    print("  🎥 Camera ready...\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame   = cv2.flip(frame, 1)
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_model.process(rgb)

        pose_ok = results.pose_landmarks is not None
        if pose_ok:
            mp_draw.draw_landmarks(
                frame, results.pose_landmarks, mp_pose_sol.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 100), thickness=2, circle_radius=4),
                mp_draw.DrawingSpec(color=(255, 255, 0), thickness=2),
            )

        kpts = extract_keypoints_from_results(results, LANDMARKS_USED)
        if recording and kpts is not None:
            # ── POSE QUALITY GATE ──────────────────────────────
            # LANDMARKS_USED = [0,11,12,13,14,15,16,...] so:
            #   left wrist  = index 5 → kpts[5*3+2] = kpts[17]
            #   right wrist = index 6 → kpts[6*3+2] = kpts[20]
            # Reject frame if both wrists are invisible (arm behind torso / out of frame)
            vis_l = kpts[17]   # left wrist visibility
            vis_r = kpts[20]   # right wrist visibility
            if max(vis_l, vis_r) >= 0.5:
                record_buffer.append(kpts)
            # else: silently skip corrupted/occluded frame

        if status_msg and (time.time() - status_timer) > 2.5:
            status_msg = ""

        frame = draw_hud(
            frame, current_class, recording, len(record_buffer),
            sample_counts, pose_ok, status_msg,
        )
        cv2.imshow("Boxing AI — Live Trainer", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('r'):
            print("\n🔄 Retraining...")
            cv2.destroyAllWindows()
            retrain_tcn(epochs=80)
            cv2.namedWindow("Boxing AI — Live Trainer")

        elif key == ord('d'):
            sample_counts = count_samples()
            print("\n📊 Dataset Status:")
            for cls in CLASSES:
                cnt    = sample_counts[cls]
                target = CLASS_TARGETS[cls]
                bar    = "█" * (cnt // 5)
                status = "✅" if cnt >= target else ("⚠️ " if cnt >= target//2 else "❌")
                print(f"  {status} {cls:<10}: {cnt:4d}/{target}  {bar}")
            print(f"  Session: { dict(session_counts) }\n")

        elif key in KEY_TO_CLASS:
            cls = KEY_TO_CLASS[key]
            if not recording:
                recording     = True
                current_class = cls
                record_buffer = []

        else:
            # Key released — save if we were recording
            if recording and current_class:
                # Idle uses its own minimum frame count
                needed = IDLE_MIN_FRAMES if current_class == "idle" else min_frames

                if len(record_buffer) >= needed:
                    # ── VARIETY WARNING (before save) ──────────────
                    is_dup, max_sim, _ = check_variety(record_buffer, current_class)
                    if is_dup:
                        print(f"  ⚠️  [{current_class.upper()}] Very similar to existing sample "
                              f"(similarity={max_sim:.3f}) — try different stance/speed/distance")
                        status_msg   = f"Similar to existing! Try vary your position"
                        status_timer = time.time()

                    _, n_saved = save_sample(record_buffer, current_class)
                    sample_counts[current_class] += n_saved
                    session_counts[current_class] += n_saved
                    total = sample_counts[current_class]

                    extra = f" ({n_saved} windows)" if current_class == "idle" and n_saved > 1 else ""
                    dup_tag = " ⚠️ SIMILAR" if is_dup else ""
                    print(f"  ✅ [{current_class.upper()}] saved{extra}{dup_tag} → #{total} total")
                    if not is_dup:
                        status_msg   = f"✓ {current_class.upper()} saved! Total: {total}"
                        status_timer = time.time()
                else:
                    tip = "Hold longer — stay still!" if current_class == "idle" else "Hold longer next time"
                    print(f"  ⚠️  Too short ({len(record_buffer)} frames < {needed}) — not saved")
                    status_msg   = tip
                    status_timer = time.time()

                recording     = False
                current_class = None
                record_buffer = []

    cap.release()
    cv2.destroyAllWindows()
    pose_model.close()

    # ── Session summary ───────────────────────────────────────
    print(f"\n{'═'*55}")
    print(f"  SESSION SUMMARY")
    print(f"{'═'*55}")
    final = count_samples()
    for cls in CLASSES:
        added  = session_counts.get(cls, 0)
        total  = final[cls]
        target = CLASS_TARGETS[cls]
        status = "✅" if total >= target else ("⚠️ " if total >= target//2 else "❌")
        print(f"  {status} {cls:<10}: +{added:3d}  →  {total:4d}/{target}")

    # ── PER-CLASS VARIETY SCORE ───────────────────────────────
    # Average pairwise cosine distance across all saved samples.
    # Higher = more variety = better model generalisation.
    # Low score means you recorded too many similar samples.
    print(f"\n  VARIETY SCORES (higher = better, aim for > 0.05):")
    for cls in CLASSES:
        score = compute_variety_score(cls)
        if score < 0:
            print(f"    {cls:<10}: not enough samples")
        elif score < 0.02:
            print(f"    {cls:<10}: {score:.4f}  ❌ Very low — too many similar samples")
        elif score < 0.05:
            print(f"    {cls:<10}: {score:.4f}  ⚠️  Low — try more varied positions/speeds")
        else:
            print(f"    {cls:<10}: {score:.4f}  ✅ Good variety")

    total_added = sum(session_counts.values())
    if total_added > 0 and auto_retrain:
        print(f"\n  Added {total_added} samples this session.")

        idle_total = final["idle"]
        if idle_total < 10:
            print(f"  ⚠️  WARNING: Only {idle_total} idle samples.")
            print(f"     Retraining without enough idle will hurt recognition quality.")
            print(f"     Consider recording more idle (hold I while standing still).")

        ans = input("  Retrain TCN now? [Y/n]: ").strip().lower()
        if ans != 'n':
            retrain_tcn(epochs=80)
            print("  ✅ Retrained! Run stage4_realtime.py to test.")
        else:
            print("  Run 'python stage2_train_tcn.py' when ready.")
    else:
        print("  Run 'python stage2_train_tcn.py' to retrain.")


# ──────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Boxing AI — Live Trainer")
    parser.add_argument("--min_frames", type=int, default=MIN_FRAMES)
    parser.add_argument("--no_retrain", action="store_true")
    args = parser.parse_args()
    run(min_frames=args.min_frames, auto_retrain=not args.no_retrain)