"""
╔══════════════════════════════════════════════════════════════╗
║   BOXING AI — VIDEO → DATASET EXTRACTOR                      ║
║   Handles multi-rep videos (15-20 reps per file)             ║
╚══════════════════════════════════════════════════════════════╝

IMPORTANT: Each video contains 15-20 repetitions of the SAME move
with pauses between reps (e.g. jab01.mp4 = 15-20 jabs).

HOW IT WORKS:
  1. Run MediaPipe on every frame → extract wrist keypoints
  2. Compute wrist velocity (speed of movement) per frame
  3. Find PEAKS in velocity → each peak = one punch rep
  4. Extract SEQ_LEN frames centred on each peak → one .npy sample
  5. Skip frames during pauses (low velocity = idle, not saved)

This gives you ONE clean sample per rep, so 15-20 reps per video
= 15-20 training samples per file. No idle contamination.

Expected folder structure:
    videos/
      jab/       jab01.mp4, jab02.mp4, ...
      cross/     cross01.mp4, ...
      hook/      hook01.mp4, ...
      uppercut/  uppercut01.mp4, ...
      idle/      idle01.mp4, ...  (optional)

Run:
    python extract_from_videos.py
    python extract_from_videos.py --dry_run      ← preview only, no files written
    python extract_from_videos.py --sensitivity 0.3  ← lower = detect weaker punches
    python extract_from_videos.py --video_dir my_vids
"""

import cv2
import mediapipe as mp
import numpy as np
import argparse
from pathlib import Path
from scipy.signal import find_peaks   # pip install scipy

from models_def import (
    CLASSES, SEQ_LEN, BASE_FEATURES,
    LANDMARKS_USED, extract_keypoints_from_results,
)

VIDEO_DIR   = Path("videos")
DATASET_DIR = Path("dataset")

# ── Wrist landmark indices in LANDMARKS_USED ──────────────────
# LANDMARKS_USED = [0,11,12,13,14,15,16,17,18,19,20,23,24]
# Index 15 = left wrist, Index 16 = right wrist in MediaPipe
# In our 13-landmark array:
#   idx 5 = landmark 15 (left wrist)
#   idx 6 = landmark 16 (right wrist)
# Each landmark occupies 3 values (x, y, visibility) in the flat array
# So in the (30, 39) array:
#   left wrist x,y  = columns [15, 16]
#   right wrist x,y = columns [18, 19]
L_WRIST_X = 5 * 3      # = 15
L_WRIST_Y = 5 * 3 + 1  # = 16
R_WRIST_X = 6 * 3      # = 18
R_WRIST_Y = 6 * 3 + 1  # = 19


def compute_wrist_speed(frames: np.ndarray) -> np.ndarray:
    """
    frames: (N, 39) raw pose array
    Returns: (N,) float array of combined wrist speed per frame.
    Speed = magnitude of wrist displacement between frames.
    Uses both wrists and takes the max (whichever hand is punching).
    """
    N = len(frames)
    if N < 2:
        return np.zeros(N)

    speed = np.zeros(N)
    for i in range(1, N):
        # Left wrist displacement
        lx = frames[i, L_WRIST_X] - frames[i-1, L_WRIST_X]
        ly = frames[i, L_WRIST_Y] - frames[i-1, L_WRIST_Y]
        l_speed = np.sqrt(lx**2 + ly**2)

        # Right wrist displacement
        rx = frames[i, R_WRIST_X] - frames[i-1, R_WRIST_X]
        ry = frames[i, R_WRIST_Y] - frames[i-1, R_WRIST_Y]
        r_speed = np.sqrt(rx**2 + ry**2)

        # Take max of both wrists
        speed[i] = max(l_speed, r_speed)

    # Smooth with a 3-frame window to reduce noise
    kernel = np.ones(3) / 3
    speed  = np.convolve(speed, kernel, mode='same')
    return speed


def detect_rep_peaks(speed: np.ndarray, sensitivity: float = 0.5) -> list[int]:
    """
    Find the frame index of each punch rep peak.

    sensitivity: 0.0-1.0
      Higher = only detect strong/fast punches
      Lower  = also detect slower/lighter punches
      Default 0.5 works well for most boxing videos.

    Returns list of frame indices, one per detected rep.
    """
    if len(speed) == 0:
        return []

    # Threshold: fraction of the max speed
    # At sensitivity=0.5: peak must be > 35% of max speed
    # At sensitivity=0.3: peak must be > 20% of max speed
    threshold = np.max(speed) * (0.15 + sensitivity * 0.4)

    # min_distance: minimum frames between two separate reps
    # At 30fps, fastest realistic combo = 0.4s apart = 12 frames
    min_distance = max(12, SEQ_LEN // 2)

    peaks, props = find_peaks(
        speed,
        height=threshold,
        distance=min_distance,
        prominence=threshold * 0.3,   # each peak must stand out from surroundings
    )
    return list(peaks)


def extract_window_at(frames: np.ndarray, peak: int) -> np.ndarray:
    """
    Extract SEQ_LEN frames centred on the peak frame.
    The peak is the moment of maximum speed (impact/extension).
    We want frames BEFORE the peak (wind-up) and AFTER (retraction).
    Centre: peak at 60% of the window (slightly after mid) to capture
    more of the wind-up than the retraction.
    """
    N      = len(frames)
    before = int(SEQ_LEN * 0.60)   # 18 frames before peak
    after  = SEQ_LEN - before      # 12 frames after peak

    start = peak - before
    end   = peak + after

    # If window goes out of bounds, clamp and pad
    if start < 0:
        pad_front = -start
        start     = 0
    else:
        pad_front = 0

    if end > N:
        pad_back = end - N
        end      = N
    else:
        pad_back = 0

    window = frames[start:end]

    # Pad by repeating edge frames
    if pad_front > 0:
        window = np.vstack([np.tile(window[:1], (pad_front, 1)), window])
    if pad_back > 0:
        window = np.vstack([window, np.tile(window[-1:], (pad_back, 1))])

    assert window.shape == (SEQ_LEN, BASE_FEATURES), \
        f"Window shape {window.shape} != ({SEQ_LEN}, {BASE_FEATURES})"
    return window.astype(np.float32)


def process_video(video_path: Path, class_name: str,
                  pose_model, sensitivity: float,
                  dry_run: bool = False) -> int:
    """
    Full pipeline for one video:
    1. Extract all pose frames
    2. Compute wrist speed
    3. Detect rep peaks
    4. Extract window per peak
    5. Save .npy files

    Returns number of samples saved (or would be saved in dry_run).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"    ⚠️  Cannot open: {video_path.name}")
        return 0

    fps_vid = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_model.process(rgb)
        kpts    = extract_keypoints_from_results(results, LANDMARKS_USED)
        # If pose not detected, interpolate by repeating last frame
        if kpts is not None:
            all_frames.append(kpts)
        elif all_frames:
            all_frames.append(all_frames[-1].copy())   # repeat last known pose
    cap.release()

    if len(all_frames) < SEQ_LEN:
        print(f"    ⚠️  {video_path.name}: only {len(all_frames)} pose frames — skipping")
        return 0

    frames_arr = np.array(all_frames, dtype=np.float32)

    # Compute wrist speed and find rep peaks
    speed = compute_wrist_speed(frames_arr)
    peaks = detect_rep_peaks(speed, sensitivity=sensitivity)

    if not peaks:
        print(f"    ⚠️  {video_path.name}: no reps detected "
              f"(try --sensitivity {max(0.1, sensitivity-0.1):.1f})")
        return 0

    duration = len(all_frames) / fps_vid
    print(f"    📹 {video_path.name}: "
          f"{len(all_frames)} frames ({duration:.1f}s) → "
          f"{len(peaks)} reps detected at frames {peaks}")

    if dry_run:
        return len(peaks)

    # Save one .npy per rep
    out_dir  = DATASET_DIR / class_name
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(out_dir.glob("video_*.npy"))
    idx      = len(existing)
    stem     = video_path.stem[:20]
    saved    = 0

    for i, peak in enumerate(peaks):
        window = extract_window_at(frames_arr, peak)
        path   = out_dir / f"video_{stem}_rep{i:02d}.npy"
        np.save(str(path), window)
        saved += 1

    print(f"       → saved {saved} samples")
    return saved


def run(video_dir: str = "videos", sensitivity: float = 0.5, dry_run: bool = False):
    video_root = Path(video_dir)

    if not video_root.exists():
        print(f"❌ Video folder not found: {video_root.resolve()}")
        print(f"\n  Expected structure:")
        for cls in CLASSES:
            print(f"    {video_dir}/{cls}/   ← put .mp4 files here")
        return

    # Check scipy is available
    try:
        from scipy.signal import find_peaks
    except ImportError:
        print("❌ scipy not installed. Run:  pip install scipy")
        return

    print("╔══════════════════════════════════════════════╗")
    print("║   BOXING AI — Video → Dataset Extractor      ║")
    print("║   Mode: multi-rep detection                  ║")
    if dry_run:
        print("║   DRY RUN — no files written                 ║")
    print("╚══════════════════════════════════════════════╝\n")
    print(f"  Video dir  : {video_root.resolve()}")
    print(f"  Dataset    : {DATASET_DIR.resolve()}")
    print(f"  Sensitivity: {sensitivity}  (higher = stricter peak detection)")
    print(f"  Window     : {SEQ_LEN} frames per rep (peak at 60%)\n")
    print(f"  Each video should have 15-20 reps of the same move.\n")

    pose_model = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    total_added  = 0
    class_counts = {}

    for cls in CLASSES:
        cls_dir = video_root / cls
        if not cls_dir.exists():
            print(f"  ⚠️  No folder: {cls_dir}  — skipping {cls}")
            class_counts[cls] = 0
            continue

        videos = (list(cls_dir.glob("*.mp4")) +
                  list(cls_dir.glob("*.MP4")) +
                  list(cls_dir.glob("*.avi")) +
                  list(cls_dir.glob("*.mov")))

        if not videos:
            print(f"  ⚠️  No video files in {cls_dir}")
            class_counts[cls] = 0
            continue

        print(f"  ── {cls.upper()} ({len(videos)} video{'s' if len(videos)>1 else ''}) ──")
        cls_added = 0

        for vp in sorted(videos):
            n = process_video(vp, cls, pose_model, sensitivity, dry_run=dry_run)
            cls_added += n

        class_counts[cls] = cls_added
        total_added       += cls_added

        # Count existing live samples
        existing_live = len(list((DATASET_DIR / cls).glob("live_*.npy"))) \
                        if (DATASET_DIR / cls).exists() else 0
        print(f"  → {cls_added} video samples {'would be ' if dry_run else ''}added  "
              f"({existing_live} live samples already exist)\n")

    pose_model.close()

    # ── Summary ───────────────────────────────────────────────
    print(f"{'═'*55}")
    print(f"  SUMMARY {'(DRY RUN — nothing written)' if dry_run else ''}")
    print(f"{'═'*55}")
    for cls in CLASSES:
        n        = class_counts.get(cls, 0)
        existing = len(list((DATASET_DIR / cls).glob("*.npy"))) \
                   if (DATASET_DIR / cls).exists() else 0
        total    = existing + (n if not dry_run else 0)
        target   = 80 if cls == "idle" else 60
        status   = "✅" if total >= target else ("⚠️ " if total >= target//2 else "❌")
        print(f"  {status} {cls:<10}: +{n:3d} from video  →  {total:4d} total  (target {target}+)")

    print(f"\n  Total new samples: {total_added}")

    if not dry_run and total_added > 0:
        print(f"\n  ✅ Done! Next steps:")
        print(f"     1. python live_trainer.py    ← collect idle samples (hold I)")
        print(f"     2. python stage2_train_tcn.py ← retrain with all data")
    elif dry_run:
        print(f"\n  Run without --dry_run to actually extract.")
        print(f"  If rep count looks wrong, adjust --sensitivity")
        print(f"    Too few reps detected  → lower  (e.g. --sensitivity 0.3)")
        print(f"    Too many reps detected → higher (e.g. --sensitivity 0.7)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract individual punch reps from multi-rep labelled videos")
    parser.add_argument("--video_dir",   default="videos",
                        help="Root folder with class subfolders (default: videos)")
    parser.add_argument("--sensitivity", type=float, default=0.5,
                        help="Peak detection sensitivity 0.1-0.9 (default: 0.5). "
                             "Lower = detect weaker/slower punches.")
    parser.add_argument("--dry_run",     action="store_true",
                        help="Preview rep counts without writing any files")
    args = parser.parse_args()
    run(video_dir=args.video_dir, sensitivity=args.sensitivity, dry_run=args.dry_run)