"""
╔══════════════════════════════════════════════════════════════╗
║       BOXING AI - STAGE 1: DATA COLLECTION & SEGMENTATION   ║
║       Works with GTX 1650 | MediaPipe + Auto-Segmentation    ║
╚══════════════════════════════════════════════════════════════╝

HOW TO USE:
-----------
MODE A - Process your existing videos:
    python stage1_data_collector.py --mode extract --video_dir "path/to/your/videos"

MODE B - Record new samples live from camera:
    python stage1_data_collector.py --mode record

MODE C - Review what you collected:
    python stage1_data_collector.py --mode review

FOLDER STRUCTURE EXPECTED FOR YOUR VIDEOS:
    videos/
        jab/         ← put your jab videos here
        cross/       ← put your cross videos here
        hook/        ← put your hook videos here
        uppercut/    ← put your uppercut videos here

OUTPUT:
    dataset/
        jab/         ← auto-segmented .npy pose files
        cross/
        hook/
        uppercut/
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import argparse
import time
import json
from pathlib import Path

# ─────────────────────────────────────────────
#  CONFIGURATION — tweak these if needed
# ─────────────────────────────────────────────
CONFIG = {
    "window_size": 30,          # frames per sample (30 = 1 sec at 30fps)
    "overlap": 15,              # overlap between windows when auto-segmenting
    "min_movement_thresh": 0.015,  # sensitivity for detecting a rep started
    "classes": ["jab", "cross", "hook", "uppercut"],
    "output_dir": "dataset",
    "landmarks_used": [         # only upper body — reduces noise
        0,   # nose
        11, 12,  # shoulders
        13, 14,  # elbows
        15, 16,  # wrists
        17, 18,  # pinky knuckles
        19, 20,  # index knuckles
        23, 24,  # hips (used for normalization center)
    ]
}

# ─────────────────────────────────────────────
#  MEDIAPIPE SETUP
# ─────────────────────────────────────────────
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

def get_pose_model():
    return mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,          # 0=fast, 1=balanced, 2=accurate
        smooth_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

# ─────────────────────────────────────────────
#  KEYPOINT EXTRACTION
# ─────────────────────────────────────────────
def extract_keypoints(results):
    """
    Extract and NORMALIZE keypoints from MediaPipe results.
    Normalization = subtract hip center so position doesn't matter.
    Returns a flat array of shape (num_landmarks * 3,) — x, y, visibility
    """
    if not results.pose_landmarks:
        return None

    landmarks = results.pose_landmarks.landmark
    lm_array = np.array([[lm.x, lm.y, lm.visibility] for lm in landmarks])

    # ── Hip-center normalization (THE KEY FIX for camera distance) ──
    left_hip  = lm_array[23, :2]
    right_hip = lm_array[24, :2]
    hip_center = (left_hip + right_hip) / 2.0

    # Normalize XY relative to hip center
    lm_array[:, 0] -= hip_center[0]
    lm_array[:, 1] -= hip_center[1]

    # Extract only the landmarks we care about
    selected = lm_array[CONFIG["landmarks_used"]]  # shape: (15, 3)
    return selected.flatten()  # shape: (45,)

def compute_wrist_velocity(frame_buffer):
    """Compute how fast the wrists are moving — used to detect punch start."""
    if len(frame_buffer) < 2:
        return 0.0
    prev = frame_buffer[-2]
    curr = frame_buffer[-1]
    # wrist indices in our extracted vector: landmarks 6,7 (left/right wrist)
    # each landmark = 3 values (x, y, vis), wrist = index 6*3=18 and 7*3=21
    prev_wrists = prev[[18, 19, 21, 22]]  # x,y of both wrists
    curr_wrists = curr[[18, 19, 21, 22]]
    velocity = np.linalg.norm(curr_wrists - prev_wrists)
    return velocity

# ─────────────────────────────────────────────
#  MODE A: EXTRACT FROM EXISTING VIDEOS
# ─────────────────────────────────────────────
def extract_from_videos(video_dir):
    """
    Processes all videos in the folder structure:
        video_dir/jab/*.mp4
        video_dir/cross/*.mp4
        etc.
    Auto-segments using sliding window + movement detection.
    """
    print("\n🥊 STAGE 1: Extracting pose data from your videos...\n")
    pose = get_pose_model()
    total_samples = 0

    for class_name in CONFIG["classes"]:
        class_video_dir = Path(video_dir) / class_name
        class_out_dir   = Path(CONFIG["output_dir"]) / class_name
        class_out_dir.mkdir(parents=True, exist_ok=True)

        if not class_video_dir.exists():
            print(f"  ⚠️  Folder not found: {class_video_dir} — skipping")
            continue

        video_files = list(class_video_dir.glob("*.mp4")) + \
                      list(class_video_dir.glob("*.avi")) + \
                      list(class_video_dir.glob("*.mov"))

        if not video_files:
            print(f"  ⚠️  No videos found in {class_video_dir} — skipping")
            continue

        print(f"  📂 Processing [{class_name.upper()}] — {len(video_files)} video(s)")
        class_samples = 0

        for vid_path in video_files:
            cap = cv2.VideoCapture(str(vid_path))
            frame_buffer = []    # rolling buffer of keypoint arrays
            sample_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Flip for mirror effect + convert color
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)
                keypoints = extract_keypoints(results)

                if keypoints is None:
                    # No pose detected — reset buffer
                    frame_buffer = []
                    continue

                frame_buffer.append(keypoints)

                # ── Sliding window segmentation ──
                if len(frame_buffer) >= CONFIG["window_size"]:
                    window = np.array(frame_buffer[-CONFIG["window_size"]:])

                    # Only save if there's meaningful movement in this window
                    velocity = compute_wrist_velocity(frame_buffer)
                    if velocity > CONFIG["min_movement_thresh"]:
                        out_path = class_out_dir / f"{vid_path.stem}_sample_{sample_count:04d}.npy"
                        np.save(str(out_path), window)
                        sample_count += 1

                    # Slide window forward by overlap amount
                    frame_buffer = frame_buffer[CONFIG["overlap"]:]

            cap.release()
            class_samples += sample_count
            print(f"    ✅ {vid_path.name} → {sample_count} samples saved")

        total_samples += class_samples
        print(f"  Total [{class_name}]: {class_samples} samples\n")

    pose.close()
    print(f"✨ Done! Total samples collected: {total_samples}")
    print(f"📁 Saved to: {CONFIG['output_dir']}/\n")
    save_config()

# ─────────────────────────────────────────────
#  MODE B: LIVE RECORDING
# ─────────────────────────────────────────────
def record_live():
    """
    Opens your camera. You select a class, then press SPACE to record a rep.
    Shows a live skeleton overlay so you know pose is being detected.
    """
    print("\n🎥 LIVE RECORDING MODE")
    print("Controls:")
    print("  [1] Jab  [2] Cross  [3] Hook  [4] Uppercut")
    print("  [SPACE] Record one rep  [Q] Quit\n")

    pose = get_pose_model()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    current_class = "jab"
    class_map = {"1": "jab", "2": "cross", "3": "hook", "4": "uppercut"}
    recording = False
    record_buffer = []
    countdown = 0
    sample_counts = {c: len(list((Path(CONFIG["output_dir"]) / c).glob("*.npy")))
                     if (Path(CONFIG["output_dir"]) / c).exists() else 0
                     for c in CONFIG["classes"]}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        keypoints = extract_keypoints(results)

        # Draw skeleton
        if results.pose_landmarks:
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(255,255,0), thickness=2))

        # ── Recording logic ──
        if recording and keypoints is not None:
            record_buffer.append(keypoints)
            progress = len(record_buffer) / CONFIG["window_size"]
            bar = "█" * int(progress * 20) + "░" * (20 - int(progress * 20))
            cv2.putText(frame, f"REC [{bar}]", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if len(record_buffer) >= CONFIG["window_size"]:
                # Save sample
                out_dir = Path(CONFIG["output_dir"]) / current_class
                out_dir.mkdir(parents=True, exist_ok=True)
                idx = sample_counts[current_class]
                np.save(str(out_dir / f"live_{idx:04d}.npy"),
                        np.array(record_buffer[:CONFIG["window_size"]]))
                sample_counts[current_class] += 1
                print(f"  ✅ Saved [{current_class}] sample #{idx}")
                recording = False
                record_buffer = []

        # ── UI Overlay ──
        color = (0, 255, 0) if not recording else (0, 0, 255)
        cv2.rectangle(frame, (0, 0), (640, 40), (0, 0, 0), -1)
        cv2.putText(frame, f"Class: {current_class.upper()}  |  Samples: {sample_counts[current_class]}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if keypoints is None:
            cv2.putText(frame, "⚠ No pose detected", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        cv2.imshow("Boxing AI - Live Recording", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif chr(key) in class_map:
            current_class = class_map[chr(key)]
            print(f"  🥊 Switched to: {current_class}")
        elif key == ord(' ') and not recording and keypoints is not None:
            recording = True
            record_buffer = []
            print(f"  🔴 Recording [{current_class}]...")

    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    print(f"\n✅ Recording done. Samples per class: {sample_counts}")
    save_config()

# ─────────────────────────────────────────────
#  MODE C: REVIEW DATASET
# ─────────────────────────────────────────────
def review_dataset():
    """Print a summary of what's been collected so far."""
    print("\n📊 DATASET REVIEW\n")
    print(f"{'Class':<12} {'Samples':>8} {'Status':>10}")
    print("─" * 35)

    total = 0
    for class_name in CONFIG["classes"]:
        class_dir = Path(CONFIG["output_dir"]) / class_name
        if class_dir.exists():
            samples = list(class_dir.glob("*.npy"))
            count = len(samples)
            status = "✅ Good" if count >= 50 else ("⚠️ Low" if count >= 20 else "❌ Need more")
        else:
            count = 0
            status = "❌ Empty"
        total += count
        print(f"{class_name:<12} {count:>8}    {status}")

    print("─" * 35)
    print(f"{'TOTAL':<12} {total:>8}\n")
    print("Recommendation: aim for 50+ samples per class for reliable training.")
    print("200+ samples per class = great accuracy.\n")

# ─────────────────────────────────────────────
#  SAVE CONFIG FOR STAGE 2
# ─────────────────────────────────────────────
def save_config():
    config_path = Path(CONFIG["output_dir"]) / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(CONFIG, f, indent=2)
    print(f"⚙️  Config saved to {config_path} (used by Stage 2 trainer)")

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Boxing AI - Stage 1: Data Collection")
    parser.add_argument("--mode", choices=["extract", "record", "review"],
                        default="review", help="Which mode to run")
    parser.add_argument("--video_dir", default="videos",
                        help="Root folder containing class subfolders of videos")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════╗")
    print("║   BOXING AI — Stage 1: Data Collection   ║")
    print("╚══════════════════════════════════════════╝")

    if args.mode == "extract":
        extract_from_videos(args.video_dir)
    elif args.mode == "record":
        record_live()
    elif args.mode == "review":
        review_dataset()