import os
import cv2
import numpy as np
import pickle
from ultralytics import YOLO
from tqdm import tqdm

# ==========================================
#   AUTO-SLICER (Compatible with FAST SORTER)
# ==========================================
# This now looks recursively into subfolders (jab/, cross/, etc.)
VIDEO_ROOT = r"data\raw_videos\self_recorded" 
OUTPUT_FILE = "data/boxing_ultra_dataset.pkl"
MODEL_NAME = "yolo11n-pose.pt"

MOTION_THRESHOLD = 0.02
COOLDOWN_FRAMES = 12

# Explicit Mapping for your Sorter's folder names
LABEL_MAP = {
    "jab": "jab",
    "cross": "cross",
    "hook": "hook",       # Catches "hook", "left_hook", "right_hook" filenames
    "uppercut": "uppercut",
    "slip": "slip_left",
    "duck": "duck",
    "block": "block",
    "idle": "idle"
}

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    denominator = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6
    cosine_angle = np.dot(ba, bc) / denominator
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

print(f"Loading {MODEL_NAME}...")
model = YOLO(MODEL_NAME)

try:
    with open(OUTPUT_FILE, "rb") as f:
        X_exist, y_exist = pickle.load(f)
    print(f"Merging with existing dataset: {len(X_exist)} clips")
except:
    X_exist, y_exist = [], []
    print("No existing data — starting fresh")

new_data = []
new_labels = []

# --- UPDATED FILE FINDER (Recursive) ---
video_files = []
for root, dirs, files in os.walk(VIDEO_ROOT):
    for file in files:
        if file.lower().endswith(".mp4"):
            video_files.append(os.path.join(root, file))

print(f"Found {len(video_files)} videos in {VIDEO_ROOT}...")

for vid_path in video_files:
    # 1. Detect Label from FOLDER NAME or FILENAME
    # This makes it compatible with your sorter folders
    path_parts = vid_path.lower().replace("\\", "/").split("/")
    folder_name = path_parts[-2] # e.g., "jab" from "self_recorded/jab/vid.mp4"
    filename = path_parts[-1]
    
    label = "unknown"
    
    # Check folder name first (High Priority)
    if folder_name in LABEL_MAP:
        label = LABEL_MAP[folder_name]
    else:
        # Check filename as backup
        for key, val in LABEL_MAP.items():
            if key in filename:
                label = val
                break
    
    if label == "unknown":
        print(f"⚠ Skipping {filename}: Unknown label.")
        continue

    print(f"Processing {filename} as '{label.upper()}'...")
    
    cap = cv2.VideoCapture(vid_path)
    
    is_recording = False
    cooldown_counter = 0
    current_clip = []
    prev_kpts = None
    prev_vel = None
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for _ in tqdm(range(frame_count), desc="Scanning", leave=False):
        ret, frame = cap.read()
        if not ret: break
        
        results = model.predict(frame, verbose=False, half=True, device=0)
        if not results or len(results[0].keypoints.xy) == 0: continue
        
        kpts_xy = results[0].keypoints.xy[0].cpu().numpy()
        h, w = frame.shape[:2]
        kpts_norm = kpts_xy.flatten() / np.array([w, h] * 17)
        
        vel = kpts_norm - prev_kpts if prev_kpts is not None else np.zeros_like(kpts_norm)
        acc = vel - prev_vel if prev_vel is not None else np.zeros_like(vel)
        
        wrist_speed = np.linalg.norm(vel[18:20]) + np.linalg.norm(vel[20:22])
        
        angles = np.zeros(5)
        if len(kpts_xy) >= 13:
            angles[0] = calculate_angle(kpts_xy[5], kpts_xy[7], kpts_xy[9])
            angles[1] = calculate_angle(kpts_xy[6], kpts_xy[8], kpts_xy[10])
            angles[2] = calculate_angle(kpts_xy[11], kpts_xy[5], kpts_xy[7])
            angles[3] = calculate_angle(kpts_xy[12], kpts_xy[6], kpts_xy[8])
            angles[4] = np.linalg.norm(kpts_xy[9]-kpts_xy[10]) / (np.linalg.norm(kpts_xy[5]-kpts_xy[6]) + 1e-6)

        energy_val = np.sum(np.abs(acc)) / len(acc)
        feats = np.concatenate([kpts_norm, vel, acc, angles, [energy_val]])
        
        if not is_recording:
            if wrist_speed > MOTION_THRESHOLD:
                is_recording = True
                current_clip = [feats]
                cooldown_counter = 0
        else:
            current_clip.append(feats)
            if wrist_speed < MOTION_THRESHOLD: cooldown_counter += 1
            else: cooldown_counter = 0 
            
            if cooldown_counter > COOLDOWN_FRAMES:
                is_recording = False
                if len(current_clip) > 8:
                    seq_array = np.array(current_clip)
                    if len(seq_array) >= 30:
                        mid = len(seq_array) // 2
                        final_seq = seq_array[mid-15:mid+15]
                    else:
                        pad_len = 30 - len(seq_array)
                        pad = np.zeros((pad_len, feats.shape[0]))
                        final_seq = np.vstack([seq_array, pad])
                    
                    new_data.append(final_seq)
                    new_labels.append(label)
                current_clip = []

        prev_kpts = kpts_norm
        prev_vel = vel

    cap.release()

if new_data:
    X_new = np.array(new_data)
    y_new = np.array(new_labels)
    
    X_final = np.concatenate([X_exist, X_new], axis=0) if len(X_exist) > 0 else X_new
    y_final = np.concatenate([y_exist, y_new], axis=0) if len(y_exist) > 0 else y_new
    
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump((X_final, y_final), f)
    
    print(f"\n✅ SUCCESS! Added {len(new_data)} clips from your sorted folders.")
else:
    print("❌ No clips found.")