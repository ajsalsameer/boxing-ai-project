import os
import cv2
import pandas as pd
import numpy as np
import pickle
from ultralytics import YOLO
from tqdm import tqdm

# ==========================================
#        SURGICAL SLICER CONFIG
# ==========================================
DATASET_ROOT = r"data\external\BoxingVI"
VIDEO_DIR = os.path.join(DATASET_ROOT, "RGB_videos")
ANNOT_DIR = os.path.join(DATASET_ROOT, "Annotation_files")

OUTPUT_FILE = "data/boxing_ultra_dataset.pkl"
MODEL_NAME = "yolo11n-pose.pt"

# Map Excel Labels to God-Tier Classes
LABEL_MAP = {
    "jab": "jab",
    "cross": "cross", "straight": "cross", "punch": "cross",
    "lead hook": "hook", "rear hook": "hook", "hook": "hook",
    "lead uppercut": "uppercut", "rear uppercut": "uppercut", "uppercut": "uppercut",
    "overhand": "overhand",
    "slip": "slip_left", "duck": "duck", "block": "block"
}

print(f"Loading {MODEL_NAME}...")
model = YOLO(MODEL_NAME)

# Load Existing Data
try:
    with open(OUTPUT_FILE, "rb") as f:
        X, y = pickle.load(f)
    print(f"Loaded existing dataset: {len(X)} clips.")
except:
    X, y = [], []

new_data = []
new_labels = []

# Get list of Excel files (V1.xlsx, V2.xlsx...)
annot_files = [f for f in os.listdir(ANNOT_DIR) if f.endswith(".xlsx")]

print(f"Found {len(annot_files)} annotation files. Starting Surgery...")

for annot_file in annot_files:
    # 1. MATCH EXCEL TO VIDEO
    file_id = os.path.splitext(annot_file)[0] # "V1"
    video_path = os.path.join(VIDEO_DIR, f"{file_id}.mp4")
    excel_path = os.path.join(ANNOT_DIR, annot_file)

    if not os.path.exists(video_path):
        print(f"⚠ Warning: Video {file_id}.mp4 not found. Skipping.")
        continue

    # 2. READ EXCEL
    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        print(f"❌ Error reading {annot_file}: {e}")
        continue

    # 3. AUTO-DETECT COLUMNS
    # We look for keywords like "start", "end", "label", "class"
    start_col, end_col, label_col = None, None, None
    
    for col in df.columns:
        c = str(col).lower()
        if "start" in c or "begin" in c or "init" in c: start_col = col
        if "end" in c or "finish" in c or "final" in c: end_col = col
        if "label" in c or "class" in c or "type" in c: label_col = col

    if not (start_col and end_col and label_col):
        print(f"❌ Could not understand columns in {annot_file}.")
        print(f"   Columns found: {list(df.columns)}")
        print("   Please rename columns in Excel to: 'Start', 'End', 'Label'")
        continue

    print(f"🔪 Slicing {file_id}.mp4 ({len(df)} moves)...")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 4. PROCESS EACH ROW (Each Punch)
    for index, row in tqdm(df.iterrows(), total=len(df), leave=False):
        raw_label = str(row[label_col]).lower()
        
        # Clean Label
        final_label = None
        for k, v in LABEL_MAP.items():
            if k in raw_label:
                final_label = v
                break
        if not final_label: continue

        # Get Start/End
        # IMPORTANT: Check if Excel uses Frames or Seconds
        # Most datasets use Frames. If numbers are small (< 600) might be seconds.
        # Assuming Frames for now.
        try:
            start_frame = int(row[start_col])
            end_frame = int(row[end_col])
        except:
            continue # Skip bad data

        if end_frame > total_frames: end_frame = total_frames
        if start_frame >= end_frame: continue

        # 5. EXTRACT FRAMES FOR THIS PUNCH
        # Jump to start
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        sequence = []
        prev_kpts = None
        prev_vel = None
        
        # Read until End Frame
        for f_idx in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret: break
            
            # YOLO
            results = model(frame, verbose=False, half=True, device=0)[0]
            if len(results.keypoints.xy) > 0:
                h, w = frame.shape[:2]
                kpts = results.keypoints.xy[0].cpu().numpy().flatten() / np.array([w, h] * 17)
                
                vel = kpts - prev_kpts if prev_kpts is not None else np.zeros_like(kpts)
                acc = vel - prev_vel if 'prev_vel' in locals() and prev_vel is not None else np.zeros_like(vel)
                
                feats = np.concatenate([kpts, vel, acc, np.zeros(6)])
                sequence.append(feats)
                prev_kpts = kpts; prev_vel = vel
        
        # 6. SAVE IF VALID
        # We need exactly 30 frames for the brain.
        # If clip is short, we pad. If long, we take middle.
        if len(sequence) > 10: # Minimum 10 frames to be valid
            # Resample or Slice to exactly 30 frames
            if len(sequence) >= 30:
                # Take the middle 30 frames (usually the "hit")
                mid = len(sequence) // 2
                final_seq = sequence[mid-15 : mid+15]
            else:
                # Pad with zeros if too short (rare)
                pad = [np.zeros_like(sequence[0])] * (30 - len(sequence))
                final_seq = sequence + pad
            
            new_data.append(final_seq)
            new_labels.append(final_label)

    cap.release()

# SAVE
if len(new_data) > 0:
    X_final = np.concatenate([X, np.array(new_data)]) if len(X) > 0 else np.array(new_data)
    y_final = np.concatenate([y, np.array(new_labels)]) if len(y) > 0 else np.array(new_labels)
    
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump((X_final, y_final), f)
    print(f"\n✅ SUCCESS! Extracted {len(new_data)} specific moves from the timeline.")
    print(f"   Total Dataset Size: {len(X_final)}")
else:
    print("\n❌ No moves extracted. Check Excel Column names!")