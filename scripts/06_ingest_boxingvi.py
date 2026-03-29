import os
import cv2
import pandas as pd
import numpy as np
import pickle
import sys
from ultralytics import YOLO
from tqdm import tqdm

# ==========================================
#        FINAL BOXINGVI INGESTION (Visual)
# ==========================================
DATASET_ROOT = r"data\external\BoxingVI"
VIDEO_DIR = os.path.join(DATASET_ROOT, "RGB_videos")
ANNOT_DIR = os.path.join(DATASET_ROOT, "Annotation_files")
OUTPUT_FILE = "data/boxing_ultra_dataset.pkl"
MODEL_NAME = "yolo11n-pose.pt"

LABEL_MAP = {
    "jab": "jab",
    "cross": "cross", "straight": "cross", "rear straight": "cross",
    "lead hook": "hook", "rear hook": "hook", "hook": "hook",
    "lead uppercut": "uppercut", "rear uppercut": "uppercut", "uppercut": "uppercut",
    "overhand": "overhand",
    "slip": "slip_left", "roll": "duck", "duck": "duck", "weave": "duck",
    "block": "block", "guard": "block"
}

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    denominator = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6
    cosine_angle = np.dot(ba, bc) / denominator
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# Suppress YOLO logs
os.environ["YOLO_VERBOSE"] = "False"

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

annot_files = [f for f in os.listdir(ANNOT_DIR) if f.lower().endswith(('.xlsx', '.xls', '.ods'))]
print(f"Found {len(annot_files)} annotation files.")

# OUTER LOOP: Files
for annot_file in annot_files:
    video_id = os.path.splitext(annot_file)[0]
    video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
    
    if not os.path.exists(video_path): continue
    
    try:
        df = pd.read_excel(os.path.join(ANNOT_DIR, annot_file))
    except: continue
    
    # Column Detection
    start_col = end_col = label_col = None
    if 'Start_Frame' in df.columns: start_col = 'Start_Frame'
    if 'Ending_Frame' in df.columns: end_col = 'Ending_Frame'
    if 'Class' in df.columns: label_col = 'Class'
    
    if not (start_col and end_col and label_col):
        # Fallback auto-detect
        for col in df.columns:
            c = str(col).lower()
            if not start_col and "start" in c: start_col = col
            if not end_col and "end" in c: end_col = col
            if not label_col and "class" in c: label_col = col
            
    if not all([start_col, end_col, label_col]): continue
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nProcessing {annot_file} ({len(df)} punches)...")

    # INNER LOOP: Punches (Now with Progress Bar!)
    # This bar will move for every punch, so you know it's working.
    pbar = tqdm(df.iterrows(), total=len(df), unit="punch", leave=False)
    
    for _, row in pbar:
        try:
            start = int(row[start_col])
            end = int(row[end_col])
            raw_label = str(row[label_col]).strip().lower()
        except: continue
        
        if start >= end or end > total_frames or start < 0: continue
        
        label = "unknown"
        for k, v in LABEL_MAP.items():
            if k in raw_label:
                label = v
                break
        if label == "unknown": continue
        
        # Update Description so you see what it's learning
        pbar.set_description(f"Learning {label.upper()}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start))
        sequence = []
        prev_kpts = None
        prev_vel = None
        
        while cap.get(cv2.CAP_PROP_POS_FRAMES) < end:
            ret, frame = cap.read()
            if not ret or frame is None: break
            
            # Predict
            results = model.predict(frame, verbose=False, half=True, device=0)
            if not results or len(results[0].keypoints.xy) == 0: continue
            
            # Extract
            r = results[0]
            h, w = frame.shape[:2]
            kpts_xy = r.keypoints.xy[0].cpu().numpy()
            kpts_norm = kpts_xy.flatten() / np.array([w, h] * 17)
            
            vel = kpts_norm - prev_kpts if prev_kpts is not None else np.zeros_like(kpts_norm)
            acc = vel - prev_vel if prev_vel is not None else np.zeros_like(vel)
            
            angles = np.zeros(5)
            if len(kpts_xy) >= 13:
                angles[0] = calculate_angle(kpts_xy[5], kpts_xy[7], kpts_xy[9])
                angles[1] = calculate_angle(kpts_xy[6], kpts_xy[8], kpts_xy[10])
                angles[2] = calculate_angle(kpts_xy[11], kpts_xy[5], kpts_xy[7])
                angles[3] = calculate_angle(kpts_xy[12], kpts_xy[6], kpts_xy[8])
                angles[4] = np.linalg.norm(kpts_xy[9]-kpts_xy[10]) / (np.linalg.norm(kpts_xy[5]-kpts_xy[6]) + 1e-6)

            energy = np.sum(np.abs(acc)) / len(acc)
            feats = np.concatenate([kpts_norm, vel, acc, angles, [energy]])
            sequence.append(feats)
            prev_kpts = kpts_norm; prev_vel = vel
        
        # Save Window
        if len(sequence) >= 10:
            seq_array = np.array(sequence)
            if len(seq_array) >= 30:
                mid = len(seq_array) // 2
                final_seq = seq_array[mid-15:mid+15]
            else:
                pad_len = 30 - len(seq_array)
                pad = np.zeros((pad_len, feats.shape[0]))
                final_seq = np.vstack([seq_array, pad])
            new_data.append(final_seq)
            new_labels.append(label)
    
    cap.release()

if new_data:
    X_new = np.array(new_data)
    y_new = np.array(new_labels)
    X_final = np.concatenate([X_exist, X_new], axis=0) if len(X_exist) > 0 else X_new
    y_final = np.concatenate([y_exist, y_new], axis=0) if len(y_exist) > 0 else y_new
    
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump((X_final, y_final), f)
    print(f"\n✅ SUCCESS! Added {len(new_data)} pro clips.")
else:
    print("❌ No clips added.")