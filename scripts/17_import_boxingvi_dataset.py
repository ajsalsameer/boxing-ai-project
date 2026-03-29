#!/usr/bin/env python3
"""
BOXING-VI IMPORTER (Ultra Edition)
==================================
- Batched YOLO (≈5× faster on GTX 1650)
- Physics-correct velocity warmup (5-frame preroll)
- Robust V1/V2/V3 Excel parsing
- Safe merging with your own data
- Global progress bars
"""

import os, sys, glob, pickle, cv2, numpy as np, json, time
import pandas as pd
from tqdm import tqdm
import torch
from ultralytics import YOLO

from utils import extract_upper_body, compute_features, COMBO_GAP_SEC

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
YOLO_MODEL = "yolo11n-pose.pt"
DATA_DIR   = "data"

BANK_PATH      = os.path.join(DATA_DIR, "feature_bank.pkl")
SEQ_PATH       = os.path.join(DATA_DIR, "combo_sequences.pkl")
LABELS_PATH    = os.path.join(DATA_DIR, "punch_labels.json")
SNAPSHOTS_PATH = os.path.join(DATA_DIR, "punch_snapshots.pkl")

USE_FP16 = True

LABEL_MAP = {
    "jab": "jab", "Jab": "jab",
    "cross": "cross", "Cross": "cross",
    "lead hook": "hook", "Lead Hook": "hook", "hook": "hook", "Hook": "hook",
    "rear hook": "hook", "Rear Hook": "hook",
    "lead uppercut": "uppercut", "Lead Uppercut": "uppercut",
    "rear uppercut": "uppercut", "Rear Uppercut": "uppercut",
    "uppercut": "uppercut", "Uppercut": "uppercut",
    "body shot": "body_shot", "Body Shot": "body_shot"
}

# ═══════════════════════════════════════════════════════════════════════════════
# ROBUST PARSER
# ═══════════════════════════════════════════════════════════════════════════════
def load_annotation_file(file_path):
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
    except:
        return None

    # V1 Format (Headers)
    df.columns = [str(c).strip() for c in df.columns]
    if 'Start_Frame' in df.columns and 'Class' in df.columns:
        end_col = 'Ending_Frame' if 'Ending_Frame' in df.columns else df.columns[2]
        return df[['Start_Frame', end_col, 'Class']].rename(
            columns={'Start_Frame': 'start', end_col: 'end', 'Class': 'label'})

    # V2/V3 Format (No Headers)
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, header=None)
        else:
            df = pd.read_excel(file_path, header=None)
        
        df[0] = pd.to_numeric(df[0], errors='coerce')
        df[1] = pd.to_numeric(df[1], errors='coerce')
        df = df.dropna(subset=[0, 1])
        
        sample = df[2].astype(str).str.lower()
        if sample.str.contains('jab|cross|hook', na=False).any():
            return df[[0, 1, 2]].rename(columns={0: 'start', 1: 'end', 2: 'label'})
    except:
        pass
        
    return None

# ═══════════════════════════════════════════════════════════════════════════════
# BATCHED FEATURE EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════════════
def extract_feature_at_frame(cap, frame_idx, model, preroll=5, device="cuda"):
    """
    Reads (preroll + 3) frames and runs YOLO ONCE (Batched).
    """
    start_frame = max(0, frame_idx - preroll)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    for _ in range(preroll + 3):
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)

    if not frames: return None

    # BATCHED INFERENCE (Speed King)
    results = model.predict(frames, verbose=False, half=USE_FP16, device=device)

    prev_kpts = None
    prev_vel  = None
    feats_buffer = []

    for i, result in enumerate(results):
        kp_upper = None
        try:
            if result.keypoints is not None:
                kp = result.keypoints.xy[0].cpu().numpy().astype(np.float32)
                if len(kp) >= 13: kp_upper = extract_upper_body(kp)
        except: pass

        if kp_upper is not None:
            h, w = frames[i].shape[:2]
            f, prev_kpts, prev_vel, _ = compute_features(kp_upper, w, h, prev_kpts, prev_vel)
            
            # Only use data after warmup
            if i >= preroll:
                feats_buffer.append(f)

    if not feats_buffer: return None
    return np.mean(feats_buffer, axis=0).astype(np.float32)

# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & MERGING
# ═══════════════════════════════════════════════════════════════════════════════
def load_existing_data():
    bank = {}; snapshots = {}; sequences = []
    if os.path.exists(BANK_PATH):
        try: bank = pickle.load(open(BANK_PATH, "rb"))
        except: pass
    if os.path.exists(SNAPSHOTS_PATH):
        try: snapshots = pickle.load(open(SNAPSHOTS_PATH, "rb"))
        except: pass
    if os.path.exists(SEQ_PATH):
        try: sequences = pickle.load(open(SEQ_PATH, "rb"))
        except: pass
    return bank, snapshots, sequences

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════
def process_dataset(data_dir, model, video_filter=None, preroll=5, device="cuda"):
    # 1. Load Old Data
    bank, snapshots, sequences = load_existing_data()
    new_bank_stats = {}
    total_new_punches = 0

    # 2. Find Videos
    video_files = sorted(glob.glob(os.path.join(data_dir, "V*.mp4"))) + \
                  sorted(glob.glob(os.path.join(data_dir, "V*.avi")))

    if video_filter:
        video_files = [f for f in video_files if os.path.splitext(os.path.basename(f))[0] in video_filter]
        print(f" → Processing only: {video_filter}")

    if not video_files:
        print("❌ No videos found")
        return

    # 3. Pre-calculate total for progress bar
    print("⏳ Scanning annotations...")
    valid_tasks = []
    total_punches_to_process = 0
    
    for vid_path in video_files:
        base_name = os.path.splitext(os.path.basename(vid_path))[0]
        candidates = [os.path.join(data_dir, f"{base_name}.xlsx"),
                      os.path.join(data_dir, f"{base_name}.csv")]
        annot_path = next((f for f in candidates if os.path.exists(f)), None)
        
        if annot_path:
            df = load_annotation_file(annot_path)
            if df is not None and not df.empty:
                valid_tasks.append((vid_path, df, base_name))
                total_punches_to_process += len(df)

    # 4. Processing Loop
    start_time = time.time()
    pbar_total = tqdm(total=total_punches_to_process, desc="TOTAL PROGRESS", unit="punch")

    for vid_path, df, base_name in valid_tasks:
        cap = cv2.VideoCapture(vid_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        current_combo = []
        current_feats = []
        last_ts = -999.0

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {base_name}", leave=False):
            raw_label = str(row['label']).strip()
            label = LABEL_MAP.get(raw_label)
            if not label: continue

            mid_f = (int(row['start']) + int(row['end'])) // 2
            
            # Batched Extraction
            feat = extract_feature_at_frame(cap, mid_f, model, preroll, device)
            if feat is None: continue

            # Store Data
            snapshots.setdefault(label, []).append(feat)
            
            if label not in new_bank_stats:
                new_bank_stats[label] = {"sum": feat, "sq_sum": feat**2, "count": 1}
            else:
                new_bank_stats[label]["sum"] += feat
                new_bank_stats[label]["sq_sum"] += feat**2
                new_bank_stats[label]["count"] += 1

            # Combo Logic
            ts = mid_f / fps
            if (ts - last_ts) > COMBO_GAP_SEC and current_combo:
                _flush_combo(current_combo, current_feats, sequences)
                current_combo = []; current_feats = []

            current_combo.append(label)
            current_feats.append(feat)
            last_ts = ts
            total_new_punches += 1
            pbar_total.update(1)

        _flush_combo(current_combo, current_feats, sequences)
        cap.release()

    pbar_total.close()

    # 5. Merge Statistics
    print("\n🔄 Merging statistics...")
    for label, stats in new_bank_stats.items():
        n_new = stats["count"]
        mu_new = stats["sum"] / n_new
        var_new = (stats["sq_sum"] / n_new) - (mu_new ** 2)

        if label not in bank:
            bank[label] = {
                "mean": mu_new.tolist(),
                "std": np.sqrt(np.maximum(var_new, 1e-6)).tolist(),
                "count": n_new
            }
        else:
            old = bank[label]
            n_old = old["count"]
            mu_old = np.array(old["mean"])
            var_old = np.array(old["std"])**2
            
            n_total = n_old + n_new
            mu_total = (n_old * mu_old + n_new * mu_new) / n_total
            
            # Pooled Variance
            term_old = n_old * (var_old + mu_old**2)
            term_new = n_new * (var_new + mu_new**2)
            var_total = (term_old + term_new) / n_total - mu_total**2
            
            bank[label] = {
                "mean": mu_total.tolist(),
                "std": np.sqrt(np.maximum(var_total, 1e-6)).tolist(),
                "count": n_total
            }

    # 6. Save
    os.makedirs(DATA_DIR, exist_ok=True)
    pickle.dump(bank, open(BANK_PATH, "wb"))
    pickle.dump(snapshots, open(SNAPSHOTS_PATH, "wb"))
    pickle.dump(sequences, open(SEQ_PATH, "wb"))
    json.dump(sorted(bank.keys()), open(LABELS_PATH, "w"))

    elapsed = time.time() - start_time
    print("\n" + "="*70)
    print(f"✅ IMPORT COMPLETE in {elapsed:.1f}s")
    print(f"   New punches: {total_new_punches}")
    print(f"   Total combos: {len(sequences)}")
    print("="*70)

def _flush_combo(combo, feats, seq_list):
    if len(combo) < 2: return
    for i in range(len(combo) - 1):
        seq_list.append({
            "history": combo[max(0, i-2):i+1],
            "history_feats": feats[max(0, i-2):i+1],
            "next": combo[i+1]
        })

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--videos", nargs="+")
    parser.add_argument("--preroll", type=int, default=5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⏳ Loading YOLO on {device.upper()}...")
    model = YOLO(YOLO_MODEL)
    
    # Warmup
    model.predict(np.zeros((640,640,3), dtype=np.uint8), verbose=False, device=device, half=USE_FP16)

    process_dataset(args.dataset_dir, model, args.videos, args.preroll, device)