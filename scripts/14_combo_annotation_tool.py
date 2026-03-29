#!/usr/bin/env python3
"""
PHASE 1 — Video → Training Data (GTX 1650 Optimized, Final Version)
=====================================================================
Changes from previous version:
- Removed augment_features (unnecessary noise)
- Fixed GPU detection (torch.cuda)
- Consistent thresholds from utils.py
- Better progress reporting
- Saves punch snapshots for debugging

Usage:
    python 17_combo_annotation_tool.py
    python 17_combo_annotation_tool.py --skip 2  # Process every 2nd frame (2x faster)
"""

import cv2, numpy as np, pickle, json, os, sys
import torch  # For GPU detection
from collections import deque
from ultralytics import YOLO
from utils import (
    FEAT_DIM, CLASS_WINDOW, ENERGY_THRESHOLD, PUNCH_COOLDOWN,
    COMBO_GAP_SEC, extract_upper_body, compute_features,
    classify_with_bank, MIN_CONFIDENCE
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
YOLO_MODEL  = "yolo11n-pose.pt"
VIDEOS_ROOT = "videos"
DATA_DIR    = "data"

BANK_PATH      = os.path.join(DATA_DIR, "feature_bank.pkl")
SEQ_PATH       = os.path.join(DATA_DIR, "combo_sequences.pkl")
LABELS_PATH    = os.path.join(DATA_DIR, "punch_labels.json")
SNAPSHOTS_PATH = os.path.join(DATA_DIR, "punch_snapshots.pkl")

COMBO_FOLDER = "combo"
VIDEO_EXTS   = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# GPU Settings
USE_FP16     = True   # Half precision (30% faster on GTX 1650)
YOLO_IMG_SIZE = 416   # Reduced from 640 (faster)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _load_pkl(path, default):
    return pickle.load(open(path, "rb")) if os.path.exists(path) else default

def _save_pkl(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pickle.dump(obj, open(path, "wb"))

def _scan_videos(root: str) -> dict:
    """Scan directory for video files organized in punch-type folders"""
    layout = {}
    if not os.path.isdir(root):
        sys.exit(f"❌  Videos folder not found: {root}\n"
                 f"    Create it with subfolders: jab/, cross/, hook/, etc.")
    
    for entry in sorted(os.listdir(root)):
        full = os.path.join(root, entry)
        if not os.path.isdir(full):
            continue
        
        label = entry.strip().lower()
        vids  = [
            os.path.join(full, f)
            for f in sorted(os.listdir(full))
            if os.path.splitext(f)[1].lower() in VIDEO_EXTS
        ]
        if vids:
            layout[label] = vids
    
    return layout


# ═══════════════════════════════════════════════════════════════════════════════
# VIDEO PROCESSOR (GPU Optimized)
# ═══════════════════════════════════════════════════════════════════════════════

class VideoProcessor:
    """
    Process video with YOLO pose estimation and punch detection.
    Optimized for GTX 1650 with FP16 and frame skipping.
    """
    def __init__(self, video_path: str, yolo_model, frame_skip: int = 1):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print(f"  ⚠️   Cannot open {video_path}")
            self.valid = False
            return
        
        self.valid      = True
        self.fps        = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total      = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        self.yolo       = yolo_model
        self.frame_skip = max(frame_skip, 1)
        
        # GPU detection (torch is more reliable than cv2.cuda)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # Adjust cooldown for frame skipping
        self.cooldown = max(PUNCH_COOLDOWN // self.frame_skip, 5)
    
    def detect_punches(self):
        """
        Generator that yields (frame_idx, timestamp, feat_vector) 
        for each detected punch.
        """
        if not self.valid:
            return
        
        seq        = deque(maxlen=CLASS_WINDOW)
        prev_kpts  = None
        prev_vel   = None
        last_frame = -999
        frame_idx  = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_idx += 1
            
            # Frame skipping for speed
            if frame_idx % self.frame_skip != 0:
                continue
            
            h, w = frame.shape[:2]
            
            # Progress indicator
            if frame_idx % 200 == 0:
                pct = frame_idx / self.total * 100
                print(f"      {frame_idx}/{self.total}  ({pct:.0f}%)", end="\r")
            
            # ── YOLO Pose Estimation ──────────────────────────────────────
            results = self.yolo.predict(
                frame,
                verbose=False,
                device=self.device,
                half=USE_FP16,
                imgsz=YOLO_IMG_SIZE
            )
            
            kp_upper = None
            try:
                if results[0].keypoints is not None:
                    kpts = results[0].keypoints.xy[0].cpu().numpy().astype(np.float32)
                    if len(kpts) >= 13:
                        kp_upper = extract_upper_body(kpts)
            except Exception:
                pass
            
            if kp_upper is None:
                continue
            
            # ── Feature Extraction ────────────────────────────────────────
            feats, prev_kpts, prev_vel, energy = compute_features(
                kp_upper, w, h, prev_kpts, prev_vel)
            seq.append(feats)
            
            # ── Punch Detection ───────────────────────────────────────────
            if energy < ENERGY_THRESHOLD:
                continue
            if (frame_idx - last_frame) <= self.cooldown:
                continue
            if len(seq) < CLASS_WINDOW:
                continue
            
            # Average features over window for stability
            avg_feat = np.mean(list(seq), axis=0).astype(np.float32)
            
            last_frame = frame_idx
            yield frame_idx, round(frame_idx / self.fps, 3), avg_feat
        
        self.cap.release()


# ═══════════════════════════════════════════════════════════════════════════════
# PASS 1: Build Feature Bank from Drill Videos
# ═══════════════════════════════════════════════════════════════════════════════

def pass1_build_feature_bank(layout: dict, yolo_model, frame_skip: int) -> dict:
    """
    Process single-punch drill videos to build the feature bank.
    Each folder becomes a "centroid" in the classifier.
    """
    bank      = _load_pkl(BANK_PATH, {})
    snapshots = _load_pkl(SNAPSHOTS_PATH, {})
    
    for label, vid_paths in layout.items():
        if label == COMBO_FOLDER:
            continue
        
        print(f"\n  📁  {label.upper()}/ ({len(vid_paths)} video{'s' if len(vid_paths)>1 else ''})")
        new_feats = []
        
        for vp in vid_paths:
            print(f"      🎬 {os.path.basename(vp)}")
            proc = VideoProcessor(vp, yolo_model, frame_skip=frame_skip)
            
            for _, _, feat in proc.detect_punches():
                new_feats.append(feat)
            
            print(f"          → {len(new_feats)} punches detected so far")
        
        if not new_feats:
            print(f"      ⚠️   No punches detected in {label}/")
            print(f"           Try: lowering ENERGY_THRESHOLD or recording clearer videos")
            continue
        
        # Save raw features for debugging
        snapshots.setdefault(label, []).extend(new_feats)
        
        # Compute statistics
        feats_arr = np.stack(new_feats)
        new_mean  = feats_arr.mean(axis=0)
        new_std   = feats_arr.std(axis=0)
        new_count = len(new_feats)
        
        # Merge with existing bank (if re-running)
        if label in bank:
            old   = bank[label]
            total = old["count"] + new_count
            
            # Weighted average
            bank[label] = {
                "mean":  ((np.array(old["mean"]) * old["count"] + new_mean * new_count) / total).tolist(),
                "std":   (np.sqrt((np.array(old["std"])**2 * old["count"] + new_std**2 * new_count) / total)).tolist(),
                "count": total
            }
        else:
            bank[label] = {
                "mean":  new_mean.tolist(),
                "std":   new_std.tolist(),
                "count": new_count
            }
        
        print(f"      ✅  bank[{label}]  total samples = {bank[label]['count']}")
    
    _save_pkl(BANK_PATH, bank)
    _save_pkl(SNAPSHOTS_PATH, snapshots)
    print(f"\n  ✅  Feature bank saved → {BANK_PATH}")
    
    return bank


# ═══════════════════════════════════════════════════════════════════════════════
# PASS 2: Auto-Label Combo Videos Using Feature Bank
# ═══════════════════════════════════════════════════════════════════════════════

def pass2_build_sequences(combo_paths: list, 
                          yolo_model, 
                          feature_bank: dict, 
                          frame_skip: int):
    """
    Process combo videos: classify each punch using feature bank,
    then group into sequences for next-move prediction training.
    """
    if not combo_paths:
        print("\n  ℹ️   No combo/ folder (or empty) — skipping Pass 2")
        print("      Sequences will be collected from live game sessions")
        return
    
    if not feature_bank:
        print("\n  ⚠️   Feature bank is empty — can't label combo videos")
        print("      Process drill videos first (Pass 1)")
        return
    
    print(f"\n  📁  combo/ ({len(combo_paths)} video{'s' if len(combo_paths)>1 else ''})")
    print(f"      Using feature bank: {list(feature_bank.keys())}")
    print(f"      Confidence threshold: {MIN_CONFIDENCE:.0%}")
    
    existing_seqs = _load_pkl(SEQ_PATH, [])
    new_pairs     = 0
    rejected      = 0
    
    for vp in combo_paths:
        print(f"\n      🎬 {os.path.basename(vp)}")
        proc = VideoProcessor(vp, yolo_model, frame_skip=frame_skip)
        
        current_combo  = []
        current_feats  = []
        last_ts        = -999.0
        
        for fidx, ts, feat in proc.detect_punches():
            # Classify using feature bank
            label, conf = classify_with_bank(feat, feature_bank, min_confidence=MIN_CONFIDENCE)
            
            if label == "unknown":
                rejected += 1
                print(f"        [{ts:7.2f}s] ⚠️  LOW CONFIDENCE (rejected)")
                continue
            
            # Combo gap detection
            if current_combo and (ts - last_ts) > COMBO_GAP_SEC:
                new_pairs += _flush_combo(current_combo, current_feats, existing_seqs)
                current_combo = []
                current_feats = []
            
            current_combo.append(label)
            current_feats.append(feat.tolist())
            last_ts = ts
            
            print(f"        [{ts:7.2f}s] {label:14s} conf={conf:.2f}")
        
        # Flush final combo
        new_pairs += _flush_combo(current_combo, current_feats, existing_seqs)
    
    _save_pkl(SEQ_PATH, existing_seqs)
    
    print(f"\n  ✅  Sequences saved → {SEQ_PATH}")
    print(f"      +{new_pairs} new pairs, {len(existing_seqs)} total")
    if rejected > 0:
        print(f"      ⚠️  {rejected} low-confidence punches rejected")
        print(f"          (Improve by recording more drill videos)")


def _flush_combo(combo: list, feats: list, seq_list: list) -> int:
    """Generate training pairs from completed combo"""
    if len(combo) < 2:
        return 0
    
    added = 0
    for i in range(len(combo) - 1):
        history       = combo[max(0, i - 2): i + 1]
        history_feats = feats[max(0, i - 2): i + 1]
        
        seq_list.append({
            "history":       history,
            "history_feats": history_feats,
            "next":          combo[i + 1]
        })
        added += 1
    
    return added


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Phase 1 — GTX 1650 Optimized")
    ap.add_argument("--videos_dir", "-d", default=VIDEOS_ROOT,
                    help="Root folder containing punch subfolders")
    ap.add_argument("--skip", "-s", type=int, default=1,
                    help="Process every Nth frame (1=all, 2=half speed)")
    args = ap.parse_args()
    
    # ── Scan Videos ───────────────────────────────────────────────────────
    layout = _scan_videos(args.videos_dir)
    if not layout:
        print(f"❌  No video subfolders found in {args.videos_dir}/")
        print(f"\n    Expected structure:")
        print(f"      {args.videos_dir}/")
        print(f"      ├── jab/")
        print(f"      │   └── jab_drill.mp4")
        print(f"      ├── cross/")
        print(f"      │   └── cross_drill.mp4")
        print(f"      ├── hook/")
        print(f"      └── combo/  ← optional")
        sys.exit(1)
    
    print("=" * 70)
    print("  PHASE 1 — GTX 1650 OPTIMIZED VIDEO PROCESSING")
    print("=" * 70)
    print(f"\n  📂  Scanned: {args.videos_dir}/")
    
    for label, paths in layout.items():
        tag = " ← combo (Pass 2)" if label == COMBO_FOLDER else ""
        print(f"      {label:16s} : {len(paths)} video{'s' if len(paths)>1 else ''}{tag}")
    
    print(f"\n  ⚙️  Settings:")
    print(f"      Energy threshold:  {ENERGY_THRESHOLD} (strict, reduces ghost punches)")
    print(f"      Punch cooldown:    {PUNCH_COOLDOWN} frames (~{PUNCH_COOLDOWN/30:.1f}s)")
    print(f"      Frame skip:        every {args.skip} frame{'s' if args.skip>1 else ''}")
    print(f"      FP16 mode:         {'ON' if USE_FP16 else 'OFF'}")
    print(f"      YOLO image size:   {YOLO_IMG_SIZE}px")
    print(f"      Min confidence:    {MIN_CONFIDENCE:.0%}")
    
    # ── Save Labels ───────────────────────────────────────────────────────
    labels = sorted(l for l in layout if l != COMBO_FOLDER)
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(LABELS_PATH, "w") as f:
        json.dump(labels, f)
    
    print(f"\n  ✅  Labels: {labels}")
    print(f"      saved → {LABELS_PATH}")
    
    # ── Load YOLO ─────────────────────────────────────────────────────────
    print("\n  ⏳  Loading YOLO ...")
    yolo = YOLO(YOLO_MODEL)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"      Device: {device}")
    
    if device == "cpu":
        print("      ⚠️  WARNING: GPU not detected!")
        print("          Check: CUDA installed? GPU drivers updated?")
        print("          Processing will be MUCH slower on CPU.")
    
    # ── PASS 1: Build Feature Bank ────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  PASS 1 — Building Feature Bank from Drill Videos")
    print("─" * 70)
    
    bank = pass1_build_feature_bank(layout, yolo, args.skip)
    
    # ── PASS 2: Auto-Label Combos ─────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  PASS 2 — Auto-Labeling Combo Videos with Feature Bank")
    print("─" * 70)
    
    combo_vids = layout.get(COMBO_FOLDER, [])
    pass2_build_sequences(combo_vids, yolo, bank, args.skip)
    
    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  ✅ DONE")
    print("=" * 70)
    print(f"  Feature bank : {BANK_PATH}")
    print(f"  Sequences    : {SEQ_PATH}")
    print(f"  Labels       : {LABELS_PATH}")
    print(f"  Snapshots    : {SNAPSHOTS_PATH}  (for debugging)")
    
    if bank:
        print(f"\n  📊 Feature Bank Summary:")
        for label, stats in bank.items():
            print(f"      {label:12s} : {stats['count']:5d} samples")
    
    print(f"\n  🎯 NEXT STEP:  python 18_train_next_move_predictor.py")
    print("=" * 70)


if __name__ == "__main__":
    main()