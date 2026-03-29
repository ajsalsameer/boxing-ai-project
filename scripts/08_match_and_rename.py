import os
import cv2  # <--- THIS WAS MISSING
import pandas as pd
from tqdm import tqdm
import shutil
from difflib import get_close_matches  # For fuzzy matching

# ==========================================
# GOD-TIER MATCHER (URL-FREE EDITION)
# ==========================================
DATASET_ROOT = r"data\external\BoxingVI"
VIDEO_DIR = os.path.join(DATASET_ROOT, "RGB_videos")
ANNOT_DIR = os.path.join(DATASET_ROOT, "Annotation_files")
REPORT_FILE = os.path.join(DATASET_ROOT, "match_report.txt")

def get_video_duration(path):
    """Quick duration check for fuzzy matching (seconds)"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frames / fps if fps > 0 else 0

def main():
    print("==================================================")
    print(" 🕵️ GOD-TIER MATCHER (URL-FREE EDITION)")
    print("==================================================")
    
    if not os.path.exists(ANNOT_DIR) or not os.path.exists(VIDEO_DIR):
        print(f"❌ Directories not found!")
        return
    
    annot_files = [f for f in os.listdir(ANNOT_DIR) if f.endswith(".xlsx")]
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")]
    
    print(f"📂 Found {len(annot_files)} Excel files.")
    print(f"📂 Found {len(video_files)} Video files.")
    
    if len(video_files) == 0:
        print("❌ NO VIDEOS FOUND!")
        return
    
    print(f"Example Video: '{video_files[0]}'")
    
    matches = 0
    failures = []
    report_lines = []
    
    print("\n--- PREFIX MATCHING (V1.xlsx → V1.mp4) ---")
    
    for annot_file in tqdm(annot_files):
        target_name = os.path.splitext(annot_file)[0] + ".mp4"  # V1.mp4
        target_path = os.path.join(VIDEO_DIR, target_name)
        
        if os.path.exists(target_path):
            # print(f"✅ {target_name} already matched.")
            matches += 1
            continue
        
        # Open Excel
        excel_path = os.path.join(ANNOT_DIR, annot_file)
        try:
            df = pd.read_excel(excel_path)
        except Exception as e:
            failures.append(f"{annot_file}: {e}")
            continue
        
        # Auto-detect columns (for duration calc)
        start_col = next((c for c in df.columns if 'start' in str(c).lower()), None)
        end_col = next((c for c in df.columns if 'end' in str(c).lower()), None)
        
        # Prefix search
        found = False
        for vid_file in video_files:
            # Check if filename roughly matches ID
            if target_name in vid_file or os.path.splitext(vid_file)[0] == os.path.splitext(annot_file)[0]:
                old_path = os.path.join(VIDEO_DIR, vid_file)
                try:
                    # Backup
                    backup_path = old_path + ".bak"
                    if not os.path.exists(backup_path):
                        shutil.copy2(old_path, backup_path)
                    
                    # Rename
                    os.rename(old_path, target_path)
                    print(f"✅ PREFIX MATCH: {vid_file} → {target_name}")
                    matches += 1
                    report_lines.append(f"MATCHED: {annot_file} → {vid_file}")
                    found = True
                    break
                except Exception as e:
                    failures.append(f"Rename error {vid_file}: {e}")
        
        if not found:
            # Fuzzy match by duration (using detected columns)
            total_duration = 0
            if start_col and end_col:
                try:
                    # Calculate total duration in seconds based on frames or seconds
                    # Assuming frames for calculation safety
                    max_frame = df[end_col].max()
                    # Rough heuristic: if max frame > 1000, assumes 30fps to get seconds
                    # This is just a heuristic to match video length
                    total_duration = max_frame / 30.0 
                except:
                    total_duration = 0

            best_match = None
            best_diff = float('inf')
            
            # Compare against all videos
            for vid_file in video_files:
                vid_path = os.path.join(VIDEO_DIR, vid_file)
                vid_dur = get_video_duration(vid_path)
                
                # If the duration is within 10% or 30 seconds
                diff = abs(vid_dur - total_duration)
                if diff < best_diff:
                    best_diff = diff
                    best_match = vid_file
            
            # If match is close enough (e.g. within 60 seconds tolerance)
            if best_match and best_diff < 60: 
                old_path = os.path.join(VIDEO_DIR, best_match)
                try:
                    shutil.copy2(old_path, old_path + ".bak")
                    os.rename(old_path, target_path)
                    print(f"✅ FUZZY MATCH (duration): {best_match} → {target_name} (diff: {best_diff:.1f}s)")
                    matches += 1
                    report_lines.append(f"FUZZY MATCH: {annot_file} → {best_match}")
                    found = True
                except Exception as e:
                    failures.append(f"Fuzzy rename error: {e}")
            
            if not found:
                failures.append(f"{annot_file}: No match found")
                report_lines.append(f"MANUAL NEEDED: {annot_file}")
    
    # Report
    with open(REPORT_FILE, "w") as f:
        f.write("BoxingVI Match Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total Matches: {matches}\n")
        f.write(f"Total Failures: {len(failures)}\n\n")
        f.write("MATCHES:\n")
        for line in report_lines:
            f.write(f"  {line}\n")
    
    print("\n==================================================")
    print(f"✅ Renamed {matches} videos. Report: {REPORT_FILE}")
    print("==================================================")
    
    final_files = [f for f in os.listdir(VIDEO_DIR) if f.startswith("V") and f.endswith(".mp4")]
    if len(final_files) >= len(annot_files) * 0.8:
        print("🎉 READY FOR INGESTION: python scripts/06_ingest_boxingvi_godtier.py")
    else:
        print("⚠ Manual renames needed. Open report.txt")

if __name__ == "__main__":
    main()