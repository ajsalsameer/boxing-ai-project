import os
import pandas as pd
import yt_dlp
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
#        TURBO HYDRATOR (V1.mp4 EXACT NAMES)
# ==========================================
DATASET_ROOT = r"data\external\BoxingVI\RGB_videos"
ODS_FILE = os.path.join(DATASET_ROOT, "Meta_data.ods")
MAX_WORKERS = 4 

def download_single_video(task):
    """Worker: Download URL to exact fname (V1.mp4)."""
    url, save_path = task
    
    # Skip if exists and >1MB
    if os.path.exists(save_path) and os.path.getsize(save_path) > 1000000:
        return f"Skipped: {os.path.basename(save_path)}"
    
    # yt-dlp opts: 720p MP4
    ydl_opts = {
        'format': 'best[ext=mp4][height<=720]/best[ext=mp4]',
        'outtmpl': save_path,
        'quiet': True,
        'no_warnings': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return f"Success: {os.path.basename(save_path)}"
    except Exception as e:
        return f"Failed: {e}"

def main():
    print("==================================================")
    print("  🚀 TURBO HYDRATOR (Exact V1.mp4 Names)")
    print("==================================================")

    if not os.path.exists(ODS_FILE):
        print(f"❌ Meta_data.ods not found in {DATASET_ROOT}")
        return

    # 1. Load ODS
    print("Loading spreadsheet...")
    try:
        # pip install odfpy
        df = pd.read_excel(ODS_FILE, engine='odf')
    except Exception as e:
        print(f"❌ Error reading ODS: {e}. Run 'pip install odfpy'")
        return

    # 2. Handle 'Video' and 'Link' columns specifically
    print(f"Columns found: {list(df.columns)}")
    
    # Check if we have the right columns
    if 'Video' not in df.columns or 'Link' not in df.columns:
        print("❌ Expected columns 'Video' and 'Link' not found.")
        return

    tasks = []
    print(f"Preparing {len(df)} downloads...")

    for index, row in df.iterrows():
        # Get Filename from 'Video' column (e.g., "V1")
        video_name = str(row['Video']).strip()
        if not video_name.endswith(".mp4"):
            video_name += ".mp4"
            
        save_path = os.path.join(DATASET_ROOT, video_name)
        
        # Get URL from 'Link' column
        url = str(row['Link']).strip()

        if "http" not in url:
            print(f"Skipping {video_name}: No URL")
            continue

        tasks.append((url, save_path))

    if not tasks:
        print("❌ No valid URLs found.")
        return

    print(f"Queueing {len(tasks)} downloads with {MAX_WORKERS} threads...")

    # 3. Parallel Download
    success = 0
    skipped = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(download_single_video, t) for t in tasks]
        
        for future in tqdm(as_completed(futures), total=len(tasks)):
            result = future.result()
            if "Success" in result: success += 1
            elif "Skipped" in result: skipped += 1
            else: failed += 1

    print("==================================================")
    print(f"  ✅ Downloaded: {success}")
    print(f"  ⏩ Skipped: {skipped}")
    print(f"  ❌ Failed: {failed}")
    print("==================================================")
    
    if success + skipped > 0:
        print("🎉 HYDRATION COMPLETE! Run ingestion: python scripts/06_ingest_boxingvi_godtier.py")

if __name__ == "__main__":
    main()