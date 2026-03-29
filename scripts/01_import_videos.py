import os
import shutil
import msvcrt
import time
from datetime import datetime

# === CONFIGURATION ===
INCOMING_DIR = "data/raw_videos/incoming"
OUTPUT_DIR = "data/raw_videos/self_recorded"

# SIMPLIFIED CLASSES (No Left/Right Hook)
CLASSES = [
    "jab", "cross", "hook", "uppercut", "overhand",
    "slip_left", "slip_right", "duck", "block", "idle"
]

VIDEO_EXTENSIONS = [".mp4", ".mov", ".avi", ".mkv", ".webm"]

# Ensure directories exist
os.makedirs(INCOMING_DIR, exist_ok=True)
for cls in CLASSES:
    os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)

def get_video_files(folder):
    return [f for f in os.listdir(folder) if any(f.lower().endswith(ext) for ext in VIDEO_EXTENSIONS)]

def main():
    print("===============================================")
    print("  🥊 FAST VIDEO SORTER (SIMPLE MODE) 🥊")
    print("===============================================")
    print(f"1. Drag & Drop videos into: {INCOMING_DIR}")
    print(f"2. Press keys to sort.")
    print("===============================================")
    print("  [J] Jab       [S] Slip Left")
    print("  [C] Cross     [D] Slip Right")
    print("  [H] Hook      [W] Duck")
    print("  [U] Uppercut  [B] Block")
    print("  [O] Overhand  [N] Idle")
    print("===============================================")

    while True:
        videos = get_video_files(INCOMING_DIR)
        if not videos:
            print("\nWaiting for videos...", end='\r')
            time.sleep(2)
            continue

        filename = videos[0]
        src_path = os.path.join(INCOMING_DIR, filename)
        print(f"\n🎥 PROCESSING: {filename}")
        print(">>> Press Key: ", end='', flush=True)

        key_raw = msvcrt.getch()
        try: key = key_raw.decode('utf-8').lower()
        except: continue
        
        if key == 'q': break
        if key == 'x': continue
            
        label = None
        if key == 'j': label = "jab"
        elif key == 'c': label = "cross"
        elif key == 'h': label = "hook"       # Just Hook!
        elif key == 'u': label = "uppercut"
        elif key == 'o': label = "overhand"
        elif key == 's': label = "slip_left"
        elif key == 'd': label = "slip_right"
        elif key == 'w': label = "duck"
        elif key == 'b': label = "block"
        elif key == 'n': label = "idle"
        
        if label:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = os.path.splitext(filename)[1]
            new_name = f"{label}_imported_{timestamp}{ext}"
            dest_path = os.path.join(OUTPUT_DIR, label, new_name)
            
            try:
                shutil.move(src_path, dest_path)
                print(f" ✅ Moved to [{label.upper()}]")
            except Exception as e:
                print(f" ❌ Error: {e}")

if __name__ == "__main__":
    main()