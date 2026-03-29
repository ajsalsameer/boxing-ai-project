import cv2
import os
import time
from datetime import datetime

# ==========================================
#        GOD-TIER RECORDER CONFIG
# ==========================================
OUTPUT_DIR = "data/raw_videos/self_recorded"

# The 10 Essential Classes
CLASSES = [
    "jab", "cross", "hook", "uppercut", "overhand",  # Attacks
    "slip_left", "slip_right", "duck", "block",      # Defense
    "idle"                                           # Essential Stance
]

RECORD_SECONDS = 2.5   # 2.5s captures punch + retraction
CAMERA_INDEX = 1       # Try 1 for Phone Link, 0 for Webcam

# Create directories automatically
for cls in CLASSES:
    os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)

# Setup Camera (MSMF for Windows Speed)
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_MSMF)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# Fallback
if not cap.isOpened():
    print(f"Camera {CAMERA_INDEX} failed. Trying Index 0...")
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)

print(f"==================================================")
print(f"  🥊 GOD-TIER DATA RECORDER (COMBINED) 🥊")
print(f"==================================================")
print(f"  --- ATTACKS ---")
print(f"  [J] JAB       [C] CROSS")
print(f"  [H] HOOK      [U] UPPERCUT")
print(f"  [O] OVERHAND")
print(f"")
print(f"  --- DEFENSE ---")
print(f"  [S] SLIP L    [D] SLIP R")
print(f"  [W] DUCK      [B] BLOCK")
print(f"")
print(f"  --- ESSENTIAL ---")
print(f"  [N] IDLE (Stance/Bouncing) - CRITICAL!")
print(f"  [Q] Quit")
print(f"==================================================")

recording = False
current_label = ""
start_time = 0
video_writer = None

while True:
    ret, frame = cap.read()
    if not ret: continue

    display_frame = frame.copy()

    # ==========================
    #    RECORDING STATE
    # ==========================
    if recording:
        elapsed = time.time() - start_time
        remaining = RECORD_SECONDS - elapsed
        
        if elapsed >= RECORD_SECONDS:
            # STOP RECORDING
            recording = False
            video_writer.release()
            print(f" ✅ Saved {current_label.upper()}")
        else:
            # WRITE FRAME
            video_writer.write(frame)
            
            # --- COMBINED VISUAL FEEDBACK ---
            # 1. Red Border
            cv2.rectangle(display_frame, (0,0), (640,480), (0,0,255), 10)
            
            # 2. Red Dot (Rec indicator)
            cv2.circle(display_frame, (50, 50), 20, (0, 0, 255), -1)
            
            # 3. Text Label
            cv2.putText(display_frame, f"REC: {current_label.upper()}", (80, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 4. Countdown Timer
            cv2.putText(display_frame, f"{remaining:.1f}s", (550, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # 5. Progress Bar Line (Bottom)
            progress_width = int((elapsed / RECORD_SECONDS) * 640)
            cv2.line(display_frame, (0, 470), (progress_width, 470), (0, 0, 255), 10)

    # ==========================
    #      IDLE STATE
    # ==========================
    else:
        # Instructions
        cv2.putText(display_frame, "Press Keys: J,C,H,U,O | S,D,W,B | N", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Grid Display for File Counts
        y_pos = 70
        x_pos = 20
        
        for i, cls in enumerate(CLASSES):
            count = len(os.listdir(os.path.join(OUTPUT_DIR, cls)))
            
            # Dynamic Colors
            color = (0, 255, 255)            # Yellow (Need more)
            if count >= 30: color = (0, 255, 0)      # Green (Good)
            if count >= 50: color = (0, 215, 255)    # Gold (God-Tier)

            # Split into 2 columns
            if i == 5: 
                x_pos = 320
                y_pos = 70
            
            cv2.putText(display_frame, f"{cls.upper()}: {count}", (x_pos, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            y_pos += 35

    cv2.imshow("God-Tier Recorder", display_frame)
    key = cv2.waitKey(1) & 0xFF

    # ==========================
    #      CONTROLS
    # ==========================
    if key == ord('q'):
        break
    
    elif not recording:
        new_label = None
        # Attack Mapping
        if key == ord('j'): new_label = "jab"
        elif key == ord('c'): new_label = "cross"
        elif key == ord('h'): new_label = "hook"
        elif key == ord('u'): new_label = "uppercut"
        elif key == ord('o'): new_label = "overhand"
        
        # Defense Mapping
        elif key == ord('s'): new_label = "slip_left"
        elif key == ord('d'): new_label = "slip_right"
        elif key == ord('w'): new_label = "duck"
        elif key == ord('b'): new_label = "block"
        
        # Idle Mapping
        elif key == ord('n'): new_label = "idle"

        if new_label:
            current_label = new_label
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{OUTPUT_DIR}/{current_label}/{current_label}_{timestamp}.mp4"
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(filename, fourcc, 30.0, (640, 480))
            
            recording = True
            start_time = time.time()

if video_writer is not None:
    video_writer.release()
cap.release()
cv2.destroyAllWindows()