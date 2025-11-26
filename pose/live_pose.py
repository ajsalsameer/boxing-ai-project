from ultralytics import YOLO
import cv2
import time
import torch
import numpy as np

# ---------------- CONFIG ----------------
MODEL_NAME = "yolo11n-pose.pt"      # fastest YOLO11 pose model
CONFIDENCE_THRESHOLD = 0.5
IMG_SIZE = 416                      # try 320 if you want even more FPS

# ------------- LOAD MODEL -------------
try:
    model = YOLO(MODEL_NAME)
except Exception:
    print(f"Could not find {MODEL_NAME}, using yolov8n-pose.pt fallback...")
    model = YOLO("yolov8n-pose.pt")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.fuse()  # small speed boost

print(f"Using {MODEL_NAME} on {device}")

# ------------- CAMERA SETUP -------------
# Try camera 1 (external) first, then fallback to 0 (default)
cap = cv2.VideoCapture(1, cv2.CAP_MSMF)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

# Ask camera for 60 FPS at 640x480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Camera opened. Press 'q' to quit.")

# COCO skeleton connections for drawing
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),       # Face
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (11, 12), (5, 11), (6, 12),           # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)    # Legs
]

# FPS counters
prev_sec = time.time()
fps = 0
frame_count = 0

# ------------- MAIN LOOP -------------
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_count += 1
    now = time.time()
    if now - prev_sec >= 1.0:
        fps = frame_count
        frame_count = 0
        prev_sec = now

    # ---- INFERENCE ----
    # half=True -> FP16, much faster on GPU
    results = model(
        frame,
        imgsz=IMG_SIZE,
        half=True,
        device=device,
        verbose=False,
        conf=CONFIDENCE_THRESHOLD
    )[0]

    # ---- DRAW KEYPOINTS & SKELETON ----
    if len(results.keypoints.xy) > 0:
        kpts = results.keypoints.xy[0].cpu().numpy()  # (num_kpts, 2)

        # Draw skeleton lines
        for p1, p2 in SKELETON:
            if p1 < len(kpts) and p2 < len(kpts):
                x1, y1 = int(kpts[p1][0]), int(kpts[p1][1])
                x2, y2 = int(kpts[p2][0]), int(kpts[p2][1])

                if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                    cv2.line(frame, (x1, y1), (x2, y2),
                             (0, 255, 0), 2)

        # Simple jab detection logic using wrists and shoulders
        if len(kpts) >= 11:
            # Right wrist (10), Left wrist (9)
            rw_x, rw_y = int(kpts[10][0]), int(kpts[10][1])
            lw_x, lw_y = int(kpts[9][0]), int(kpts[9][1])

            # Right shoulder (6), Left shoulder (5)
            rs_x = kpts[6][0]
            ls_x = kpts[5][0]

            # Draw wrists
            if rw_x > 0:
                cv2.circle(frame, (rw_x, rw_y), 10, (0, 255, 255), -1)
            if lw_x > 0:
                cv2.circle(frame, (lw_x, lw_y), 10, (0, 255, 255), -1)

            # Jab conditions (very simple for now)
            if rw_x > rs_x + 80 and rw_x > 0:
                cv2.putText(frame, "RIGHT JAB!", (50, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

            if lw_x < ls_x - 80 and lw_x > 0:
                cv2.putText(frame, "LEFT JAB!", (50, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

    # ---- DISPLAY FPS ----
    cv2.putText(frame, f"FPS: {fps}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    cv2.imshow("YOLO11 Pose TURBO", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
