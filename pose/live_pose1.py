from ultralytics import YOLO
import cv2
import time
import torch
import numpy as np

# ---------------- CONFIG ----------------
MODEL_NAME = "yolo11n-pose.pt"      # fastest YOLO11 pose model
CONFIDENCE_THRESHOLD = 0.5
IMG_SIZE = 320                      # smaller = faster (try 256 if needed)

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
# Force camera index 0
cap = cv2.VideoCapture(0, cv2.CAP_MSMF)

# Ask camera for 640x480 @ 60 FPS (actual FPS depends on webcam)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)

if not cap.isOpened():
    print("Error: Could not open camera 0.")
    exit()

print("Camera 0 opened. Press 'q' to quit.")

# Minimal skeleton connections
SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (11, 12), (5, 11), (6, 12),              # torso
    (11, 13), (13, 15), (12, 14), (14, 16)   # legs
]

# FPS smoothing
last_time = time.time()
fps = 0.0
alpha = 0.9  # for exponential moving average

# ------------- MAIN LOOP -------------
while True:
    grabbed, frame = cap.read()
    if not grabbed:
        continue

    start_t = time.time()

    # ---- INFERENCE ----
    results = model(
        frame,
        imgsz=IMG_SIZE,
        half=True,
        device=device,
        verbose=False,
        conf=CONFIDENCE_THRESHOLD
    )[0]

    # ---- DRAW KEYPOINTS & SKELETON (LIGHT) ----
    if len(results.keypoints.xy) > 0:
        kpts = results.keypoints.xy[0].cpu().numpy()  # (num_kpts, 2)

        # Draw only basic skeleton (less drawing = more FPS)
        for p1, p2 in SKELETON:
            if p1 < len(kpts) and p2 < len(kpts):
                x1, y1 = int(kpts[p1][0]), int(kpts[p1][1])
                x2, y2 = int(kpts[p2][0]), int(kpts[p2][1])
                if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                    cv2.line(frame, (x1, y1), (x2, y2),
                             (0, 255, 0), 2)

    # ---- FPS CALC (smooth) ----
    dt = time.time() - start_t
    inst_fps = 1.0 / dt if dt > 0 else 0.0
    fps = alpha * fps + (1 - alpha) * inst_fps

    cv2.putText(frame, f"FPS: {int(fps)}", (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    cv2.imshow("YOLO11 Pose TURBO", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
