import cv2
import time
import torch
print("[STEP 1] Importing libraries... DONE")

try:
    from ultralytics import YOLO
    print("[STEP 2] Libraries imported successfully.")
except ImportError:
    print("ERROR: Ultralytics not installed. Run 'pip install ultralytics'")
    exit()

print("------------------------------------------------")
print("[STEP 3] Loading AI Model... (If this hangs, it's downloading)")
try:
    # We use verbose=True so you can see the download progress bar
    model = YOLO("yolo11n-pose.pt")
    print("[STEP 4] AI Model Loaded Successfully!")
except Exception as e:
    print(f"\n[ERROR] YOLO11 failed to load. Trying YOLOv8 fallback...")
    try:
        model = YOLO("yolov8n-pose.pt")
        print("[STEP 4] Fallback Model (YOLOv8) Loaded!")
    except:
        print(f"[FATAL] Could not load any model. Check internet connection.\nError: {e}")
        exit()

print("------------------------------------------------")
print("[STEP 5] Opening Camera 1 (Phone Link)...")

# Try MSMF first (Best for Phone Link)
cap = cv2.VideoCapture(1, cv2.CAP_MSMF)

if not cap.isOpened():
    print("[WARNING] Camera 1 failed. Trying Camera 0...")
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    if not cap.isOpened():
        print("[FATAL] No cameras found. Is 'Phone Link' active on your PC?")
        exit()

print("[STEP 6] Camera Connection Established!")
print("------------------------------------------------")
print("[STEP 7] Starting Video Feed... (Press 'q' to quit)")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[WAITING] Camera is connected but sending no data...", end='\r')
        time.sleep(0.1)
        continue

    # Show raw camera first to prove it works
    cv2.putText(frame, "CAMERA WORKS!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Debug Mode", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\n[FINISHED] Script closed successfully.")