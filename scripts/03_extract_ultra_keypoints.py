import cv2
import os
import numpy as np
import pickle
from ultralytics import YOLO
from tqdm import tqdm

# ==========================================
#        GOD-TIER EXTRACTOR CONFIG
# ==========================================
RAW_FOLDER = "data/raw_videos/self_recorded"
OUTPUT_FILE = "data/boxing_ultra_dataset.pkl"
MODEL_NAME = "yolo11n-pose.pt"

# The 10 Classes
CLASSES = [
    "jab", "cross", "hook", "uppercut", "overhand",
    "slip_left", "slip_right", "duck", "block",
    "idle"
]

# Load YOLO (Pretrained Eyes)
print(f"Loading {MODEL_NAME}...")
model = YOLO(MODEL_NAME)

data = []
labels = []

# --- HELPER: CRASH-PROOF ANGLE CALCULATION ---
def calculate_angle(a, b, c):
    """Calculates angle ABC (in degrees) safely."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    
    # Avoid divide by zero
    denominator = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6
    cosine_angle = np.dot(ba, bc) / denominator
    
    # Clip to valid range (-1 to 1) to prevent NaN errors
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

print("==================================================")
print("  GOD-TIER EXTRACTOR: POS + VEL + ACC + ANGLES")
print("==================================================")

for label in CLASSES:
    folder = os.path.join(RAW_FOLDER, label)
    if not os.path.exists(folder):
        print(f"⚠ Skipping {label} (Folder missing or empty)")
        continue
    
    files = [f for f in os.listdir(folder) if f.endswith(('.mp4', '.mov', '.avi'))]
    
    for video_file in tqdm(files, desc=f"Processing {label.upper()}"):
        path = os.path.join(folder, video_file)
        cap = cv2.VideoCapture(path)
        
        sequence = []
        prev_kpts = None
        prev_vel = None
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # 1. YOLO Extraction (Fast Mode)
            results = model(frame, verbose=False, half=True, device=0)[0]
            
            if len(results.keypoints.xy) > 0:
                h, w = frame.shape[:2]
                kpts = results.keypoints.xy[0].cpu().numpy() # Shape (17, 2)
                
                # --- FEATURE ENGINEERING ---
                
                # A. Position (Normalized 0.0-1.0)
                norm_kpts = kpts.flatten() / np.array([w, h] * 17)
                
                # B. Velocity (Current - Previous)
                if prev_kpts is not None:
                    velocity = norm_kpts - prev_kpts
                else:
                    velocity = np.zeros_like(norm_kpts)
                
                # C. Acceleration (Vel - Prev_Vel)
                if prev_vel is not None:
                    acceleration = velocity - prev_vel
                else:
                    acceleration = np.zeros_like(velocity)
                
                # D. Biomechanical Angles (5 Key Angles)
                angles = np.zeros(5)
                # Ensure we have enough confidence/points
                if len(kpts) >= 13: 
                    # 1. Left Elbow (Shoulder-Elbow-Wrist)
                    angles[0] = calculate_angle(kpts[5], kpts[7], kpts[9])
                    # 2. Right Elbow
                    angles[1] = calculate_angle(kpts[6], kpts[8], kpts[10])
                    # 3. Left Shoulder (Hip-Shoulder-Elbow)
                    angles[2] = calculate_angle(kpts[11], kpts[5], kpts[7])
                    # 4. Right Shoulder
                    angles[3] = calculate_angle(kpts[12], kpts[6], kpts[8])
                    # 5. Guard Width (Wrist to Wrist relative to Shoulder width) - approximated
                    angles[4] = np.linalg.norm(kpts[9] - kpts[10]) / (np.linalg.norm(kpts[5] - kpts[6]) + 1e-6)

                # E. Energy Estimate (Sum of Absolute Acceleration)
                energy = np.sum(np.abs(acceleration))
                
                # COMBINE FEATURES: 34 + 34 + 34 + 5 + 1 = 108 Features
                features = np.concatenate([norm_kpts, velocity, acceleration, angles, [energy]])
                sequence.append(features)
                
                # Update History
                prev_kpts = norm_kpts
                prev_vel = velocity
        
        cap.release()
        
        # --- SLIDING WINDOW & AUGMENTATION ---
        WINDOW_SIZE = 30
        STEP = 5
        
        # 1. Save Normal Data
        if len(sequence) >= WINDOW_SIZE:
            for i in range(0, len(sequence) - WINDOW_SIZE + 1, STEP):
                window = sequence[i : i + WINDOW_SIZE]
                data.append(window)
                labels.append(label)

        # 2. AUTO-FEINT GENERATION (The Secret Sauce)
        # If this is an attack, create a "fake" version where we kill the velocity/energy halfway through
        if label in ["jab", "cross", "hook", "uppercut", "overhand"]:
            feint_seq = np.array(sequence).copy()
            # Reduce velocity and acceleration by 70% to simulate a "hesitation" or feint
            feint_seq[:, 34:68] *= 0.3 # Dampen Velocity
            feint_seq[:, 68:102] *= 0.3 # Dampen Acceleration
            feint_seq[:, 107] *= 0.2    # Kill Energy
            
            if len(feint_seq) >= WINDOW_SIZE:
                for i in range(0, len(feint_seq) - WINDOW_SIZE + 1, STEP):
                    window = feint_seq[i : i + WINDOW_SIZE]
                    data.append(window)
                    # We create a NEW class called "feint"
                    labels.append("feint") 

# Convert to Numpy
X = np.array(data)
y = np.array(labels)

print(f"\n==================================================")
print(f"  EXTRACTION COMPLETE")
print(f"==================================================")
print(f"  Total Windows: {X.shape[0]}")
print(f"  Sequence Length: {X.shape[1]}")
print(f"  Features per Frame: {X.shape[2]}")
print(f"  Unique Classes: {np.unique(y)}")
print(f"==================================================")

# Save
with open(OUTPUT_FILE, "wb") as f:
    pickle.dump((X, y), f)

print(f"✅ Dataset saved to {OUTPUT_FILE}")