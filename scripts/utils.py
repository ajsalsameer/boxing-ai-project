"""
utils.py - Optimized for Complete Boxing AI System
===================================================
Tuned for:
- Single punch recognition (Step 1)
- Combo recording (Step 2)  
- Live prediction game (Step 3)
- GTX 1650 performance
"""

import numpy as np
import cv2
from collections import Counter

# ═══════════════════════════════════════════════════════════════════════════════
# KEYPOINTS
# ═══════════════════════════════════════════════════════════════════════════════
UPPER_BODY_IDX = list(range(13))
NUM_KPT = 13
NUM_ANGLES = 5
FEAT_DIM = NUM_KPT * 2 * 3 + NUM_ANGLES + 1  # 84

# ═══════════════════════════════════════════════════════════════════════════════
# THRESHOLDS (Balanced for all 3 steps)
# ═══════════════════════════════════════════════════════════════════════════════
CLASS_WINDOW = 8           # Frames to average
ENERGY_THRESHOLD = 0.010   # LOWERED - More sensitive for detection
PUNCH_COOLDOWN = 15        # LOWERED - Faster detection (was 30)
COMBO_GAP_SEC = 2.0        # Gap between combos
SINGLE_MOVE_THRESHOLD = 0.80

# Classification (Balanced for accuracy + responsiveness)
MIN_CONFIDENCE = 0.50      # LOWERED - Was 0.7 (too strict), now 0.5 (better detection)
TEMPORAL_WINDOW = 3        # LOWERED - Was 5 (too laggy), now 3 (responsive)

# ═══════════════════════════════════════════════════════════════════════════════
# VISUALS
# ═══════════════════════════════════════════════════════════════════════════════
DEFAULT_LABELS = ["jab", "cross", "hook", "uppercut"]

SKELETON_EDGES = [
    (0,1), (0,2), (1,3), (2,4),
    (3,5), (4,6), (5,6),
    (5,7), (6,8), (7,9), (8,10),
    (5,11), (6,12), (11,12),
]

PUNCH_COLOURS = {
    "jab":        (0, 230, 255),    # Yellow
    "cross":      (0, 100, 255),    # Orange  
    "hook":       (0, 255, 180),    # Cyan
    "uppercut":   (255, 80, 80),    # Red
}
DEFAULT_COLOUR = (255, 255, 255)

# ═══════════════════════════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_angle(a, b, c) -> float:
    """Calculate angle at vertex b (in degrees)"""
    try:
        a, b, c = (np.asarray(x, dtype=np.float64) for x in (a, b, c))
        ba, bc = a - b, c - b
        denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8
        cosine = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosine)))
    except:
        return 0.0

def extract_upper_body(all_kpts: np.ndarray) -> np.ndarray:
    """Extract upper body keypoints (13 out of 17)"""
    if len(all_kpts) < 13:
        return np.zeros((13, 2), dtype=np.float32)
    return all_kpts[UPPER_BODY_IDX]

def compute_features(kp_upper, w, h, prev_kpts, prev_vel):
    """
    Extract 84-dim feature vector:
    - Position (normalized)
    - Velocity
    - Acceleration  
    - Joint angles
    - Energy
    
    Returns: (features, prev_kpts, prev_vel, energy)
    """
    if len(kp_upper) < 13:
        return np.zeros(FEAT_DIM, dtype=np.float32), prev_kpts, prev_vel, 0.0
    
    kpts_norm = (kp_upper / np.array([w, h], dtype=np.float32)).flatten()
    
    if prev_kpts is None:
        prev_kpts = kpts_norm.copy()
        prev_vel = np.zeros_like(kpts_norm)
    
    vel = kpts_norm - prev_kpts
    acc = vel - prev_vel
    
    # Energy (weighted for punch detection)
    wrist_energy = float(np.linalg.norm(acc[18:22]))  # L+R wrists
    elbow_energy = float(np.linalg.norm(acc[14:18]))  # L+R elbows
    energy = wrist_energy * 0.4 + elbow_energy * 0.6
    
    # Joint angles
    angles = np.zeros(NUM_ANGLES, dtype=np.float32)
    try:
        angles[0] = calculate_angle(kp_upper[5], kp_upper[7], kp_upper[9])   # L elbow
        angles[1] = calculate_angle(kp_upper[6], kp_upper[8], kp_upper[10])  # R elbow
        angles[2] = calculate_angle(kp_upper[7], kp_upper[5], kp_upper[6])   # L shoulder
        angles[3] = calculate_angle(kp_upper[8], kp_upper[6], kp_upper[5])   # R shoulder
        mid_hip = (kp_upper[11] + kp_upper[12]) / 2.0
        mid_shld = (kp_upper[5] + kp_upper[6]) / 2.0
        angles[4] = calculate_angle(kp_upper[5], mid_shld, mid_hip)          # Torso
    except:
        pass
    
    # Concatenate
    feats = np.concatenate([kpts_norm, vel, acc, angles, [energy]])
    
    # Ensure correct dimension
    if feats.shape[0] < FEAT_DIM:
        feats = np.concatenate([feats, np.zeros(FEAT_DIM - feats.shape[0])])
    feats = feats[:FEAT_DIM]
    
    return feats.astype(np.float32), kpts_norm, vel, energy

def draw_skeleton(frame, kp_upper, colour=(0, 255, 100), thickness=2):
    """Draw skeleton with anti-aliasing"""
    for a, b in SKELETON_EDGES:
        if a < len(kp_upper) and b < len(kp_upper):
            p1 = (int(kp_upper[a][0]), int(kp_upper[a][1]))
            p2 = (int(kp_upper[b][0]), int(kp_upper[b][1]))
            if p1 != (0, 0) and p2 != (0, 0):
                cv2.line(frame, p1, p2, colour, thickness, cv2.LINE_AA)
    
    for pt in kp_upper:
        p = (int(pt[0]), int(pt[1]))
        if p != (0, 0):
            cv2.circle(frame, p, 4, colour, -1, cv2.LINE_AA)

def classify_video_type(punch_labels: list) -> str:
    """Determine if video is single-move drill or combo"""
    if not punch_labels:
        return "combo"
    most_common_count = Counter(punch_labels).most_common(1)[0][1]
    ratio = most_common_count / len(punch_labels)
    return "single_move" if ratio >= SINGLE_MOVE_THRESHOLD else "combo"

# ═══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION (Mahalanobis Distance)
# ═══════════════════════════════════════════════════════════════════════════════

def classify_with_bank(feat_vector: np.ndarray, 
                       feature_bank: dict, 
                       min_confidence: float = MIN_CONFIDENCE) -> tuple:
    """
    Classify punch using Mahalanobis distance.
    
    Why Mahalanobis > Euclidean:
    - Normalizes by standard deviation
    - Wrist speed (high variance) and elbow angle (low variance) equally weighted
    
    Returns: (label, confidence) or ("unknown", conf)
    """
    if not feature_bank:
        return "unknown", 0.0
    
    feat = np.asarray(feat_vector, dtype=np.float32)
    
    dists = {}
    for label, stats in feature_bank.items():
        centroid = np.array(stats["mean"], dtype=np.float32)
        std = np.maximum(np.array(stats["std"], dtype=np.float32), 1e-4)
        
        # Mahalanobis: (x - μ) / σ
        normalized_diff = (feat - centroid) / std
        dist = float(np.linalg.norm(normalized_diff))
        
        dists[label] = dist
    
    if not dists:
        return "unknown", 0.0
    
    # Best = minimum distance
    best_label = min(dists, key=dists.get)
    best_dist = dists[best_label]
    
    # Confidence: separation from worst match
    worst_dist = max(dists.values()) + 1e-8
    confidence = (worst_dist - best_dist) / worst_dist
    
    # Reject low confidence
    if confidence < min_confidence:
        return "unknown", round(confidence, 3)
    
    return best_label, round(confidence, 3)

# ═══════════════════════════════════════════════════════════════════════════════
# TEMPORAL SMOOTHING
# ═══════════════════════════════════════════════════════════════════════════════

def temporal_smooth(history: list, 
                    new_label: str, 
                    new_conf: float, 
                    window: int = TEMPORAL_WINDOW) -> tuple:
    """
    Smooth classifications over time using confidence-weighted voting.
    
    Example:
        history = []
        label1, conf1 = classify_with_bank(feat1, bank)
        smoothed1, history = temporal_smooth(history, label1, conf1)
        
        label2, conf2 = classify_with_bank(feat2, bank)  
        smoothed2, history = temporal_smooth(history, label2, conf2)
    
    Returns: (smoothed_label, updated_history)
    """
    if new_label == "unknown":
        return new_label, history
    
    # Append new
    history = list(history) + [(new_label, new_conf)]
    history = history[-window:]
    
    # Weighted voting
    votes = {}
    for lbl, conf in history:
        votes[lbl] = votes.get(lbl, 0) + conf
    
    smoothed = max(votes, key=votes.get)
    return smoothed, history