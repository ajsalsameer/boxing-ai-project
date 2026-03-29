#!/usr/bin/env python3
"""
Build Early Prediction Dataset - COMPLETE FIXED VERSION
With proper shuffling to prevent sorted class order
"""
import pickle
import numpy as np
import os
from collections import defaultdict

EARLY_WINDOW = 5
CLASS_WINDOW = 8
MAX_WINDOW = 12
EXTRACT_WINDOWS = [CLASS_WINDOW]

# Class-specific energy thresholds
ENERGY_THRESHOLDS = {
    'jab': 0.012,        # Fast, low energy
    'cross': 0.016,      # Moderate energy
    'hook': 0.020,       # High energy
    'left_hook': 0.020,
    'right_hook': 0.020,
    'uppercut': 0.022,   # Very high energy
}
DEFAULT_ENERGY_THRESHOLD = 0.018
MIN_ENERGY_RISE = 0.008

COOLDOWN_FRAMES = 15
LOOKBACK_FRAMES = 3

MERGE_HOOKS = True
MIN_SAMPLES_PER_CLASS = 250
MAX_SAMPLES_PER_CLASS = 400

INPUT_FILE = "data/boxing_ultra_dataset.pkl"
OUTPUT_FILE = "data/boxing_oracle_dataset.pkl"

def calculate_wrist_acceleration(video_sequence):
    """Calculate wrist-only acceleration across frames"""
    positions = video_sequence[:, :34].reshape(-1, 17, 2)
    wrist_positions = positions[:, [9, 10], :].reshape(-1, 4)
    velocity = np.diff(wrist_positions, axis=0, prepend=wrist_positions[0:1])
    acceleration = np.diff(velocity, axis=0, prepend=velocity[0:1])
    return np.linalg.norm(acceleration, axis=1)

def find_energy_spike_onset(energies, threshold, min_rise):
    """Find the START of energy rise (wind-up), not the peak"""
    spikes = []
    in_spike = False
    onset = None
    cooldown = 0
    
    for t in range(2, len(energies) - 2):
        if cooldown > 0:
            cooldown -= 1
            continue
        
        current = energies[t]
        prev = energies[t-1]
        next_val = energies[t+1]
        
        if not in_spike and current > threshold:
            onset = t
            for look_t in range(t-1, max(0, t-5), -1):
                if energies[look_t+1] - energies[look_t] > min_rise:
                    onset = look_t
                else:
                    break
            in_spike = True
        
        elif in_spike and (current >= next_val or current < threshold):
            peak = t
            spikes.append((onset, peak))
            in_spike = False
            cooldown = COOLDOWN_FRAMES
            onset = None
    
    return spikes

def extract_single_window(video, onset, peak, window_size, label):
    """Extract ONE window of specific size from a punch"""
    T = len(video)
    start = max(0, onset - LOOKBACK_FRAMES)
    end = start + window_size
    
    if end > T:
        slice_data = video[start:]
        if len(slice_data) == 0:
            return None
        padding_needed = end - T
        padding = np.tile(slice_data[-1], (padding_needed, 1))
        slice_data = np.vstack([slice_data, padding])
    else:
        slice_data = video[start:end]
    
    if slice_data.shape[0] != window_size:
        return None
    
    return (slice_data.astype(np.float32), label)

def smart_upsample(samples, target_count):
    """Intelligently upsample minority class with noise"""
    if len(samples) >= target_count:
        return samples
    if len(samples) == 0:
        return []
    
    upsampled = list(samples)
    needed = target_count - len(samples)
    
    for i in range(needed):
        idx = i % len(samples)
        original = samples[idx].copy()
        noise = np.random.normal(0, 0.008, original.shape).astype(np.float32)
        upsampled.append(original + noise)
        if len(upsampled) >= target_count:
            break
    
    return upsampled[:target_count]

def main():
    print(f"📂 Loading: {INPUT_FILE}")
    
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")
    
    with open(INPUT_FILE, "rb") as f:
        X, y = pickle.load(f)
    
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    
    print(f"Raw dataset: X={X.shape}, y={y.shape}")
    
    # Merge hook variants
    if MERGE_HOOKS:
        print("\n🔧 Merging hook variants...")
        y_before = y.copy()
        y = np.array([
            'hook' if 'hook' in str(label).lower() else str(label).lower()
            for label in y
        ])
        merged_count = np.sum(y == 'hook')
        original_count = np.sum(['hook' in str(label).lower() for label in y_before])
        print(f"   Merged {original_count} hook variants → {merged_count} total 'hook' samples")
    
    # Track samples per class per window size
    class_samples_by_window = {ws: defaultdict(list) for ws in EXTRACT_WINDOWS}
    
    print("\n" + "="*60)
    print("EXTRACTING WIND-UP WINDOWS (CLASS-SPECIFIC THRESHOLDS)")
    print("="*60)
    
    defense_labels = {"idle", "block", "duck", "slip_left", "slip_right", "slip_right_back", "slip_left_back"}
    
    total_processed = 0
    total_windows = 0
    
    for i in range(len(X)):
        video = X[i]
        label = str(y[i]).lower()
        
        # Skip defense moves
        if label in defense_labels:
            continue
        
        # Skip too-short sequences
        if len(video) < max(EXTRACT_WINDOWS):
            continue
        
        total_processed += 1
        
        # Calculate wrist acceleration
        try:
            energies = calculate_wrist_acceleration(video)
        except Exception:
            continue
        
        # Use class-specific threshold
        threshold = ENERGY_THRESHOLDS.get(label, DEFAULT_ENERGY_THRESHOLD)
        
        # Find energy spike onsets
        spikes = find_energy_spike_onset(energies, threshold, MIN_ENERGY_RISE)
        
        if len(spikes) == 0:
            continue
        
        # Extract windows from each spike
        for onset, peak in spikes:
            for window_size in EXTRACT_WINDOWS:
                result = extract_single_window(video, onset, peak, window_size, label)
                if result is not None:
                    class_samples_by_window[window_size][label].append(result[0])
                    total_windows += 1
    
    print(f"\n📊 Extraction Summary:")
    print(f"  Videos processed: {total_processed}")
    print(f"  Windows extracted: {total_windows}")
    
    # Balance dataset
    print("\n" + "="*60)
    print("BALANCING DATASET")
    print("="*60)
    
    for window_size in EXTRACT_WINDOWS:
        class_samples = class_samples_by_window[window_size]
        
        if len(class_samples) == 0:
            print(f"⚠️  No samples for window size {window_size}")
            continue
        
        print(f"\nWindow size: {window_size} frames")
        print("\nClass distribution (RAW):")
        for label in sorted(class_samples.keys()):
            count = len(class_samples[label])
            print(f"  {label:12s}: {count:4d} samples")
        
        # Calculate target count
        counts = [len(samples) for samples in class_samples.values()]
        target = max(int(np.median(counts)), MIN_SAMPLES_PER_CLASS)
        target = min(target, MAX_SAMPLES_PER_CLASS)
        
        print(f"\n🎯 Target: {target} samples per class")
        
        X_balanced = []
        y_balanced = []
        
        print("\nBalancing:")
        for label in sorted(class_samples.keys()):
            samples = class_samples[label]
            original_count = len(samples)
            
            if original_count == 0:
                continue
            
            # Upsample if needed
            if original_count < target:
                samples = smart_upsample(samples, target)
                print(f"  {label:12s}: {len(samples):4d} (upsampled from {original_count}, +{len(samples)-original_count})")
            
            # Downsample if needed
            elif original_count > target:
                indices = np.random.choice(original_count, target, replace=False)
                samples = [samples[i] for i in indices]
                print(f"  {label:12s}: {len(samples):4d} (downsampled from {original_count}, -{original_count-len(samples)})")
            
            else:
                print(f"  {label:12s}: {len(samples):4d} (unchanged)")
            
            X_balanced.extend(samples)
            y_balanced.extend([label] * len(samples))
        
        # Convert to numpy arrays
        X_final = np.array(X_balanced, dtype=np.float32)
        y_final = np.array(y_balanced)
        
        print(f"\nBefore shuffle: {X_final.shape}")
        
        # CRITICAL: SHUFFLE THE DATASET
        print("\n" + "="*60)
        print("SHUFFLING DATASET")
        print("="*60)
        
        shuffle_idx = np.random.RandomState(42).permutation(len(X_final))
        X_final = X_final[shuffle_idx]
        y_final = y_final[shuffle_idx]
        
        print(f"✅ Dataset shuffled with seed=42")
        
        # Verify shuffle worked
        print("\nVerification (first 20 labels):")
        print("  ", list(y_final[:20]))
        print("\nVerification (last 20 labels):")
        print("  ", list(y_final[-20:]))
        
        # Check distribution
        unique_first = np.unique(y_final[:100])
        unique_last = np.unique(y_final[-100:])
        
        if len(unique_first) >= 3 and len(unique_last) >= 3:
            print("\n✅ Shuffle successful - multiple classes in both ends")
        else:
            print("\n⚠️  Shuffle may have issues - check manually")
        
        # Save
        print("\n" + "="*60)
        print("SAVING DATASET")
        print("="*60)
        
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        
        with open(OUTPUT_FILE, "wb") as f:
            pickle.dump((X_final, y_final), f)
        
        print(f"\nFinal shape: X={X_final.shape}, y={y_final.shape}")
        
        # Final statistics
        unique, counts = np.unique(y_final, return_counts=True)
        
        print(f"\n📊 Final Class Distribution:")
        for cls, cnt in zip(unique, counts):
            pct = (cnt / len(y_final)) * 100
            print(f"  {cls:12s}: {cnt:4d} samples ({pct:5.1f}%)")
        
        imbalance_ratio = max(counts) / min(counts)
        print(f"\nImbalance ratio: {imbalance_ratio:.2f}:1", end="")
        
        if imbalance_ratio <= 1.1:
            print(" ✅ PERFECT")
        elif imbalance_ratio <= 1.3:
            print(" ✅ EXCELLENT")
        elif imbalance_ratio <= 1.5:
            print(" ✅ GOOD")
        else:
            print(" ⚠️  NEEDS WORK")
        
        print(f"\n✅ Saved to: {OUTPUT_FILE}")
        
        # Validation
        print("\n" + "="*60)
        print("VALIDATION")
        print("="*60)
        
        if np.any(np.isnan(X_final)) or np.any(np.isinf(X_final)):
            print("❌ Dataset contains NaN/Inf!")
        else:
            print("✅ No NaN/Inf values")
        
        print(f"✅ Consistent shape: {X_final.shape[1:]}")
        print(f"\nFeature range: [{X_final.min():.4f}, {X_final.max():.4f}]")
        
        print("\n" + "="*60)
        print("✅ DATASET BUILD COMPLETE!")
        print("="*60)
        print(f"\nNext steps:")
        print(f"1. Train: python train_early_prediction_oracle.py --data {OUTPUT_FILE}")
        print(f"2. Verify: python scripts/test_model.py")
        print(f"3. Live test: python scripts/13_live_counter.py --debug")

if __name__ == "__main__":
    main()