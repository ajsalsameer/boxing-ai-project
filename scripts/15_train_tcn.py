#!/usr/bin/env python3
"""
PHASE 2 — Train Next-Move Predictor (ABSOLUTE PATHS VERSION)
=============================================================
CRITICAL FIX: Uses absolute paths so it works from any directory

Run from ANYWHERE:
  python scripts/15_train_next_move_predictor.py
  python 15_train_next_move_predictor.py
  
Both will save to: D:/projects/boxing-ai-project/models/
"""

import os, sys, json, pickle, numpy as np
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# ═══════════════════════════════════════════════════════════════════════════════
# CRITICAL FIX: Absolute paths
# ═══════════════════════════════════════════════════════════════════════════════
# Get the project root (parent of 'scripts' or current dir if no scripts/)
if Path(__file__).parent.name == 'scripts':
    PROJECT_ROOT = Path(__file__).parent.parent
else:
    PROJECT_ROOT = Path(__file__).parent

DATA_DIR  = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

SEQ_PATH    = DATA_DIR / "combo_sequences.pkl"
LABELS_PATH = DATA_DIR / "punch_labels.json"

print(f"📂  Project root: {PROJECT_ROOT}")
print(f"📂  Data dir:     {DATA_DIR}")
print(f"📂  Model dir:    {MODEL_DIR}\n")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
HISTORY_LEN   = 3
FEAT_DIM      = 84
EPOCHS        = 300
BATCH_SIZE    = 32
VAL_SPLIT     = 0.15
PATIENCE      = 40

LEARNING_RATE = 0.0008
DROPOUT_RATE  = 0.25
L2_LAMBDA     = 0.001


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_data():
    if not LABELS_PATH.exists() or not SEQ_PATH.exists():
        print(f"❌  Missing files:")
        print(f"    Labels:    {LABELS_PATH} (exists: {LABELS_PATH.exists()})")
        print(f"    Sequences: {SEQ_PATH} (exists: {SEQ_PATH.exists()})")
        sys.exit(1)
    
    with open(LABELS_PATH) as f:
        label_list = json.load(f)
    label_to_idx = {l: i for i, l in enumerate(label_list)}
    num_classes  = len(label_list)
    
    with open(SEQ_PATH, "rb") as f:
        raw = pickle.load(f)
    
    print(f"  Total sequences: {len(raw)}")
    
    # Check next-label distribution
    next_labels = [s["next"] for s in raw if s["next"] in label_to_idx]
    print("\n  Next-punch distribution:")
    for lbl, cnt in Counter(next_labels).most_common():
        print(f"    {lbl:12s} : {cnt:5d}  ({cnt/len(next_labels)*100:.1f}%)")
    
    # Check for missing classes
    missing_classes = set(label_list) - set(Counter(next_labels).keys())
    if missing_classes:
        print(f"\n  ⚠️   Classes with 0 samples: {missing_classes}")
        print(f"       (These will be excluded from the model)")
        
        # Remove empty classes from label list
        label_list = [l for l in label_list if l not in missing_classes]
        label_to_idx = {l: i for i, l in enumerate(label_list)}
        num_classes = len(label_list)
        print(f"       Active classes: {label_list}")
    
    X_pose = []
    X_hist = []
    y      = []
    
    for item in raw:
        if item["next"] not in label_to_idx:
            continue
        if any(l not in label_to_idx for l in item["history"]):
            continue
        
        # Pose feature
        pose_feat = np.zeros(FEAT_DIM, dtype=np.float32)
        if "history_feats" in item and item["history_feats"]:
            pose_feat = np.array(item["history_feats"][-1], dtype=np.float32)[:FEAT_DIM]
        
        # One-hot history
        one_hot = np.zeros(num_classes * HISTORY_LEN, dtype=np.float32)
        padded  = [None] * (HISTORY_LEN - len(item["history"])) + item["history"][-HISTORY_LEN:]
        for slot_i, lbl in enumerate(padded):
            if lbl is not None and lbl in label_to_idx:
                one_hot[slot_i * num_classes + label_to_idx[lbl]] = 1.0
        
        X_pose.append(pose_feat)
        X_hist.append(one_hot)
        y.append(label_to_idx[item["next"]])
    
    X_pose = np.array(X_pose, dtype=np.float32)
    X_hist = np.array(X_hist, dtype=np.float32)
    y      = np.array(y, dtype=np.int32)
    
    print(f"\n  Final dataset: {len(X_pose)} samples")
    print(f"  Pose shape: {X_pose.shape}")
    print(f"  History shape: {X_hist.shape}")
    
    return X_pose, X_hist, y, label_list, num_classes


def compute_class_weights(y_train, num_classes):
    counts = Counter(y_train)
    max_count = max(counts.values())
    
    weights = {}
    for i in range(num_classes):
        cnt = counts.get(i, 1)
        weights[i] = float(np.sqrt(max_count / cnt))
    
    print("\n  Class weights:")
    for i, w in sorted(weights.items()):
        print(f"    class {i}: {w:.3f}")
    
    return weights


def build_model(input_dim: int, num_classes: int):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(256, activation="relu", kernel_regularizer=l2(L2_LAMBDA)),
        BatchNormalization(),
        Dropout(DROPOUT_RATE),
        Dense(128, activation="relu", kernel_regularizer=l2(L2_LAMBDA)),
        BatchNormalization(),
        Dropout(DROPOUT_RATE),
        Dense(64,  activation="relu", kernel_regularizer=l2(L2_LAMBDA)),
        Dropout(DROPOUT_RATE * 0.6),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def train():
    print("=" * 70)
    print("  PHASE 2 — TRAIN HYBRID MLP (Absolute Paths)")
    print("=" * 70 + "\n")
    
    X_pose, X_hist, y, labels, num_classes = load_data()
    
    X_pose_train, X_pose_val, X_hist_train, X_hist_val, y_train, y_val = \
        train_test_split(X_pose, X_hist, y, test_size=VAL_SPLIT,
                         random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_pose_train_scaled = scaler.fit_transform(X_pose_train)
    X_pose_val_scaled   = scaler.transform(X_pose_val)
    
    print(f"\n  Pose features after scaling:")
    print(f"    Train mean: {X_pose_train_scaled.mean():.4f}  "
          f"std: {X_pose_train_scaled.std():.4f}")
    
    X_train = np.concatenate([X_pose_train_scaled, X_hist_train], axis=1)
    X_val   = np.concatenate([X_pose_val_scaled,   X_hist_val],   axis=1)
    
    input_dim = X_train.shape[1]
    print(f"\n  Input dim: {input_dim}  (pose={FEAT_DIM} + hist={num_classes}×{HISTORY_LEN})")
    print(f"  Train: {len(X_train)}   Val: {len(X_val)}\n")
    
    class_weights = compute_class_weights(y_train, num_classes)
    
    model = build_model(input_dim, num_classes)
    model.summary()
    
    print("\n🏋️  Training …\n")
    
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=[
            EarlyStopping(
                monitor='val_accuracy',
                patience=PATIENCE,
                mode='max',
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_accuracy',
                patience=PATIENCE // 2,
                factor=0.5,
                mode='max',
                min_lr=1e-6,
                verbose=1
            ),
        ],
        verbose=1,
    )
    
    print("\n" + "=" * 70)
    print("  EVALUATION")
    print("=" * 70)
    
    _, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"  Best validation accuracy: {val_acc:.1%}")
    
    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    
    # FIXED: Only use labels that actually appear in y_val
    active_label_indices = sorted(set(y_val))
    active_labels = [labels[i] for i in active_label_indices]
    
    print("\n" + classification_report(
        y_val, y_pred,
        labels=active_label_indices,
        target_names=active_labels,
        zero_division=0
    ))
    
    cm = confusion_matrix(y_val, y_pred, labels=active_label_indices)
    print("  Confusion Matrix:")
    print("         " + "  ".join(f"{l:>8s}" for l in active_labels))
    for i, lbl_idx in enumerate(active_label_indices):
        print(f"  {labels[lbl_idx]:>8s}  " + "  ".join(f"{x:>8d}" for x in cm[i]))
    
    # ── SAVE (with absolute paths) ───────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    model_path   = MODEL_DIR / "next_move_predictor.keras"
    scaler_path  = MODEL_DIR / "pose_scaler.pkl"
    mapping_path = MODEL_DIR / "label_to_idx.json"
    config_path  = MODEL_DIR / "model_config.json"
    
    print(f"\n  💾  Saving to: {MODEL_DIR}")
    
    model.save(str(model_path))
    print(f"      ✅  Model saved")
    
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"      ✅  Scaler saved")
    
    with open(mapping_path, "w") as f:
        json.dump({
            "label_to_idx": {l: i for i, l in enumerate(labels)},
            "idx_to_label": {str(i): l for i, l in enumerate(labels)}
        }, f, indent=2)
    print(f"      ✅  Label mapping saved")
    
    with open(config_path, "w") as f:
        json.dump({
            "model_type":   "hybrid_mlp",
            "input_dim":    int(input_dim),
            "num_classes":  num_classes,
            "feat_dim":     FEAT_DIM,
            "history_len":  HISTORY_LEN,
            "labels":       labels,
            "val_accuracy": round(float(val_acc), 4),
            "uses_scaler":  True,
        }, f, indent=2)
    print(f"      ✅  Config saved")
    
    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
        a1.plot(history.history["loss"],         label="Train")
        a1.plot(history.history["val_loss"],     label="Val")
        a1.set_title("Loss")
        a1.legend()
        
        a2.plot(history.history["accuracy"],     label="Train")
        a2.plot(history.history["val_accuracy"], label="Val")
        a2.set_title("Accuracy")
        a2.legend()
        
        fig.tight_layout()
        fig.savefig(str(MODEL_DIR / "training_curves.png"), dpi=120)
        print(f"      ✅  Training curves saved")
    except Exception as e:
        print(f"      ⚠️   Could not save plot: {e}")
    
    print(f"\n  🎯  Files saved successfully to: {MODEL_DIR}")
    print(f"      • next_move_predictor.keras")
    print(f"      • pose_scaler.pkl")
    print(f"      • label_to_idx.json")
    print(f"      • model_config.json")
    print(f"\n  🎯  NEXT:  streamlit run scripts/16_live_next_move_counter.py")
    print("=" * 70)


if __name__ == "__main__":
    train()