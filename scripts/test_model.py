#!/usr/bin/env python3
"""
Diagnostic script to verify all classes are being predicted
"""
import pickle
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import sys

def main():
    print("Starting diagnostic...")
    
    # Step 1: Load model
    print("\n[1/5] Loading classifier model...")
    try:
        model_class = tf.keras.models.load_model("classifier/boxing_oracle_classifier.h5", compile=False)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Step 2: Load dataset
    print("\n[2/5] Loading dataset...")
    try:
        with open("data/boxing_oracle_dataset.pkl", "rb") as f:
            X, y = pickle.load(f)
        print(f"✅ Dataset loaded: X={X.shape}, y={y.shape}")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return

    # Step 3: Load labels
    print("\n[3/5] Loading labels...")
    try:
        with open("classifier/labels_oracle.pkl", "rb") as f:
            classes = pickle.load(f)
        print(f"✅ Classes loaded: {classes}")
    except Exception as e:
        print(f"❌ Failed to load labels: {e}")
        return

    # Step 4: Load normalization
    print("\n[4/5] Loading normalization...")
    try:
        norm = np.load("classifier/boxing_oracle.norm.npz")
        mean = norm["mean"]
        std = norm["std"]
        print(f"✅ Normalization loaded: mean shape={mean.shape}, std shape={std.shape}")
    except Exception as e:
        print(f"❌ Failed to load normalization: {e}")
        return

    # Step 5: Normalize and predict
    print("\n[5/5] Running predictions...")
    try:
        # Normalize entire dataset
        X_norm = (X - mean) / std
        print(f"✅ Dataset normalized")
        
        # Use last 320 samples as test set (20% of 1600)
        X_test = X_norm[-320:]
        y_test_labels = y[-320:]
        
        # Convert labels to indices
        label_to_idx = {label: i for i, label in enumerate(classes)}
        y_test = np.array([label_to_idx[str(label).lower()] for label in y_test_labels])
        
        print(f"✅ Test set prepared: {len(X_test)} samples")
        
        # Predict (use first 8 frames for 8-frame classifier)
        print("Running model predictions...")
        preds = model_class.predict(X_test[:, :8, :], verbose=0)
        y_pred = np.argmax(preds, axis=1)
        
        print(f"✅ Predictions complete")
        
    except Exception as e:
        print(f"❌ Failed during prediction: {e}")
        import traceback
        traceback.print_exc()
        return

    # Report results
    print("\n" + "="*60)
    print("PER-CLASS PERFORMANCE ON TEST SET")
    print("="*60)
    try:
        report = classification_report(y_test, y_pred, target_names=classes, digits=3)
        print(report)
    except Exception as e:
        print(f"Could not generate report: {e}")

    # Prediction distribution
    print("\n" + "="*60)
    print("PREDICTION DISTRIBUTION")
    print("="*60)
    
    pred_counts = np.bincount(y_pred, minlength=len(classes))
    
    for cls, cnt in zip(classes, pred_counts):
        pct = (cnt / len(y_pred)) * 100
        bar = "█" * max(1, int(pct / 2))
        print(f"  {cls:12s}: {cnt:3d} predictions ({pct:5.1f}%) {bar}")

    # Confusion Matrix
    print("\n" + "="*60)
    print("CONFUSION MATRIX")
    print("="*60)
    
    cm = confusion_matrix(y_test, y_pred)
    
    # Header
    header_str = "True \\ Pred"
    print(f"\n{header_str:12s}", end="")
    for cls in classes:
        print(f"{cls:>10s}", end="")
    print()
    print("-" * 60)
    
    # Rows
    for i, cls in enumerate(classes):
        print(f"{cls:12s}", end="")
        for j in range(len(classes)):
            val = cm[i][j]
            if i == j and val > 50:
                marker = "✅"
            elif val > 10:
                marker = "⚠️ "
            else:
                marker = "   "
            print(f"{marker}{val:>6d}", end="")
        print()

    # Actual vs Predicted counts
    print("\n" + "="*60)
    print("ACTUAL vs PREDICTED COUNTS")
    print("="*60)
    
    actual_counts = np.bincount(y_test, minlength=len(classes))
    
    print(f"{'Class':12s} {'Actual':>8s} {'Predicted':>10s} {'Status':>10s}")
    print("-" * 60)
    for i, cls in enumerate(classes):
        actual = actual_counts[i]
        predicted = pred_counts[i]
        diff = abs(actual - predicted)
        
        if predicted == 0:
            status = "❌ IGNORED"
        elif diff < 10:
            status = "✅ GOOD"
        else:
            status = "⚠️  BIASED"
        
        print(f"{cls:12s} {actual:>8d} {predicted:>10d} {status:>10s}")

    # Final verdict
    print("\n" + "="*60)
    print("FINAL VERDICT")
    print("="*60)
    
    min_pred = pred_counts.min()
    max_pred = pred_counts.max()
    
    if min_pred == 0:
        ignored = [classes[i] for i in range(len(classes)) if pred_counts[i] == 0]
        print(f"\n❌ MODEL IS IGNORING CLASSES: {ignored}")
        print("\n🔧 FIX REQUIRED:")
        print("   1. Increase class weights for ignored classes")
        print("   2. Retrain with: cw[jab_idx] *= 2.0, cw[cross_idx] *= 1.8")
    elif min_pred < 20:
        underrep = classes[np.argmin(pred_counts)]
        print(f"\n⚠️  LOW PREDICTIONS FOR: {underrep} (only {min_pred} predictions)")
        print(f"\n✅ But model is working! All classes detected.")
        print("   Test live - it should work, just watch for bias.")
    else:
        imbalance = max_pred / min_pred
        print(f"\n✅ ALL CLASSES WELL PREDICTED!")
        print(f"   Min: {min_pred}, Max: {max_pred}, Ratio: {imbalance:.2f}:1")
        print("\n🚀 Model is ready for live testing!")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)