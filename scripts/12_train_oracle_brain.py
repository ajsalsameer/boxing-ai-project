#!/usr/bin/env python3
"""
Train Early Prediction Oracle - COMPLETE FIXED VERSION
With jab/cross boosting to prevent class collapse

Usage:
    python train_early_prediction_oracle.py --data data/boxing_oracle_dataset.pkl --epochs 60 --batch 32 --augment
"""
import os
import argparse
import pickle
import json
import random
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

EARLY_WINDOW = 5
CLASS_WINDOW = 8
MAX_WINDOW = 12
FEAT_DIM = 108

def augment_data(X, y, noise_std=0.015, speed_range=(0.85, 1.15)):
    X_aug = []
    y_aug = []
    for i in range(len(X)):
        seq = X[i]
        label = y[i]
        X_aug.append(seq)
        y_aug.append(label)
        noisy = seq + np.random.normal(0, noise_std, seq.shape).astype(np.float32)
        X_aug.append(noisy)
        y_aug.append(label)
        speed = np.random.uniform(*speed_range)
        T = len(seq)
        indices = np.linspace(0, T-1, int(T * speed))
        indices = np.clip(indices, 0, T-1).astype(int)
        warped = seq[indices]
        if len(warped) < T:
            pad = np.tile(warped[-1], (T - len(warped), 1))
            warped = np.vstack([warped, pad])
        else:
            warped = warped[:T]
        X_aug.append(warped.astype(np.float32))
        y_aug.append(label)
    return np.array(X_aug, dtype=np.float32), np.array(y_aug)

def build_early_warning_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(32, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs=inputs, outputs=outputs, name='early_warning')

def build_classifier_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(48, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Bidirectional(layers.LSTM(48))(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs=inputs, outputs=outputs, name='classifier')

def asymmetric_focal_loss(gamma=2.0, alpha=0.25, beta=1.5):
    def loss_fn(y_true, y_pred):
        y_true_ohe = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1])
        p_t = tf.reduce_sum(y_pred * y_true_ohe, axis=-1)
        ce = tf.keras.losses.categorical_crossentropy(y_true_ohe, y_pred)
        weight = tf.where(p_t < 0.5, beta, 1.0)
        focal = alpha * tf.pow(1.0 - p_t, gamma) * ce * weight
        return focal
    return loss_fn

def extract_early_windows(X, y, window_size):
    N, T, D = X.shape
    if T < window_size:
        pad = np.zeros((N, window_size - T, D), dtype=np.float32)
        return np.concatenate([X, pad], axis=1)
    else:
        return X[:, :window_size, :]

def train_models(args):
    print(f"Loading data: {args.data}")
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Data file not found: {args.data}")
    
    with open(args.data, "rb") as f:
        X, y = pickle.load(f)
    
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    
    print(f"Raw data: X={X.shape}, y={y.shape}")
    
    y = np.array(['hook' if 'hook' in str(label).lower() else str(label).lower() for label in y])
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes = le.classes_.tolist()
    print(f"Classes: {classes}")
    
    Path(os.path.dirname(args.labels) or ".").mkdir(parents=True, exist_ok=True)
    with open(args.labels, "wb") as f:
        pickle.dump(classes, f)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=SEED
    )
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    if args.augment:
        print("Augmenting training data...")
        X_train, y_train = augment_data(X_train, y_train)
        print(f"After augmentation: {len(X_train)} samples")
    
    X_flat = X_train.reshape(-1, X_train.shape[-1])
    mean = X_flat.mean(axis=0, keepdims=True).astype(np.float32)
    std = X_flat.std(axis=0, keepdims=True).astype(np.float32)
    std[std < 1e-6] = 1.0
    
    norm_path = args.model.replace('.h5', '.norm.npz')
    np.savez(norm_path, mean=mean, std=std)
    
    def normalize(x):
        return ((x - mean) / std).astype(np.float32)
    
    X_train_norm = normalize(X_train)
    X_test_norm = normalize(X_test)
    
    # CRITICAL: Compute class weights with JAB/CROSS BOOSTING
    cw_vals = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    cw = {i: float(w) for i, w in enumerate(cw_vals)}
    
    # BOOST JAB AND CROSS (prevent class collapse)
    try:
        jab_idx = list(classes).index('jab')
        cross_idx = list(classes).index('cross')
        cw[jab_idx] *= 1.5   # 50% boost for jabs
        cw[cross_idx] *= 1.3  # 30% boost for crosses
        print(f"✅ Boosted jab (×1.5) and cross (×1.3) weights")
    except ValueError:
        print("⚠️  Warning: Could not find jab/cross in classes")
    
    print(f"Final class weights: {cw}")
    
    # Train Early Warning Model (5 frames)
    print("\n" + "="*60)
    print("TRAINING EARLY WARNING MODEL (5-frame wind-up)")
    print("="*60)
    
    X_early_train = extract_early_windows(X_train_norm, y_train, EARLY_WINDOW)
    X_early_test = extract_early_windows(X_test_norm, y_test, EARLY_WINDOW)
    
    model_early = build_early_warning_model((EARLY_WINDOW, FEAT_DIM), len(classes))
    
    opt_early = optimizers.Adam(learning_rate=1e-3)
    model_early.compile(
        optimizer=opt_early,
        loss=asymmetric_focal_loss(gamma=2.0, alpha=0.25, beta=1.5),
        metrics=['accuracy']
    )
    
    model_early.summary()
    
    early_path = args.model.replace('.h5', '_early.h5')
    
    cbs_early = [
        callbacks.ModelCheckpoint(early_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        callbacks.EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1)
    ]
    
    history_early = model_early.fit(
        X_early_train, y_train,
        validation_data=(X_early_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch,
        class_weight=cw,
        callbacks=cbs_early,
        verbose=2
    )
    
    print(f"\n✅ Early Warning Model Saved: {early_path}")
    print(f"Best val_accuracy: {max(history_early.history['val_accuracy'])*100:.2f}%")
    
    # Train Classifier Model (8 frames)
    print("\n" + "="*60)
    print("TRAINING CLASSIFIER MODEL (8-frame extension)")
    print("="*60)
    
    X_class_train = extract_early_windows(X_train_norm, y_train, CLASS_WINDOW)
    X_class_test = extract_early_windows(X_test_norm, y_test, CLASS_WINDOW)
    
    model_class = build_classifier_model((CLASS_WINDOW, FEAT_DIM), len(classes))
    
    opt_class = optimizers.Adam(learning_rate=8e-4)
    model_class.compile(
        optimizer=opt_class,
        loss=asymmetric_focal_loss(gamma=2.0, alpha=0.25, beta=1.2),
        metrics=['accuracy']
    )
    
    model_class.summary()
    
    class_path = args.model.replace('.h5', '_classifier.h5')
    
    cbs_class = [
        callbacks.ModelCheckpoint(class_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        callbacks.EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1)
    ]
    
    history_class = model_class.fit(
        X_class_train, y_train,
        validation_data=(X_class_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch,
        class_weight=cw,
        callbacks=cbs_class,
        verbose=2
    )
    
    print(f"\n✅ Classifier Model Saved: {class_path}")
    print(f"Best val_accuracy: {max(history_class.history['val_accuracy'])*100:.2f}%")
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Early Warning: {max(history_early.history['val_accuracy'])*100:.2f}% @ 5 frames (83ms)")
    print(f"Classifier:    {max(history_class.history['val_accuracy'])*100:.2f}% @ 8 frames (133ms)")
    print(f"\nNormalization saved: {norm_path}")
    print(f"Classes: {classes}")
    print(f"\nNext: python scripts/13_live_counter.py --debug")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", "-d", default="data/boxing_oracle_dataset.pkl")
    p.add_argument("--model", "-m", default="classifier/boxing_oracle.h5")
    p.add_argument("--labels", "-l", default="classifier/labels_oracle.pkl")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--augment", action="store_true", default=True)
    args = p.parse_args()
    
    train_models(args)