import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

# ==========================================
#   ULTIMATE BRAIN (SIMPLE & CLEAN)
# ==========================================
DATA_FILE = "data/boxing_ultra_dataset.pkl"
MODEL_FILE = "classifier/boxing_brain_hybrid.h5"
LABEL_FILE = "classifier/labels_ultra.pkl"

print(f"Loading Dataset: {DATA_FILE}")
with open(DATA_FILE, "rb") as f:
    X, y = pickle.load(f)

# --- CLEANUP: Force any lingering 'left/right' labels to 'hook' ---
y = np.array(['hook' if 'hook' in label else label for label in y])

print("⚖️ Auto-Balancing Dataset...")
unique_classes, counts = np.unique(y, return_counts=True)
max_count = max(counts)

X_balanced, y_balanced = [], []
for cls in unique_classes:
    indices = np.where(y == cls)[0]
    X_class, y_class = X[indices], y[indices]
    if len(X_class) < max_count:
        X_class, y_class = resample(X_class, y_class, replace=True, n_samples=max_count, random_state=42)
    X_balanced.append(X_class)
    y_balanced.append(y_class)

X = np.concatenate(X_balanced)
y = np.concatenate(y_balanced)

le = LabelEncoder()
y_encoded = le.fit_transform(y)
classes = le.classes_
print(f"Classes: {classes}") # Should NOT see left_hook/right_hook anymore

with open(LABEL_FILE, "wb") as f:
    pickle.dump(classes, f)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

# --- BRAIN ARCHITECTURE ---
def squeeze_excite_block(input_tensor, ratio=16):
    filters = input_tensor.shape[-1]
    se = layers.GlobalAveragePooling1D()(input_tensor)
    se = layers.Dense(filters // ratio, activation='relu', use_bias=False)(se)
    se = layers.Dense(filters, activation='sigmoid', use_bias=False)(se)
    se = layers.Reshape((1, filters))(se)
    x = layers.Multiply()([input_tensor, se])
    return x

def hybrid_block(inputs):
    x_skip = layers.Dense(64)(inputs) 
    x = layers.Conv1D(filters=64, kernel_size=3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = squeeze_excite_block(x) 
    x = layers.Add()([x, x_skip]) 
    x = layers.Activation('relu')(x) 
    x = layers.Dense(64, activation="linear")(x) 
    attn_output = layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    attn_output = layers.Dropout(0.2)(attn_output)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output) 
    ffn = layers.Conv1D(filters=64, kernel_size=1, activation="relu")(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ffn) 
    return x

inputs = layers.Input(shape=(30, 108))
x = hybrid_block(inputs)
x = hybrid_block(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.001))(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(classes), activation="softmax")(x)

model = models.Model(inputs=inputs, outputs=outputs)
optimizer = optimizers.AdamW(learning_rate=0.001, weight_decay=0.004)
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

print("\n🧠 TRAINING (CLEAN HOOKS)...")
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

history = model.fit(X_train, y_train, batch_size=32, epochs=120, validation_data=(X_test, y_test), callbacks=[early_stop, reduce_lr], verbose=1)

model.save(MODEL_FILE)
print(f"✅ CLEAN BRAIN SAVED. Best Val Accuracy: {max(history.history['val_accuracy'])*100:.2f}%")