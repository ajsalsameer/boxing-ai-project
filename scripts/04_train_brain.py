import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization, MultiHeadAttention, LayerNormalization, Input, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ==========================================
#        GRU BRAIN CONFIGURATION
# ==========================================
DATA_FILE = "data/boxing_ultra_dataset.pkl"
MODEL_FILE = "classifier/boxing_brain_ultra.h5"
LABEL_FILE = "classifier/labels_ultra.pkl"

print(f"Loading Physics Dataset from {DATA_FILE}...")
with open(DATA_FILE, "rb") as f:
    X, y = pickle.load(f)

# Encode Labels (jab -> 0, cross -> 1...)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
classes = le.classes_
print(f"Classes found: {classes}")

# Save Labels for the Live System
with open(LABEL_FILE, "wb") as f:
    pickle.dump(classes, f)

# Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded)

# ==========================================
#        THE PHYSICS-AWARE GRU MODEL
# ==========================================
def build_gru_attention_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    
    # 1. ATTENTION LAYER (The "Focus")
    # This helps the AI ignore noise and focus on the "Snap" of the punch
    # We use MultiHeadAttention to look at the physics features
    attention = MultiHeadAttention(num_heads=4, key_dim=input_shape[-1])(inputs, inputs)
    x = LayerNormalization(epsilon=1e-6)(attention + inputs) # Add & Norm

    # 2. FAST GRU LAYERS (The "Pattern Recognizer")
    # GRU is faster than LSTM and great for velocity/acceleration data
    x = GRU(128, return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = GRU(128, return_sequences=False)(x) # Last layer summarizes the clip
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # 3. DECISION LAYERS
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Build it
input_shape = X_train.shape[1:] # (30 frames, 108 features)
model = build_gru_attention_model(input_shape, len(classes))

model.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

model.summary()

# ==========================================
#        TRAINING
# ==========================================
print("\n🥊 TRAINING PHYSICS-AWARE GRU...")

history = model.fit(
    X_train, y_train, 
    epochs=50, 
    batch_size=32, 
    validation_data=(X_test, y_test)
)

# ==========================================
#        SAVE
# ==========================================
model.save(MODEL_FILE)
print(f"\n✅ GRU BRAIN SAVED TO: {MODEL_FILE}")
print(f"✅ LABELS SAVED TO: {LABEL_FILE}")