import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
import os

print("="*60)
print("🔧 REBUILDING MODEL FOR FLASK COMPATIBILITY")
print("="*60)

# ============================================================
# CONFIG
# ============================================================
OLD_MODEL = '/content/drive/MyDrive/FruitRipeness/models/best_finetune_v3.h5'
NEW_KERAS = '/content/drive/MyDrive/FruitRipeness/models/mobilenet_flask.keras'
NEW_H5 = '/content/drive/MyDrive/FruitRipeness/models/mobilenet_flask.h5'

# ============================================================
# LOAD OLD MODEL
# ============================================================
print("\n🔄 Step 1: Loading old model...")
try:
    old_model = tf.keras.models.load_model(OLD_MODEL, compile=False)
    print("✅ Model loaded successfully")
    print(f"   Input: {old_model.input_shape}")
    print(f"   Output: {old_model.output_shape}")

    # Extract weights
    old_weights = old_model.get_weights()
    print(f"✅ Extracted {len(old_weights)} weight arrays")

except Exception as e:
    print(f"❌ Failed to load model: {e}")
    raise

# ============================================================
# REBUILD ARCHITECTURE
# ============================================================
print("\n🏗 Step 2: Rebuilding architecture...")

# MobileNet base (no ImageNet weights)
base = MobileNet(
    weights=None,
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze/unfreeze layers
for layer in base.layers[:50]:
    layer.trainable = False
for layer in base.layers[50:]:
    layer.trainable = True

# Complete model
model = Sequential([
    base,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(6, activation='softmax')
], name='MobileNet_FruitClassifier')

print("✅ Architecture rebuilt")
print(f"   Total params: {model.count_params():,}")

# ============================================================
# TRANSFER WEIGHTS
# ============================================================
print("\n📦 Step 3: Transferring weights...")
model.set_weights(old_weights)
print("✅ Weights transferred successfully")

# ============================================================
# COMPILE
# ============================================================
print("\n⚙ Step 4: Compiling model...")
model.compile(
    optimizer=tf.keras.optimizers.Adam(2e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print("✅ Model compiled")

# ============================================================
# SAVE FORMAT 1: .keras
# ============================================================
print(f"\n💾 Step 5a: Saving as .keras...")
model.save(NEW_KERAS)
keras_size = os.path.getsize(NEW_KERAS) / (1024*1024)
print(f"✅ Saved: {NEW_KERAS}")
print(f"   Size: {keras_size:.2f} MB")

# ============================================================
# SAVE FORMAT 2: .h5
# ============================================================
print(f"\n💾 Step 5b: Saving as .h5...")
model.save(NEW_H5)
h5_size = os.path.getsize(NEW_H5) / (1024*1024)
print(f"✅ Saved: {NEW_H5}")
print(f"   Size: {h5_size:.2f} MB")

# ============================================================
# VERIFY
# ============================================================
print("\n🧪 Step 6: Verifying...")

# Test .keras
test1 = tf.keras.models.load_model(NEW_KERAS)
print(f"✅ .keras loads OK: {test1.output_shape}")

# Test .h5
test2 = tf.keras.models.load_model(NEW_H5)
print(f"✅ .h5 loads OK: {test2.output_shape}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*60)
print("✅ SUCCESS! MODEL REBUILD COMPLETE")
print("="*60)
print(f"\n📁 Files saved:")
print(f"   1. {NEW_KERAS} ({keras_size:.2f} MB)")
print(f"   2. {NEW_H5} ({h5_size:.2f} MB)")
print(f"\n📥 Download EITHER file:")
print(f"   • For TF 2.12+: Use .h5 file")
print(f"   • For TF 2.15+: Use .keras file")
print("="*60)