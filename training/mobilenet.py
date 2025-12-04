# ============================================================
# 🍌🥭 BALANCED FRUIT RIPENESS CLASSIFICATION - FINE-TUNING v3 (Optimized)
# ============================================================

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
import pickle
from google.colab import drive

# ============================================================
# 🔧 SETUP
# ============================================================
drive.mount('/content/drive')
base_path = '/content/drive/MyDrive/FruitRipeness'
data_dir = os.path.join(base_path, 'dataset_balanced_250')
model_path = os.path.join(base_path, 'models', 'best_finetune_v3.h5')

os.makedirs(os.path.dirname(model_path), exist_ok=True)

print("\n✅ Google Drive mounted!")
print(f"📂 Dataset: {data_dir}")

# ============================================================
# 📊 DATASET SETUP
# ============================================================
img_size = (224, 224)
batch_size = 32

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.2,
    zoom_range=0.2,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    validation_split=0.2
)

test_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = train_gen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_data = test_gen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

num_classes = train_data.num_classes
print(f"\n✅ Total Classes: {num_classes}")

# ============================================================
# 🧠 BUILD FINETUNED MODEL (MOBILENET)
# ============================================================
base = MobileNet(weights='imagenet', include_top=False, input_shape=(224,224,3))

# Unfreeze more layers (fine-tune deeper)
for layer in base.layers[:50]:
    layer.trainable = False
for layer in base.layers[50:]:
    layer.trainable = True

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
    Dense(num_classes, activation='softmax')
])

# ============================================================
# ⚙ COMPILE
# ============================================================
optimizer = Adam(learning_rate=2e-4)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ============================================================
# 📞 CALLBACKS
# ============================================================
callbacks = [
    ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
]

# ============================================================
# 🚀 TRAINING
# ============================================================
print("\n============================================================")
print("🚀 STARTING TRAINING (Fine-Tuning v3 - Optimized)")
print("============================================================")

start = time.time()
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=120,
    callbacks=callbacks
)
end = time.time()

print(f"\n⏱ Training time: {(end-start)/3600:.2f} hours")

# ============================================================
# 💾 SAVE MODEL
# ============================================================
model.save(model_path)
print(f"✅ Model saved at: {model_path}")

# ============================================================
# 🧪 EVALUATION
# ============================================================
print("\n📊 Evaluating on test data...")
test_loss, test_acc = model.evaluate(val_data)
print(f"🎯 Test Accuracy: {test_acc*100:.2f}%")

# Predictions
val_data.reset()
pred = model.predict(val_data)
pred_classes = np.argmax(pred, axis=1)
true_classes = val_data.classes
labels = list(val_data.class_indices.keys())

print("\n📋 Classification Report:")
print(classification_report(true_classes, pred_classes, target_names=labels))

# Create results directory
results_dir = os.path.join(base_path, 'results')
os.makedirs(results_dir, exist_ok=True)

print("\n📊 Saving results...")

# ============================================================
# 1. SAVE TRAINING HISTORY
# ============================================================
history_df = pd.DataFrame({
    'epoch': range(1, len(history.history['accuracy']) + 1),
    'accuracy': history.history['accuracy'],
    'val_accuracy': history.history['val_accuracy'],
    'loss': history.history['loss'],
    'val_loss': history.history['val_loss'],
    'learning_rate': history.history.get('learning_rate', [None] * len(history.history['accuracy']))
})

history_df.to_csv(os.path.join(results_dir, 'training_history.csv'), index=False)
print("✅ Training history saved to CSV")

# Save as pickle (untuk load nanti)
with open(os.path.join(results_dir, 'history.pkl'), 'wb') as f:
    pickle.dump(history.history, f)
print("✅ History pickle saved")

# ============================================================
# 2. SAVE TRAINING CURVES
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Accuracy
ax1.plot(history.history['accuracy'], label='Training', linewidth=2, color='#2ecc71')
ax1.plot(history.history['val_accuracy'], label='Validation', linewidth=2, color='#e74c3c')
ax1.set_title('Model Accuracy - Fine-Tuning v3', fontsize=16, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)

# Loss
ax2.plot(history.history['loss'], label='Training', linewidth=2, color='#2ecc71')
ax2.plot(history.history['val_loss'], label='Validation', linewidth=2, color='#e74c3c')
ax2.set_title('Model Loss', fontsize=16, fontweight='bold')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
print("✅ Training curves saved")
plt.show()

# ============================================================
# 3. SAVE CONFUSION MATRIX
# ============================================================
cm = confusion_matrix(true_classes, pred_classes)

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=labels,
    yticklabels=labels,
    square=True,
    cbar_kws={'label': 'Count'},
    annot_kws={'size': 12, 'weight': 'bold'}
)
plt.title(f'Confusion Matrix - Test Accuracy: {test_acc*100:.2f}%',
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
print("✅ Confusion matrix saved")
plt.show()

# ============================================================
# 4. SAVE CLASSIFICATION REPORT
# ============================================================
report_dict = classification_report(true_classes, pred_classes, target_names=labels, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv(os.path.join(results_dir, 'classification_report.csv'))
print("✅ Classification report saved to CSV")

# ============================================================
# 5. SAVE SUMMARY METRICS
# ============================================================
summary = {
    'test_accuracy': test_acc,
    'test_loss': test_loss,
    'total_epochs': len(history.history['accuracy']),
    'best_val_accuracy': max(history.history['val_accuracy']),
    'best_val_loss': min(history.history['val_loss']),
    'training_time_hours': (end-start)/3600,
    'num_classes': num_classes,
    'total_samples': train_data.samples + val_data.samples
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv(os.path.join(results_dir, 'summary_metrics.csv'), index=False)
print("✅ Summary metrics saved")

# ============================================================
# 6. SAVE PER-CLASS METRICS BAR CHART
# ============================================================
from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(true_classes, pred_classes, average=None)
recall = recall_score(true_classes, pred_classes, average=None)
f1 = f1_score(true_classes, pred_classes, average=None)

x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(14, 7))
bars1 = ax.bar(x - width, precision, width, label='Precision', color='#3498db')
bars2 = ax.bar(x, recall, width, label='Recall', color='#2ecc71')
bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#e74c3c')

ax.set_xlabel('Class', fontsize=14, fontweight='bold')
ax.set_ylabel('Score', fontsize=14, fontweight='bold')
ax.set_title('Per-Class Performance Metrics', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.legend(fontsize=12)
ax.set_ylim([0.9, 1.02])
ax.grid(True, alpha=0.3, axis='y')

for bar in bars1 + bars2 + bars3:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'per_class_metrics.png'), dpi=300, bbox_inches='tight')
print("✅ Per-class metrics saved")
plt.show()

# ============================================================
# 7. SAVE SAMPLE PREDICTIONS
# ============================================================
val_data.reset()
X_sample, y_sample = next(val_data)
y_pred_sample = model.predict(X_sample, verbose=0)

num_samples = min(12, len(X_sample))
indices = np.random.choice(len(X_sample), num_samples, replace=False)

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.ravel()

for i, idx in enumerate(indices):
    img = X_sample[idx]
    true_idx = np.argmax(y_sample[idx])
    pred_idx = np.argmax(y_pred_sample[idx])
    confidence = y_pred_sample[idx][pred_idx]

    true_label = labels[true_idx]
    pred_label = labels[pred_idx]

    color = 'green' if true_label == pred_label else 'red'

    axes[i].imshow(img)
    axes[i].set_title(
        f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1%}",
        fontsize=10,
        color=color,
        fontweight='bold'
    )
    axes[i].axis('off')

plt.suptitle('Sample Predictions', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'sample_predictions.png'), dpi=300, bbox_inches='tight')
print("✅ Sample predictions saved")
plt.show()

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*60)
print("✅ ALL RESULTS SAVED SUCCESSFULLY!")
print("="*60)
print(f"\n📁 Location: {results_dir}")
print("\n📄 Files saved:")
print("   ✓ training_history.csv")
print("   ✓ history.pkl")
print("   ✓ training_curves.png")
print("   ✓ confusion_matrix.png")
print("   ✓ classification_report.csv")
print("   ✓ summary_metrics.csv")
print("   ✓ per_class_metrics.png")
print("   ✓ sample_predictions.png")
print("\n🎉 Ready for thesis documentation!")
print("="*60)

# jika training/mobilenet.py ini gagal, coba pakai training/mobilenet_alt.py yang sudah tersedia