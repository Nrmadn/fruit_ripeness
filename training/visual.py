# ============================================================
# 📊 GENERATE VISUALIZATIONS ONLY (NO TRAINING)
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from google.colab import drive

# ============================================================
# 🔧 SETUP
# ============================================================
drive.mount('/content/drive')
base_path = '/content/drive/MyDrive/FruitRipeness'
model_path = os.path.join(base_path, 'models', 'best_finetune_v3.h5')
data_dir = os.path.join(base_path, 'dataset_balanced_250')
results_dir = os.path.join(base_path, 'results')

os.makedirs(results_dir, exist_ok=True)

print("✅ Loading model...")
model = keras.models.load_model(model_path)
print("✅ Model loaded successfully!")

# ============================================================
# 📂 LOAD TEST DATA
# ============================================================
print("\n📂 Loading test data...")
test_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

val_data = test_gen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

labels = list(val_data.class_indices.keys())
print(f"✅ Found {len(labels)} classes: {labels}")

# ============================================================
# 🧪 EVALUATE MODEL
# ============================================================
print("\n📊 Evaluating model...")
test_loss, test_acc = model.evaluate(val_data, verbose=1)
print(f"\n🎯 Test Accuracy: {test_acc*100:.2f}%")
print(f"🎯 Test Loss: {test_loss:.4f}")

# Get predictions
val_data.reset()
pred = model.predict(val_data, verbose=1)
pred_classes = np.argmax(pred, axis=1)
true_classes = val_data.classes

print("\n📋 Classification Report:")
print(classification_report(true_classes, pred_classes, target_names=labels))

# ============================================================
# 1. CONFUSION MATRIX
# ============================================================
print("\n📈 Creating confusion matrix...")
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
plt.savefig(os.path.join(results_dir, 'confusion_matrix_v3.png'), dpi=300, bbox_inches='tight')
print("✅ Confusion matrix saved!")
plt.show()

# ============================================================
# 2. PER-CLASS METRICS
# ============================================================
print("\n📈 Creating per-class metrics...")
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
plt.savefig(os.path.join(results_dir, 'per_class_metrics_v3.png'), dpi=300, bbox_inches='tight')
print("✅ Per-class metrics saved!")
plt.show()

# ============================================================
# 3. SAMPLE PREDICTIONS (BALANCED - ALL CLASSES) - FIXED
# ============================================================
print("\n📈 Creating sample predictions (2 per class)...")

# Collect predictions
val_data.reset()
all_predictions = []
all_images = []
all_true_labels = []

# Get all predictions WITH STOP CONDITION
total_batches = len(val_data)  # ← PENTING!
batch_count = 0

print(f"Processing {total_batches} batches...")

for X_batch, y_batch in val_data:
    pred_batch = model.predict(X_batch, verbose=0)
    for i in range(len(X_batch)):
        all_images.append(X_batch[i])
        all_true_labels.append(np.argmax(y_batch[i]))
        all_predictions.append(pred_batch[i])

    batch_count += 1
    if batch_count % 5 == 0:
        print(f"  Processed {batch_count}/{total_batches} batches...")

    # CRITICAL: Stop after all batches
    if batch_count >= total_batches:
        break

print(f"✅ Processed all {batch_count} batches")

# Convert to arrays
all_images = np.array(all_images)
all_true_labels = np.array(all_true_labels)
all_predictions = np.array(all_predictions)

print(f"✅ Total samples collected: {len(all_images)}")

# Select 2 samples per class (balanced)
selected_indices = []
for class_idx in range(len(labels)):
    # Find all samples of this class
    class_mask = all_true_labels == class_idx
    class_indices = np.where(class_mask)[0]

    print(f"  Class {labels[class_idx]}: {len(class_indices)} samples available")

    if len(class_indices) >= 2:
        # Randomly pick 2
        selected = np.random.choice(class_indices, 2, replace=False)
        selected_indices.extend(selected)
    elif len(class_indices) > 0:
        # If less than 2, take all available
        selected_indices.extend(class_indices)
        print(f"    ⚠ Only {len(class_indices)} sample(s) available")

print(f"\n✅ Selected {len(selected_indices)} samples for visualization")

# Shuffle selected indices
np.random.shuffle(selected_indices)

# Create visualization
num_samples = len(selected_indices)
rows = (num_samples + 3) // 4  # Calculate rows needed
fig, axes = plt.subplots(rows, 4, figsize=(16, 4*rows))
axes = axes.ravel()

for i, idx in enumerate(selected_indices):
    img = all_images[idx]
    true_idx = all_true_labels[idx]
    pred_idx = np.argmax(all_predictions[idx])
    confidence = all_predictions[idx][pred_idx]

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

# Hide unused subplots
for i in range(num_samples, len(axes)):
    axes[i].axis('off')

plt.suptitle('Sample Predictions (Balanced - All Classes)', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'sample_predictions_v3_balanced.png'), dpi=300, bbox_inches='tight')
print("✅ Balanced sample predictions saved!")
plt.show()

# ============================================================
# 4. CLASSIFICATION REPORT CSV
# ============================================================
print("\n📄 Saving classification report...")
report_dict = classification_report(true_classes, pred_classes, target_names=labels, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv(os.path.join(results_dir, 'classification_report_v3.csv'))
print("✅ Classification report saved!")

# ============================================================
# 5. SUMMARY METRICS
# ============================================================
print("\n📄 Saving summary metrics...")
summary = {
    'test_accuracy': test_acc,
    'test_loss': test_loss,
    'num_classes': len(labels),
    'total_test_samples': len(true_classes),
    'macro_precision': precision.mean(),
    'macro_recall': recall.mean(),
    'macro_f1': f1.mean()
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv(os.path.join(results_dir, 'summary_metrics_v3.csv'), index=False)
print("✅ Summary metrics saved!")

# ============================================================
# 6. OVERALL METRICS BAR CHART
# ============================================================
print("\n📈 Creating overall metrics chart...")
fig, ax = plt.subplots(figsize=(10, 6))

metrics = {
    'Test Accuracy': test_acc,
    'Macro Precision': precision.mean(),
    'Macro Recall': recall.mean(),
    'Macro F1-Score': f1.mean()
}

colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
bars = ax.barh(list(metrics.keys()), list(metrics.values()), color=colors)

ax.set_xlabel('Score', fontsize=14, fontweight='bold')
ax.set_title('Overall Model Performance Summary', fontsize=16, fontweight='bold')
ax.set_xlim([0.9, 1.0])
ax.grid(True, alpha=0.3, axis='x')

for bar, value in zip(bars, metrics.values()):
    ax.text(value + 0.002, bar.get_y() + bar.get_height()/2,
            f'{value:.2%}',
            va='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'overall_metrics_v3.png'), dpi=300, bbox_inches='tight')
print("✅ Overall metrics saved!")
plt.show()

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*60)
print("✅ ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
print("="*60)
print(f"\n📁 Location: {results_dir}")
print("\n📄 Files saved:")
print("   ✓ confusion_matrix_v3.png")
print("   ✓ per_class_metrics_v3.png")
print("   ✓ sample_predictions_v3.png")
print("   ✓ classification_report_v3.csv")
print("   ✓ summary_metrics_v3.csv")
print("   ✓ overall_metrics_v3.png")
print(f"\n🎯 Test Accuracy: {test_acc*100:.2f}%")
print("\n🎉 Ready for thesis documentation!")
print("="*60)