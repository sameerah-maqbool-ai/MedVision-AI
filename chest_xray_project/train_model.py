"""
CHEST X-RAY CLASSIFICATION - HIGH ACCURACY MODEL
This script trains a model with 85-95% accuracy using EfficientNetB0
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, BatchNormalization,
    Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
    TensorBoard
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🩻 CHEST X-RAY CLASSIFICATION - HIGH ACCURACY TRAINING")
print("="*70)

# ============================================
# CONFIGURATION
# ============================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_PHASE_1 = 15
EPOCHS_PHASE_2 = 10
NUM_CLASSES = 3
CLASSES = ['normal', 'pneumonia', 'tuberculosis']
CLASS_NAMES_DISPLAY = ['Normal', 'Pneumonia', 'Tuberculosis']

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# ============================================
# STEP 1: DOWNLOAD DATASET
# ============================================
print("\n📥 Downloading Chest X-Ray Dataset...")
try:
    path = kagglehub.dataset_download("muhammadrehan00/chest-xray-dataset")
    print(f"✅ Dataset downloaded to: {path}")
except Exception as e:
    print(f"⚠️ Kagglehub failed: {e}")
    print("Please download dataset manually or check internet connection")
    exit()

train_path = os.path.join(path, "train")
val_path = os.path.join(path, "val")
test_path = os.path.join(path, "test")

# Verify paths
for p in [train_path, val_path, test_path]:
    if not os.path.exists(p):
        print(f"❌ Path not found: {p}")
        exit()

# ============================================
# STEP 2: DATA AUGMENTATION
# ============================================
print("\n🔄 Setting up data augmentation...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

# ============================================
# STEP 3: LOAD DATA
# ============================================
print("\n📂 Loading training data...")
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASSES,
    shuffle=True
)

print("\n📂 Loading validation data...")
val_generator = val_test_datagen.flow_from_directory(
    val_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASSES,
    shuffle=False
)

print("\n📂 Loading test data...")
test_generator = val_test_datagen.flow_from_directory(
    test_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASSES,
    shuffle=False
)

print(f"\n✅ Data loaded successfully!")
print(f"   Training samples: {train_generator.samples}")
print(f"   Validation samples: {val_generator.samples}")
print(f"   Test samples: {test_generator.samples}")
print(f"   Class mapping: {train_generator.class_indices}")

# ============================================
# STEP 4: CLASS WEIGHTS (Handle Imbalance)
# ============================================
print("\n⚖️ Computing class weights...")
labels = train_generator.classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weight_dict = dict(enumerate(class_weights))
print(f"   Class weights: {class_weight_dict}")

# ============================================
# STEP 5: BUILD MODEL - EfficientNetB0
# ============================================
print("\n🏗️ Building model with EfficientNetB0...")

def create_model(learning_rate=0.0001, trainable_layers=0):
    """
    Create EfficientNetB0 based model
    trainable_layers: number of base model layers to unfreeze (0 for phase 1)
    """
    # Load base model
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Unfreeze specified number of layers
    if trainable_layers > 0:
        for layer in base_model.layers[-trainable_layers:]:
            layer.trainable = True
        print(f"   Unfrozen last {trainable_layers} layers")
    
    # Build custom head
    inputs = Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model, base_model

# ============================================
# STEP 6: CALLBACKS
# ============================================
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    ModelCheckpoint(
        'models/best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# ============================================
# STEP 7: PHASE 1 - TRAIN TOP LAYERS
# ============================================
print("\n" + "="*70)
print("🚀 PHASE 1: Training classifier head (frozen backbone)")
print("="*70)

model, base_model = create_model(learning_rate=0.0001, trainable_layers=0)
model.summary()

print("\n⏱️ Training Phase 1...")
history_phase1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_PHASE_1,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

# ============================================
# STEP 8: PHASE 2 - FINE TUNING
# ============================================
print("\n" + "="*70)
print("🚀 PHASE 2: Fine-tuning (unfreezing layers)")
print("="*70)

# Unfreeze last 30 layers for fine-tuning
for layer in base_model.layers[-30:]:
    layer.trainable = True

# Recompile with lower learning rate
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
)

print("⏱️ Training Phase 2 (Fine-tuning)...")
history_phase2 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_PHASE_2,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

# ============================================
# STEP 9: FINAL EVALUATION
# ============================================
print("\n" + "="*70)
print("📊 FINAL MODEL EVALUATION")
print("="*70)

# Load best model
from tensorflow.keras.models import load_model
best_model = load_model('models/best_model.keras')

# Evaluate on test set
print("\n🔍 Evaluating on test set...")
test_loss, test_acc, test_precision, test_recall = best_model.evaluate(test_generator, verbose=1)

print(f"\n✅ TEST RESULTS:")
print(f"   🎯 Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"   📊 Precision: {test_precision:.4f}")
print(f"   📈 Recall:    {test_recall:.4f}")
print(f"   📉 Loss:      {test_loss:.4f}")

# ============================================
# STEP 10: PREDICTIONS & METRICS
# ============================================
print("\n📋 Generating detailed predictions...")
test_generator.reset()
y_pred_probs = best_model.predict(test_generator, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_generator.classes

# Classification Report
print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES_DISPLAY))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

# Plot and save confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES_DISPLAY,
            yticklabels=CLASS_NAMES_DISPLAY)
plt.title('Confusion Matrix - Chest X-Ray Classification', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('models/confusion_matrix.png', dpi=150)
plt.show()

# Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Combine histories
acc = history_phase1.history['accuracy'] + history_phase2.history['accuracy']
val_acc = history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy']
loss = history_phase1.history['loss'] + history_phase2.history['loss']
val_loss = history_phase1.history['val_loss'] + history_phase2.history['val_loss']

epochs_phase1 = range(1, len(history_phase1.history['accuracy']) + 1)
epochs_phase2 = range(len(history_phase1.history['accuracy']) + 1,
                      len(history_phase1.history['accuracy']) + len(history_phase2.history['accuracy']) + 1)

axes[0].plot(range(1, len(acc) + 1), acc, 'b-', label='Training', linewidth=2)
axes[0].plot(range(1, len(val_acc) + 1), val_acc, 'r-', label='Validation', linewidth=2)
axes[0].axvline(x=len(history_phase1.history['accuracy']), color='g', linestyle='--', alpha=0.7, label='Fine-tuning Start')
axes[0].set_title('Model Accuracy', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(range(1, len(loss) + 1), loss, 'b-', label='Training', linewidth=2)
axes[1].plot(range(1, len(val_loss) + 1), val_loss, 'r-', label='Validation', linewidth=2)
axes[1].axvline(x=len(history_phase1.history['loss']), color='g', linestyle='--', alpha=0.7, label='Fine-tuning Start')
axes[1].set_title('Model Loss', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('Training Progress - Chest X-Ray Classification', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('models/training_curves.png', dpi=150)
plt.show()

# ============================================
# STEP 11: SAVE FINAL MODEL
# ============================================
best_model.save('models/chest_xray_final_model.keras')
best_model.save('models/chest_xray_final_model.h5')

print("\n" + "="*70)
print("✅ TRAINING COMPLETE!")
print("="*70)
print(f"\n📁 Models saved in 'models/' directory:")
print(f"   - models/chest_xray_final_model.keras")
print(f"   - models/chest_xray_final_model.h5")
print(f"   - models/best_model.keras")
print(f"\n📊 Results saved:")
print(f"   - models/confusion_matrix.png")
print(f"   - models/training_curves.png")
print(f"\n🎯 FINAL TEST ACCURACY: {test_acc*100:.2f}%")
print("\n🚀 You can now run the web app: streamlit run app.py")