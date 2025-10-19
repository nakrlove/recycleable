# EffNetV2S_M1_safe_train.py
# M1-safe training script: EfficientNetV2S + class-weight + TF-preproc-augmentation + Top-1/3/5 + Confusion Matrix
# Usage: python EffNetV2S_M1_safe_train.py

import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, metrics
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# 1) Config (변경 가능)
# -------------------------
DATA_DIR = "dataset_10000"   # train/val/test 폴더를 포함한 루트
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

IMG_SIZE = (224, 224)
BATCH_SIZE = 8               # M1 안전 권장 (메모리 여유 있으면 16으로 올려도 됨)
EPOCHS_STAGE1 = 12
EPOCHS_STAGE2 = 8
FINE_TUNE_FRACTION = 0.25    # 상위 25% 레이어만 미세조정
SHUFFLE_BUFFER = 256         # 안전한 shuffle 버퍼
PREFETCH_SIZE = 2            # 과도한 prefetch 방지
LEARNING_RATE_STAGE1 = 1e-3
LEARNING_RATE_STAGE2 = 1e-5
MODEL_SAVE_STAGE1 = "EffNetV2S_stage1_best.keras"
MODEL_SAVE_STAGE2 = "EffNetV2S_stage2_best.keras"
FINAL_MODEL = "EffNetV2S_final.keras"
SEED = 42

# -------------------------
# 2) Device info
# -------------------------
print("TensorFlow version:", tf.__version__)
print("Physical devices:", tf.config.list_physical_devices())
# (Optional) limit thread parallelism if you want:
# tf.config.threading.set_inter_op_parallelism_threads(2)
# tf.config.threading.set_intra_op_parallelism_threads(2)

# -------------------------
# 3) Datasets (batched)
# -------------------------
print("\nLoading datasets (this lists files and builds pipelines)...")
train_ds = image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True,
    seed=SEED
)

val_ds = image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

test_ds = image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Found classes ({num_classes}):", class_names)
# compute counts from filesystem for class_weight (more reliable than dataset.cardinality)
class_counts = []
for cname in class_names:
    p = os.path.join(TRAIN_DIR, cname)
    n = 0
    if os.path.isdir(p):
        n = sum(1 for _ in os.listdir(p) if os.path.isfile(os.path.join(p, _)))
    class_counts.append(n)
class_counts = np.array(class_counts, dtype=np.int64)
total_train = int(class_counts.sum())
print("Class counts (train):", dict(zip(class_names, class_counts.tolist())))
print("Total train samples:", total_train)

# safe class weights with clipping
raw_weights = total_train / (num_classes * np.maximum(class_counts, 1))
clipped = np.clip(raw_weights, a_min=None, a_max=10.0)  # cap to 10
class_weight = {i: float(w) for i, w in enumerate(clipped)}
print("Class Weights:", class_weight)

steps_per_epoch = math.ceil(total_train / BATCH_SIZE)
print(f"Batch size: {BATCH_SIZE}, Steps/epoch: {steps_per_epoch}")

# -------------------------
# 4) Data pipeline (lightweight)
#    - no heavy cache
#    - no python-level augmentation
#    - small shuffle + prefetch
# -------------------------
# Keep the dataset pipeline minimal; augmentation handled in-model by preprocessing layers
train_ds = train_ds.shuffle(buffer_size=SHUFFLE_BUFFER, seed=SEED).prefetch(PREFETCH_SIZE)
val_ds = val_ds.prefetch(PREFETCH_SIZE)
test_ds = test_ds.prefetch(PREFETCH_SIZE)

# -------------------------
# 5) Model with tf.keras preprocessing layers
#    Preprocessing inside model runs on device (Metal) and lowers CPU overhead.
# -------------------------
# Build augmentation stack (inside model)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.08),
    layers.RandomContrast(0.08),
    # If you want translation/crop, be careful with shape; keep it simple for stability.
], name="data_augmentation")

# Input + optional rescaling (EffNetV2 expects inputs scaled - we will use preprocess_input)
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

inputs = layers.Input(shape=IMG_SIZE + (3,), name="input_image")
x = layers.Lambda(lambda img: tf.cast(img, tf.float32))(inputs)     # ensure float32
x = layers.Lambda(preprocess_input, name="preprocess")(x)          # EfficientNetV2 preprocess (-1..1)
x = data_augmentation(x)                                           # augmentation (runs on device)
# Backbone
backbone = EfficientNetV2S(include_top=False, input_shape=IMG_SIZE+(3,), weights="imagenet")
backbone.trainable = False
x = backbone(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.35)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs, name="EffNetV2S_m1safe")

# Compile stage1
model.compile(
    optimizer=optimizers.Adam(learning_rate=LEARNING_RATE_STAGE1),
    loss="categorical_crossentropy",
    metrics=[
        "accuracy",
        metrics.TopKCategoricalAccuracy(k=3, name="top3"),
        metrics.TopKCategoricalAccuracy(k=5, name="top5")
    ],
)

model.summary()

# -------------------------
# 6) Callbacks
# -------------------------
cb_stage1 = [
    callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
    callbacks.ModelCheckpoint(MODEL_SAVE_STAGE1, save_best_only=True, monitor="val_loss", verbose=1)
]

# -------------------------
# 7) Stage 1: Train (feature extraction)
# -------------------------
print("\n=== Stage 1: Feature extraction training ===")
history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE1,
    steps_per_epoch=steps_per_epoch,
    class_weight=class_weight,
    callbacks=cb_stage1,
    verbose=2
)

# -------------------------
# 8) Stage 2: Fine-tune (unfreeze top layers)
# -------------------------
print("\n=== Stage 2: Fine-tuning ===")
# load best stage1 weights if callback saved them
if os.path.exists(MODEL_SAVE_STAGE1):
    print("Loading best weights from stage1:", MODEL_SAVE_STAGE1)
    model.load_weights(MODEL_SAVE_STAGE1)

# Unfreeze top fraction of backbone
num_layers = len(backbone.layers)
fine_tune_at = int(num_layers * (1 - FINE_TUNE_FRACTION))
for i, layer in enumerate(backbone.layers):
    layer.trainable = (i >= fine_tune_at)
print(f"Unfreezing layers from {fine_tune_at} / {num_layers} (trainable count: {sum([1 for l in backbone.layers if l.trainable])})")

# recompile with lower LR
model.compile(
    optimizer=optimizers.Adam(learning_rate=LEARNING_RATE_STAGE2),
    loss="categorical_crossentropy",
    metrics=[
        "accuracy",
        metrics.TopKCategoricalAccuracy(k=3, name="top3"),
        metrics.TopKCategoricalAccuracy(k=5, name="top5")
    ],
)

cb_stage2 = [
    callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
    callbacks.ModelCheckpoint(MODEL_SAVE_STAGE2, save_best_only=True, monitor="val_loss", verbose=1)
]

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE2,
    steps_per_epoch=steps_per_epoch,
    class_weight=class_weight,
    callbacks=cb_stage2,
    verbose=2
)

# -------------------------
# 9) Save final (inference) model (no optimizer state)
# -------------------------
final_path = FINAL_MODEL
model.save(final_path, include_optimizer=False, save_format="keras")
print("Saved final model to", final_path)

# -------------------------
# 10) Evaluation: Top-1/3/5 + Confusion Matrix + Classification Report
# -------------------------
# Ensure best weights from stage2 loaded
if os.path.exists(MODEL_SAVE_STAGE2):
    model.load_weights(MODEL_SAVE_STAGE2)

y_true = []
y_pred_top1 = []
y_pred_top3_hits = []
y_pred_top5_hits = []

print("\nRunning evaluation on test set...")
for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    true_idx = np.argmax(labels.numpy(), axis=1)
    y_true.extend(true_idx.tolist())
    top1 = np.argmax(preds, axis=1)
    y_pred_top1.extend(top1.tolist())
    top3 = np.argsort(preds, axis=1)[:, -3:]
    top5 = np.argsort(preds, axis=1)[:, -5:]
    # top-k hits (1/0)
    y_pred_top3_hits.extend([1 if t in top else 0 for top, t in zip(top3, true_idx)])
    y_pred_top5_hits.extend([1 if t in top else 0 for top, t in zip(top5, true_idx)])

y_true = np.array(y_true)
y_pred_top1 = np.array(y_pred_top1)
top1_acc = np.mean(y_true == y_pred_top1)
top3_acc = np.mean(y_pred_top3_hits)
top5_acc = np.mean(y_pred_top5_hits)

print(f"Test Top-1 Accuracy: {top1_acc*100:.2f}%")
print(f"Test Top-3 Accuracy: {top3_acc*100:.2f}%")
print(f"Test Top-5 Accuracy: {top5_acc*100:.2f}%")

# Confusion matrix (top-1)
cm = confusion_matrix(y_true, y_pred_top1)
plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=False, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Top-1)")
plt.tight_layout()
plt.show()

print("\nClassification Report (Top-1):")
print(classification_report(y_true, y_pred_top1, target_names=class_names, zero_division=0))
