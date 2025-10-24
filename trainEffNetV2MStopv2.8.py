import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, metrics
from tensorflow.keras.applications import EfficientNetV2S, efficientnet_v2
from tensorflow.keras.preprocessing import image_dataset_from_directory
import tensorflow_addons as tfa

# -------------------------
# Config
# -------------------------
DATA_DIR = "dataset"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_STAGE1 = 20
EPOCHS_STAGE2 = 15
FINE_TUNE_FRACTION = 0.5  # Stage2 backbone trainable fraction
SEED = 42

LEARNING_RATE_STAGE1 = 1e-3
LEARNING_RATE_STAGE2 = 1e-4

MODEL_SAVE_STAGE1 = "EffNetV2S_stage1_best.keras"
MODEL_SAVE_STAGE2 = "EffNetV2S_stage2_best.keras"
FINAL_MODEL = "EffNetV2S_final.keras"

# -------------------------
# Dataset
# -------------------------
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

print("✅ trainEffNetV2MStopv2.8 version ✅")
# -------------------------
# Class Weights (imbalance 대응)
# -------------------------
class_counts = np.array([len(os.listdir(os.path.join(TRAIN_DIR, c))) for c in class_names])
total_train = class_counts.sum()
weights = total_train / (num_classes * np.maximum(class_counts, 1))
weights = np.clip(weights, None, 20.0)  # 소수 클래스 더 강하게
class_weight = {i: float(w) for i, w in enumerate(weights)}

# -------------------------
# Data Augmentation (강화)
# -------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomBrightness(0.2)
], name="data_augmentation")

# -------------------------
# Prefetch
# -------------------------
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(1024).prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# -------------------------
# Build Model
# -------------------------
inputs = layers.Input(shape=IMG_SIZE + (3,))
x = layers.Rescaling(1./255)(inputs)
x = efficientnet_v2.preprocess_input(x)
x = data_augmentation(x)

backbone = EfficientNetV2S(include_top=False, input_shape=IMG_SIZE + (3,), weights="imagenet")
backbone.trainable = False  # Stage1 frozen
x = backbone(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs)
model.compile(
    optimizer=optimizers.Adam(learning_rate=LEARNING_RATE_STAGE1),
    loss=tfa.losses.SigmoidFocalCrossEntropy(),
    metrics=["accuracy", metrics.TopKCategoricalAccuracy(k=3, name="top3")]
)

# -------------------------
# Stage1 Training
# -------------------------
cb_stage1 = [
    callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    callbacks.ModelCheckpoint(MODEL_SAVE_STAGE1, save_best_only=True, monitor="val_loss"),
]

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE1,
    class_weight=class_weight,
    callbacks=cb_stage1,
    verbose=2
)

# -------------------------
# Stage2 Fine-tune
# -------------------------
model.load_weights(MODEL_SAVE_STAGE1)
num_layers = len(backbone.layers)
fine_tune_at = int(num_layers * (1 - FINE_TUNE_FRACTION))
for i, layer in enumerate(backbone.layers):
    layer.trainable = (i >= fine_tune_at)

model.compile(
    optimizer=optimizers.Adam(learning_rate=LEARNING_RATE_STAGE2),
    loss=tfa.losses.SigmoidFocalCrossEntropy(),
    metrics=["accuracy", metrics.TopKCategoricalAccuracy(k=3, name="top3")]
)

cb_stage2 = [
    callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    callbacks.ModelCheckpoint(MODEL_SAVE_STAGE2, save_best_only=True, monitor="val_loss"),
]

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE2,
    class_weight=class_weight,
    callbacks=cb_stage2,
    verbose=2
)

# -------------------------
# Save final model
# -------------------------
model.save(FINAL_MODEL)
print("✅ Model saved:", FINAL_MODEL)
