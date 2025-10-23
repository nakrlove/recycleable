import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, metrics
from tensorflow.keras.applications import EfficientNetV2S, efficientnet_v2
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os, math, re
# -------------------------
# 1) Config
# -------------------------
DATA_DIR = "dataset_2000"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS_STAGE1 = 16
EPOCHS_STAGE2 = 10
FINE_TUNE_FRACTION = 0.25
SHUFFLE_BUFFER = 256
PREFETCH_SIZE = 2
LEARNING_RATE_STAGE1 = 1e-3
LEARNING_RATE_STAGE2 = 1e-5
MODEL_SAVE_STAGE1 = "EffNetV2S_stage1_best_nolambda.keras"
MODEL_SAVE_STAGE2 = "EffNetV2S_stage2_best_nolambda.keras"
FINAL_MODEL = "EffNetV2S_final_nolambda.keras"
SEED = 42



# ============================================================
# 2️⃣ 한글 파일/폴더 처리 (기존 코드 사용)
# ============================================================
def has_korean(text): return bool(re.search(r'[가-힣]', text))
CUSTOM_MAP = {"_김장현_":"_kimjanghyun_","플라스틱":"plastic","비닐":"vinyl","종이":"paper","유리":"glass","금속":"metal"}
def safe_name(name, counter):
    cleaned = re.sub(r'[가-힣]+', '', name)
    cleaned = re.sub(r'\s+', '_', cleaned)
    cleaned = re.sub(r'[^a-zA-Z0-9_.-]', '', cleaned)
    return cleaned if cleaned.strip() else f"korean_file_{counter:03d}"
def find_korean_dirs(base_path="."):
    return [os.path.join(root, d) for root, dirs, _ in os.walk(base_path) for d in dirs if has_korean(d)]
def rename_korean_files(base_path="."):
    counter, renamed = 1, []
    for root, dirs, files in os.walk(base_path, topdown=False):
        for filename in files:
            old_path = os.path.join(root, filename)
            new_filename = filename
            for k, v in CUSTOM_MAP.items():
                new_filename = new_filename.replace(k, v)
            if has_korean(new_filename):
                name, ext = os.path.splitext(new_filename)
                new_filename = safe_name(name, counter) + ext
                counter += 1
            new_path = os.path.join(root, new_filename)
            if new_path != old_path:
                os.rename(old_path, new_path)
                renamed.append((old_path, new_path))
        for dirname in dirs:
            old_dir = os.path.join(root, dirname)
            new_dirname = dirname
            for k, v in CUSTOM_MAP.items():
                new_dirname = new_dirname.replace(k, v)
            if has_korean(new_dirname):
                new_dirname = safe_name(new_dirname, counter)
                counter += 1
            new_dir = os.path.join(root, new_dirname)
            if new_dir != old_dir:
                os.rename(old_dir, new_dir)
                renamed.append((old_dir, new_dir))
rename_korean_files(DATA_DIR)

# -------------------------
# 2) Dataset
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
print(f"Found {num_classes} classes: {class_names}")

# -------------------------
# 3) Compute class weights
# -------------------------
class_counts = np.array([
    len(os.listdir(os.path.join(TRAIN_DIR, c))) for c in class_names
])
total_train = class_counts.sum()
weights = total_train / (num_classes * np.maximum(class_counts, 1))
weights = np.clip(weights, None, 10.0)
class_weight = {i: float(w) for i, w in enumerate(weights)}

# -------------------------
# 4) Dataset Prefetch
# -------------------------
train_ds = train_ds.shuffle(buffer_size=SHUFFLE_BUFFER).prefetch(PREFETCH_SIZE)
val_ds = val_ds.prefetch(PREFETCH_SIZE)
test_ds = test_ds.prefetch(PREFETCH_SIZE)

# -------------------------
# 5) Build Model (🚫 No Lambda)
# -------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.08),
    layers.RandomContrast(0.08),
], name="data_augmentation")

inputs = layers.Input(shape=IMG_SIZE + (3,), name="input_image")

# 대신 아래처럼 Lambda 없이 전처리
x = layers.Rescaling(1./255)(inputs)  # float32 변환 + 정규화
x = efficientnet_v2.preprocess_input(x)  # -1~1 스케일 (EffNetV2 전용)
x = data_augmentation(x)

backbone = EfficientNetV2S(include_top=False, input_shape=IMG_SIZE + (3,), weights="imagenet")
backbone.trainable = False

x = backbone(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.35)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs, name="EffNetV2S_noLambda")

model.compile(
    optimizer=optimizers.Adam(learning_rate=LEARNING_RATE_STAGE1),
    loss="categorical_crossentropy",
    metrics=[
        "accuracy",
        metrics.TopKCategoricalAccuracy(k=3, name="top3"),
        metrics.TopKCategoricalAccuracy(k=5, name="top5"),
    ]
)
model.summary()

# -------------------------
# 6) Stage 1 Training
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
# 7) Stage 2 Fine-tune
# -------------------------
if os.path.exists(MODEL_SAVE_STAGE1):
    model.load_weights(MODEL_SAVE_STAGE1)

num_layers = len(backbone.layers)
fine_tune_at = int(num_layers * (1 - FINE_TUNE_FRACTION))
for i, layer in enumerate(backbone.layers):
    layer.trainable = (i >= fine_tune_at)

model.compile(
    optimizer=optimizers.Adam(learning_rate=LEARNING_RATE_STAGE2),
    loss="categorical_crossentropy",
    metrics=[
        "accuracy",
        metrics.TopKCategoricalAccuracy(k=3, name="top3"),
        metrics.TopKCategoricalAccuracy(k=5, name="top5"),
    ]
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
# 8) Save final (no Lambda!)
# -------------------------
model.save(FINAL_MODEL, include_optimizer=True)
print("✅ Saved clean model:", FINAL_MODEL)
