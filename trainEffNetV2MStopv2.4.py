# ================== trainEffNetV2MStopv2.4.py ==================
# ============================================================
# ✅ EfficientNetV2M 학습 로직 (불균형 대응 / 대규모 데이터셋용)
# ============================================================

import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, metrics
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import EfficientNetV2M
from sklearn.utils.class_weight import compute_class_weight

# ============================================================
# 1. 경로 및 기본 설정
# ============================================================
# BASE_DIR = "dataset_sp"  # 원본 데이터셋
BASE_DIR = "dataset_sp"  # 원본 데이터셋
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 40
SEED = 42
LR = 1e-3

# ============================================================
# 2. 데이터셋 로드
# ============================================================
train_ds = image_dataset_from_directory(
    os.path.join(BASE_DIR, "train"),
    image_size=IMG_SIZE,
    batch_size=None,  # ⚠️ 불균형 보정 위해 batch 나중에 설정
    label_mode="categorical",
    shuffle=True,
    seed=SEED
)

val_ds = image_dataset_from_directory(
    os.path.join(BASE_DIR, "val"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

test_ds = image_dataset_from_directory(
    os.path.join(BASE_DIR, "test"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"클래스 개수: {num_classes}")

# ============================================================
# 3. 클래스별 개수 계산 (불균형 보정용)
# ============================================================
count_dict = {name: 0 for name in class_names}

for _, y in train_ds:
    idx = np.argmax(y)
    count_dict[class_names[idx]] += 1

class_counts = np.array(list(count_dict.values()))
print("Class Counts:", class_counts)

max_count = np.max(class_counts)

# ============================================================
# 4. 소수 클래스 증강용 Oversampling
# ============================================================
class_datasets = []
for cls_idx, cls_name in enumerate(class_names):
    ds_cls = train_ds.filter(lambda x, y: tf.argmax(y) == cls_idx)
    repeat_factor = max(1, max_count // class_counts[cls_idx])
    ds_cls = ds_cls.repeat(repeat_factor)
    class_datasets.append(ds_cls)

train_balanced_ds = tf.data.Dataset.sample_from_datasets(class_datasets)
train_balanced_ds = (
    train_balanced_ds
    .map(lambda x, y: (tf.image.random_flip_left_right(x), y))
    .shuffle(1000)
    .batch(BATCH_SIZE)
    .prefetch(1)
)

effective_train_samples = max_count * len(class_names)
print(f"Effective train samples after oversampling: {effective_train_samples}")

# ============================================================
# 5. 모델 구성
# ============================================================
base_model = EfficientNetV2M(include_top=False, input_shape=IMG_SIZE + (3,), weights="imagenet")
base_model.trainable = False  # 전이학습 1단계

inputs = layers.Input(shape=IMG_SIZE + (3,))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = models.Model(inputs, outputs)

# ============================================================
# 6. 컴파일
# ============================================================
model.compile(
    optimizer=optimizers.Adam(learning_rate=LR),
    loss="categorical_crossentropy",
    metrics=["accuracy", metrics.TopKCategoricalAccuracy(k=3, name="top3"),
             metrics.TopKCategoricalAccuracy(k=5, name="top5")]
)

# ============================================================
# 7. 콜백
# ============================================================
cb = [
    callbacks.ModelCheckpoint("keras/best_model.keras", monitor="val_top3", save_best_only=True, mode="max"),
    callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
]

# ============================================================
# 8. 학습
# ============================================================
history = model.fit(
    train_balanced_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=cb
)
