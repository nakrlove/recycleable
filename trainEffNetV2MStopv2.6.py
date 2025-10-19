# ============================================================
# ✅ EfficientNetV2M 학습 최종 소스 코드 (CLASS_COUNTS 자동 계산)
# ============================================================

import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, metrics
from tensorflow.keras.applications import EfficientNetV2M
from tensorflow.keras.preprocessing import image_dataset_from_directory
from pathlib import Path
# ============================================================
# 1. 환경 설정 및 하이퍼파라미터
# ============================================================
SEED = 42
BATCH_SIZE = 64
IMAGE_SIZE = (224, 224)
AUTOTUNE = tf.data.AUTOTUNE
EPOCHS_STAGE1 = 100
EPOCHS_STAGE2 = 50
LR_STAGE1 = 1e-4
LR_STAGE2 = 5e-6
FINE_TUNE_FRACTION = 0.2
PATIENCE_STAGE1 = 5
PATIENCE_STAGE2 = 6

BASE_PATH = Path("dataset_10000")
train_dir = BASE_PATH / "train"
val_dir   = BASE_PATH / "val"

# ============================================================
# 2. CLASS_COUNTS 자동 계산 기능 추가
# ============================================================
print("\n[데이터 카운트 중...]")
class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
class_counts_list = []

for class_name in class_names:
    class_path = os.path.join(train_dir, class_name)
    count = len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])
    class_counts_list.append(count)
    print(f"{class_name}: {count}")

CLASS_COUNTS = np.array(class_counts_list)
total_train_samples = CLASS_COUNTS.sum()
num_classes = len(CLASS_COUNTS)

print("\n✅ CLASS_COUNTS 배열 자동 생성 완료:")
print(CLASS_COUNTS)
print(f"총 클래스 수: {num_classes}, 총 학습 샘플 수: {total_train_samples}\n")

# ============================================================
# 3. Dataset 로드 및 증강
# ============================================================
train_ds = image_dataset_from_directory(
    train_dir,
    image_size=IMAGE_SIZE,
    batch_size=None,
    label_mode="categorical",
    shuffle=True,
    seed=SEED
)

val_ds = image_dataset_from_directory(
    val_dir,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

# 💡 데이터 증강 함수
def augmentation_fn(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    image = tf.image.random_brightness(image, max_delta=0.2)
    return image, label

train_batched_ds = (
    train_ds
    .map(augmentation_fn, num_parallel_calls=AUTOTUNE)
    .shuffle(25000)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

val_ds = val_ds.prefetch(AUTOTUNE)

# ============================================================
# 4. Class Weight 및 Steps per Epoch 계산
# ============================================================
raw_weights = total_train_samples / (num_classes * CLASS_COUNTS)
clipped_weights = np.clip(raw_weights, a_min=None, a_max=10.0)
class_weights_dict = {i: float(w) for i, w in enumerate(clipped_weights)}

steps_per_epoch = math.ceil(total_train_samples / BATCH_SIZE)
print(f"총 훈련 샘플: {total_train_samples}, 배치 크기: {BATCH_SIZE}, Epoch당 스텝 수: {steps_per_epoch}")

# ============================================================
# 5. 모델 정의 및 Stage 1 학습
# ============================================================
base_model = EfficientNetV2M(include_top=False, input_shape=IMAGE_SIZE + (3,), weights="imagenet")
base_model.trainable = False

inputs = layers.Input(shape=IMAGE_SIZE + (3,))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = models.Model(inputs, outputs)

model.compile(
    optimizer=optimizers.Adam(learning_rate=LR_STAGE1),
    loss="categorical_crossentropy",
    metrics=["accuracy", metrics.TopKCategoricalAccuracy(k=3, name="top3"), metrics.TopKCategoricalAccuracy(k=5, name="top5")]
)

cb_stage1 = [
    callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE_STAGE1, restore_best_weights=True),
    callbacks.ModelCheckpoint("model_stage1_best.keras", monitor="val_loss", save_best_only=True, mode='min')
]

print("\n[Stage 1] 전이 학습 시작...")
history1 = model.fit(
    train_batched_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE1,
    steps_per_epoch=steps_per_epoch,
    class_weight=class_weights_dict,
    callbacks=cb_stage1
)

# ============================================================
# 6. Fine-tuning 단계 (Stage 2)
# ============================================================
try:
    model.load_weights("model_stage1_best.keras")
    print("✅ Stage 1 최적 가중치 로드 성공. Fine-tuning 시작.")
except Exception as e:
    print(f"❌ Stage 1 가중치 로드 실패: {e}. 현재 상태로 진행.")

num_layers = len(base_model.layers)
fine_tune_at = int(num_layers * (1.0 - FINE_TUNE_FRACTION))
for i, layer in enumerate(base_model.layers):
    layer.trainable = (i >= fine_tune_at)

print(f"총 {num_layers}개 레이어 중 {fine_tune_at}번째 ({base_model.layers[fine_tune_at].name})부터 Fine-tuning 시작.")

model.compile(
    optimizer=optimizers.Adam(learning_rate=LR_STAGE2),
    loss="categorical_crossentropy",
    metrics=["accuracy", metrics.TopKCategoricalAccuracy(k=3, name="top3"), metrics.TopKCategoricalAccuracy(k=5, name="top5")]
)

cb_stage2 = [
    callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE_STAGE2, restore_best_weights=True),
    callbacks.ModelCheckpoint("model_stage2_best.keras", monitor="val_loss", save_best_only=True, mode='min')
]

print("\n[Stage 2] Fine-tuning 시작...")
history2 = model.fit(
    train_batched_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE2,
    steps_per_epoch=steps_per_epoch,
    class_weight=class_weights_dict,
    callbacks=cb_stage2
)

# ============================================================
# 7. 최종 모델 저장
# ============================================================
model.save("model_final.keras", save_format="keras")
print("✅ 학습 완료 및 최종 모델 저장 완료 (model_final.keras)")
