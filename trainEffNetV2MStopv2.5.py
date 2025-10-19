# ============================================================
# ✅ EfficientNetV2M 학습 최종 소스 코드
# (Class Weight 적용, steps_per_epoch 명시, 안정적인 2단계 Fine-tuning)
# ============================================================

import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, metrics
from tensorflow.keras.applications import EfficientNetV2M
from tensorflow.keras.preprocessing import image_dataset_from_directory

# ============================================================
# 1. 환경 설정 및 하이퍼파라미터
# ============================================================
SEED = 42
BATCH_SIZE = 64
IMAGE_SIZE = (224, 224)
AUTOTUNE = tf.data.AUTOTUNE
EPOCHS_STAGE1 = 10
EPOCHS_STAGE2 = 10
LR_STAGE1 = 1e-4
LR_STAGE2 = 5e-6       # 💡 Fine-tuning LR은 안정성을 위해 5e-6으로 낮춤
FINE_TUNE_FRACTION = 0.2
PATIENCE_STAGE1 = 5    # Early Stopping Patience 상향 조정
PATIENCE_STAGE2 = 6    # Early Stopping Patience 상향 조정

train_dir = "/content/dataset_sp/train"
val_dir = "/content/dataset_sp/val"

# 💡 훈련 데이터의 실제 클래스별 샘플 개수 (제공된 로그 기반)
CLASS_COUNTS = np.array([
    13420, 13444, 23593, 19661, 10415, 19064, 10824, 24075, 6180,
    1922, 20107, 21424, 20706, 13699, 14598, 1040, 12100, 22527,
    19875, 2617, 20958, 13455, 19323, 23366, 15348, 1702, 22892,
    10860, 21714
])
total_train_samples = CLASS_COUNTS.sum()
num_classes = len(CLASS_COUNTS)

# ============================================================
# 2. Dataset 로드 및 증강
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

class_names = train_ds.class_names

# 💡 데이터 증강 함수 (Augmentation 강화)
def augmentation_fn(image, label):
    # 좌우 반전
    image = tf.image.random_flip_left_right(image)
    # 90도 단위 회전
    image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    # 밝기 조절
    image = tf.image.random_brightness(image, max_delta=0.2)
    return image, label

# 훈련 Dataset 파이프라인
train_batched_ds = (
    train_ds
    .map(augmentation_fn, num_parallel_calls=AUTOTUNE)
    .shuffle(25000)  # 💡 셔플 버퍼 확대
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

val_ds = val_ds.prefetch(AUTOTUNE)

# ============================================================
# 3. Class Weight 및 Steps per Epoch 계산
# ============================================================
# Class Weight 계산 및 Clipping
raw_weights = total_train_samples / (num_classes * CLASS_COUNTS)
clipped_weights = np.clip(raw_weights, a_min=None, a_max=10.0)
class_weights_dict = {i: float(w) for i, w in enumerate(clipped_weights)}

# 💡 Steps per Epoch 계산 (진행 상태 로그 출력을 위해 필수)
steps_per_epoch = math.ceil(total_train_samples / BATCH_SIZE)
print(f"총 훈련 샘플: {total_train_samples}, 배치 크기: {BATCH_SIZE}, Epoch당 스텝 수: {steps_per_epoch}")

# ============================================================
# 4. 모델 정의 및 Stage 1 학습
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
    callbacks.ModelCheckpoint("model_stage1_best.keras", monitor="val_loss", save_best_only=True, mode='min') # 💡 확장자 .keras 변경
]

print("\n[Stage 1] 전이 학습 시작...")
history1 = model.fit(
    train_batched_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE1,
    steps_per_epoch=steps_per_epoch, # ✅ steps_per_epoch 명시
    class_weight=class_weights_dict,
    callbacks=cb_stage1
)


# ============================================================
# 5. Fine-tuning 단계 (Stage 2)
# ============================================================

# ✅ 1단계 최적 가중치 명시적 로드 (안정성 확보)
try:
    # 💡 .keras 확장자로 변경된 파일 로드 시도
    model.load_weights("model_stage1_best.keras")
    print("✅ Stage 1 최적 가중치 로드 성공. Fine-tuning 시작.")
except Exception as e:
    print(f"❌ Stage 1 가중치 로드 실패: {e}. 현재 메모리 상태로 Fine-tuning 진행.")

# 💡 Fine-tuning 범위 설정: 전체 레이어의 상위 20%만 학습
num_layers = len(base_model.layers)
fine_tune_at = int(num_layers * (1.0 - FINE_TUNE_FRACTION))

for i, layer in enumerate(base_model.layers):
    layer.trainable = (i >= fine_tune_at)

print(f"총 {num_layers}개 레이어 중 {fine_tune_at}번째 ({base_model.layers[fine_tune_at].name})부터 Fine-tuning 시작.")

# 낮은 LR로 재컴파일
model.compile(
    optimizer=optimizers.Adam(learning_rate=LR_STAGE2), # 💡 5e-6로 매우 낮춤
    loss="categorical_crossentropy",
    metrics=["accuracy", metrics.TopKCategoricalAccuracy(k=3, name="top3"), metrics.TopKCategoricalAccuracy(k=5, name="top5")]
)

cb_stage2 = [
    callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE_STAGE2, restore_best_weights=True),
    callbacks.ModelCheckpoint("model_stage2_best.keras", monitor="val_loss", save_best_only=True, mode='min') # 💡 확장자 .keras 변경
]

print("\n[Stage 2] Fine-tuning 시작...")
history2 = model.fit(
    train_batched_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE2,
    steps_per_epoch=steps_per_epoch, # ✅ steps_per_epoch 명시
    class_weight=class_weights_dict,
    callbacks=cb_stage2
)

# ============================================================
# 6. 최종 모델 저장
# ============================================================
# 💡 최신 Keras 표준에 따라 .keras 확장자로 저장
model.save("model_final.keras", save_format="keras")
print("✅ 학습 완료 및 최종 모델 저장 완료 (model_final.keras)")