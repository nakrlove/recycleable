# ================== trainEffNetV2MStopv2.5.py ==================
# ============================================================
# ✅ EfficientNetV2M 학습 로직 (균형 데이터셋 전용, dataset_sp500)
# ============================================================

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, metrics
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import EfficientNetV2M

# ============================================================
# 1. 경로 및 기본 설정
# ============================================================
BASE_DIR = "dataset_10000"  # 균형 데이터셋
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 120
SEED = 42
LR = 1e-4

# ============================================================
# 2. 데이터셋 로드 (batch=None으로 불균형 고려 X)
# ============================================================
train_ds = image_dataset_from_directory(
    os.path.join(BASE_DIR, "train"),
    image_size=IMG_SIZE,
    batch_size=None,  # batch는 나중에 설정
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
# 3. train dataset 준비
# ============================================================
# dataset_sp500은 각 클래스 500개로 균형
# 증강은 필요시 추가, 기본은 좌우 반전 정도
# train_balanced_ds = (
#     train_ds
#     .map(lambda x, y: (tf.image.random_flip_left_right(x), y))  # 간단 증강
#     .shuffle(500)  # 전체 데이터보다 작은 shuffle buffer도 가능
#     .batch(BATCH_SIZE)
#     .prefetch(tf.data.AUTOTUNE)
# )

train_balanced_ds = (
    train_ds
    .map(lambda x, y: (tf.image.random_flip_left_right(x), y))  # 좌우 반전

    # ✅ 추가 증강 기법 적용:
    # 1. 무작위 회전 (Rotation)
    .map(lambda x, y: (tf.image.rot90(x, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)), y)) 
    # 2. 무작위 확대/축소/이동 (Random Zoom/Shift) - Keras Layers 활용이 더 편리
    # 3. 밝기/대비 변화 (Brightness/Contrast)
    .map(lambda x, y: (tf.image.random_brightness(x, max_delta=0.2), y))
    .map(lambda x, y: (tf.image.random_contrast(x, lower=0.8, upper=1.2), y))
    
    .shuffle(500) 
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# ============================================================
# 4. 전체 샘플 수 확인
# ============================================================
train_samples = 500 * num_classes
val_samples = 62 * num_classes
test_samples = 62 * num_classes

print(f"Train samples: {train_samples}, steps per epoch: {train_samples // BATCH_SIZE}")
print(f"Val samples: {val_samples}, steps: {val_samples // BATCH_SIZE}")
print(f"Test samples: {test_samples}, steps: {test_samples // BATCH_SIZE}")

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
    metrics=["accuracy",
             metrics.TopKCategoricalAccuracy(k=3, name="top3"),
             metrics.TopKCategoricalAccuracy(k=5, name="top5")]
)

# ============================================================
# 7. 콜백 설정
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


# ============================================================
# 8-2. 2단계 미세 조정 (추가)
# ============================================================
print("--- 2단계: EfficientNetV2M 미세 조정 시작 ---")
base_model.trainable = True # 전체 EfficientNetV2M 레이어 학습 가능하도록 설정

# 매우 낮은 Learning Rate 설정
FINE_TUNE_LR = 1e-5 

model.compile(
    optimizer=optimizers.Adam(learning_rate=FINE_TUNE_LR),
    loss="categorical_crossentropy",
    metrics=["accuracy", metrics.TopKCategoricalAccuracy(k=3, name="top3"), metrics.TopKCategoricalAccuracy(k=5, name="top5")]
)

# 2단계 전용 콜백 (Val Loss가 더 이상 개선되지 않을 때까지)
cb_ft = [
    callbacks.ModelCheckpoint("keras/best_finetuned_model.keras", monitor="val_top3", save_best_only=True, mode="max"),
    callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True) # patience를 낮춰도 좋음
]

# 1단계에서 학습하지 않은 나머지 Epoch를 활용하여 추가 학습
history_ft = model.fit(
    train_balanced_ds,
    validation_data=val_ds,
    epochs=50, # 미세 조정을 위한 Epoch 수 재설정 (예: 50)
    callbacks=cb_ft
)

# ============================================================
# 9. 평가 (Test)
# ============================================================
test_results = model.evaluate(test_ds)
print("Test Results:", test_results)
