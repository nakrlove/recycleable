# ============================================================
# trainEffNetV2M_finetune.py
# ============================================================

import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, metrics, callbacks
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import EfficientNetV2M

# ============================================================
# 0. 기본 설정
# ============================================================
BASE_DIR = "dataset_sp500"  # 데이터셋 루트
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

# ============================================================
# 1. 데이터셋 로드
# ============================================================
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(BASE_DIR, "train"),
    image_size=IMG_SIZE,
    batch_size=None,
    label_mode="categorical",
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(BASE_DIR, "val"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
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
# 2. 데이터 증강 및 배치 처리
# ============================================================
train_balanced_ds = (
    train_ds
    .map(lambda x, y: (tf.image.random_flip_left_right(x), y))
    .map(lambda x, y: (tf.image.rot90(x, k=tf.random.uniform([], 0, 4, tf.int32)), y))
    .map(lambda x, y: (tf.image.random_brightness(x, max_delta=0.2), y))
    .map(lambda x, y: (tf.image.random_contrast(x, 0.8, 1.2), y))
    .shuffle(500)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# ============================================================
# 3. 1단계 모델 학습 여부 확인
# ============================================================
best_model_path = "./keras/best_model.keras"

if os.path.exists(best_model_path):
    print("1단계 학습 모델 불러오기...")
    model = load_model(best_model_path)
else:
    print("1단계 모델 없음. 새로 생성 및 학습 시작...")

    # ============================================================
    # 4. 모델 구성 (1단계)
    # ============================================================
    base_model = EfficientNetV2M(include_top=False, input_shape=IMG_SIZE + (3,), weights="imagenet")
    base_model.trainable = False  # 1단계에서는 freeze

    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)

    # 1단계 학습 설정
    LR = 1e-4
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LR),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            metrics.TopKCategoricalAccuracy(k=3, name="top3"),
            metrics.TopKCategoricalAccuracy(k=5, name="top5")
        ]
    )

    # 콜백
    cb = [
        callbacks.ModelCheckpoint(best_model_path, monitor="val_top3", save_best_only=True, mode="max"),
        callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    ]

    # 1단계 학습
    history = model.fit(
        train_balanced_ds,
        validation_data=val_ds,
        epochs=120,
        callbacks=cb
    )

# ============================================================
# 5. 2단계 미세조정
# ============================================================
print("--- 2단계: EfficientNetV2M 미세 조정 시작 ---")

# 모델 구조에서 base_model 찾아 trainable=True 설정
# 일반적으로 model.layers[1]이 base_model임
base_model = model.layers[1]
base_model.trainable = True

# 매우 낮은 Learning Rate
FINE_TUNE_LR = 1e-5
model.compile(
    optimizer=optimizers.Adam(learning_rate=FINE_TUNE_LR),
    loss="categorical_crossentropy",
    metrics=[
        "accuracy",
        metrics.TopKCategoricalAccuracy(k=3, name="top3"),
        metrics.TopKCategoricalAccuracy(k=5, name="top5")
    ]
)

# 콜백
cb_ft = [
    callbacks.ModelCheckpoint(
        "./keras/best_finetuned_model.keras",
        monitor="val_top3", save_best_only=True, mode="max"
    ),
    callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
]

# 2단계 학습
history_ft = model.fit(
    train_balanced_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=cb_ft
)

# ============================================================
# 6. 평가
# ============================================================
test_results = model.evaluate(test_ds)
print("Test Results:", test_results)
