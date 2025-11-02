# ============================================================
# trainEffNetV2S_prod_ready_v3.3.py
# ============================================================

import os
import json
import math
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, metrics, callbacks
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import EfficientNetV2S

# ============================================================
# 0. GPU / Mixed Precision 설정
# ============================================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("[INFO] GPU(s) detected and memory growth enabled.")
    except RuntimeError as e:
        print(e)

# Mixed Precision 활성화
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print("[INFO] Mixed Precision Enabled")

# ============================================================
# 1. 기본 설정
# ============================================================
BASE_DIR = "dataset_25000"
IMG_SIZE = (224, 224)
BATCH_SIZE = 8        # 안정적 학습용, VRAM에 따라 조절 가능
ACCUM_STEPS = 2       # Gradient Accumulation step 수
SEED = 42

# ============================================================
# 2. 데이터셋 로드
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
print(f"[INFO] num_classes = {num_classes}")
print("[INFO] class_names =", class_names)

# class_names JSON 저장
os.makedirs("./keras", exist_ok=True)
with open("./keras/class_indices.json", "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=4)
print("[INFO] Saved class_indices.json")

# ============================================================
# 3. 데이터 파이프라인 (증강 + prefetch + parallel)
# ============================================================
AUTOTUNE = tf.data.AUTOTUNE

def augment(x, y):
    x = tf.image.random_flip_left_right(x)
    x = tf.image.rot90(x, k=tf.random.uniform([], 0, 4, tf.int32))
    x = tf.image.random_brightness(x, max_delta=0.2)
    x = tf.image.random_contrast(x, 0.8, 1.2)
    return x, y

train_ds = train_ds.map(augment, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.shuffle(800).batch(BATCH_SIZE).prefetch(AUTOTUNE)

val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# ============================================================
# 4. class_weights 계산 (파일 수 기반)
# ============================================================
train_path = os.path.join(BASE_DIR, "train")
train_counts = [len(os.listdir(os.path.join(train_path, c))) for c in class_names]
total = sum(train_counts)
class_weights = {i: total / (num_classes * count) for i, count in enumerate(train_counts)}
print("[INFO] class_weights:", class_weights)

# ============================================================
# 5. 모델 생성
# ============================================================
best_model_path = "./keras/best_model.keras"

if os.path.exists(best_model_path):
    print("[INFO] Loading existing model...")
    model = load_model(best_model_path)
else:
    print("[INFO] Creating new model...")
    base_model = EfficientNetV2S(include_top=False, input_shape=IMG_SIZE + (3,), weights="imagenet")
    base_model.trainable = False

    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)  # mixed_precision 보정
    model = models.Model(inputs, outputs)

    LR = 1e-4
    model.compile(
        optimizer=optimizers.Adam(LR),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            metrics.TopKCategoricalAccuracy(k=3, name="top3"),
            metrics.TopKCategoricalAccuracy(k=5, name="top5")
        ]
    )

    cb = [
        callbacks.ModelCheckpoint(best_model_path, monitor="val_top3", save_best_only=True, mode="max"),
        callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    ]

    steps_per_epoch = math.ceil(197374 / BATCH_SIZE)   # 명시적 설정
    print(f"[INFO] steps_per_epoch = {steps_per_epoch}")

    # Gradient Accumulation 적용시, Custom Training Loop 필요
    # 여기서는 .fit() 기준 안전하게 배치 사이즈로만 학습
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=50,
        class_weight=class_weights,
        steps_per_epoch=steps_per_epoch,
        callbacks=cb
    )

# ============================================================
# 6. Stage2 Fine-tuning
# ============================================================
print("[INFO] Stage2 Fine-tuning...")
base_model = model.layers[1]
base_model.trainable = True

FINE_TUNE_LR = 1e-5
model.compile(
    optimizer=optimizers.Adam(FINE_TUNE_LR),
    loss="categorical_crossentropy",
    metrics=[
        "accuracy",
        metrics.TopKCategoricalAccuracy(k=3, name="top3"),
        metrics.TopKCategoricalAccuracy(k=5, name="top5")
    ]
)

cb_ft = [
    callbacks.ModelCheckpoint("./keras/best_finetuned_model.keras", monitor="val_top3", save_best_only=True, mode="max"),
    callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
]

steps_per_epoch_ft = math.ceil(197374 / BATCH_SIZE)
history_ft = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,   # Stage2 epoch 낮춤
    class_weight=class_weights,
    steps_per_epoch=steps_per_epoch_ft,
    callbacks=cb_ft
)

# ============================================================
# 7. 테스트
# ============================================================
test_results = model.evaluate(test_ds)
print("[INFO] Test Results:", test_results)
