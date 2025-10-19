# ============================================================
# GPT 비교 성능에서 최고라함
# ✅ EfficientNetV2M 최적화 학습 소스
# (Class Weight 적용, 2단계 Fine-tuning, 과적합 방지, 최적 성능)
# ============================================================

import os, math, re
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, metrics
from tensorflow.keras.applications import EfficientNetV2M
from tensorflow.keras.preprocessing import image_dataset_from_directory

# ============================================================
# 1️⃣ 환경 설정 및 하이퍼파라미터
# ============================================================
SEED = 42
BATCH_SIZE = 64
IMAGE_SIZE = (224, 224)
AUTOTUNE = tf.data.AUTOTUNE

# Epoch & Learning rate 설정
EPOCHS_STAGE1 = 20
EPOCHS_STAGE2 = 15
LR_STAGE1 = 1e-4
LR_STAGE2 = 5e-6
FINE_TUNE_FRACTION = 0.2
PATIENCE_STAGE1 = 5
PATIENCE_STAGE2 = 6

BASE_PATH = Path("dataset_10000")
train_dir = BASE_PATH / "train"
val_dir = BASE_PATH / "val"

# 클래스별 샘플 수
# CLASS_COUNTS = np.array([
#     20958, 10824, 2617, 1922, 13444, 13699, 13420,
#     19661, 12100, 19064, 19875, 19323, 21714, 21424, 6180,
#     23593, 14598, 20107, 22892, 15348, 24075, 23366, 10860,
#     10415, 22527, 1040, 20706, 1702, 13455
# ])
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
rename_korean_files(BASE_PATH)

# ============================================================
# 3️⃣ Dataset 로드 + 최적화 파이프라인
# ============================================================
train_ds = image_dataset_from_directory(
    train_dir, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode="categorical", shuffle=True, seed=SEED
)
val_ds = image_dataset_from_directory(
    val_dir, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode="categorical", shuffle=False
)

class_names = train_ds.class_names

# 증강 함수 (GPU 연산 활용, 효율적)
def augmentation_fn(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_saturation(image, 0.8, 1.2)
    image = tf.image.random_hue(image, 0.02)
    image = tf.image.rot90(image, k=tf.random.uniform([],0,4,dtype=tf.int32))
    image = tf.clip_by_value(image, 0, 255)
    return image, label

# pipeline 최적화
train_batched_ds = (train_ds
    .map(augmentation_fn, num_parallel_calls=AUTOTUNE)
    .shuffle(buffer_size=25000, seed=SEED)
    .repeat()  # Stage 2까지 연속 학습 가능
    .prefetch(AUTOTUNE)
)
val_ds = val_ds.prefetch(AUTOTUNE)

# ============================================================
# 4️⃣ Class Weight 및 Steps per Epoch 계산
# ============================================================
raw_weights = total_train_samples / (num_classes * CLASS_COUNTS)
clipped_weights = np.clip(raw_weights, a_min=None, a_max=10.0)
class_weights_dict = {i: float(w) for i, w in enumerate(clipped_weights)}

steps_per_epoch = math.ceil(total_train_samples / BATCH_SIZE)
print(f"총 훈련 샘플: {total_train_samples}, 배치 크기: {BATCH_SIZE}, Epoch당 스텝 수: {steps_per_epoch}")

# ============================================================
# 5️⃣ 모델 정의
# ============================================================
base_model = EfficientNetV2M(include_top=False, input_shape=IMAGE_SIZE+(3,), weights="imagenet")
base_model.trainable = False

inputs = layers.Input(shape=IMAGE_SIZE+(3,))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = models.Model(inputs, outputs)

model.compile(
    optimizer=optimizers.Adam(LR_STAGE1),
    loss="categorical_crossentropy",
    metrics=["accuracy",
             metrics.TopKCategoricalAccuracy(k=3, name="top3"),
             metrics.TopKCategoricalAccuracy(k=5, name="top5")]
)

# callbacks
cb_stage1 = [
    callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE_STAGE1, restore_best_weights=True),
    callbacks.ModelCheckpoint("model_stage1_best.keras", monitor="val_loss", save_best_only=True, mode='min')
]

# ============================================================
# 6️⃣ Stage 1 학습
# ============================================================
history1 = model.fit(
    train_batched_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE1,
    steps_per_epoch=steps_per_epoch,
    class_weight=class_weights_dict,
    callbacks=cb_stage1,
    verbose=2
)

# ============================================================
# 7️⃣ Stage 2 Fine-tuning
# ============================================================
# Stage 1 최적 가중치 로드
model.load_weights("model_stage1_best.keras")

# 상위 20% 레이어만 학습
num_layers = len(base_model.layers)
fine_tune_at = int(num_layers*(1-FINE_TUNE_FRACTION))
for i, layer in enumerate(base_model.layers):
    layer.trainable = (i>=fine_tune_at)

print(f"Fine-tuning: {fine_tune_at}/{num_layers} 레이어 학습 가능")

# 재컴파일
model.compile(
    optimizer=optimizers.Adam(LR_STAGE2),
    loss="categorical_crossentropy",
    metrics=["accuracy",
             metrics.TopKCategoricalAccuracy(k=3, name="top3"),
             metrics.TopKCategoricalAccuracy(k=5, name="top5")]
)

cb_stage2 = [
    callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE_STAGE2, restore_best_weights=True),
    callbacks.ModelCheckpoint("model_stage2_best.keras", monitor="val_loss", save_best_only=True, mode='min')
]

history2 = model.fit(
    train_batched_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE2,
    steps_per_epoch=steps_per_epoch,
    class_weight=class_weights_dict,
    callbacks=cb_stage2,
    verbose=2
)

# ============================================================
# 8️⃣ 최종 모델 저장
# ============================================================
model.save("model_final.keras")
print("✅ 학습 완료, 최종 모델 저장 완료")
