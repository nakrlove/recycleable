# trainEffNetV2_stable_fixed.py
# 수정자: (assistant)
# 목적: 메모리 안전한 데이터 파이프라인 + 모델내 데이터 증강 + (옵션) 디스크 기반 클래스 보강

import os
import math
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image as kp_image
from tensorflow.keras.utils import image_dataset_from_directory
from sklearn.utils.class_weight import compute_class_weight

# ================== 사용자 설정 ==================
# 사용자 설정
# ROOT = "drive/MyDrive/"
ROOT = ""
BASE_DIR = "dataset_sp"     
FULL_BASE_DIR = os.path.join(ROOT, BASE_DIR)  # 합쳐진 경로

train_dir = os.path.join(FULL_BASE_DIR, "train")
val_dir   = os.path.join(FULL_BASE_DIR, "val")
test_dir  = os.path.join(FULL_BASE_DIR, "test")

IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 30
SEED = 42

# 옵션: 모델 내부에서 실시간 증강 수행 (권장)
AUGMENT_IN_MODEL = True

# 옵션: 부족 클래스에 대해 디스크에 증강본을 생성해서 개수를 맞춤 (False 권장; 디스크 많이 사용)
ON_DISK_AUGMENT = False
# ON_DISK_AUGMENT 사용 시 전략: 'median'|'mean'|'max' 또는 정수 (target count)
AUGMENT_TARGET_STRATEGY = 'median'
# 디스크 증강 시 한 원본 이미지로부터 생성할 최대 augmentation 복제 수 (안전범위)
MAX_AUG_PER_IMAGE = 10

# ==================================================

# 안전체크: 폴더 존재 여부
for p in [train_dir, val_dir, test_dir]:
    if not os.path.exists(p):
        print(f"경고: 경로가 존재하지 않습니다: {p}")

# ================== 0) 유틸: 폴더별 이미지 수 집계 ==================
def get_class_counts(base_train_dir):
    classes = sorted([d for d in os.listdir(base_train_dir) if os.path.isdir(os.path.join(base_train_dir, d))])
    counts = {}
    for c in classes:
        p = os.path.join(base_train_dir, c)
        files = [f for f in os.listdir(p) if os.path.isfile(os.path.join(p, f))]
        counts[c] = len(files)
    return counts

train_counts = get_class_counts(train_dir)
print("Train class counts (sample):", list(train_counts.items())[:6], "... total classes:", len(train_counts))

# ================== 1) (옵션) 디스크 기반 증강 함수 ==================
# 주의: 디스크 사용량 및 시간 증가. 기본은 OFF.
def augment_on_disk_to_target(train_dir, strategy='median', target_count=None):
    """
    소수 클래스에 대해 증강 이미지를 생성해서 각 클래스의 파일 개수를 target으로 맞춘다.
    strategy: 'median'|'mean'|'max' 또는 int (target)
    """
    print(">> 시작: 디스크 기반 증강 (주의: 시간이 오래 걸리고 디스크 사용이 늘어납니다)")
    counts = get_class_counts(train_dir)
    class_names = list(counts.keys())
    arr = np.array(list(counts.values()))
    if target_count is None:
        if strategy == 'median':
            target = int(np.median(arr))
        elif strategy == 'mean':
            target = int(np.mean(arr))
        elif strategy == 'max':
            target = int(np.max(arr))
        elif isinstance(strategy, int):
            target = int(strategy)
        else:
            raise ValueError("strategy must be 'median'|'mean'|'max'|int")
    else:
        target = int(target_count)

    print(f"증강 목표(target per class) = {target} (strategy={strategy})")
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.08,
        height_shift_range=0.08,
        shear_range=0.08,
        zoom_range=0.08,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    for cls in class_names:
        cur = counts[cls]
        if cur >= target:
            continue
        need = target - cur
        src_dir = os.path.join(train_dir, cls)
        img_files = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
        if not img_files:
            print(f"경고: 클래스 {cls}에 이미지가 하나도 없습니다. 건너뜁니다.")
            continue
        print(f"증강: 클래스 '{cls}' 현재 {cur} -> 목표 {target} (생성 필요: {need})")
        i = 0
        # 임의로 원본 이미지를 반복하면서 증강본 생성
        idx = 0
        while i < need:
            src = img_files[idx % len(img_files)]
            idx += 1
            # load and expand
            try:
                img = kp_image.load_img(src, target_size=IMG_SIZE)
                x = kp_image.img_to_array(img)
                x = x.reshape((1,) + x.shape)  # (1, h, w, c)
                gen = datagen.flow(x, batch_size=1)
                # generate up to MAX_AUG_PER_IMAGE or remaining need
                gen_times = min(MAX_AUG_PER_IMAGE, need - i)
                for _ in range(gen_times):
                    batch = gen.next()
                    out_img = batch[0].astype('uint8')
                    out_name = f"aug_{i}_{os.path.basename(src)}"
                    out_path = os.path.join(src_dir, out_name)
                    kp_image.save_img(out_path, out_img)
                    i += 1
                    if i >= need:
                        break
            except Exception as e:
                print("이미지 처리 오류:", src, e)
                continue
        print(f"완료: '{cls}' 증강 완료, 생성된 수: {need}")

# ================== 2) (선택실행) 디스크 증강 ==================
if ON_DISK_AUGMENT:
    augment_on_disk_to_target(train_dir, strategy=AUGMENT_TARGET_STRATEGY)

# ================== 3) 데이터셋 로드 (tf.data) ==================
# image_dataset_from_directory에서 shuffle=False로 불러온 뒤, 직접 shuffle로 제어
train_ds = image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False  # 직접 shuffle
)

val_ds = image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

test_ds = image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

# class_names는 train으로부터
class_names = train_ds.class_names
num_classes = len(class_names)
print(f"클래스 개수: {num_classes}, class_names sample: {class_names[:6]}")

# 안전 체크: train에 어떤 클래스가 없지는 않은가
missing_in_train = []
for d in sorted(os.listdir(os.path.join(BASE_DIR, "train"))):
    p = os.path.join(BASE_DIR, "train", d)
    if os.path.isdir(p):
        if len([f for f in os.listdir(p) if os.path.isfile(os.path.join(p, f))]) == 0:
            missing_in_train.append(d)
if missing_in_train:
    print("경고: train에서 파일이 하나도 없는 클래스가 있습니다:", missing_in_train)

# ================== 4) 데이터 파이프라인 최적화 (메모리 안전) ==================
AUTOTUNE = tf.data.AUTOTUNE

# train_count 계산 (파일 수)
train_count = sum(len(files) for _, _, files in os.walk(train_dir))
val_count = sum(len(files) for _, _, files in os.walk(val_dir))
test_count = sum(len(files) for _, _, files in os.walk(test_dir))
print(f"Train samples: {train_count}, Val samples: {val_count}, Test samples: {test_count}")

# shuffle buffer: 너무 크게 잡으면 메모리 이슈 -> 적절히 제한
shuffle_buffer = min(1000, max(100, train_count))  # 최소 100, 최대 1000

# shuffle, prefetch만 (cache 제거)
train_ds = train_ds.shuffle(buffer_size=shuffle_buffer, seed=SEED).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

# ================ 5) 클래스 가중치 계산 (train 디렉토리 기준) ================
all_labels = []
for idx, cname in enumerate(class_names):
    class_path = os.path.join(train_dir, cname)
    cnt = len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])
    all_labels += [idx] * cnt

if len(all_labels) == 0:
    raise RuntimeError("Train 데이터가 비어있습니다. train_dir을 확인하세요.")

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(all_labels),
    y=np.array(all_labels)
)
class_weight_dict = dict(enumerate(class_weights))
print("Class Weights sample:", list(class_weight_dict.items())[:6])

# ================ 6) 모델 정의 (데이터 증강을 모델 내부에 적용) ============
base_model = tf.keras.applications.EfficientNetV2B0(
    include_top=False,
    input_shape=IMG_SIZE + (3,),
    weights='imagenet'
)
base_model.trainable = False  # stage 1: freeze

inputs = keras.Input(shape=IMG_SIZE + (3,))

# 데이터 증강 레이어 (학습시에만 적용)
if AUGMENT_IN_MODEL:
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.03),
        layers.RandomZoom(0.08),
        layers.RandomTranslation(0.05, 0.05),
        layers.RandomContrast(0.05)
    ], name="data_augmentation")
    x = data_augmentation(inputs)
else:
    x = inputs

# preprocess for EfficientNetV2
x = tf.keras.applications.efficientnet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)
model = keras.Model(inputs, outputs)

# ================ 7) 컴파일 ===================
optimizer = keras.optimizers.Adam(learning_rate=1e-3)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ================ 8) 콜백 ====================
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)
]

# ================ 9) Steps 계산 =================
steps_per_epoch = math.ceil(train_count / BATCH_SIZE)
validation_steps = math.ceil(val_count / BATCH_SIZE)

print(f"Steps per epoch: {steps_per_epoch}, Validation steps: {validation_steps}")

# ================ 10) 학습 (Feature Extractor) =================
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=validation_steps,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

# ================ 11) Fine-tuning =================
base_model.trainable = True
# fine-tune의 경우 너무 많은 층을 풀지 않도록 안전하게 일부만 학습가능
for layer in base_model.layers[:200]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_finetune = model.fit(
    train_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=validation_steps,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

# ================ 12) 저장 & 평가 =================
model_save_path = os.path.join(BASE_DIR, "efficientnetv2_stable_final.keras")
model.save(model_save_path, save_format="keras")
print(f"✅ 모델 저장 완료: {model_save_path}")

test_loss, test_acc = model.evaluate(test_ds, verbose=1)
print(f"✅ Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")
