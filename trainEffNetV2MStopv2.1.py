import os
import math
import json
import random
import tensorflow as tf
from tensorflow.keras import layers, optimizers, callbacks, metrics
from tensorflow.keras.applications import EfficientNetV2M
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# ================== 경로 설정 ==================
# ROOT = "drive/MyDrive/"
# BASE_DIR = "dataset_sp"

ROOT = ""
BASE_DIR = "dataset_sp1000"
FULL_BASE_DIR = os.path.join(ROOT, BASE_DIR)
train_dir = os.path.join(FULL_BASE_DIR, "train")
val_dir   = os.path.join(FULL_BASE_DIR, "val")
test_dir  = os.path.join(FULL_BASE_DIR, "test")

IMG_SIZE = (224,224)
BATCH_SIZE = 16
EPOCHS = 40
SEED = 42

# ================== 데이터셋 로드 ==================
train_ds = image_dataset_from_directory(
    train_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="categorical", shuffle=True, seed=SEED
)
val_ds = image_dataset_from_directory(
    val_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="categorical", shuffle=False, seed=SEED
)
test_ds = image_dataset_from_directory(
    test_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="categorical", shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"클래스 개수: {num_classes}")

# ================== 클래스 가중치 ==================
all_labels = []
for idx, cls_name in enumerate(class_names):
    cls_path = os.path.join(train_dir, cls_name)
    count = len(os.listdir(cls_path))
    all_labels += [idx] * count

class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
class_weight_dict = dict(enumerate(class_weights))

# ================== 데이터 증강 ==================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.25),
    layers.RandomZoom(0.25),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.15),
    layers.RandomTranslation(0.1,0.1),
], name="data_augmentation")

# ================== 불균형 클래스 대응 ==================
train_ds_unbatched = train_ds.unbatch()
class_counts = {cls: len(os.listdir(os.path.join(train_dir, cls))) for cls in class_names}
max_count = max(class_counts.values())

class_datasets = []
for cls_idx, cls_name in enumerate(class_names):
    ds_cls = train_ds_unbatched.filter(lambda x,y: tf.argmax(y)==cls_idx)
    repeat_factor = max(1, max_count // class_counts[cls_name])
    ds_cls = ds_cls.repeat(repeat_factor)
    ds_cls = ds_cls.map(lambda x,y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    class_datasets.append(ds_cls)

train_ds_balanced = tf.data.experimental.sample_from_datasets(class_datasets)
train_ds_balanced = train_ds_balanced.shuffle(3000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ================== MixUp + CutMix (tfp 제거 버전) ==================
def sample_beta(alpha, beta, shape):
    # Beta(alpha, beta) ~ Gamma(alpha)/(Gamma(alpha)+Gamma(beta))
    gamma1 = tf.random.gamma(shape, alpha)
    gamma2 = tf.random.gamma(shape, beta)
    return gamma1 / (gamma1 + gamma2)

def mixup_cutmix(ds, alpha=0.2):
    def _mixup_cutmix(x, y):
        batch_size = tf.shape(x)[0]
        idx = tf.random.shuffle(tf.range(batch_size))
        x2 = tf.gather(x, idx)
        y2 = tf.gather(y, idx)

        # MixUp
        l = sample_beta(alpha, alpha, [batch_size,1,1,1])
        y_l = sample_beta(alpha, alpha, [batch_size,1])
        x_mix = x*l + x2*(1-l)
        y_mix = y*y_l + y2*(1-y_l)

        # CutMix
        H, W = IMG_SIZE
        def _cutmix_image(a, b):
            cx = tf.random.uniform([], 0, W, dtype=tf.int32)
            cy = tf.random.uniform([], 0, H, dtype=tf.int32)
            rw = tf.random.uniform([], 32, W//2, dtype=tf.int32)
            rh = tf.random.uniform([], 32, H//2, dtype=tf.int32)
            x1 = tf.clip_by_value(cx-rw//2,0,W)
            y1 = tf.clip_by_value(cy-rh//2,0,H)
            x2_ = tf.clip_by_value(cx+rw//2,0,W)
            y2_ = tf.clip_by_value(cy+rh//2,0,H)
            mask = tf.pad(tf.ones([y2_-y1, x2_-x1,3]), [[y1,H-y2_],[x1,W-x2_],[0,0]])
            return a*(1-mask) + b*mask
        x_final = tf.map_fn(lambda a: _cutmix_image(a[0],a[1]), (x_mix,x2), dtype=tf.float32)
        return x_final, y_mix
    return ds.map(_mixup_cutmix, num_parallel_calls=tf.data.AUTOTUNE)

train_ds_balanced = mixup_cutmix(train_ds_balanced, alpha=0.2)

# val/test는 증강 없이 prefetch/cache
val_ds = val_ds.prefetch(tf.data.AUTOTUNE).cache()
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

# ================== 모델 정의 ==================
base_model = EfficientNetV2M(include_top=False, input_shape=IMG_SIZE+(3,), weights='imagenet')
base_model.trainable = False

inputs = tf.keras.Input(shape=IMG_SIZE+(3,))
x = tf.keras.applications.efficientnet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(512, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)
optimizer = optimizers.Adam(learning_rate=1e-3)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy','TopKCategoricalAccuracy'])

callbacks_list = [
    callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
]

# ================== 1차 학습 ==================
print("✅ 1차 학습: Feature Extractor 시작")
history = model.fit(
    train_ds_balanced,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weight_dict,
    callbacks=callbacks_list,
    verbose=1
)

# ================== 2차 학습: Fine-tuning ==================
base_model.trainable = True
for layer in base_model.layers[:150]:
    layer.trainable = False

model.compile(optimizer=optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy','TopKCategoricalAccuracy'])

history_ft = model.fit(
    train_ds_balanced,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weight_dict,
    callbacks=callbacks_list,
    verbose=1
)

# ================== 모델 저장 ==================
model_save_path = os.path.join(FULL_BASE_DIR, "efficientnetv2_maxaccuracy_mixup_cutmix_noTFP.keras")
model.save(model_save_path, save_format="keras")
print(f"✅ 모델 저장 완료: {model_save_path}")

# ================== 테스트 평가 ==================
test_loss, test_acc, test_top3 = model.evaluate(test_ds, verbose=1)
print(f"✅ Test Accuracy: {test_acc:.4f}, Top-3 Accuracy: {test_top3:.4f}, Test Loss: {test_loss:.4f}")
