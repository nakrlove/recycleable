import os
import math
import tensorflow as tf
from tensorflow.keras import layers, optimizers, callbacks, metrics
from tensorflow.keras.applications import EfficientNetV2M
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# ================== 경로 설정 ==================
KERS_DATA = "keras"
KERS_DATA_DIR = os.path.join(KERS_DATA)
ROOT = ""
BASE_DIR = "dataset_sp500"
FULL_BASE_DIR = os.path.join(ROOT, BASE_DIR)
train_dir = os.path.join(FULL_BASE_DIR, "train")
val_dir   = os.path.join(FULL_BASE_DIR, "val")
test_dir  = os.path.join(FULL_BASE_DIR, "test")

IMG_SIZE = (224,224)
BATCH_SIZE = 16
EPOCHS = 40
SEED = 42

# ================== 데이터셋 로드 (원본: 배치 해제 전) ==================
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

# ================== 클래스 카운트 & class_weight ==================
# (train 디렉토리의 파일 수로 계산)
class_counts = {cls: len(os.listdir(os.path.join(train_dir, cls))) for cls in class_names}
all_labels = []
for idx, cls_name in enumerate(class_names):
    all_labels += [idx] * class_counts[cls_name]

class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
class_weight_dict = dict(enumerate(class_weights))

# ================== 증강 정의 ==================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.25),
    layers.RandomZoom(0.25),
    layers.RandomTranslation(0.1,0.1),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.15),
], name="data_augmentation")

# ================== 밸런스(오버샘플링) 처리: 언배치 후 클래스별 데이터셋 생성 ==================
# 주의: image_dataset_from_directory 가 반환하는 y는 one-hot (label_mode="categorical") 이므로 tf.argmax 사용 가능
train_ds_unbatched = train_ds.unbatch()

class_datasets = []
repeat_factors = {}
max_count = max(class_counts.values())

for cls_idx, cls_name in enumerate(class_names):
    cnt = class_counts[cls_name]
    # repeat factor: 적은 클래스는 더 많이 반복
    repeat_factor = max(1, max_count // cnt)
    repeat_factors[cls_name] = repeat_factor

    ds_cls = train_ds_unbatched.filter(lambda x, y, _idx=cls_idx: tf.argmax(y) == _idx)
    ds_cls = ds_cls.repeat(repeat_factor)  # 각 클래스별로 반복
    # 캐스팅 및 전처리(이미지 -> float32) 수행 (preprocess_input은 색공간/정규화 처리)
    ds_cls = ds_cls.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=tf.data.AUTOTUNE)
    # 증강은 배치 이전에도 가능 (입력이 float)
    ds_cls = ds_cls.map(lambda x, y: (data_augmentation(x, training=True), y),
                        num_parallel_calls=tf.data.AUTOTUNE)
    class_datasets.append(ds_cls)

# effective total samples after oversampling
effective_samples = sum(class_counts[cls] * repeat_factors[cls] for cls in class_names)
print(f"Effective train samples after oversampling: {effective_samples}")

# 샘플링으로 클래스 섞기 -> batch -> prefetch -> repeat(무한)
# 메모리가 제한적이면 shuffle buffer를 작게 잡습니다.
SHUFFLE_BUFFER = min(1000, effective_samples)
train_ds_balanced = (
    tf.data.experimental.sample_from_datasets(class_datasets, seed=SEED)
    .shuffle(SHUFFLE_BUFFER)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
    .repeat()  # 무한 반복: fit()에서 steps_per_epoch로 제어
)

# ================== MixUp + CutMix (batch 단위에서 동작하도록 수정) ==================
def sample_beta_scalar(alpha, shape):
    # shape: scalar batch_size
    # 반환: [batch_size, 1, 1, 1] or [batch_size, 1] 등으로 reshape은 호출부에서 처리
    gamma1 = tf.random.gamma(shape, alpha)
    gamma2 = tf.random.gamma(shape, alpha)
    lam = gamma1 / (gamma1 + gamma2)
    return lam

def mixup_cutmix(ds, alpha=0.2):
    def _map(x, y):
        x = tf.cast(x, tf.float32)  # ensure float
        batch_size = tf.shape(x)[0]

        # shuffle within batch
        idx = tf.random.shuffle(tf.range(batch_size))
        x2 = tf.gather(x, idx)
        y2 = tf.gather(y, idx)

        # MixUp lambda (per-sample)
        lam = sample_beta_scalar(alpha, [batch_size])
        lam_x = tf.reshape(lam, [batch_size, 1, 1, 1])
        lam_y = tf.reshape(lam, [batch_size, 1])

        x_mix = x * lam_x + x2 * (1.0 - lam_x)
        y_mix = y * lam_y + y2 * (1.0 - lam_y)

        # CutMix region (random rectangle per sample)
        H = IMG_SIZE[0]
        W = IMG_SIZE[1]

        def _cutmix_single(args):
            img1, img2 = args
            cx = tf.random.uniform([], 0, W, dtype=tf.int32)
            cy = tf.random.uniform([], 0, H, dtype=tf.int32)
            rw = tf.random.uniform([], 32, max(32, W // 2), dtype=tf.int32)
            rh = tf.random.uniform([], 32, max(32, H // 2), dtype=tf.int32)
            x1 = tf.clip_by_value(cx - rw // 2, 0, W)
            y1 = tf.clip_by_value(cy - rh // 2, 0, H)
            x2_ = tf.clip_by_value(cx + rw // 2, 0, W)
            y2_ = tf.clip_by_value(cy + rh // 2, 0, H)
            h = y2_ - y1
            w = x2_ - x1
            # create mask with ones in the box
            mask = tf.pad(tf.ones([h, w, 1], dtype=tf.float32),
                          [[y1, H - y2_], [x1, W - x2_], [0, 0]])
            return img1 * (1.0 - mask) + img2 * mask

        x_cutmix = tf.map_fn(_cutmix_single, (x_mix, x2), dtype=tf.float32)
        # final: randomly choose per-sample whether to use MixUp or CutMix
        use_cutmix = tf.random.uniform([batch_size], 0, 1) > 0.5
        use_cutmix = tf.cast(tf.reshape(use_cutmix, [batch_size, 1, 1, 1]), tf.float32)
        x_final = x_mix * (1.0 - use_cutmix) + x_cutmix * use_cutmix

        return x_final, y_mix

    return ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)

train_ds_balanced = mixup_cutmix(train_ds_balanced, alpha=0.2)

# ================== val/test prefetch/cache ==================
# val은 소형이거나 메모리가 충분하면 cache() 권장
val_ds = val_ds.map(lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y))
val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.map(lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y))
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

# ================== 모델 정의 ==================
base_model = EfficientNetV2M(include_top=False, input_shape=IMG_SIZE+(3,), weights='imagenet')
base_model.trainable = False

inputs = tf.keras.Input(shape=IMG_SIZE+(3,))
# 이미 파이프라인에서 preprocess_input 했으므로 여기서는 바로 base_model에 넣음
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(512, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

# ================== LR Warmup + CosineDecay ==================
class WarmUpCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, total_steps, warmup_steps):
        super().__init__()
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        base_lr = tf.cast(self.base_lr, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)
        lr = tf.cond(
            step < warmup_steps,
            lambda: base_lr * (step / warmup_steps),
            lambda: base_lr * 0.5 * (1 + tf.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
        )
        return lr

    def get_config(self):
        return {"base_lr": self.base_lr, "total_steps": self.total_steps, "warmup_steps": self.warmup_steps}

# ================== steps_per_epoch 계산 (정확히) ==================
steps_per_epoch = math.ceil(effective_samples / BATCH_SIZE)
total_steps = steps_per_epoch * EPOCHS
warmup_steps = int(0.1 * total_steps)
base_lr = 1e-3

lr_schedule = WarmUpCosine(base_lr, total_steps, warmup_steps)
optimizer = optimizers.Adam(learning_rate=lr_schedule)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy',
             tf.keras.metrics.TopKCategoricalAccuracy(k=3,name='top3'),
             tf.keras.metrics.TopKCategoricalAccuracy(k=5,name='top5')]
)

# ================== callbacks ==================
callbacks_list = [
    callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
    callbacks.ModelCheckpoint(os.path.join(KERS_DATA_DIR, "best_model.keras"),
                              monitor='val_top3', save_best_only=True, mode='max', verbose=1)
]

# ================== 1차 학습 (steps_per_epoch 명시로 Unknown 제거) ==================
history = model.fit(
    train_ds_balanced,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=math.ceil(tf.data.experimental.cardinality(val_ds).numpy()),
    callbacks=callbacks_list,
    verbose=1
)

# ================== Fine-tuning ==================
base_model.trainable = True
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

optimizer_ft = optimizers.Adam(learning_rate=1e-4)
model.compile(
    optimizer=optimizer_ft,
    loss='categorical_crossentropy',
    metrics=['accuracy',
             tf.keras.metrics.TopKCategoricalAccuracy(k=3,name='top3'),
             tf.keras.metrics.TopKCategoricalAccuracy(k=5,name='top5')]
)

# Fine-tuning 에서도 같은 train_ds를 사용; class_weight는 오버샘플링 했기 때문에 보통 불필요
history_ft = model.fit(
    train_ds_balanced,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=math.ceil(tf.data.experimental.cardinality(val_ds).numpy()),
    callbacks=callbacks_list,
    verbose=1
)

# ================== 저장 및 평가 ==================
model_save_path = os.path.join(KERS_DATA_DIR, "efficientnetv2_maxaccuracy_full_eager.keras")
model.save(model_save_path, save_format="keras")
print(f"✅ 모델 저장 완료: {model_save_path}")

test_loss, test_acc, test_top3, test_top5 = model.evaluate(test_ds, verbose=1)
print(f"✅ Test Accuracy: {test_acc:.4f}, Top-3: {test_top3:.4f}, Top-5: {test_top5:.4f}, Test Loss: {test_loss:.4f}")
