# trainEffNetV2M_finetune_fixed.py
"""
수정완성본:
 - batch_size 고정 (초기 로드부터)
 - .DS_Store, 빈 디렉토리 필터링 체크
 - class_indices.json 저장
 - model load 시 output dim 검사 -> mismatch면 새 모델 생성
 - EfficientNetV2M preprocess_input 적용
 - class_weight 계산하여 fit에 적용
 - GPU memory growth 안전 설정
"""

import os
import sys
import json
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, metrics, callbacks
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2M, preprocess_input

# Optional: sklearn for class_weight; fallback if not installed
try:
    from sklearn.utils.class_weight import compute_class_weight
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

# ---------------------------
# 0. 설정값
# ---------------------------
BASE_DIR = "dataset_25000"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42
EPOCHS_STAGE1 = 50
EPOCHS_STAGE2 = 30
KERAS_DIR = "./keras"
BEST_MODEL_PATH = os.path.join(KERAS_DIR, "best_model.keras")
BEST_FINETUNE_PATH = os.path.join(KERAS_DIR, "best_finetuned_model.keras")
CLASS_JSON = os.path.join(KERAS_DIR, "class_indices.json")

os.makedirs(KERAS_DIR, exist_ok=True)

# ---------------------------
# 0-1. GPU 메모리 growth 설정 (있을 때만)
# ---------------------------
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print("[INFO] GPU found. Enabled memory growth for GPUs.")
    except Exception as e:
        print("[WARN] Could not set memory growth:", e)
else:
    print("[INFO] No GPU/Metal device found (or not visible). Running on CPU.")


# ---------------------------
# Utility: macOS .DS_Store 및 빈 폴더 체크
# ---------------------------
def clean_dataset_dirs(base_dir):
    # remove .DS_Store files
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f == ".DS_Store":
                try:
                    os.remove(os.path.join(root, f))
                except Exception:
                    pass
    # find empty class folders under train/val/test and warn
    removed = False
    for split in ("train", "val", "test"):
        split_dir = os.path.join(base_dir, split)
        if not os.path.exists(split_dir):
            continue
        for cls in os.listdir(split_dir):
            cls_path = os.path.join(split_dir, cls)
            if os.path.isdir(cls_path) and not any(os.scandir(cls_path)):
                print(f"[WARN] Empty folder detected and removed: {cls_path}")
                try:
                    shutil.rmtree(cls_path)
                    removed = True
                except Exception as e:
                    print("[WARN] failed to remove empty folder:", e)
    if removed:
        print("[INFO] Some empty class folders were removed; re-check your dataset if necessary.")

clean_dataset_dirs(BASE_DIR)


# ---------------------------
# 1. 데이터셋 로드 (batch_size 적용)
#    label_mode="int" -> 이후 one_hot 처리
# ---------------------------
def load_dataset(split, batch_size=BATCH_SIZE, shuffle=False):
    path = os.path.join(BASE_DIR, split)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset split not found: {path}")
    ds = tf.keras.utils.image_dataset_from_directory(
        path,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="int",
        shuffle=shuffle,
        seed=SEED
    )
    return ds

train_ds = load_dataset("train", shuffle=True)
val_ds = load_dataset("val", shuffle=False)
test_ds = load_dataset("test", shuffle=False)

# class names & counts
class_names = train_ds.class_names
num_classes = len(class_names)
print(f"[INFO] num_classes = {num_classes}")
print(f"[INFO] class_names = {class_names}")

# save class names JSON for inference service
with open(CLASS_JSON, "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)
print(f"[INFO] Saved class indices to {CLASS_JSON}")

# ---------------------------
# 2. 전처리 / augmentation 함수
# ---------------------------
AUTOTUNE = tf.data.AUTOTUNE

def preprocess_and_onehot(image, label):
    # image: uint8 [0,255]
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)  # EfficientNetV2M preprocessing
    label = tf.one_hot(label, depth=num_classes)
    return image, label

def augment_batch(images, labels):
    # images: batch tensor
    images = tf.image.random_flip_left_right(images)
    images = tf.image.random_flip_up_down(images)
    # random brightness/contrast per image in batch
    images = tf.image.random_brightness(images, 0.15)
    images = tf.image.random_contrast(images, 0.8, 1.2)
    # random zoom using central crop/resizing
    def random_zoom(img):
        # random scale between 0.9 and 1.05
        scale = tf.random.uniform([], 0.9, 1.05)
        shape = tf.shape(img)
        h = shape[0]; w = shape[1]
        new_h = tf.cast(tf.cast(h, tf.float32) * scale, tf.int32)
        new_w = tf.cast(tf.cast(w, tf.float32) * scale, tf.int32)
        img = tf.image.resize(img, [new_h, new_w])
        img = tf.image.resize_with_crop_or_pad(img, h, w)
        return img
    images = tf.map_fn(lambda im: random_zoom(im), images)
    return images, labels

# create prepared datasets
train_ds = (
    train_ds
    .map(lambda x, y: (tf.image.resize(x, IMG_SIZE), y), num_parallel_calls=AUTOTUNE)
    .map(preprocess_and_onehot, num_parallel_calls=AUTOTUNE)
    .map(lambda x, y: (tf.identity(x), tf.identity(y)))  # ensure shapes known
)

# For augmentation, operate on batched dataset (we already loaded batched)
train_ds = train_ds.map(lambda x, y: augment_batch(x, y), num_parallel_calls=AUTOTUNE)
train_ds = train_ds.shuffle(1000, seed=SEED).prefetch(AUTOTUNE)

val_ds = (
    val_ds
    .map(lambda x, y: (tf.image.resize(x, IMG_SIZE), y), num_parallel_calls=AUTOTUNE)
    .map(preprocess_and_onehot, num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
)

test_ds = (
    test_ds
    .map(lambda x, y: (tf.image.resize(x, IMG_SIZE), y), num_parallel_calls=AUTOTUNE)
    .map(preprocess_and_onehot, num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
)

# ---------------------------
# 3. 클래스 가중치 계산 (class imbalance)
# ---------------------------
# gather labels from original unshuffled dataset; safe because labels are ints
all_labels = []
for _, batch_labels in tf.keras.utils.image_dataset_from_directory(
        os.path.join(BASE_DIR, "train"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False):
    all_labels.append(batch_labels.numpy().ravel())
if all_labels:
    all_labels = np.concatenate(all_labels, axis=0)
else:
    raise RuntimeError("No labels found in train dataset")

# compute weights
if _HAS_SKLEARN:
    class_weights = compute_class_weight("balanced", classes=np.arange(num_classes), y=all_labels)
else:
    counts = np.bincount(all_labels, minlength=num_classes)
    total = counts.sum()
    # weight inversely proportional to frequency
    class_weights = total / (num_classes * (counts + 1e-6))
class_weights_dict = {i: float(w) for i, w in enumerate(class_weights)}
print("[INFO] class_weights:", class_weights_dict)


# ---------------------------
# 4. 모델 생성 / 또는 로드 (안전 검사 포함)
# ---------------------------
def make_model(num_classes, input_shape=IMG_SIZE + (3,), dropout_rate=0.3):
    base = EfficientNetV2M(include_top=False, input_shape=input_shape, weights="imagenet")
    x = layers.Input(shape=input_shape)
    y = base(x, training=False)
    y = layers.GlobalAveragePooling2D()(y)
    y = layers.Dropout(dropout_rate)(y)
    out = layers.Dense(num_classes, activation="softmax")(y)
    m = models.Model(inputs=x, outputs=out)
    return m, base

model = None
if os.path.exists(BEST_MODEL_PATH):
    try:
        print("[INFO] Found existing model. Attempting to load:", BEST_MODEL_PATH)
        model = load_model(BEST_MODEL_PATH)
        # verify output dim
        output_dim = model.output_shape[-1]
        print(f"[INFO] Loaded model output dim: {output_dim}, dataset num_classes: {num_classes}")
        if output_dim != num_classes:
            print("[WARN] Model output dim != num_classes. Recreating new model from scratch.")
            model = None
    except Exception as e:
        print("[WARN] Could not load model (will create a new one):", e)
        model = None

if model is None:
    model, base_model = make_model(num_classes)
    base_model.trainable = False
    LR = 1e-4
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LR),
        loss="categorical_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3"),
                 tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5")]
    )
    print("[INFO] New model created.")
else:
    # If loaded successfully, freeze base for stage1 (fine-tune later)
    # try to find base model to freeze (if nested model)
    # naive: set all layers except last Dense to trainable=False for stage1
    for layer in model.layers[:-2]:
        layer.trainable = False
    print("[INFO] Existing model loaded and partially frozen for stage1.")

model.summary()

# ---------------------------
# 5. callbacks
# ---------------------------
cb_stage1 = [
    callbacks.ModelCheckpoint(BEST_MODEL_PATH, monitor="val_top3", save_best_only=True, mode="max"),
    callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1)
]

# ---------------------------
# 6. Stage1 training (feature head only)
# ---------------------------
print("[INFO] Starting Stage 1 training (head only).")
history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE1,
    class_weight=class_weights_dict,
    callbacks=cb_stage1
)

# ---------------------------
# 7. Stage2 fine-tune (unfreeze base)
# ---------------------------
print("[INFO] Starting Stage 2 fine-tuning (unfreeze base).")
# try to find base model inside current model robustly:
base_candidate = None
for layer in model.layers:
    if isinstance(layer, tf.keras.Model) and layer is not model:
        base_candidate = layer
        break

if base_candidate is not None:
    print("[INFO] Found nested base model; setting trainable=True on base.")
    base_candidate.trainable = True
else:
    # if no nested model, set all layers trainable
    for layer in model.layers:
        layer.trainable = True

FINE_LR = 1e-5
model.compile(
    optimizer=optimizers.Adam(learning_rate=FINE_LR),
    loss="categorical_crossentropy",
    metrics=["accuracy",
             tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3"),
             tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5")]
)

cb_stage2 = [
    callbacks.ModelCheckpoint(BEST_FINETUNE_PATH, monitor="val_top3", save_best_only=True, mode="max"),
    callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=1)
]

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE2,
    class_weight=class_weights_dict,
    callbacks=cb_stage2
)

# ---------------------------
# 8. Evaluation
# ---------------------------
print("[INFO] Evaluating on test set...")
test_metrics = model.evaluate(test_ds)
print("[INFO] Test metrics:", test_metrics)

# Save final model if not already
final_save_path = BEST_FINETUNE_PATH if os.path.exists(BEST_FINETUNE_PATH) else BEST_MODEL_PATH
model.save(final_save_path)
print(f"[INFO] Final model saved to {final_save_path}")
