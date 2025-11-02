# trainEffNetV2M_prod_ready.py
"""
Production-ready training script with:
 - tf.data pipeline optimizations (map parallelism, prefetch)
 - optional caching (disabled by default for large datasets)
 - optional mixed precision
 - stronger EarlyStopping (min_delta + smaller patience)
 - periodic checkpointing (every N epochs and every M steps)
 - batch/time logging callback for profiling
 - macOS 'caffeinate' 안내 for preventing sleep
 - dataset pipeline micro-benchmark function
 - recommended epoch defaults lowered for quicker iteration
"""

import os
import sys
import time
import json
import shutil
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2M, preprocess_input

try:
    from sklearn.utils.class_weight import compute_class_weight
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

# -------------------------
# CONFIGURABLE HYPERPARAMS
# -------------------------
BASE_DIR = "dataset_25000"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

# Performance tuning flags
USE_MIXED_PRECISION = False   # Set True to try mixed float16 (environment-dependent)
USE_CACHE = False             # Set True ONLY if dataset fits in memory (or you want OS disk cache)
NUM_PARALLEL_CALLS = tf.data.AUTOTUNE
PREFETCH = tf.data.AUTOTUNE
SHUFFLE_BUFFER = 2000

# Epoch recommendations (reduced for large dataset; you can raise later)
EPOCHS_STAGE1 = 15   # Stage1: head-only. One epoch here is expensive due to dataset size
EPOCHS_STAGE2 = 30   # Stage2: fine-tune full model

# EarlyStopping tuning (stricter)
ES_MIN_DELTA = 1e-4
ES_PATIENCE_STAGE1 = 3
ES_PATIENCE_STAGE2 = 3

# Checkpointing / Backup
KERAS_DIR = "./keras"
BEST_MODEL_PATH = os.path.join(KERAS_DIR, "best_model.keras")
BEST_FINETUNE_PATH = os.path.join(KERAS_DIR, "best_finetuned_model.keras")
BACKUP_CHECKPOINT_DIR = os.path.join(KERAS_DIR, "backups")
os.makedirs(KERAS_DIR, exist_ok=True)
os.makedirs(BACKUP_CHECKPOINT_DIR, exist_ok=True)

# Periodic backup settings
BACKUP_EPOCH_INTERVAL = 2   # save a backup every N epochs
BACKUP_STEP_INTERVAL = None # optionally save every M steps (None to disable)

CLASS_JSON = os.path.join(KERAS_DIR, "class_indices.json")

# -------------------------
# GPU / precision setup
# -------------------------
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print("[INFO] GPU(s) detected. Enabled memory growth.")
    except Exception as e:
        print("[WARN] Could not set memory growth:", e)
else:
    print("[INFO] No GPU detected. Running on CPU (slower).")

if USE_MIXED_PRECISION:
    try:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')
        print("[INFO] Mixed precision set to mixed_float16.")
    except Exception as e:
        print("[WARN] Mixed precision unavailable:", e)

# -------------------------
# Utilities: clean macOS .DS_Store and empty dirs
# -------------------------
def clean_dataset_dirs(base_dir):
    removed = False
    for root, dirs, files in os.walk(base_dir):
        for f in list(files):
            if f == ".DS_Store":
                try:
                    os.remove(os.path.join(root, f))
                except Exception:
                    pass
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
        print("[INFO] Removed empty folders. Re-run dataset scanning.")

clean_dataset_dirs(BASE_DIR)

# -------------------------
# Dataset loaders (batching from the start)
# -------------------------
def load_dataset(split, batch_size=BATCH_SIZE, shuffle=False):
    path = os.path.join(BASE_DIR, split)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset split not found: {path}")
    ds = tf.keras.utils.image_dataset_from_directory(
        path,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="int",   # int labels, convert to one-hot later
        shuffle=shuffle,
        seed=SEED
    )
    return ds

train_ds = load_dataset("train", shuffle=True)
val_ds = load_dataset("val", shuffle=False)
test_ds = load_dataset("test", shuffle=False)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"[INFO] num_classes = {num_classes}")
print(f"[INFO] class_names = {class_names}")

with open(CLASS_JSON, "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)
print(f"[INFO] Saved class indices to {CLASS_JSON}")

# -------------------------
# Preprocess & augmentation (batched)
# -------------------------
AUTOTUNE = tf.data.AUTOTUNE

def preprocess_batch(images, labels):
    # images: uint8 [B, H, W, C]
    images = tf.image.resize(images, IMG_SIZE)
    images = tf.cast(images, tf.float32)
    images = preprocess_input(images)   # EfficientNetV2 preprocessing (float32)
    labels = tf.one_hot(labels, depth=num_classes)
    return images, labels

# Lightweight augmentation applied per batch (vectorized)
def augment_batch(images, labels):
    images = tf.image.random_flip_left_right(images)
    images = tf.image.random_flip_up_down(images)
    images = tf.image.random_brightness(images, 0.12)
    images = tf.image.random_contrast(images, 0.85, 1.15)
    # random zoom per image
    def random_zoom(img):
        scale = tf.random.uniform([], 0.92, 1.05)
        h = tf.shape(img)[0]; w = tf.shape(img)[1]
        new_h = tf.cast(tf.cast(h, tf.float32) * scale, tf.int32)
        new_w = tf.cast(tf.cast(w, tf.float32) * scale, tf.int32)
        img = tf.image.resize(img, [new_h, new_w])
        img = tf.image.resize_with_crop_or_pad(img, h, w)
        return img
    images = tf.map_fn(lambda im: random_zoom(im), images, fn_output_signature=tf.float32)
    return images, labels

def prepare_dataset(ds, training=False, use_cache=USE_CACHE):
    ds = ds.map(lambda x, y: (tf.image.resize(x, IMG_SIZE), y),
                num_parallel_calls=NUM_PARALLEL_CALLS)
    ds = ds.map(preprocess_batch, num_parallel_calls=NUM_PARALLEL_CALLS)
    if use_cache:
        print("[INFO] Caching dataset in memory (ensure enough RAM).")
        ds = ds.cache()
    if training:
        ds = ds.map(augment_batch, num_parallel_calls=NUM_PARALLEL_CALLS)
        ds = ds.shuffle(SHUFFLE_BUFFER, seed=SEED)
    ds = ds.prefetch(PREFETCH)
    return ds

train_ds = prepare_dataset(train_ds, training=True)
val_ds = prepare_dataset(val_ds, training=False)
test_ds = prepare_dataset(test_ds, training=False)

# -------------------------
# Class weights
# -------------------------
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

if _HAS_SKLEARN:
    class_weights = compute_class_weight("balanced", classes=np.arange(num_classes), y=all_labels)
else:
    counts = np.bincount(all_labels, minlength=num_classes)
    total = counts.sum()
    class_weights = total / (num_classes * (counts + 1e-6))
class_weights_dict = {i: float(w) for i, w in enumerate(class_weights)}
print("[INFO] class_weights:", class_weights_dict)

# -------------------------
# Model creation / safe load
# -------------------------
def make_model(num_classes, input_shape=IMG_SIZE + (3,), dropout_rate=0.3):
    base = EfficientNetV2M(include_top=False, input_shape=input_shape, weights="imagenet")
    inp = layers.Input(shape=input_shape)
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs=inp, outputs=out)
    return model, base

model = None
if os.path.exists(BEST_MODEL_PATH):
    try:
        print("[INFO] Loading existing model:", BEST_MODEL_PATH)
        model = load_model(BEST_MODEL_PATH)
        if model.output_shape[-1] != num_classes:
            print("[WARN] Loaded model output_dim != dataset classes -> will recreate model")
            model = None
    except Exception as e:
        print("[WARN] Failed to load model:", e)
        model = None

if model is None:
    model, base_model = make_model(num_classes)
    # freeze base for stage1
    for layer in model.layers:
        layer.trainable = False
    # make head trainable (dense)
    for layer in model.layers[-4:]:
        layer.trainable = True
    print("[INFO] New model created. Base frozen for stage1.")
else:
    print("[INFO] Using loaded model; freezing base for stage1.")
    for layer in model.layers[:-2]:
        layer.trainable = False

model.summary()

# -------------------------
# Callbacks: stronger EarlyStopping, periodic backup, step-time logging
# -------------------------
class BatchTimingCallback(callbacks.Callback):
    def __init__(self, log_every_n_steps=200, out_file=None):
        super().__init__()
        self.log_every = log_every_n_steps
        self.out_file = out_file
        self.step_times = []
        self._step_start = None

    def on_train_batch_begin(self, batch, logs=None):
        self._step_start = time.time()

    def on_train_batch_end(self, batch, logs=None):
        t = time.time() - (self._step_start or time.time())
        self.step_times.append(t)
        if len(self.step_times) % self.log_every == 0:
            arr = np.array(self.step_times[-self.log_every:])
            msg = (f"[BATCHTIMING] last {self.log_every} steps: mean={arr.mean():.3f}s, "
                   f"median={np.median(arr):.3f}s, p95={np.percentile(arr,95):.3f}s")
            print(msg)
            if self.out_file:
                with open(self.out_file, "a") as f:
                    f.write(msg + "\n")

class PeriodicBackupCallback(callbacks.Callback):
    def __init__(self, backup_dir=BACKUP_CHECKPOINT_DIR, epoch_interval=BACKUP_EPOCH_INTERVAL, step_interval=BACKUP_STEP_INTERVAL):
        super().__init__()
        self.backup_dir = backup_dir
        self.epoch_interval = epoch_interval
        self.step_interval = step_interval
        self._step_count = 0

    def on_epoch_end(self, epoch, logs=None):
        if self.epoch_interval and ((epoch+1) % self.epoch_interval == 0):
            fname = os.path.join(self.backup_dir, f"backup_epoch{epoch+1}.keras")
            self.model.save(fname)
            print(f"[BACKUP] Saved backup at epoch {epoch+1} -> {fname}")

    def on_train_batch_end(self, batch, logs=None):
        self._step_count += 1
        if self.step_interval and (self._step_count % self.step_interval == 0):
            fname = os.path.join(self.backup_dir, f"backup_step{self._step_count}.keras")
            self.model.save(fname)
            print(f"[BACKUP] Saved backup at step {self._step_count} -> {fname}")

# standard callbacks
cb_stage1 = [
    callbacks.ModelCheckpoint(BEST_MODEL_PATH, monitor="val_top3", save_best_only=True, mode="max", save_weights_only=False),
    callbacks.EarlyStopping(monitor="val_loss", patience=ES_PATIENCE_STAGE1, min_delta=ES_MIN_DELTA, restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
    BatchTimingCallback(log_every_n_steps=500, out_file=os.path.join(KERAS_DIR,"batch_timing.log")),
    PeriodicBackupCallback(epoch_interval=BACKUP_EPOCH_INTERVAL)
]

cb_stage2 = [
    callbacks.ModelCheckpoint(BEST_FINETUNE_PATH, monitor="val_top3", save_best_only=True, mode="max", save_weights_only=False),
    callbacks.EarlyStopping(monitor="val_loss", patience=ES_PATIENCE_STAGE2, min_delta=ES_MIN_DELTA, restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
    BatchTimingCallback(log_every_n_steps=500, out_file=os.path.join(KERAS_DIR,"batch_timing_stage2.log")),
    PeriodicBackupCallback(epoch_interval=BACKUP_EPOCH_INTERVAL)
]

# -------------------------
# Compile & Stage1 fit (head)
# -------------------------
LR_STAGE1 = 1e-4
model.compile(
    optimizer=optimizers.Adam(learning_rate=LR_STAGE1),
    loss="categorical_crossentropy",
    metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3"), tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5")]
)

print("[INFO] Starting Stage1 training (head-only)")
history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE1,
    class_weight=class_weights_dict,
    callbacks=cb_stage1,
    verbose=1
)

# -------------------------
# Stage2: unfreeze and fine-tune
# -------------------------
print("[INFO] Starting Stage2: unfreeze base and fine-tune")
for layer in model.layers:
    layer.trainable = True

LR_STAGE2 = 1e-5
model.compile(
    optimizer=optimizers.Adam(learning_rate=LR_STAGE2),
    loss="categorical_crossentropy",
    metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3"), tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5")]
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE2,
    class_weight=class_weights_dict,
    callbacks=cb_stage2,
    verbose=1
)

# -------------------------
# Final evaluation & save
# -------------------------
print("[INFO] Evaluating on test set...")
test_metrics = model.evaluate(test_ds)
print("[INFO] Test metrics:", test_metrics)

final_path = BEST_FINETUNE_PATH if os.path.exists(BEST_FINETUNE_PATH) else BEST_MODEL_PATH
model.save(final_path)
print(f"[INFO] Saved final model to {final_path}")

# -------------------------
# Pipeline benchmark util (manual call)
# -------------------------
def benchmark_dataset(ds, num_batches=50):
    print(f"[BENCH] Running dataset benchmark for {num_batches} batches...")
    times = []
    it = iter(ds)
    for i in range(num_batches):
        t0 = time.time()
        try:
            _ = next(it)
        except StopIteration:
            break
        times.append(time.time() - t0)
    times = np.array(times)
    print(f"[BENCH] mean={times.mean():.3f}s, median={np.median(times):.3f}s, p95={np.percentile(times,95):.3f}s")
    return times

# Optional: run small benchmark to inspect pipeline speed
# runs only if explicitly invoked at end of script
if "--bench" in sys.argv:
    benchmark_dataset(train_ds, num_batches=50)

# -------------------------
# System sleep prevention note (outside python)
# -------------------------
print("\n[INFO] To prevent macOS sleep during long training, run in another terminal:")
print("    caffeinate -dims &")
print("This command keeps system awake (press Ctrl+C to stop).")
print("Alternatively: sudo pmset -a sleep 0 displaysleep 0\n")
