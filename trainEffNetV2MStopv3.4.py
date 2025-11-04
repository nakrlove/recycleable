"""
Final production-ready training script optimized for macOS M1
- tf.data pipeline (file-path based, interleave, controlled shuffle)
- MixUp & CutMix augmentation
- Label smoothing + AdamW (fallback) + weight decay
- Snapshot optional to avoid repeated heavy decoding
- Dead-stall watchdog + batch timing
- TF threading tuning for macOS M1
- Recommendations: install `tensorflow-macos` + `tensorflow-metal` for speed

Usage:
    python trainEffNetV2M_prod_final.py
    python trainEffNetV2M_prod_final.py --bench

Adjust top CONFIG values as needed.
"""

import os
import sys
import time
import json
import shutil
import random
from glob import glob

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks

# Attempt to import optional tensorflow-addons for AdamW
try:
    import tensorflow_addons as tfa
    _HAS_TFA = True
except Exception:
    tfa = None
    _HAS_TFA = False

# -------------------------
# CONFIG
# -------------------------
BASE_DIR = "dataset_25000"          # dataset root with train/val/test subdirs
IMG_SIZE = (224, 224)
BATCH_SIZE = 32                       # try 32 on 64GB RAM; lower if OOM
SEED = 42
NUM_CLASSES = None                    # auto-detected

USE_MIXED_PRECISION = False          # set True only if HW + TF support
USE_SNAPSHOT = True                   # use tf.data.experimental.snapshot when available
SNAPSHOT_DIR = "./keras/snapshot_train"

SHUFFLE_BUFFER = 1024
AUTOTUNE = tf.data.AUTOTUNE
NUM_INTERLEAVE = 16

EPOCHS_STAGE1 = 12
EPOCHS_STAGE2 = 20
ES_PATIENCE_STAGE1 = 4
ES_PATIENCE_STAGE2 = 4
ES_MIN_DELTA = 1e-4

LR_STAGE1 = 1e-4
LR_STAGE2 = 1e-5
WEIGHT_DECAY = 1e-5

KERAS_DIR = "./keras"
BEST_MODEL_PATH = os.path.join(KERAS_DIR, "best_model.keras")
BEST_FINETUNE_PATH = os.path.join(KERAS_DIR, "best_finetuned_model.keras")
BACKUP_CHECKPOINT_DIR = os.path.join(KERAS_DIR, "backups")
CLASS_JSON = os.path.join(KERAS_DIR, "class_indices.json")

os.makedirs(KERAS_DIR, exist_ok=True)
os.makedirs(BACKUP_CHECKPOINT_DIR, exist_ok=True)

# -------------------------
# Runtime tuning for macOS M1
# -------------------------
try:
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    print("[INFO] Set TF inter_op=2, intra_op=4")
except Exception as e:
    print("[WARN] Failed to set TF threading:", e)

if USE_MIXED_PRECISION:
    try:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')
        print("[INFO] Mixed precision enabled: mixed_float16")
    except Exception as e:
        print("[WARN] Mixed precision not available:", e)

# -------------------------
# Utilities
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
# File path parsing & dataset helpers
# -------------------------

def parse_directory_to_paths(base_dir, split):
    split_dir = os.path.join(base_dir, split)
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"Dataset split not found: {split_dir}")
    class_names = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
    paths = []
    labels = []
    for idx, cls in enumerate(class_names):
        cls_dir = os.path.join(split_dir, cls)
        for root, _, files in os.walk(cls_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    paths.append(os.path.join(root, f))
                    labels.append(idx)
    return paths, labels, class_names

train_paths, train_labels, class_names = parse_directory_to_paths(BASE_DIR, "train")
val_paths, val_labels, _ = parse_directory_to_paths(BASE_DIR, "val")
test_paths, test_labels, _ = parse_directory_to_paths(BASE_DIR, "test")

num_classes = len(class_names)
print(f"[INFO] num_classes = {num_classes}")
print(f"[INFO] class_names = {class_names}")
with open(CLASS_JSON, "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)

# -------------------------
# tf.data pipeline (pure TF ops)
# -------------------------

@tf.function
def _decode_and_preprocess(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.efficientnet_v2.preprocess_input(img)
    label = tf.cast(label, tf.int32)
    label = tf.one_hot(label, depth=num_classes)
    return img, label

# Augmentations: image-level (fast, vectorized)
@tf.function
def augment_image(img, label):
    # Basic randomized ops (GPU-friendly where possible)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, 0.08)
    img = tf.image.random_contrast(img, 0.9, 1.1)
    # Random rotation small angle
    angle = tf.random.uniform([], -0.05, 0.05)
    img = tfa.image.rotate(img, angles=angle) if _HAS_TFA else img
    return img, label


def make_ds_from_paths(paths, labels, batch_size=BATCH_SIZE, training=False, shuffle_buffer=SHUFFLE_BUFFER, snapshot_dir=None):
    # Convert to tensors; keep as strings
    paths = tf.constant(paths)
    labels = tf.constant(labels)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(shuffle_buffer, seed=SEED, reshuffle_each_iteration=True)
    # interleave pattern: read many files in parallel
    ds = ds.interleave(lambda p, l: tf.data.Dataset.from_tensors((p, l)), cycle_length=NUM_INTERLEAVE, num_parallel_calls=AUTOTUNE)
    ds = ds.map(_decode_and_preprocess, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(augment_image, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)

    # MixUp / CutMix applied per-batch (tf functions)
    if training:
        ds = ds.map(lambda x, y: tf.numpy_function(lambda a, b: apply_mixup_cutmix_np(a, b), [x, y], Tout=[tf.float32, tf.float32]), num_parallel_calls=AUTOTUNE)
        # Note: using numpy_function here to keep code simpler; for highest performance implement pure-TF mixup/cutmix

    if snapshot_dir and USE_SNAPSHOT:
        try:
            ds = ds.apply(tf.data.experimental.snapshot(snapshot_dir))
            print(f"[INFO] Using snapshot cache at {snapshot_dir}")
        except Exception as e:
            print("[WARN] snapshot not available:", e)

    ds = ds.prefetch(AUTOTUNE)
    return ds

# -------------------------
# MixUp & CutMix (numpy helper for simplicity + safety on macOS)
# -------------------------
from typing import Tuple

def _sample_beta(alpha=0.2):
    return np.random.beta(alpha, alpha) if alpha > 0 else 1.0


def apply_mixup_cutmix_np(images: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply MixUp or CutMix randomly per-batch. Returns mixed images and labels (one-hot).
    Implemented in numpy for reliability across TF versions on macOS.
    """
    if len(images.shape) == 3:
        images = np.expand_dims(images, 0)
    batch = images.shape[0]
    p = np.random.rand()
    # Choose operation probabilities: mixup 0.6, cutmix 0.3, none 0.1
    if p < 0.6:
        # MixUp
        lam = _sample_beta(0.2)
        idx = np.random.permutation(batch)
        x2 = images[idx]
        y2 = labels[idx]
        x = lam * images + (1 - lam) * x2
        y = lam * labels + (1 - lam) * y2
        return x.astype(np.float32), y.astype(np.float32)
    elif p < 0.9:
        # CutMix
        lam = _sample_beta(1.0)
        idx = np.random.permutation(batch)
        x2 = images[idx]
        y2 = labels[idx]
        H, W = images.shape[1], images.shape[2]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        x1 = np.clip(cx - cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        x2_copy = images.copy()
        x = images.copy()
        x[:, y1:y1+cut_h, x1:x1+cut_w, :] = x2_copy[:, y1:y1+cut_h, x1:x1+cut_w, :]
        # adjust lambda to real area
        lam_adj = 1. - (cut_h * cut_w) / (H * W)
        y = lam_adj * labels + (1. - lam_adj) * y2
        return x.astype(np.float32), y.astype(np.float32)
    else:
        return images.astype(np.float32), labels.astype(np.float32)

# -------------------------
# Class weights for imbalance
# -------------------------
all_labels = np.array(train_labels)
counts = np.bincount(all_labels, minlength=num_classes)
total = counts.sum()
class_weights = total / (num_classes * (counts + 1e-6))
class_weights_dict = {i: float(w) for i, w in enumerate(class_weights)}
print("[INFO] class_weights:", class_weights_dict)

# -------------------------
# Model creation
# -------------------------
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import EfficientNetV2M


def make_model(num_classes, input_shape=IMG_SIZE + (3,), dropout_rate=0.3):
    base = EfficientNetV2M(include_top=False, input_shape=input_shape, weights='imagenet')
    inp = layers.Input(shape=input_shape)
    x = base(inp, training=False)  # freeze BN in stage1
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
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
    base_model.trainable = False
    print("[INFO] New model created. Base frozen for stage1.")
else:
    print("[INFO] Using loaded model; freezing base for stage1.")
    for layer in model.layers[:-2]:
        layer.trainable = False

model.summary()

# -------------------------
# Callbacks
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
    def __init__(self, backup_dir=BACKUP_CHECKPOINT_DIR, epoch_interval=2, step_interval=None):
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

class PipelineWatchdogCallback(callbacks.Callback):
    def __init__(self, max_step_time=6.0, window=40):
        super().__init__()
        self.max_step_time = max_step_time
        self.window = window
        self.last_times = []
        self._start = None

    def on_train_batch_begin(self, batch, logs=None):
        self._start = time.time()

    def on_train_batch_end(self, batch, logs=None):
        t = time.time() - (self._start or time.time())
        self.last_times.append(t)
        if len(self.last_times) > self.window:
            self.last_times.pop(0)
        avg = sum(self.last_times) / len(self.last_times)
        if avg > self.max_step_time:
            fname = os.path.join(BACKUP_CHECKPOINT_DIR, "watchdog_save.keras")
            try:
                self.model.save(fname)
            except Exception as e:
                print("[WATCHDOG] failed to save model:", e)
            print(f"[WATCHDOG] avg step {avg:.2f}s > {self.max_step_time}s. Saved model -> {fname}. Stopping training.")
            self.model.stop_training = True

# callbacks lists
cb_stage1 = [
    callbacks.ModelCheckpoint(BEST_MODEL_PATH, monitor="val_top3", save_best_only=True, mode="max", save_weights_only=False),
    callbacks.EarlyStopping(monitor="val_loss", patience=ES_PATIENCE_STAGE1, min_delta=ES_MIN_DELTA, restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
    BatchTimingCallback(log_every_n_steps=500, out_file=os.path.join(KERAS_DIR, "batch_timing.log")),
    PeriodicBackupCallback(epoch_interval=2),
    PipelineWatchdogCallback(max_step_time=6.0, window=40),
]

cb_stage2 = [
    callbacks.ModelCheckpoint(BEST_FINETUNE_PATH, monitor="val_top3", save_best_only=True, mode="max", save_weights_only=False),
    callbacks.EarlyStopping(monitor="val_loss", patience=ES_PATIENCE_STAGE2, min_delta=ES_MIN_DELTA, restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
    BatchTimingCallback(log_every_n_steps=500, out_file=os.path.join(KERAS_DIR, "batch_timing_stage2.log")),
    PeriodicBackupCallback(epoch_interval=2),
    PipelineWatchdogCallback(max_step_time=6.0, window=40),
]

# -------------------------
# Compile helpers: optimizer + loss
# -------------------------

# choose AdamW if available
def get_optimizer(lr=LR_STAGE1, weight_decay=WEIGHT_DECAY):
    if _HAS_TFA:
        print("[INFO] Using tfa.optimizers.AdamW")
        return tfa.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
    else:
        # try TF experimental AdamW
        try:
            from tensorflow.keras.optimizers.experimental import AdamW as AdamWExp
            print("[INFO] Using tf.keras.optimizers.experimental.AdamW")
            return AdamWExp(learning_rate=lr, weight_decay=weight_decay)
        except Exception:
            print("[WARN] AdamW not available; falling back to Adam (no explicit weight decay)")
            return tf.keras.optimizers.Adam(learning_rate=lr)

loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
metrics = ["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3"), tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5")]

# -------------------------
# Build datasets
# -------------------------
if not os.path.exists(SNAPSHOT_DIR):
    try:
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    except Exception:
        pass

train_ds = make_ds_from_paths(train_paths, train_labels, batch_size=BATCH_SIZE, training=True, shuffle_buffer=SHUFFLE_BUFFER, snapshot_dir=SNAPSHOT_DIR if USE_SNAPSHOT else None)
val_ds = make_ds_from_paths(val_paths, val_labels, batch_size=BATCH_SIZE, training=False, shuffle_buffer=SHUFFLE_BUFFER)
test_ds = make_ds_from_paths(test_paths, test_labels, batch_size=BATCH_SIZE, training=False, shuffle_buffer=SHUFFLE_BUFFER)

# small benchmark util
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

if "--bench" in sys.argv:
    benchmark_dataset(train_ds, num_batches=50)
    sys.exit(0)

# -------------------------
# Stage1: compile and train head-only
# -------------------------
optimizer_stage1 = get_optimizer(LR_STAGE1, WEIGHT_DECAY)
model.compile(optimizer=optimizer_stage1, loss=loss_fn, metrics=metrics)

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
# Stage2: unfreeze base and fine-tune
# -------------------------
print("[INFO] Starting Stage2: unfreeze base and fine-tune")
# Unfreeze entire base and enable BN updates (trainable True)
try:
    base_model.trainable = True
except Exception:
    # fallback: set all layers trainable
    for layer in model.layers:
        layer.trainable = True

optimizer_stage2 = get_optimizer(LR_STAGE2, WEIGHT_DECAY)
model.compile(optimizer=optimizer_stage2, loss=loss_fn, metrics=metrics)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE2,
    class_weight=class_weights_dict,
    callbacks=cb_stage2,
    verbose=1
)

# -------------------------
# Final evaluate & save
# -------------------------
print("[INFO] Evaluating on test set...")
test_metrics = model.evaluate(test_ds)
print("[INFO] Test metrics:", test_metrics)

final_path = BEST_FINETUNE_PATH if os.path.exists(BEST_FINETUNE_PATH) else BEST_MODEL_PATH
model.save(final_path)
print(f"[INFO] Saved final model to {final_path}")

print("\n[INFO] To prevent macOS sleep during long training, run in another terminal:")
print("    caffeinate -dims &")
print("Alternatively: sudo pmset -a sleep 0 displaysleep 0")
