# trainEffNetV2M_fixed_resume_stall_v1.py
"""
Fixed production-ready training script (M1 friendly)
- Resolves Keras class_weight / y.shape.rank NoneType error by providing sample_weight from dataset
- Uses sparse labels (int) and sparse loss/metrics
- Keep snapshot, watchdog, backup, dataset optimizations
- MixUp/CutMix disabled by default to avoid label/weight complexity (enable with caution)
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

# optional tfa
try:
    import tensorflow_addons as tfa
    _HAS_TFA = True
except Exception:
    tfa = None
    _HAS_TFA = False

# -------------------------
# CONFIG
# -------------------------
BASE_DIR = "dataset_25000"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

USE_MIXED_PRECISION = False   # enable only if tensorflow-macos + metal installed and tested
USE_SNAPSHOT = True
SNAPSHOT_DIR = "./keras/snapshot_train"

SHUFFLE_BUFFER = 50000
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
TRAIN_STATE = os.path.join(KERAS_DIR, "train_state.json")

os.makedirs(KERAS_DIR, exist_ok=True)
os.makedirs(BACKUP_CHECKPOINT_DIR, exist_ok=True)

# MixUp toggle (disabled by default since we use sample_weight integer labels)
USE_MIXUP = False

# -------------------------
# TF threading & precision for M1
# -------------------------
try:
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(4)
except Exception:
    pass

if USE_MIXED_PRECISION:
    try:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")
        print("[INFO] Mixed precision enabled")
    except Exception as e:
        print("[WARN] Mixed precision unavailable:", e)

# -------------------------
# Utilities
# -------------------------
def clean_dataset_dirs(base_dir):
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
                try:
                    shutil.rmtree(cls_path)
                except Exception:
                    pass

clean_dataset_dirs(BASE_DIR)

# -------------------------
# Parse dataset to paths
# -------------------------
def parse_directory_to_paths(base_dir, split):
    split_dir = os.path.join(base_dir, split)
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"Dataset split not found: {split_dir}")
    class_names = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
    paths, labels = [], []
    for idx, cls in enumerate(class_names):
        cls_dir = os.path.join(split_dir, cls)
        for root, _, files in os.walk(cls_dir):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
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
# Compute class weights (clipped)
# -------------------------
all_labels = np.array(train_labels)
counts = np.bincount(all_labels, minlength=num_classes)
total = counts.sum()
raw_class_weights = total / (num_classes * (counts + 1e-6))
# Clip extreme weights to avoid instability
class_weights = np.clip(raw_class_weights, 0.5, 8.0).astype(np.float32)
class_weights_dict = {i: float(class_weights[i]) for i in range(num_classes)}
print("[INFO] class_weights (clipped):", class_weights_dict)
# create TF vector for dataset gather
class_weight_vector = tf.constant(class_weights, dtype=tf.float32)

# -------------------------
# Dataset pipeline: yield (x, y_int, sample_weight)
# -------------------------
def _decode_and_preprocess_to_int(path, label):
    """
    Returns: img(float32), label(int scalar), sample_weight(float scalar)
    """
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.efficientnet_v2.preprocess_input(img)
    label = tf.cast(label, tf.int32)
    # ensure shapes so TF/Keras won't see None rank
    img.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])
    label = tf.reshape(label, [])  # scalar
    # sample weight from class weight vector
    sw = tf.gather(class_weight_vector, label)
    sw = tf.reshape(sw, [])  # scalar
    return img, label, sw

@tf.function
def augment_image_int(img, label, sw):
    # augment only images, keep label & sw unchanged
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.06)
    img = tf.image.random_contrast(img, 0.95, 1.05)
    # small rotation if tfa available
    if _HAS_TFA:
        angle = tf.random.uniform([], -0.03, 0.03)
        img = tfa.image.rotate(img, angles=angle)
    return img, label, sw

def make_ds_from_paths_with_weights(paths, labels, batch_size=BATCH_SIZE, training=False, shuffle_buffer=SHUFFLE_BUFFER, snapshot_dir=None):
    paths_t = tf.constant(paths)
    labels_t = tf.constant(labels)
    ds = tf.data.Dataset.from_tensor_slices((paths_t, labels_t))
    if training:
        ds = ds.shuffle(shuffle_buffer, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.interleave(lambda p, l: tf.data.Dataset.from_tensors((p, l)), cycle_length=NUM_INTERLEAVE, num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda p, l: tf.py_function(func=_decode_and_preprocess_to_int, inp=[p, l], Tout=[tf.float32, tf.int32, tf.float32]),
                num_parallel_calls=AUTOTUNE)
    # py_function loss: must set shapes
    def set_shapes(img, lbl, sw):
        img.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])
        lbl.set_shape([])
        sw.set_shape([])
        return img, lbl, sw
    ds = ds.map(set_shapes, num_parallel_calls=AUTOTUNE)

    if training:
        ds = ds.map(augment_image_int, num_parallel_calls=AUTOTUNE)

    if snapshot_dir and USE_SNAPSHOT:
        try:
            # tf.data.Dataset.snapshot is recommended in newer TF; using experimental for compatibility
            ds = ds.apply(tf.data.experimental.snapshot(snapshot_dir))
            print(f"[INFO] Using snapshot cache at {snapshot_dir}")
        except Exception as e:
            print("[WARN] snapshot not available:", e)

    ds = ds.batch(batch_size, drop_remainder=False)

    # Optionally apply MixUp/CutMix here — careful: mixup would need to produce soft labels and mixed sample weights.
    # Because we are using integer labels + sample_weight, mixing labels would require changing loss/fit handling.
    # For now we keep MixUp disabled by default.
    if USE_MIXUP:
        def mixup_np(x, y, w):
            # x: np array images, y: np int labels -> convert to one-hot and mix, return mixed images and soft labels and mixed weights
            x2, y2, w2 = apply_mixup_cutmix_np_with_weights_np(x, y, w)
            return x2, y2, w2
        # Implementing mixup here would require changing model loss to categorical and dataset yields accordingly.
        # Skipping to avoid class_weight/sparse mismatch.

    ds = ds.prefetch(AUTOTUNE)
    return ds

# -------------------------
# MixUp/CutMix numpy helpers (kept for reference)
# -------------------------
from typing import Tuple

def _sample_beta(alpha=0.2):
    return np.random.beta(alpha, alpha) if alpha > 0 else 1.0

def apply_mixup_cutmix_np_with_weights_np(images: np.ndarray, labels_int: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Helper for an alternate pipeline that uses one-hot soft labels and mixed weights.
    Not used by default in this script (kept for later use).
    """
    batch = images.shape[0]
    # convert labels_int -> one-hot
    labels = np.eye(num_classes, dtype=np.float32)[labels_int]
    p = np.random.rand()
    if p < 0.6:
        lam = _sample_beta(0.2)
        idx = np.random.permutation(batch)
        x2 = images[idx]; y2 = labels[idx]; w2 = weights[idx]
        x = lam * images + (1 - lam) * x2
        y = lam * labels + (1 - lam) * y2
        w = lam * weights + (1 - lam) * w2
        return x.astype(np.float32), y.astype(np.float32), w.astype(np.float32)
    elif p < 0.9:
        lam = _sample_beta(1.0)
        idx = np.random.permutation(batch)
        x2 = images[idx]; y2 = labels[idx]; w2 = weights[idx]
        H, W = images.shape[1], images.shape[2]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat); cut_h = int(H * cut_rat)
        cx = np.random.randint(W); cy = np.random.randint(H)
        x1 = np.clip(cx - cut_w // 2, 0, W); y1 = np.clip(cy - cut_h // 2, 0, H)
        x = images.copy()
        x[:, y1:y1+cut_h, x1:x1+cut_w, :] = x2[:, y1:y1+cut_h, x1:x1+cut_w, :]
        lam_adj = 1. - (cut_h * cut_w) / (H * W)
        y = lam_adj * labels + (1. - lam_adj) * y2
        w = lam_adj * weights + (1. - lam_adj) * w2
        return x.astype(np.float32), y.astype(np.float32), w.astype(np.float32)
    else:
        return images.astype(np.float32), labels.astype(np.float32), weights.astype(np.float32)

# -------------------------
# Model creation
# -------------------------
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import EfficientNetV2M

def make_model(num_classes, input_shape=IMG_SIZE + (3,), dropout_rate=0.3):
    base = EfficientNetV2M(include_top=False, input_shape=input_shape, weights="imagenet")
    inp = layers.Input(shape=input_shape)
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    out = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    model = models.Model(inputs=inp, outputs=out)
    return model, base

# -------------------------
# Build datasets (train yields x,y_int, sample_weight)
# -------------------------
if not os.path.exists(SNAPSHOT_DIR):
    try:
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    except Exception:
        pass

train_ds = make_ds_from_paths_with_weights(train_paths, train_labels, batch_size=BATCH_SIZE, training=True,
                                          shuffle_buffer=SHUFFLE_BUFFER, snapshot_dir=SNAPSHOT_DIR if USE_SNAPSHOT else None)

# validation/test: produce (x, y) tuples (no sample weight required for val)
def build_eval_ds(paths, labels, batch_size=BATCH_SIZE):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda p, l: (tf.io.read_file(p), tf.cast(l, tf.int32)), num_parallel_calls=AUTOTUNE)
    # decode & preprocess
    def decode_and_prep_bytes(p_bytes, lbl):
        img = tf.image.decode_jpeg(p_bytes, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32)
        img = tf.keras.applications.efficientnet_v2.preprocess_input(img)
        img.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])
        lbl = tf.reshape(tf.cast(lbl, tf.int32), [])
        return img, lbl
    ds = ds.map(lambda p, l: tf.py_function(func=decode_and_prep_bytes, inp=[p, l], Tout=[tf.float32, tf.int32]), num_parallel_calls=AUTOTUNE)
    def set_shapes_eval(img, lbl):
        img.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])
        lbl.set_shape([])
        return img, lbl
    ds = ds.map(set_shapes_eval, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds

val_ds = build_eval_ds(val_paths, val_labels, batch_size=BATCH_SIZE)
test_ds = build_eval_ds(test_paths, test_labels, batch_size=BATCH_SIZE)

# -------------------------
# Callbacks: timing, backup, watchdog
# -------------------------
class BatchTimingCallback(callbacks.Callback):
    def __init__(self, log_every_n_steps=200, out_file=None):
        super().__init__()
        self.log_every = log_every_n_steps
        self.out_file = out_file
        self.step_times = []
        self._start = None
    def on_train_batch_begin(self, batch, logs=None):
        self._start = time.time()
    def on_train_batch_end(self, batch, logs=None):
        t = time.time() - (self._start or time.time())
        self.step_times.append(t)
        if len(self.step_times) % self.log_every == 0:
            arr = np.array(self.step_times[-self.log_every:])
            msg = f"[BATCHTIMING] last {self.log_every} steps: mean={arr.mean():.3f}s median={np.median(arr):.3f}s p95={np.percentile(arr,95):.3f}s"
            print(msg)
            if self.out_file:
                with open(self.out_file, "a") as f:
                    f.write(msg + "\n")

class PeriodicBackupCallback(callbacks.Callback):
    def __init__(self, backup_dir=BACKUP_CHECKPOINT_DIR, epoch_interval=2):
        super().__init__()
        self.backup_dir = backup_dir
        self.epoch_interval = epoch_interval
    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1) % self.epoch_interval == 0:
            fname = os.path.join(self.backup_dir, f"backup_epoch{epoch+1}.keras")
            try:
                self.model.save(fname)
                print("[BACKUP] saved", fname)
            except Exception as e:
                print("[WARN] backup failed:", e)
            st = {"last_epoch": int(epoch+1), "timestamp": time.time()}
            try:
                with open(TRAIN_STATE, "w") as f:
                    json.dump(st, f)
            except Exception:
                pass

class PipelineWatchdogCallback(callbacks.Callback):
    def __init__(self, max_step_time=6.0, window=40):
        super().__init__()
        self.max_step_time = max_step_time
        self.window = window
        self.times = []
        self._start = None
    def on_train_batch_begin(self, batch, logs=None):
        self._start = time.time()
    def on_train_batch_end(self, batch, logs=None):
        t = time.time() - (self._start or time.time())
        self.times.append(t)
        if len(self.times) > self.window:
            self.times.pop(0)
        avg = sum(self.times) / len(self.times)
        if avg > self.max_step_time:
            fname = os.path.join(BACKUP_CHECKPOINT_DIR, "watchdog_save.keras")
            try:
                self.model.save(fname)
            except Exception:
                pass
            print(f"[WATCHDOG] avg step {avg:.2f}s > {self.max_step_time}s. Saved {fname} and stopping training")
            try:
                st = {}
                if os.path.exists(TRAIN_STATE):
                    st = json.load(open(TRAIN_STATE))
                st.update({"watchdog_triggered": True, "watchdog_avg": avg, "timestamp": time.time()})
                with open(TRAIN_STATE, "w") as f:
                    json.dump(st, f)
            except Exception:
                pass
            self.model.stop_training = True

# callback lists
cb_stage1 = [
    callbacks.ModelCheckpoint(BEST_MODEL_PATH, monitor="val_sparse_top3", save_best_only=True, mode="max", save_weights_only=False),
    callbacks.EarlyStopping(monitor="val_loss", patience=ES_PATIENCE_STAGE1, min_delta=ES_MIN_DELTA, restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
    BatchTimingCallback(log_every_n_steps=500, out_file=os.path.join(KERAS_DIR, "batch_timing.log")),
    PeriodicBackupCallback(epoch_interval=2),
    PipelineWatchdogCallback(max_step_time=6.0, window=40),
]

cb_stage2 = [
    callbacks.ModelCheckpoint(BEST_FINETUNE_PATH, monitor="val_sparse_top3", save_best_only=True, mode="max", save_weights_only=False),
    callbacks.EarlyStopping(monitor="val_loss", patience=ES_PATIENCE_STAGE2, min_delta=ES_MIN_DELTA, restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
    BatchTimingCallback(log_every_n_steps=500, out_file=os.path.join(KERAS_DIR, "batch_timing_stage2.log")),
    PeriodicBackupCallback(epoch_interval=2),
    PipelineWatchdogCallback(max_step_time=6.0, window=40),
]

# -------------------------
# Optimizer & loss (sparse)
# -------------------------
def get_optimizer(lr=LR_STAGE1, weight_decay=WEIGHT_DECAY):
    if _HAS_TFA:
        return tfa.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
    else:
        try:
            from tensorflow.keras.optimizers.experimental import AdamW as AdamWExp
            return AdamWExp(learning_rate=lr, weight_decay=weight_decay)
        except Exception:
            return tf.keras.optimizers.Adam(learning_rate=lr)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
metrics = [
    tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
    tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="sparse_top3"),
    tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="sparse_top5")
]

# -------------------------
# Build or load model
# -------------------------
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
    base_model.trainable = False
    print("[INFO] New model created. Base frozen for stage1.")
else:
    print("[INFO] Using loaded model; freezing base for stage1.")
    for layer in model.layers[:-2]:
        layer.trainable = False

model.summary()

# -------------------------
# Stage1 training (head-only)
# -------------------------
optimizer_stage1 = get_optimizer(LR_STAGE1, WEIGHT_DECAY)
model.compile(optimizer=optimizer_stage1, loss=loss_fn, metrics=metrics)

print("[INFO] Starting Stage1 training (head-only)")
# train_ds yields (x, y_int, sample_weight)
# Keras accepts dataset that yields (x, y, sample_weight)
history1 = model.fit(
    train_ds,
    validation_data=val_ds,   # val_ds yields (x, y)
    epochs=EPOCHS_STAGE1,
    callbacks=cb_stage1,
    verbose=1
)

# -------------------------
# Stage2: unfreeze & fine-tune
# -------------------------
print("[INFO] Starting Stage2: unfreeze base and fine-tune")
try:
    base_model.trainable = True
except Exception:
    for layer in model.layers:
        layer.trainable = True

optimizer_stage2 = get_optimizer(LR_STAGE2, WEIGHT_DECAY)
model.compile(optimizer=optimizer_stage2, loss=loss_fn, metrics=metrics)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE2,
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
