# train_prod_m1_aug_resume.py
"""
Production-ready training script for macOS M1
Features:
 - EfficientNetV2-S backbone (fast / accurate on M1)
 - Model-level augmentation (RandomFlip/Rotation/Zoom/Contrast)
 - Mixed precision (optional)
 - Safe tf.data pipeline (batch only once)
 - Class-weight (auto compute) + optional rare-class oversampling hook
 - Watchdog to detect dead-stall and auto-backup+stop
 - Auto-resume from latest checkpoint or backup
 - Cosine LR schedule, gradient clipping, AdamW(when available) fallback to Adam
 - Periodic backups + batch timing logs
 - SIGINT handler saves resume checkpoint
Usage:
  python train_prod_m1_aug_resume.py
"""

import os
import sys
import time
import json
import random
import signal
from glob import glob
from pathlib import Path
from collections import defaultdict

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

# -------------------------
# CONFIG
# -------------------------
DATA_ROOT = "./dataset_25000"       # root with train/val/test subfolders
IMG_SIZE = (224, 224)
BATCH_SIZE = 16                    # reduce if OOM
SEED = 42

NUM_CLASSES = None                 # auto-detect
USE_MIXED_PRECISION = True         # recommended on M1 with tensorflow-metal
USE_ADAMW = True                   # will try tfa.AdamW or tf.experimental.AdamW
WEIGHT_DECAY = 1e-5
CLIPNORM = 1.0

SHUFFLE_BUFFER = 2048
PREFETCH = tf.data.AUTOTUNE

EPOCHS_STAGE1 = 8
EPOCHS_STAGE2 = 20

ES_PATIENCE_STAGE1 = 4
ES_PATIENCE_STAGE2 = 5
ES_MIN_DELTA = 1e-4

BACKUP_DIR = "./keras/backups"
os.makedirs(BACKUP_DIR, exist_ok=True)
BEST_MODEL = os.path.join(BACKUP_DIR, "best_model.keras")
BEST_FINETUNE = os.path.join(BACKUP_DIR, "best_finetuned_model.keras")
WATCHDOG_SAVE = os.path.join(BACKUP_DIR, "watchdog_save.keras")
RESUME_PATH = os.path.join(BACKUP_DIR, "resume.keras")
CLASS_JSON = "./keras/class_indices.json"
BATCH_TIMING_LOG = "./keras/batch_timing.log"

# Watchdog config
WATCHDOG_WINDOW = 8        # number of recent batch times to average
WATCHDOG_THRESHOLD = 8.0   # seconds per step threshold -> triggers save+stop

# Resume attempts
MAX_RECOVERIES = 6
RECOVERY_WAIT = 6.0

# Oversampling toggle (optional) - keeps pipeline simple; default False
OVERSAMPLE_RARE = False
RARE_THRESHOLD = 300

# Misc
RANDOM_SEED = 42
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# -------------------------
# Optional imports
# -------------------------
try:
    import tensorflow_addons as tfa
    _HAS_TFA = True
except Exception:
    tfa = None
    _HAS_TFA = False

# Mixed precision setup
if USE_MIXED_PRECISION:
    try:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")
        print("[INFO] Mixed precision enabled: mixed_float16")
    except Exception as e:
        print("[WARN] Mixed precision unavailable:", e)
        USE_MIXED_PRECISION = False

# -------------------------
# Utility: find dataset splits
# -------------------------
def parse_dataset_split(split_dir):
    classes = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
    paths = []
    labels = []
    for idx, cls in enumerate(classes):
        cls_dir = os.path.join(split_dir, cls)
        for root, _, files in os.walk(cls_dir):
            for f in files:
                if f.lower().endswith((".jpg",".jpeg",".png")):
                    paths.append(os.path.join(root, f))
                    labels.append(idx)
    return paths, labels, classes

train_p, train_l, class_names = parse_dataset_split(os.path.join(DATA_ROOT, "train"))
val_p, val_l, _ = parse_dataset_split(os.path.join(DATA_ROOT, "val"))
test_p, test_l, _ = parse_dataset_split(os.path.join(DATA_ROOT, "test"))

NUM_CLASSES = len(class_names)
print(f"[INFO] classes ({NUM_CLASSES}): {class_names}")
with open(CLASS_JSON, "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)

# -------------------------
# Compute class weights
# -------------------------
counts = np.bincount(np.array(train_l), minlength=NUM_CLASSES)
total = counts.sum()
# smoother class weights (clip to avoid extreme)
raw = total / (NUM_CLASSES * (counts + 1e-6))
class_weights = {i: float(np.clip(raw[i], 0.5, 8.0)) for i in range(NUM_CLASSES)}
print("[INFO] class_weights (clipped):", class_weights)

# -------------------------
# Oversampling helper (optional)
# -------------------------
def make_oversampled_generator(paths, labels, rare_threshold=RARE_THRESHOLD):
    # yields (path, label)
    per_class = defaultdict(list)
    for p,l in zip(paths, labels):
        per_class[l].append(p)
    rare = {c for c, pts in per_class.items() if len(pts) < rare_threshold}
    # Build new list with mild oversampling for rare classes
    out = []
    for c, pts in per_class.items():
        rep = 1
        if c in rare:
            rep = max(2, int(rare_threshold / max(1, len(pts))))
        for _ in range(rep):
            out.extend([(p,c) for p in pts])
    random.shuffle(out)
    for p,c in out:
        yield p, c

# -------------------------
# tf.data pipeline (stable)
# -------------------------
AUTOTUNE = tf.data.AUTOTUNE
def build_ds_from_paths(paths, labels, batch_size=BATCH_SIZE, training=False, oversample=False):
    if oversample and training:
        gen = lambda: make_oversampled_generator(paths, labels)
        output_signature = (tf.TensorSpec(shape=(), dtype=tf.string), tf.TensorSpec(shape=(), dtype=tf.int32))
        ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    else:
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(SHUFFLE_BUFFER, seed=SEED, reshuffle_each_iteration=True)

    def _parse(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32)
        # Do NOT apply model preprocess here; we will call preprocess_input in model pipeline for consistency
        label = tf.cast(label, tf.int32)
        return img, label

    ds = ds.map(_parse, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=training)  # ensure fixed batch for training
    ds = ds.prefetch(PREFETCH)
    return ds

train_ds = build_ds_from_paths(train_p, train_l, training=True, oversample=OVERSAMPLE_RARE)
val_ds   = build_ds_from_paths(val_p, val_l, training=False)
test_ds  = build_ds_from_paths(test_p, test_l, training=False)

# Quick sanity check shapes
batch_example = next(iter(train_ds.take(1)))
print("[INFO] example batch shapes:", batch_example[0].shape, batch_example[1].shape)

# -------------------------
# Augmentation & Model builder
# -------------------------
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2S, preprocess_input

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.06),
    layers.RandomZoom(0.08),
    layers.RandomContrast(0.08),
], name="data_augmentation")

def make_model(num_classes=NUM_CLASSES, input_shape=IMG_SIZE+(3,), dropout_rate=0.3, label_smoothing=0.05):
    inputs = layers.Input(shape=input_shape, name="input_image")
    x = data_augmentation(inputs)               # augmentation ON during training
    x = preprocess_input(x)                     # efficientnet preprocess (float32)
    base = EfficientNetV2S(include_top=False, input_shape=input_shape, weights="imagenet")
    x = base(x, training=False)                 # keep BN in inference mode during stage1
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax",
                           kernel_regularizer=keras.regularizers.l2(1e-6))(x)
    model = keras.Model(inputs, outputs, name="EffNetV2S_aug")
    return model, base

model, base_model = make_model()

# -------------------------
# Optimizer, loss, metrics, scheduler
# -------------------------
def get_optimizer(lr=1e-4, weight_decay=WEIGHT_DECAY, clipnorm=CLIPNORM):
    if USE_ADAMW and _HAS_TFA:
        print("[INFO] Using tfa.optimizers.AdamW")
        return tfa.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay, clipnorm=clipnorm)
    # try TF experimental AdamW
    try:
        from tensorflow.keras.optimizers.experimental import AdamW as AdamWExp
        print("[INFO] Using tf.keras.optimizers.experimental.AdamW")
        return AdamWExp(learning_rate=lr, weight_decay=weight_decay, clipnorm=clipnorm)
    except Exception:
        print("[WARN] AdamW unavailable; falling back to Adam")
        return keras.optimizers.Adam(learning_rate=lr, clipnorm=clipnorm)

# Cosine LR schedule for stage2
def cosine_scheduler(initial_lr, steps_per_epoch, epochs, alpha=0.0):
    total_steps = steps_per_epoch * epochs
    return tf.keras.optimizers.schedules.CosineDecay(initial_lr, decay_steps=total_steps, alpha=alpha)

# loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(label_smoothing=0.0)  # using sparse labels
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

metrics = [
    keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
    keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3"),
    keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5"),
]

# -------------------------
# Callbacks: BatchTiming, PeriodicBackup, Watchdog, Checkpoint, EarlyStopping
# -------------------------
class BatchTiming(callbacks.Callback):
    def __init__(self, log_every=500, out_file=BATCH_TIMING_LOG):
        super().__init__()
        self.log_every = log_every
        self.out_file = out_file
        self.step_times = []
        self._s = None
    def on_train_batch_begin(self, batch, logs=None):
        self._s = time.time()
    def on_train_batch_end(self, batch, logs=None):
        t = time.time() - (self._s or time.time())
        self.step_times.append(t)
        if len(self.step_times) % self.log_every == 0:
            arr = np.array(self.step_times[-self.log_every:])
            msg = f"[BATCHTIMING] last {self.log_every}: mean={arr.mean():.3f}s median={np.median(arr):.3f}s p95={np.percentile(arr,95):.3f}s"
            print(msg)
            try:
                with open(self.out_file, "a") as f:
                    f.write(msg + "\n")
            except Exception:
                pass

class PeriodicBackup(callbacks.Callback):
    def __init__(self, backup_dir=BACKUP_DIR, epoch_interval=2):
        super().__init__()
        self.backup_dir = backup_dir
        self.epoch_interval = epoch_interval
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.epoch_interval == 0:
            fname = os.path.join(self.backup_dir, f"backup_epoch{epoch+1}.keras")
            try:
                self.model.save(fname)
                print(f"[BACKUP] saved {fname}")
            except Exception as e:
                print("[WARN] backup failed:", e)

class PipelineWatchdog(callbacks.Callback):
    def __init__(self, threshold=WATCHDOG_THRESHOLD, window=WATCHDOG_WINDOW, save_path=WATCHDOG_SAVE):
        super().__init__()
        self.threshold = threshold
        self.window = window
        self.times = []
        self._s = None
        self.save_path = save_path
    def on_train_batch_begin(self, batch, logs=None):
        self._s = time.time()
    def on_train_batch_end(self, batch, logs=None):
        dt = time.time() - (self._s or time.time())
        self.times.append(dt)
        if len(self.times) > self.window:
            self.times.pop(0)
        if len(self.times) >= max(4, self.window//2):
            avg = sum(self.times)/len(self.times)
            if avg > self.threshold:
                print(f"[WATCHDOG] avg step {avg:.1f}s > {self.threshold}s. Saving {self.save_path} and stopping training.")
                try:
                    self.model.save(self.save_path)
                except Exception as e:
                    print("[WARN] watchdog save failed:", e)
                self.model.stop_training = True

# standard callbacks generators
def get_stage1_callbacks():
    return [
        callbacks.ModelCheckpoint(BEST_MODEL, monitor="val_top3", mode="max", save_best_only=True),
        callbacks.EarlyStopping(monitor="val_loss", patience=ES_PATIENCE_STAGE1, restore_best_weights=True, min_delta=ES_MIN_DELTA, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
        BatchTiming(log_every=500),
        PeriodicBackup(epoch_interval=2),
        PipelineWatchdog()
    ]

def get_stage2_callbacks():
    return [
        callbacks.ModelCheckpoint(BEST_FINETUNE, monitor="val_top3", mode="max", save_best_only=True),
        callbacks.EarlyStopping(monitor="val_loss", patience=ES_PATIENCE_STAGE2, restore_best_weights=True, min_delta=ES_MIN_DELTA, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
        BatchTiming(log_every=500),
        PeriodicBackup(epoch_interval=2),
        PipelineWatchdog()
    ]

# -------------------------
# Checkpoint helpers
# -------------------------
def find_latest_checkpoint():
    candidates = []
    for p in (BEST_FINETUNE, BEST_MODEL, WATCHDOG_SAVE, RESUME_PATH):
        if os.path.exists(p):
            candidates.append((p, os.path.getmtime(p)))
    # also search for backups
    for f in glob(os.path.join(BACKUP_DIR, "backup_*.keras")):
        try:
            candidates.append((f, os.path.getmtime(f)))
        except Exception:
            pass
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]

def safe_load_model(path):
    try:
        m = keras.models.load_model(path)
        print("[INFO] Loaded model:", path)
        return m
    except Exception as e:
        print("[WARN] Failed to load model fully (will recreate and try to load weights partially):", e)
        try:
            # try load without compile and then load weights if shapes match
            m = make_model()[0]  # recreate model
            m.load_weights(path, by_name=True, skip_mismatch=True)
            print("[INFO] Loaded weights by name (skipped mismatches).")
            return m
        except Exception as e2:
            print("[ERROR] Partial weight load also failed:", e2)
            return None

# -------------------------
# SIGINT graceful save
# -------------------------
def sigint_handler(signum, frame):
    print("[INFO] SIGINT caught — saving resume checkpoint")
    try:
        model.save(RESUME_PATH)
        print("[INFO] Saved resume ->", RESUME_PATH)
    except Exception as e:
        print("[WARN] Failed to save resume on SIGINT:", e)
    sys.exit(0)

signal.signal(signal.SIGINT, sigint_handler)

# -------------------------
# Training flow with auto-recovery
# -------------------------
def train():
    recoveries = 0
    while recoveries <= MAX_RECOVERIES:
        # try to load latest checkpoint if exists (resume)
        ckpt = find_latest_checkpoint()
        if ckpt:
            print("[INFO] Attempting to resume from:", ckpt)
            loaded = safe_load_model(ckpt)
            if loaded is not None:
                global model, base_model
                model = loaded
                # try to find base_model if missing via attribute
                try:
                    base_model = [l for l in model.layers if 'efficientnetv2' in l.name][0]
                except Exception:
                    pass
        else:
            print("[INFO] No checkpoint found; building new model")
            model, base_model = make_model()

        # Stage1: head-only
        print("[INFO] Starting Stage1: head-only")
        try:
            # freeze base
            try:
                base_model.trainable = False
            except Exception:
                for layer in model.layers:
                    layer.trainable = False

            # compile stage1 optimizer
            steps_per_epoch = max(1, int(np.ceil(len(train_p)/BATCH_SIZE)))
            opt1 = get_optimizer(lr=1e-4)
            model.compile(optimizer=opt1, loss=loss_fn, metrics=metrics)

            history1 = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=EPOCHS_STAGE1,
                class_weight=class_weights,
                callbacks=get_stage1_callbacks(),
                verbose=1
            )
        except KeyboardInterrupt:
            print("[INFO] KeyboardInterrupt during Stage1; exiting")
            return
        except Exception as e:
            print("[ERROR] Exception during Stage1:", e)
            recoveries += 1
            print(f"[INFO] Recovery attempt {recoveries}/{MAX_RECOVERIES} after Stage1 error")
            time.sleep(RECOVERY_WAIT)
            continue

        # Stage2: unfreeze (partial fine-tune)
        print("[INFO] Starting Stage2: fine-tune (unfreeze last blocks)")
        try:
            # Partial unfreeze: unfreeze last N layers of base to save time/memory
            try:
                # Unfreeze all (safer)
                base_model.trainable = True
            except Exception:
                for layer in model.layers:
                    layer.trainable = True

            steps_per_epoch = max(1, int(np.ceil(len(train_p)/BATCH_SIZE)))
            lr2 = 5e-5
            schedule = cosine_scheduler(lr2, steps_per_epoch, EPOCHS_STAGE2)
            opt2 = get_optimizer(lr=lr2)
            model.compile(optimizer=opt2, loss=loss_fn, metrics=metrics)

            history2 = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=EPOCHS_STAGE2,
                class_weight=class_weights,
                callbacks=get_stage2_callbacks(),
                verbose=1
            )
        except KeyboardInterrupt:
            print("[INFO] KeyboardInterrupt during Stage2; exiting")
            return
        except Exception as e:
            print("[ERROR] Exception during Stage2:", e)
            recoveries += 1
            print(f"[INFO] Recovery attempt {recoveries}/{MAX_RECOVERIES} after Stage2 error")
            time.sleep(RECOVERY_WAIT)
            continue

        # Evaluate
        try:
            print("[INFO] Evaluating on test set...")
            test_metrics = model.evaluate(test_ds)
            print("[INFO] Test metrics:", test_metrics)
        except Exception as e:
            print("[WARN] test evaluation failed:", e)

        # Save final model (prefer finetune-best if exists)
        final_path = BEST_FINETUNE if os.path.exists(BEST_FINETUNE) else BEST_MODEL
        try:
            model.save(final_path)
            print("[INFO] Saved final model to", final_path)
        except Exception as e:
            print("[WARN] final save failed:", e)

        return

    print("[ERROR] Max recoveries exceeded; aborting.")

# -------------------------
# Entrypoint
# -------------------------
if __name__ == "__main__":
    train()
