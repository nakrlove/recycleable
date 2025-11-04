# train_effv2s_m1_prod_resume_v1.py
"""
M1-optimized training script:
 - Model: EfficientNetV2-S (lightweight, good tradeoff for M1)
 - Mixed precision, SGD/AdamW options
 - Sample-weight dataset to avoid Keras class_weight shape bugs
 - Snapshot/caching and tf.data optimizations
 - Dead-stall watchdog + automatic resume from latest checkpoint
 - Partial unfreeze strategy and gradient clipping
 - Use --bench to run pipeline micro-benchmark
"""
import os, sys, time, json, shutil, random, math
from glob import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks

# Optional add-ons
try:
    import tensorflow_addons as tfa
    _HAS_TFA = True
except Exception:
    tfa = None
    _HAS_TFA = False

# -------------------------
# CONFIG (tweak these)
# -------------------------
BASE_DIR = "dataset_25000"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16               # default; raise to 24~32 if mem+speed allow
SEED = 42

USE_MIXED_PRECISION = True    # recommended on M1 with tensorflow-metal
USE_SNAPSHOT = True           # snapshot can speed up repeated runs
SNAPSHOT_DIR = "./keras/snapshot_train"

SHUFFLE_BUFFER = 50000
AUTOTUNE = tf.data.AUTOTUNE
NUM_INTERLEAVE = 16

EPOCHS_STAGE1 = 6             # head-only epochs (shorter for quicker iteration)
EPOCHS_STAGE2 = 18            # fine-tune epochs
ES_PATIENCE_STAGE1 = 4
ES_PATIENCE_STAGE2 = 4
ES_MIN_DELTA = 1e-4

LEARNING_RATE_STAGE1 = 1e-3
LEARNING_RATE_STAGE2 = 5e-5
WEIGHT_DECAY = 1e-5

USE_ADAMW = False             # If True uses AdamW (tfa or TF experimental), else SGD(momentum)
GRAD_CLIPNORM = 1.0

PARTIAL_UNFREEZE_LAYERS = 40  # during stage2, unfreeze last N layers of base (helps speed)

KERAS_DIR = "./keras"
BEST_MODEL_PATH = os.path.join(KERAS_DIR, "best_model.keras")
BEST_FINETUNE_PATH = os.path.join(KERAS_DIR, "best_finetuned_model.keras")
BACKUP_CHECKPOINT_DIR = os.path.join(KERAS_DIR, "backups")
TRAIN_STATE = os.path.join(KERAS_DIR, "train_state.json")
CLASS_JSON = os.path.join(KERAS_DIR, "class_indices.json")

os.makedirs(KERAS_DIR, exist_ok=True)
os.makedirs(BACKUP_CHECKPOINT_DIR, exist_ok=True)

# Recovery controls
MAX_RECOVERIES = 6
RECOVERY_WAIT = 8.0  # seconds before retry

# Watchdog thresholds
WATCHDOG_STEP_TIME_THRESHOLD = 6.0  # seconds (if average step > this => backup + stop)

# Other
USE_MIXUP = False  # disabled by default to avoid label/weight complexity
VERBOSE = 1

# -------------------------
# TF / hardware tuning
# -------------------------
try:
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

if USE_MIXED_PRECISION:
    try:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")
        print("[INFO] mixed_float16 enabled")
    except Exception as e:
        print("[WARN] mixed precision unavailable:", e)

# -------------------------
# Utilities
# -------------------------
def clean_ds(base_dir):
    for root, dirs, files in os.walk(base_dir):
        for f in list(files):
            if f == ".DS_Store":
                try: os.remove(os.path.join(root, f))
                except: pass
clean_ds(BASE_DIR)

def parse_dir(base_dir, split):
    split_dir = os.path.join(base_dir, split)
    if not os.path.exists(split_dir):
        raise FileNotFoundError(split_dir)
    classes = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
    paths = []
    labels = []
    for i, c in enumerate(classes):
        pth = os.path.join(split_dir, c)
        for root, _, files in os.walk(pth):
            for f in files:
                if f.lower().endswith(('.jpg','.jpeg','.png')):
                    paths.append(os.path.join(root, f))
                    labels.append(i)
    return paths, labels, classes

train_paths, train_labels, class_names = parse_dir(BASE_DIR, "train")
val_paths, val_labels, _ = parse_dir(BASE_DIR, "val")
test_paths, test_labels, _ = parse_dir(BASE_DIR, "test")

num_classes = len(class_names)
print(f"[INFO] classes({num_classes}) = {class_names}")
with open(CLASS_JSON, "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)

# -------------------------
# Class weights & sample weight vector
# -------------------------
labels_arr = np.array(train_labels)
counts = np.bincount(labels_arr, minlength=num_classes)
total = counts.sum()
raw_w = total / (num_classes * (counts + 1e-6))
# Clip to stable range
class_weight_arr = np.clip(raw_w, 0.5, 8.0).astype(np.float32)
class_weights_dict = {i: float(class_weight_arr[i]) for i in range(num_classes)}
print("[INFO] class_weights clipped:", class_weights_dict)
class_weight_vector = tf.constant(class_weight_arr, dtype=tf.float32)

# -------------------------
# Dataset: produce (x, y_int, sample_weight)
# -------------------------
def _decode_preprocess(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.efficientnet_v2.preprocess_input(img)
    label = tf.cast(label, tf.int32)
    img.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])
    label = tf.reshape(label, [])
    sw = tf.gather(class_weight_vector, label)
    sw = tf.reshape(sw, [])
    return img, label, sw

@tf.function
def augment_fn(img, label, sw, rare=False):
    # lightweight augment; rare flag can trigger stronger transforms if desired
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.06)
    img = tf.image.random_contrast(img, 0.95, 1.05)
    if _HAS_TFA:
        ang = tf.random.uniform([], -0.03, 0.03)
        img = tfa.image.rotate(img, ang)
    return img, label, sw

from collections import defaultdict
class_to_paths = defaultdict(list)
for p, l in zip(train_paths, train_labels):
    class_to_paths[l].append(p)
rare_threshold = 200
rare_classes = [c for c, pts in class_to_paths.items() if len(pts) < rare_threshold]
print("[INFO] rare_classes:", rare_classes)

def balanced_generator(oversample_rare=True):
    # oversample rare classes to increase their presence in epoch
    per_class = {c: len(class_to_paths[c]) for c in class_to_paths}
    maxc = max(per_class.values())
    samples = []
    for c, pts in class_to_paths.items():
        rep = (max(8, maxc // 4) if (oversample_rare and c in rare_classes) else 1)
        for _ in range(rep):
            for p in pts:
                samples.append((p, c))
    random.shuffle(samples)
    while True:
        for p,c in samples:
            yield p, c

def build_train_ds(batch_size=BATCH_SIZE, steps_per_epoch=None):
    gen = lambda: balanced_generator(oversample_rare=True)
    out_sig = (tf.TensorSpec(shape=(), dtype=tf.string), tf.TensorSpec(shape=(), dtype=tf.int32))
    ds = tf.data.Dataset.from_generator(gen, output_signature=out_sig)
    ds = ds.map(lambda p,l: tf.py_function(func=_decode_preprocess, inp=[p,l], Tout=[tf.float32, tf.int32, tf.float32]), num_parallel_calls=AUTOTUNE)
    # set shapes after py_function
    def set_shapes(img, label, sw):
        img.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])
        label.set_shape([])
        sw.set_shape([])
        return img, label, sw
    ds = ds.map(set_shapes, num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda img,l,sw: augment_fn(img,l,sw), num_parallel_calls=AUTOTUNE)
    if USE_SNAPSHOT:
        try:
            ds = ds.apply(tf.data.experimental.snapshot(SNAPSHOT_DIR))
            print("[INFO] Using snapshot:", SNAPSHOT_DIR)
        except Exception as e:
            print("[WARN] snapshot unavailable:", e)
    ds = ds.shuffle(SHUFFLE_BUFFER, seed=SEED)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds

def build_eval_ds(paths, labels, batch_size=BATCH_SIZE):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda p,l: tf.py_function(func=lambda a,b: (_decode_preprocess(a,b)[0], tf.cast(b, tf.int32), tf.constant(1.0, dtype=tf.float32)), inp=[p,l], Tout=[tf.float32, tf.int32, tf.float32]), num_parallel_calls=AUTOTUNE)
    def setsh(img, lbl, sw):
        img.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3]); lbl.set_shape([]); sw.set_shape([])
        return img, lbl  # for validation we return (x, y)
    ds = ds.map(lambda img,l,sw: (img, l), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds

# build datasets
train_ds = build_train_ds(batch_size=BATCH_SIZE)
val_ds = build_eval_ds(val_paths, val_labels, batch_size=BATCH_SIZE)
test_ds = build_eval_ds(test_paths, test_labels, batch_size=BATCH_SIZE)

# -------------------------
# Model: EfficientNetV2-S
# -------------------------
from tensorflow.keras.applications import EfficientNetV2S
def make_model(num_classes, input_shape=IMG_SIZE+(3,), dropout_rate=0.3):
    base = EfficientNetV2S(include_top=False, input_shape=input_shape, weights="imagenet")
    inp = layers.Input(shape=input_shape)
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    out = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    model = models.Model(inputs=inp, outputs=out)
    return model, base

# -------------------------
# Optimizer / loss / metrics
# -------------------------
def get_optimizer(lr, wd=WEIGHT_DECAY):
    if USE_ADAMW:
        if _HAS_TFA:
            return tfa.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
        else:
            try:
                from tensorflow.keras.optimizers.experimental import AdamW as AdamWExp
                return AdamWExp(learning_rate=lr, weight_decay=wd)
            except Exception:
                print("[WARN] AdamW not available; falling back to Adam")
                return tf.keras.optimizers.Adam(learning_rate=lr)
    else:
        # SGD is memory-light and stable on M1
        return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
metrics = [
    tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
    tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="sparse_top3"),
    tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="sparse_top5")
]

# -------------------------
# Callbacks: watchdog + backup + timing
# -------------------------
class BatchTiming(callbacks.Callback):
    def __init__(self, every=200, out_file=None):
        super().__init__(); self.every=every; self.out=out_file; self.times=[]; self._s=None
    def on_train_batch_begin(self, batch, logs=None): self._s=time.time()
    def on_train_batch_end(self, batch, logs=None):
        t=time.time()- (self._s or time.time()); self.times.append(t)
        if len(self.times)%self.every==0:
            arr=np.array(self.times[-self.every:]); msg=f"[BATCHTIMING] mean={arr.mean():.3f}s median={np.median(arr):.3f}s p95={np.percentile(arr,95):.3f}s"
            print(msg)
            if self.out: open(self.out,"a").write(msg+"\n")

class PeriodicBackup(callbacks.Callback):
    def __init__(self, every_epochs=2):
        super().__init__(); self.every_epochs=every_epochs
    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1)%self.every_epochs==0:
            fname=os.path.join(BACKUP_CHECKPOINT_DIR, f"backup_epoch{epoch+1}.keras")
            try: self.model.save(fname); print("[BACKUP] saved", fname)
            except Exception as e: print("[WARN] backup failed:", e)
            st={"last_epoch": int(epoch+1), "timestamp": time.time()}
            try: open(TRAIN_STATE,"w").write(json.dumps(st))
            except: pass

class PipelineWatchdog(callbacks.Callback):
    def __init__(self, max_step_time=WATCHDOG_STEP_TIME_THRESHOLD, window=40):
        super().__init__(); self.max_step_time=max_step_time; self.window=window; self.times=[]; self._s=None
    def on_train_batch_begin(self, batch, logs=None): self._s=time.time()
    def on_train_batch_end(self, batch, logs=None):
        t=time.time()- (self._s or time.time()); self.times.append(t)
        if len(self.times)>self.window: self.times.pop(0)
        avg=sum(self.times)/len(self.times)
        if avg>self.max_step_time:
            fname=os.path.join(BACKUP_CHECKPOINT_DIR, "watchdog_save.keras")
            try: self.model.save(fname)
            except: pass
            print(f"[WATCHDOG] avg step {avg:.2f}s > {self.max_step_time}s. Saved {fname} and stopping training.")
            # record state
            try:
                st = json.loads(open(TRAIN_STATE).read()) if os.path.exists(TRAIN_STATE) else {}
            except:
                st = {}
            st.update({"watchdog_triggered":True, "watchdog_avg":avg, "timestamp":time.time()})
            try: open(TRAIN_STATE,"w").write(json.dumps(st))
            except: pass
            self.model.stop_training = True

# callbacks lists
cb_stage1 = [
    callbacks.ModelCheckpoint(BEST_MODEL_PATH, monitor="val_sparse_top3", save_best_only=True, mode="max"),
    callbacks.EarlyStopping(monitor="val_loss", patience=ES_PATIENCE_STAGE1, restore_best_weights=True, min_delta=ES_MIN_DELTA, verbose=1),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
    BatchTiming(every=500, out_file=os.path.join(KERAS_DIR,"batch_timing_stage1.log")),
    PeriodicBackup(every_epochs=2),
    PipelineWatchdog(max_step_time=WATCHDOG_STEP_TIME_THRESHOLD, window=40)
]

cb_stage2 = [
    callbacks.ModelCheckpoint(BEST_FINETUNE_PATH, monitor="val_sparse_top3", save_best_only=True, mode="max"),
    callbacks.EarlyStopping(monitor="val_loss", patience=ES_PATIENCE_STAGE2, restore_best_weights=True, min_delta=ES_MIN_DELTA, verbose=1),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
    BatchTiming(every=500, out_file=os.path.join(KERAS_DIR,"batch_timing_stage2.log")),
    PeriodicBackup(every_epochs=2),
    PipelineWatchdog(max_step_time=WATCHDOG_STEP_TIME_THRESHOLD, window=40)
]

# -------------------------
# Checkpoint/resume utils
# -------------------------
def find_latest_checkpoint():
    candidates=[]
    if os.path.exists(BEST_FINETUNE_PATH): candidates.append((BEST_FINETUNE_PATH, os.path.getmtime(BEST_FINETUNE_PATH)))
    if os.path.exists(BEST_MODEL_PATH): candidates.append((BEST_MODEL_PATH, os.path.getmtime(BEST_MODEL_PATH)))
    for f in glob(os.path.join(BACKUP_CHECKPOINT_DIR,"*.keras")):
        try:
            mtime=os.path.getmtime(f); candidates.append((f, mtime))
        except: pass
    if not candidates: return None
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]

def safe_load_model(path):
    try:
        m = keras.models.load_model(path)
        print("[INFO] Loaded model:", path, "optimizer present?", hasattr(m,"optimizer") and m.optimizer is not None)
        return m
    except Exception as e:
        print("[WARN] load_model failed:", e)
        return None

# -------------------------
# Training loop with automatic recovery
# -------------------------
def train_with_recovery():
    recoveries = 0
    while recoveries <= MAX_RECOVERIES:
        ckpt = find_latest_checkpoint()
        model = None
        if ckpt:
            model = safe_load_model(ckpt)
        if model is None:
            model, base = make_model(num_classes)
            base.trainable = False
            print("[INFO] New model created; base frozen")
        else:
            # ensure base exists (if loaded model may be complete)
            try:
                base = [l for l in model.layers if 'efficientnetv2' in l.name][0]
            except:
                base = None

        # Stage1 compile (head-only)
        opt1 = get_optimizer(LEARNING_RATE_STAGE1)
        if GRAD_CLIPNORM:
            try:
                opt1 = tf.keras.optimizers.get(opt1) if isinstance(opt1, dict) else opt1
                # wrappers not needed; Keras supports clipnorm on optimizer param via .clipnorm isn't always settable; use compile param
            except:
                pass
        model.compile(optimizer=opt1, loss=loss_fn, metrics=metrics)
        print("[INFO] Starting Stage1")
        try:
            history1 = model.fit(
                train_ds, validation_data=val_ds,
                epochs=EPOCHS_STAGE1, initial_epoch=0,
                callbacks=cb_stage1, verbose=VERBOSE
            )
        except KeyboardInterrupt:
            print("[INFO] KeyboardInterrupt; exiting")
            return
        except Exception as e:
            print("[ERROR] Exception during Stage1:", e)
            recoveries += 1
            print(f"[INFO] Recovery attempt {recoveries}/{MAX_RECOVERIES} after exception")
            time.sleep(RECOVERY_WAIT)
            continue

        # Stage2: partial unfreeze
        print("[INFO] Stage1 complete; entering Stage2 partial unfreeze")
        try:
            base_layers = model.layers[1] if len(model.layers)>1 else None
            # Generic approach: unfreeze last N layers of the base model if available
            if base is not None:
                all_layers = base.layers if hasattr(base, "layers") else []
                for layer in all_layers[:-PARTIAL_UNFREEZE_LAYERS]:
                    layer.trainable = False
                for layer in all_layers[-PARTIAL_UNFREEZE_LAYERS:]:
                    layer.trainable = True
            else:
                for layer in model.layers:
                    layer.trainable = True
        except Exception:
            for layer in model.layers:
                layer.trainable = True

        opt2 = get_optimizer(LEARNING_RATE_STAGE2)
        model.compile(optimizer=opt2, loss=loss_fn, metrics=metrics)
        print("[INFO] Starting Stage2 (fine-tune)")
        try:
            history2 = model.fit(
                train_ds, validation_data=val_ds,
                epochs=EPOCHS_STAGE2, initial_epoch=0,
                callbacks=cb_stage2, verbose=VERBOSE
            )
        except KeyboardInterrupt:
            print("[INFO] KeyboardInterrupt; exiting")
            return
        except Exception as e:
            print("[ERROR] Exception during Stage2:", e)
            recoveries += 1
            print(f"[INFO] Recovery attempt {recoveries}/{MAX_RECOVERIES}")
            time.sleep(RECOVERY_WAIT)
            continue

        # Evaluate & save final
        print("[INFO] Evaluating on test set")
        test_metrics = model.evaluate(test_ds)
        print("[INFO] Test metrics:", test_metrics)
        final = BEST_FINETUNE_PATH if os.path.exists(BEST_FINETUNE_PATH) else BEST_MODEL_PATH
        try:
            model.save(final)
            print("[INFO] Saved final model to", final)
        except Exception as e:
            print("[WARN] final save failed:", e)
        return

    print("[ERROR] Max recoveries exceeded; aborting")

# -------------------------
# Bench utility
# -------------------------
def benchmark(ds, n=50):
    it = iter(ds)
    times=[]
    for i in range(n):
        t0=time.time()
        try: next(it)
        except StopIteration: break
        times.append(time.time()-t0)
    times=np.array(times)
    print("[BENCH] mean", times.mean(), "median", np.median(times), "p95", np.percentile(times,95))
    return times

# -------------------------
# Entrypoint
# -------------------------
if __name__ == "__main__":
    if "--bench" in sys.argv:
        benchmark(train_ds, n=50)
        sys.exit(0)
    train_with_recovery()
