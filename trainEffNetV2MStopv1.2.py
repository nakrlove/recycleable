# =====================================
# 🧠 Colab 세션 유지 코드 (업그레이드 버전)
# =====================================
from IPython.display import Javascript, display

def keep_colab_alive(interval_min=15):
    js_code = f"""
    async function ClickConnect() {{
        console.log("[KeepAlive] Colab 세션 유지 중...");
        const btn = document.querySelector("colab-connect-button") 
                 || document.querySelector("#connect") 
                 || document.querySelector("colab-toolbar-button#connect") 
                 || document.querySelector("colab-sessions-button") 
                 || document.querySelector("#top-toolbar colab-connect-button");
        if (btn) {{
            btn.click();
            console.log("✅ 연결 유지 버튼 클릭 완료");
        }} else {{
            console.log("⚠️ Colab 연결 버튼을 찾을 수 없습니다. (UI 변경 가능성)");
        }}
    }}
    setInterval(ClickConnect, {interval_min} * 60 * 1000);
    console.log("✅ Colab Keep-Alive 활성화됨 — {interval_min}분 간격으로 클릭 중");
    """
    try:
        display(Javascript(js_code))
    except Exception as e:
        print("⚠️ Colab 환경이 아닙니다. 세션 유지 스크립트는 무시됩니다.")
        print(e)

keep_colab_alive(30)



# 안정화 학습 스크립트 (Drive root 포함)
import os
import shutil
import random
from pathlib import Path
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2M
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -----------------------------
# 설정 (필요시 조정)
# -----------------------------
ROOT = Path("drive/MyDrive")
BASE_DIR = ROOT / "dataset_sp"
TRAIN_DIR = BASE_DIR / "train"
VAL_DIR = BASE_DIR / "val"
TEST_DIR = BASE_DIR / "test"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
STAGE1_EPOCHS = 8      # feature-extractor 학습
STAGE2_EPOCHS = 20     # fine-tune
FINE_TUNE_LAST_N_LAYERS = 50
MODEL_BEST_PATH = ROOT / "recycle_classifier_v2m_best.keras"
MODEL_FINAL_PATH = ROOT / "recycle_classifier_v2m_final.keras"
RANDOM_SEED = 42

os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# -----------------------------
# 1) train -> val/test 자동 분할 (8:1:1)
#    이미 val/test가 있으면 덮어쓰지 않음
# -----------------------------
def split_train_val_test(train_dir: Path, val_dir: Path, test_dir: Path, val_ratio=0.1, test_ratio=0.1):
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    for cls in sorted(os.listdir(train_dir)):
        class_train_path = train_dir / cls
        if not class_train_path.is_dir():
            continue
        images = [f for f in os.listdir(class_train_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if len(images) == 0:
            continue
        random.shuffle(images)
        n_total = len(images)
        n_val = max(1, int(n_total * val_ratio))
        n_test = max(1, int(n_total * test_ratio))
        n_train = n_total - n_val - n_test

        # create class dirs if missing
        (val_dir / cls).mkdir(parents=True, exist_ok=True)
        (test_dir / cls).mkdir(parents=True, exist_ok=True)

        # only copy val/test images (leave train as-is)
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        for fname in val_images:
            src = class_train_path / fname
            dst = val_dir / cls / fname
            if not dst.exists():
                shutil.copy(src, dst)
        for fname in test_images:
            src = class_train_path / fname
            dst = test_dir / cls / fname
            if not dst.exists():
                shutil.copy(src, dst)

# call split (will create val/test if missing)
split_train_val_test(TRAIN_DIR, VAL_DIR, TEST_DIR, val_ratio=0.1, test_ratio=0.1)

# -----------------------------
# 2) ImageDataGenerator 설정 (train만 augmentation)
# -----------------------------
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=15,
    width_shift_range=0.08,
    height_shift_range=0.08,
    shear_range=0.05,
    zoom_range=0.08,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_test_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    seed=RANDOM_SEED
)

val_gen = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

test_gen = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# -----------------------------
# 3) 클래스 불일치 및 개수 확인
# -----------------------------
train_classes = set(train_gen.class_indices.keys())
val_classes = set(val_gen.class_indices.keys())
test_classes = set(test_gen.class_indices.keys())

print("train_classes:", len(train_classes))
print("val_classes:", len(val_classes))
print("test_classes:", len(test_classes))

missing_in_val = train_classes - val_classes
missing_in_test = train_classes - test_classes
if missing_in_val or missing_in_test:
    print("경고: train에 있지만 val/test에 없는 클래스가 있습니다.")
    if missing_in_val:
        print(" missing_in_val:", missing_in_val)
    if missing_in_test:
        print(" missing_in_test:", missing_in_test)
    # (필요 시 추가 자동 생성/알림 로직 삽입 가능)

NUM_CLASSES = len(train_gen.class_indices)
print("NUM_CLASSES (train 기준):", NUM_CLASSES)
print("class_indices:", train_gen.class_indices)

# -----------------------------
# 4) steps_per_epoch 안전 계산 (ceil)
# -----------------------------
steps_per_epoch = math.ceil(train_gen.samples / BATCH_SIZE)
validation_steps = math.ceil(val_gen.samples / BATCH_SIZE)
print(f"steps_per_epoch: {steps_per_epoch}, validation_steps: {validation_steps}")

# -----------------------------
# 5) class_weight 계산 (간단한 역비례 가중치)
# -----------------------------
# train_gen.classes : array of class indices for each sample
unique, counts = np.unique(train_gen.classes, return_counts=True)
total = train_gen.samples
class_weight = {}
for cls_idx, cnt in zip(unique, counts):
    # weight = total / (num_classes * cnt)
    class_weight[int(cls_idx)] = float(total) / (NUM_CLASSES * cnt)
print("class_weight sample:", {k: round(v,3) for k,v in list(class_weight.items())[:5]})

# -----------------------------
# 6) 모델 생성 (EfficientNetV2M)
# -----------------------------
base_model = EfficientNetV2M(weights="imagenet", include_top=False, input_shape=(*IMAGE_SIZE, 3))
base_model.trainable = False  # Stage1: freeze
x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.3)(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)
model = Model(base_model.input, outputs)

print(model.summary())  # parameter counts

# compile
model.compile(optimizer=Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])

# -----------------------------
# 7) callbacks
# -----------------------------
callbacks = [
    ModelCheckpoint(str(MODEL_BEST_PATH), monitor="val_accuracy", save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6, verbose=1),
    EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1)
]

# -----------------------------
# 8) Stage1: feature-extractor 학습
# -----------------------------
history1 = model.fit(
    train_gen,
    epochs=STAGE1_EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=validation_steps,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1
)

# -----------------------------
# 9) Stage2: fine-tune - 마지막 N 레이어 unfreeze
# -----------------------------
base_model.trainable = True
# freeze all except last N layers
if FINE_TUNE_LAST_N_LAYERS > 0:
    for layer in base_model.layers[:-FINE_TUNE_LAST_N_LAYERS]:
        layer.trainable = False
else:
    for layer in base_model.layers:
        layer.trainable = True

# recompile with lower lr
model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])

history2 = model.fit(
    train_gen,
    epochs=STAGE2_EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=validation_steps,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1
)

# -----------------------------
# 10) 저장 및 평가
# -----------------------------
# save final weights
model.save(str(MODEL_FINAL_PATH), save_format="keras")
print("Saved final model to:", MODEL_FINAL_PATH)

# load best model (optional) and eval
best_model = tf.keras.models.load_model(str(MODEL_BEST_PATH))
loss, acc = best_model.evaluate(test_gen, steps=math.ceil(test_gen.samples / BATCH_SIZE))
print(f"Best model test accuracy: {acc:.4f}, loss: {loss:.4f}")



# ================================
# 분류 함수
# ================================
# def classify_image(model, image_path, threshold=0.5):
#     img = tf.io.read_file(image_path)
#     img = tf.image.decode_jpeg(img, channels=3)
#     img = tf.cast(img, tf.float32) / 255.0
#     img = tf.expand_dims(img, axis=0)

#     preds = model.predict(img)
#     class_id = np.argmax(preds[0])
#     confidence = preds[0][class_id]

#     if confidence < threshold:
#         return "일반쓰레기"
#     else:
#         return classes[class_id]

def classify_image(model, image_path, classes, threshold=0.5):
    """
    학습용 코드(옵션1) 기반으로 안전하게 이미지 분류
    Args:
        model: 학습된 Keras 모델
        image_path: 분류할 이미지 경로
        classes: 학습 시 사용된 클래스 리스트
        threshold: confidence 임계값
    Returns:
        class_name 또는 "일반쓰레기"
    """
    # 이미지 읽기 및 디코딩 (JPEG, PNG 모두 가능)
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    
    # 학습용 코드 기준: 이미 224x224이므로 resize 불필요
    img.set_shape([224, 224, 3])
    
    # 정규화
    img = tf.cast(img, tf.float32) / 255.0
    
    # 배치 차원 추가
    img = tf.expand_dims(img, axis=0)
    
    # 예측
    preds = model.predict(img)
    class_id = np.argmax(preds[0])
    confidence = preds[0][class_id]
    
    # threshold 기준 판정
    if confidence < threshold:
        return "일반쓰레기"
    else:
        return classes[class_id], float(confidence)

# 예시
# result = classify_image(model, "sample.jpg")
# print("분류 결과:", result)
