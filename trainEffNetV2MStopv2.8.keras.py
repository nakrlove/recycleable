import os
import sys
import os, math, re
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import EfficientNetV2M
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from datetime import datetime

# --------------------------------------------------
# 1️⃣ 현재 실행 파일명 로그 출력
# --------------------------------------------------
script_name = os.path.basename(sys.argv[0])
print(f"\n🚀 Running training script: {script_name}\n")

# --------------------------------------------------
# 2️⃣ 데이터 경로 설정
# --------------------------------------------------
DATASET_PATH = "dataset_25000"
TRAIN_PATH = os.path.join(DATASET_PATH, "train")
VAL_PATH = os.path.join(DATASET_PATH, "val")
TEST_PATH = os.path.join(DATASET_PATH, "test")

# --------------------------------------------------
# 3️⃣ 데이터셋 로드
# --------------------------------------------------
BATCH_SIZE = 32
IMG_SIZE = (224, 224)



# ============================================================
# 2️⃣ 한글 파일/폴더 처리 (기존 코드 사용)
# ============================================================
def has_korean(text): return bool(re.search(r'[가-힣]', text))
CUSTOM_MAP = {"_김장현_":"_kimjanghyun_","플라스틱":"plastic","비닐":"vinyl","종이":"paper","유리":"glass","금속":"metal"}
def safe_name(name, counter):
    cleaned = re.sub(r'[가-힣]+', '', name)
    cleaned = re.sub(r'\s+', '_', cleaned)
    cleaned = re.sub(r'[^a-zA-Z0-9_.-]', '', cleaned)
    return cleaned if cleaned.strip() else f"korean_file_{counter:03d}"
def find_korean_dirs(base_path="."):
    return [os.path.join(root, d) for root, dirs, _ in os.walk(base_path) for d in dirs if has_korean(d)]
def rename_korean_files(base_path="."):
    counter, renamed = 1, []
    for root, dirs, files in os.walk(base_path, topdown=False):
        for filename in files:
            old_path = os.path.join(root, filename)
            new_filename = filename
            for k, v in CUSTOM_MAP.items():
                new_filename = new_filename.replace(k, v)
            if has_korean(new_filename):
                name, ext = os.path.splitext(new_filename)
                new_filename = safe_name(name, counter) + ext
                counter += 1
            new_path = os.path.join(root, new_filename)
            if new_path != old_path:
                os.rename(old_path, new_path)
                renamed.append((old_path, new_path))
        for dirname in dirs:
            old_dir = os.path.join(root, dirname)
            new_dirname = dirname
            for k, v in CUSTOM_MAP.items():
                new_dirname = new_dirname.replace(k, v)
            if has_korean(new_dirname):
                new_dirname = safe_name(new_dirname, counter)
                counter += 1
            new_dir = os.path.join(root, new_dirname)
            if new_dir != old_dir:
                os.rename(old_dir, new_dir)
                renamed.append((old_dir, new_dir))
rename_korean_files(DATASET_PATH)


train_ds = image_dataset_from_directory(
    TRAIN_PATH, image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=True
)
val_ds = image_dataset_from_directory(
    VAL_PATH, image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False
)
test_ds = image_dataset_from_directory(
    TEST_PATH, image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False
)

# --------------------------------------------------
# 4️⃣ 클래스 이름 저장 (Django용 등에서 사용)
# --------------------------------------------------
class_names = train_ds.class_names
with open("class_names.txt", "w", encoding="utf-8") as f:
    for name in class_names:
        f.write(name + "\n")
print("✅ Class names saved to class_names.txt")

# --------------------------------------------------
# 5️⃣ Prefetch 최적화
# --------------------------------------------------
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

# --------------------------------------------------
# 6️⃣ 모델 구성 (EfficientNetV2M)
# --------------------------------------------------
base_model = EfficientNetV2M(include_top=False, input_shape=IMG_SIZE + (3,), weights="imagenet")
base_model.trainable = False

inputs = layers.Input(shape=IMG_SIZE + (3,))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)
model = models.Model(inputs, outputs)

# --------------------------------------------------
# 7️⃣ 컴파일
# --------------------------------------------------
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# --------------------------------------------------
# 8️⃣ 콜백 설정
# --------------------------------------------------
MODEL_SAVE_PATH = "trained_model_EffNetV2M_v3.0.keras"
checkpoint = ModelCheckpoint(
    MODEL_SAVE_PATH, monitor="val_accuracy", save_best_only=True, verbose=1
)
earlystop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

# --------------------------------------------------
# 9️⃣ 학습 실행
# --------------------------------------------------
print("🚀 Start training EfficientNetV2M ...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=[checkpoint, earlystop],
    verbose=1,
)

# --------------------------------------------------
# 🔟 최종 저장 (keras 포맷)
# --------------------------------------------------
model.save(MODEL_SAVE_PATH)
print(f"✅ Model saved as: {MODEL_SAVE_PATH}")

# --------------------------------------------------
# 11️⃣ 테스트 평가
# --------------------------------------------------
test_loss, test_acc = model.evaluate(test_ds)
print(f"✅ Test Accuracy: {test_acc:.4f}, Loss: {test_loss:.4f}")
