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



import os
import shutil
import random
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2M
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ============================================================
# 1️⃣ 경로 설정 (Google Drive 기준)
# ============================================================
ROOT = Path("drive/MyDrive")
BASE_DIR = ROOT / "dataset_sp"
TRAIN_DIR = BASE_DIR / "train"
VAL_DIR = BASE_DIR / "val"
TEST_DIR = BASE_DIR / "test"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
MODEL_SAVE_PATH = ROOT / "recycle_classifier_v2m_fixed.keras"

# ============================================================
# 2️⃣ val/test 폴더 생성 및 train에서 자동 분할 (8:1:1)
# ============================================================
def split_train_val_test(train_dir, val_dir, test_dir, val_ratio=0.1, test_ratio=0.1):
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    for cls in os.listdir(train_dir):
        class_train_path = train_dir / cls
        if not class_train_path.is_dir():
            continue
        images = [f for f in os.listdir(class_train_path) if f.lower().endswith((".jpg",".png"))]
        random.shuffle(images)

        n_total = len(images)
        n_val = max(1, int(n_total * val_ratio))
        n_test = max(1, int(n_total * test_ratio))
        n_train = n_total - n_val - n_test

        val_cls_dir = val_dir / cls
        test_cls_dir = test_dir / cls
        val_cls_dir.mkdir(exist_ok=True)
        test_cls_dir.mkdir(exist_ok=True)

        # val/test에만 복사
        for f in images[n_train:n_train+n_val]:
            shutil.copy(class_train_path / f, val_cls_dir / f)
        for f in images[n_train+n_val:]:
            shutil.copy(class_train_path / f, test_cls_dir / f)

split_train_val_test(TRAIN_DIR, VAL_DIR, TEST_DIR)

# ============================================================
# 3️⃣ 데이터 제너레이터
# ============================================================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=42
)
val_generator = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)
test_generator = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

NUM_CLASSES = len(train_generator.class_indices)
print(f"클래스 수: {NUM_CLASSES}, 클래스 목록: {list(train_generator.class_indices.keys())}")

steps_per_epoch = max(1, train_generator.samples // BATCH_SIZE)
validation_steps = max(1, val_generator.samples // BATCH_SIZE)

# ============================================================
# 4️⃣ 모델 구성 (EfficientNetV2M)
# ============================================================
base_model = EfficientNetV2M(weights="imagenet", include_top=False, input_shape=(*IMAGE_SIZE,3))
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.3)(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer=Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# ============================================================
# 5️⃣ 콜백
# ============================================================
callbacks = [
    ModelCheckpoint(MODEL_SAVE_PATH, monitor="val_accuracy", save_best_only=True, mode="max"),
    ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3, verbose=1),
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
]

# ============================================================
# 6️⃣ Stage 1: Feature extractor 학습
# ============================================================
history1 = model.fit(
    train_generator,
    epochs=EPOCHS//2,
    validation_data=val_generator,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=callbacks
)

# ============================================================
# 7️⃣ Stage 2: Fine-tuning
# ============================================================
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])

history2 = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=callbacks
)

# ============================================================
# 8️⃣ 테스트 평가
# ============================================================
loss, acc = model.evaluate(test_generator)
print(f"최종 테스트 정확도: {acc:.4f}")


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
