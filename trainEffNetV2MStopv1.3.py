# =====================================
# 🧠 Colab 세션 유지 코드 (업그레이드 버전)
# =====================================

# from IPython.display import Javascript, display

# def keep_colab_alive(interval_min=15):
#     js_code = f"""
#     async function ClickConnect() {{
#         console.log("[KeepAlive] Colab 세션 유지 중...");
#         const btn = document.querySelector("colab-connect-button") 
#                  || document.querySelector("#connect") 
#                  || document.querySelector("colab-toolbar-button#connect") 
#                  || document.querySelector("colab-sessions-button") 
#                  || document.querySelector("#top-toolbar colab-connect-button");
#         if (btn) {{
#             btn.click();
#             console.log("✅ 연결 유지 버튼 클릭 완료");
#         }} else {{
#             console.log("⚠️ Colab 연결 버튼을 찾을 수 없습니다. (UI 변경 가능성)");
#         }}
#     }}
#     setInterval(ClickConnect, {interval_min} * 60 * 1000);
#     console.log("✅ Colab Keep-Alive 활성화됨 — {interval_min}분 간격으로 클릭 중");
#     """
#     try:
#         display(Javascript(js_code))
#     except Exception as e:
#         print("⚠️ Colab 환경이 아닙니다. 세션 유지 스크립트는 무시됩니다.")
#         print(e)

# keep_colab_alive(30)


# ============================================================
# ✅ 안정화 학습용 EfficientNetV2 모델 (Google Drive 경로 기반)
# ============================================================

import os
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# ============================================================
# 1. 기본 경로 설정
# ============================================================
BASE_DIR = "dataset_sp"
# root = os.path.join("drive", "MyDrive", BASE_DIR)
root = "dataset_sp"
train_dir = os.path.join(root, "train")
val_dir = os.path.join(root, "val")
test_dir = os.path.join(root, "test")

# ============================================================
# 2. 하이퍼파라미터
# ============================================================
IMG_SIZE = (224, 224)
# BATCH_SIZE = 32
BATCH_SIZE = 8
EPOCHS = 30
SEED = 42

# ============================================================
# 3. 데이터셋 로드 (.repeat() 포함)
# ============================================================
# train_ds = image_dataset_from_directory(
#     train_dir,
#     image_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     label_mode="categorical",
#     shuffle=True,
#     seed=SEED
# )

train_ds = image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True,
    seed=SEED
)

# ⚠️ class_names는 여기서 추출해야 함
class_names = train_ds.class_names
num_classes = len(class_names)
print(f"클래스 개수: {num_classes}")

# ✅ 이후에 캐시, 셔플, 프리페치 적용
train_ds = train_ds.cache().shuffle(500).prefetch(1)

val_ds = image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False,
    seed=SEED
)

val_ds = val_ds.cache().prefetch(1)

test_ds = image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)
val_ds = test_ds.cache().prefetch(1)

# ✅ 이제 generator 고갈 방지용 repeat 적용
train_ds = train_ds.repeat()
val_ds = val_ds.repeat()  # ✅ validation도 반복
# ============================================================
# 4. 클래스 가중치 계산 (불균형 데이터 보정)
# ============================================================
# 각 폴더별 데이터 개수 기반으로 계산
all_labels = []
for class_index, class_name in enumerate(class_names):
    class_path = os.path.join(train_dir, class_name)
    count = len(os.listdir(class_path))
    all_labels += [class_index] * count

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(all_labels),
    y=all_labels
)
class_weight_dict = dict(enumerate(class_weights))
print("Class Weights:", class_weight_dict)

# ============================================================
# 5. Prefetch 최적화
# ============================================================
AUTOTUNE = tf.data.AUTOTUNE


# train_ds = train_ds.prefetch(AUTOTUNE)
# val_ds = val_ds.prefetch(AUTOTUNE)
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

test_ds = test_ds.prefetch(AUTOTUNE)

# ============================================================
# 6. 모델 정의
# ============================================================
base_model = tf.keras.applications.EfficientNetV2B0(
    include_top=False,
    input_shape=IMG_SIZE + (3,),
    weights='imagenet'
)
base_model.trainable = False  # Stage 1: Feature extractor만 사용

inputs = keras.Input(shape=IMG_SIZE + (3,))
x = tf.keras.applications.efficientnet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)
model = keras.Model(inputs, outputs)

# ============================================================
# 7. 컴파일
# ============================================================
optimizer = keras.optimizers.Adam(learning_rate=1e-3)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ============================================================
# 8. 콜백 설정
# ============================================================
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)
]

# ============================================================
# 9. Steps 계산 (데이터 고갈 방지)
# ============================================================
train_count = sum(len(files) for _, _, files in os.walk(train_dir))
val_count = sum(len(files) for _, _, files in os.walk(val_dir))
steps_per_epoch = math.ceil(train_count / BATCH_SIZE)
validation_steps = math.ceil(val_count / BATCH_SIZE)

print(f"Train samples: {train_count}, Val samples: {val_count}")
print(f"Steps per epoch: {steps_per_epoch}, Validation steps: {validation_steps}")

# ============================================================
# 10. 1차 학습 (Feature Extractor 단계)
# ============================================================
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=validation_steps,
    class_weight=class_weight_dict,
    callbacks=callbacks
)

# ============================================================
# 11. 2차 학습 (Fine-tuning 단계)
# ============================================================
base_model.trainable = True
for layer in base_model.layers[:200]:  # 처음 몇 층은 그대로 둠
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_finetune = model.fit(
    train_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=validation_steps,
    class_weight=class_weight_dict,
    callbacks=callbacks
)

# ============================================================
# 12. 모델 저장
# ============================================================
model_save_path = os.path.join(root, "efficientnetv2_stable_final.keras")
model.save(model_save_path , save_format="keras")
print(f"✅ 모델 저장 완료: {model_save_path}")

# ============================================================
# 13. 평가
# ============================================================
test_loss, test_acc = model.evaluate(test_ds)
print(f"✅ Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")


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

# def classify_image(model, image_path, classes, threshold=0.5):
#     """
#     학습용 코드(옵션1) 기반으로 안전하게 이미지 분류
#     Args:
#         model: 학습된 Keras 모델
#         image_path: 분류할 이미지 경로
#         classes: 학습 시 사용된 클래스 리스트
#         threshold: confidence 임계값
#     Returns:
#         class_name 또는 "일반쓰레기"
#     """
#     # 이미지 읽기 및 디코딩 (JPEG, PNG 모두 가능)
#     img = tf.io.read_file(image_path)
#     img = tf.image.decode_image(img, channels=3)
    
#     # 학습용 코드 기준: 이미 224x224이므로 resize 불필요
#     img.set_shape([224, 224, 3])
    
#     # 정규화
#     img = tf.cast(img, tf.float32) / 255.0
    
#     # 배치 차원 추가
#     img = tf.expand_dims(img, axis=0)
    
#     # 예측
#     preds = model.predict(img)
#     class_id = np.argmax(preds[0])
#     confidence = preds[0][class_id]
    
#     # threshold 기준 판정
#     if confidence < threshold:
#         return "일반쓰레기"
#     else:
#         return classes[class_id], float(confidence)

# 예시
# result = classify_image(model, "sample.jpg")
# print("분류 결과:", result)
