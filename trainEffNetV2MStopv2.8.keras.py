import os
import sys
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
DATASET_PATH = "dataset_2000"
TRAIN_PATH = os.path.join(DATASET_PATH, "train")
VAL_PATH = os.path.join(DATASET_PATH, "val")
TEST_PATH = os.path.join(DATASET_PATH, "test")

# --------------------------------------------------
# 3️⃣ 데이터셋 로드
# --------------------------------------------------
BATCH_SIZE = 32
IMG_SIZE = (224, 224)

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
