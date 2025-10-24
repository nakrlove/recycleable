import os
import sys
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from datetime import datetime

# === 현재 실행 중인 파일명 출력 ===
current_script = os.path.basename(sys.argv[0])
print(f"\n🚀 실행 파일명: {current_script}\n")

# === 데이터 경로 설정 ===
BASE_DIR = "dataset_2000"
train_dir = os.path.join(BASE_DIR, "train")
val_dir = os.path.join(BASE_DIR, "val")
test_dir = os.path.join(BASE_DIR, "test")

# === 파라미터 설정 ===
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 30

# === 데이터 증강 (과적합 방지 + 일반화 향상) ===
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.25),
    layers.RandomZoom(0.3),
    layers.RandomContrast(0.3),
], name="data_augmentation")

# === 데이터셋 로드 ===
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

# === class_names 자동 저장 (Django에서도 사용 가능) ===
class_names = train_ds.class_names
class_names_file = os.path.join(BASE_DIR, "class_names.txt")

with open(class_names_file, "w") as f:
    for name in class_names:
        f.write(name + "\n")

print(f"✅ 클래스 목록이 '{class_names_file}' 파일로 저장되었습니다.")
print(f"클래스 개수: {len(class_names)} → {class_names}\n")

# === 데이터셋 최적화 ===
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

# === 클래스 가중치 계산 ===
labels = np.concatenate([y for x, y in train_ds], axis=0)
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)
class_weights_dict = dict(enumerate(class_weights))
print("📊 클래스 가중치:", class_weights_dict, "\n")

# === EfficientNetV2 기반 모델 구성 ===
base_model = tf.keras.applications.EfficientNetV2M(
    include_top=False,
    input_shape=IMG_SIZE + (3,),
    weights="imagenet"
)
base_model.trainable = False  # 전이학습 (Feature Extractor)

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = tf.keras.applications.efficientnet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)

model = models.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# === 콜백 설정 ===
checkpoint_path = os.path.join("checkpoints", f"EffNetV2M_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5")
os.makedirs("checkpoints", exist_ok=True)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6),
    ModelCheckpoint(checkpoint_path, save_best_only=True, monitor="val_loss")
]

# === 모델 학습 ===
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights_dict,
    callbacks=callbacks
)

# === 테스트 평가 ===
test_loss, test_acc = model.evaluate(test_ds)
print(f"\n✅ 테스트 정확도: {test_acc * 100:.2f}% | 테스트 손실: {test_loss:.4f}")

# === 모델 저장 ===
model.save("trained_model_EffNetV2M_v3.0.h5")
print("\n💾 모델 저장 완료: trained_model_EffNetV2M_v3.0.h5")
