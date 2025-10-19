# ================== trainEffNetV2MStopv1.5.py ==================
import os
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# ================== 경로 설정 ==================
ROOT = ""
# ROOT = "drive/MyDrive/"
BASE_DIR = "dataset_sp"
FULL_BASE_DIR = os.path.join(ROOT, BASE_DIR)

train_dir = os.path.join(FULL_BASE_DIR, "train")
val_dir   = os.path.join(FULL_BASE_DIR, "val")
test_dir  = os.path.join(FULL_BASE_DIR, "test")

# ================== 하이퍼파라미터 ==================
IMG_SIZE = (224, 224)
BATCH_SIZE = 2   # 일부 클래스 적은 데이터 때문에 줄임
EPOCHS = 30
SEED = 42

# ================== 데이터셋 로드 ==================
train_ds = image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True,
    seed=SEED
)
class_names = train_ds.class_names
num_classes = len(class_names)
print(f"클래스 개수: {num_classes}")

# ================== 데이터 증강 ==================
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
train_ds = train_ds.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
train_ds = train_ds.repeat()  # train repeat 필요

val_ds = image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False,
    seed=SEED
)
val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

test_ds = image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

# ================== 클래스 가중치 계산 ==================
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

# ================== Steps 계산 ==================
train_count = sum(len(files) for _, _, files in os.walk(train_dir))
val_count   = sum(len(files) for _, _, files in os.walk(val_dir))
test_count  = sum(len(files) for _, _, files in os.walk(test_dir))

steps_per_epoch = math.ceil(train_count / BATCH_SIZE)
validation_steps = max(1, math.ceil(val_count / BATCH_SIZE))

print(f"Train samples: {train_count}, Val samples: {val_count}, Test samples: {test_count}")
print(f"Steps per epoch: {steps_per_epoch}, Validation steps: {validation_steps}")

# ================== 모델 정의 ==================
base_model = keras.applications.EfficientNetV2B0(
    include_top=False,
    input_shape=IMG_SIZE + (3,),
    weights='imagenet'
)
base_model.trainable = False

inputs = keras.Input(shape=IMG_SIZE + (3,))
x = keras.applications.efficientnet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)
model = keras.Model(inputs, outputs)

optimizer = keras.optimizers.Adam(learning_rate=1e-3)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)
]

# ================== 1차 학습: Feature Extractor ==================
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=validation_steps,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

# ================== 2차 학습: Fine-tuning ==================
base_model.trainable = True
for layer in base_model.layers[:200]:
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
    callbacks=callbacks,
    verbose=1
)

# ================== 모델 저장 ==================
model_save_path = os.path.join(FULL_BASE_DIR, "efficientnetv2_stable_final.keras")
model.save(model_save_path, save_format="keras")
print(f"✅ 모델 저장 완료: {model_save_path}")

# ================== 테스트 평가 ==================
test_steps = max(1, math.ceil(test_count / BATCH_SIZE))
test_loss, test_acc = model.evaluate(test_ds, steps=test_steps)
print(f"✅ Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")
