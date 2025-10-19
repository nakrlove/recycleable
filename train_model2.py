import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# 1. 데이터 준비 모듈을 임포트합니다.
import prepare_data 

# ===== 설정 =====
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15           # 최대 50 에포크까지 훈련

ORIGINAL_DATA_DIR = "dataset/train"
SPLIT_DATA_DIR = "split_dataset"

# 2. 데이터 준비 함수 호출
DATA_DIR = prepare_data.prepare_and_split_data(
    input_dir=ORIGINAL_DATA_DIR,
    output_dir=SPLIT_DATA_DIR,
    ratio=(0.8, 0.2, 0.0)
)

#  None 검사 및 종료
if DATA_DIR is None:
    print("\n 데이터 준비 과정에서 오류가 발생했거나 원본 이미지('dataset/train')가 없습니다. 학습을 시작할 수 없습니다.")
    exit()

MODEL_DIR = "saved_model_recycle"

# 조기 종료(Early Stopping) 설정
EARLY_STOP_PATIENCE = 15  # ⬅ 넉넉하게 변경
MONITOR_METRIC = "val_loss" 

# ===== 데이터 제너레이터 (보완된 증강) =====
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,        # 20 → 15 (과도한 회전 방지)
    width_shift_range=0.1,    # 0.15 → 0.1
    height_shift_range=0.1,   # 0.15 → 0.1
    shear_range=0.1,          # 0.15 → 0.1
    zoom_range=0.1,           # 0.15 → 0.1
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

# 훈련 및 검증 데이터 로드
try:
    train_gen = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True
    )
    val_gen = val_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "val"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )
except Exception as e:
    print(f"\n 데이터 로드 실패. '{DATA_DIR}' 폴더에 'train'과 'val'이 올바르게 분리되었는지 확인하세요.")
    print(f"   오류: {e}")
    exit()

num_classes = len(train_gen.class_indices)
print(f"\n 감지된 클래스 수: {num_classes}개")

# ===== 모델 구성 (전이학습: MobileNetV2) =====
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(*IMG_SIZE, 3),
    include_top=False, 
    weights="imagenet", 
    pooling="avg" 
)
base_model.trainable = False  # Step1: 우선 Freeze 상태로 시작

inputs = layers.Input(shape=(*IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = layers.Dropout(0.4)(x) 
x = layers.Dense(512, activation="relu")(x) 
x = layers.BatchNormalization()(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs)

# Step1: 기본 학습 (동결 상태)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ===== 콜백 (학습 제어) =====
os.makedirs(MODEL_DIR, exist_ok=True)

checkpoint = callbacks.ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, "best_model.h5"),
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)
earlystop = callbacks.EarlyStopping(
    monitor=MONITOR_METRIC, 
    patience=EARLY_STOP_PATIENCE,
    restore_best_weights=True,
    verbose=1
)
reduce_lr = callbacks.ReduceLROnPlateau(   # ⬅ 러닝레이트 자동 조정
    monitor="val_loss",
    factor=0.5,
    patience=5,
    verbose=1,
    min_lr=1e-7
)

# ===== Step1: 학습 시작 (Freeze) =====
print("\n Step1: base_model Freeze 상태 학습 시작...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,   # 10 에포크까지만 학습 후 Fine-tuning 전환
    callbacks=[checkpoint, earlystop, reduce_lr],
    steps_per_epoch=train_gen.samples // BATCH_SIZE, 
    validation_steps=val_gen.samples // BATCH_SIZE
)

# ===== Step2: Fine-tuning =====
print("\n Step2: Fine-tuning 시작 (상위 레이어 일부 해제)...")

base_model.trainable = True
fine_tune_at = 100  # 마지막 100개 레이어만 학습

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # ⬅ 낮은 학습률로 재컴파일
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_finetune = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,  # 나머지 epoch 계속
    callbacks=[checkpoint, earlystop, reduce_lr],
    steps_per_epoch=train_gen.samples // BATCH_SIZE,
    validation_steps=val_gen.samples // BATCH_SIZE
)

# ===== 모델 및 클래스 정보 저장 =====
model.save(os.path.join(MODEL_DIR, "final_saved_model"))
with open(os.path.join(MODEL_DIR, "class_indices.json"), "w") as f:
    json.dump(train_gen.class_indices, f)

print(f"\n 학습 완료! 모델 및 클래스 정보가 저장됨: {MODEL_DIR}")
