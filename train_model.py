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
EPOCHS = 50           # 최대 50 에포크까지 훈련

ORIGINAL_DATA_DIR = "dataset/train"
SPLIT_DATA_DIR = "split_dataset"

# 2. 데이터 준비 함수 호출
DATA_DIR = prepare_data.prepare_and_split_data(
    input_dir=ORIGINAL_DATA_DIR,  # 인자명 소문자로 통일
    output_dir=SPLIT_DATA_DIR,    # 인자명 소문자로 통일
    ratio=(0.8, 0.2, 0.0)
)

# 🚨 None 검사 및 종료: DATA_DIR이 None이면 데이터 로드 전 종료
if DATA_DIR is None:
    print("\n🚨 데이터 준비 과정에서 오류가 발생했거나 원본 이미지('dataset/train')가 없습니다. 학습을 시작할 수 없습니다.")
    exit()

MODEL_DIR = "saved_model_recycle"

# 조기 종료(Early Stopping) 설정
EARLY_STOP_PATIENCE = 10 
MONITOR_METRIC = "val_loss" 

# ===== 데이터 제너레이터 (이미지 증강 및 전처리) =====
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
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
base_model.trainable = False 

inputs = layers.Input(shape=(*IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = layers.Dropout(0.4)(x) 
x = layers.Dense(512, activation="relu")(x) 
x = layers.BatchNormalization()(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs)

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

# ===== 학습 시작 =====
print("\n 모델 학습 시작...")
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[checkpoint, earlystop],
    steps_per_epoch=train_gen.samples // BATCH_SIZE, 
    validation_steps=val_gen.samples // BATCH_SIZE
)

# ===== 모델 및 클래스 정보 저장 =====
model.save(os.path.join(MODEL_DIR, "final_saved_model"))
with open(os.path.join(MODEL_DIR, "class_indices.json"), "w") as f:
    json.dump(train_gen.class_indices, f)

print(f"\n 학습 완료! 모델 및 클래스 정보가 저장됨: {MODEL_DIR}")