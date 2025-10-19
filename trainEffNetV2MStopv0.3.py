import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2M
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
import json

# ============================================================
# 0. CPU/GPU 설정
# ============================================================
USE_GPU = False  # True: GPU 사용, False: CPU 강제

if USE_GPU:
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
    print("GPU 모드 사용")
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print("CPU 모드 사용")

print("Physical devices:", tf.config.list_physical_devices('GPU'))

# ============================================================
# 1. 환경 설정
# ============================================================
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16 if USE_GPU else 32
DATASET_PATH = 'trash_dataset_path/train'  # train 폴더만 있으면 됨
MODEL_NAME = 'trash_classifier_efficientnetv2'

LEARNING_RATE_STEP1 = 1e-3
LEARNING_RATE_STEP2 = 1e-5
SEED = 42

# ============================================================
# 2. tf.data.Dataset으로 train / val / test 분리
# 9/3일오후 20:00 데이터 자르는 설명있음
# ============================================================

# 전체 dataset에서 validation_split 비율 적용
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,  # train: 80%, val+test: 20%
    subset='training',
    seed=SEED
)

val_test_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset='validation',
    seed=SEED
)

# val_test_dataset에서 val/test로 나누기 (50%씩)
val_batches = int(0.5 * tf.data.experimental.cardinality(val_test_dataset).numpy())
val_dataset = val_test_dataset.take(val_batches)
test_dataset = val_test_dataset.skip(val_batches)

print("✅ Dataset 준비 완료")
print(f"Train batches: {tf.data.experimental.cardinality(train_dataset).numpy()}")
print(f"Val batches: {tf.data.experimental.cardinality(val_dataset).numpy()}")
print(f"Test batches: {tf.data.experimental.cardinality(test_dataset).numpy()}")

# ============================================================
# 2-1. 클래스 이름 저장
# ============================================================
class_names = train_dataset.class_names
NUM_CLASSES = len(class_names)
with open('class_names.json', 'w', encoding='utf-8') as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)
print(f"총 클래스 수: {NUM_CLASSES}, 클래스 이름: {class_names}")
print("✅ class_names.json 파일 생성 완료")

# ============================================================
# 3. 데이터 전처리 (EfficientNetV2)
# ============================================================
def preprocess_dataset(dataset):
    return dataset.map(lambda x, y: (tf.keras.applications.efficientnet_v2.preprocess_input(x), y))

train_dataset = preprocess_dataset(train_dataset).prefetch(tf.data.AUTOTUNE)
val_dataset = preprocess_dataset(val_dataset).prefetch(tf.data.AUTOTUNE)
test_dataset = preprocess_dataset(test_dataset).prefetch(tf.data.AUTOTUNE)

# ============================================================
# 4. 모델 구축
# ============================================================
def build_efficientnet_model(input_shape, num_classes):
    base_model = EfficientNetV2M(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x) # Dropout 강화  9/3일오후 영상에서 01:07분 설명중  책431 
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model

model, base_model = build_efficientnet_model(IMAGE_SIZE + (3,), NUM_CLASSES)
model.summary()

# ============================================================
# 4-1. 커스텀 콜백: val_accuracy와 val_loss 모두 안정적이면 학습 중단
# ============================================================
class EarlyStopOnStableMetrics(Callback):
    def __init__(self, monitor_acc='val_accuracy', monitor_loss='val_loss', patience=3, delta_acc=1e-4, delta_loss=1e-4):
        super().__init__()
        self.monitor_acc = monitor_acc
        self.monitor_loss = monitor_loss
        self.patience = patience
        self.delta_acc = delta_acc
        self.delta_loss = delta_loss
        self.acc_history = []
        self.loss_history = []

    def on_epoch_end(self, epoch, logs=None):
        current_acc = logs.get(self.monitor_acc)
        current_loss = logs.get(self.monitor_loss)
        if current_acc is None or current_loss is None:
            return

        self.acc_history.append(current_acc)
        self.loss_history.append(current_loss)

        if len(self.acc_history) >= self.patience:
            recent_acc = self.acc_history[-self.patience:]
            recent_loss = self.loss_history[-self.patience:]
            acc_stable = max(recent_acc) - min(recent_acc) < self.delta_acc
            loss_stable = max(recent_loss) - min(recent_loss) < self.delta_loss
            if acc_stable and loss_stable:
                print(f"\n⚠️ {self.patience}번 연속 val_accuracy와 val_loss가 안정적이므로 학습 중단")
                self.model.stop_training = True

# ============================================================
# 5. 학습 (Step1: Head layer)
# early-stopping 조기종료. patience=5, patience=3
# 9/3일오후 53:00분 강의 영상   책437
# ============================================================
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint(f'{MODEL_NAME}_best_step1.keras', monitor='val_accuracy', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    EarlyStopOnStableMetrics(monitor_acc='val_accuracy', monitor_loss='val_loss', patience=3)
]

# ⚡ SparseCategoricalCrossentropy 사용 (정수 레이블)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_STEP1, amsgrad=False),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

history_step1 = model.fit(
    train_dataset,
    epochs=30,
    validation_data=val_dataset,
    callbacks=callbacks
)

model.load_weights(f'{MODEL_NAME}_best_step1.keras')
print("# ============================================================")
print("# Step2: Fine-tuning")
print("# ============================================================")
# ============================================================
# Step2: Fine-tuning
# ============================================================

num_layers_to_train = int(len(base_model.layers) * 0.3)
for layer in base_model.layers[-num_layers_to_train:]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_STEP2, amsgrad=False),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

callbacks_ft = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint(f'{MODEL_NAME}_best_final.keras', monitor='val_accuracy', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, min_lr=1e-7, verbose=1),
    EarlyStopOnStableMetrics(monitor_acc='val_accuracy', monitor_loss='val_loss', patience=3)
]

history_step2 = model.fit(
    train_dataset,
    epochs=50,
    validation_data=val_dataset,
    callbacks=callbacks_ft
)

model.load_weights(f'{MODEL_NAME}_best_final.keras')
print(f"\n✅ 최종 모델 가중치가 '{MODEL_NAME}_best_final.keras'에 저장되었습니다.")

# ============================================================
# 6. Test 데이터 평가
# ============================================================
test_loss, test_acc = model.evaluate(test_dataset)
print(f"테스트 정확도: {test_acc*100:.2f}%")
