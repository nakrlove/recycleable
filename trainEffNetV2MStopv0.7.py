import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import threading, time, json
from tensorflow.keras.applications import EfficientNetV2M
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from sklearn.utils.class_weight import compute_class_weight

# ============================================================
# 🔹 Colab 세션 자동유지 스레드
# ============================================================
class KeepAliveThread:
    def __init__(self, interval=300):  # 5분마다 실행
        self.interval = interval
        self.running = False
        self.thread = None

    def _keep_alive(self):
        while self.running:
            time.sleep(self.interval)
            print("[KeepAlive] Colab 세션 유지 중...")

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._keep_alive, daemon=True)
            self.thread.start()
            print("[KeepAlive] 백그라운드 세션 유지 시작됨")

    def stop(self):
        if self.running:
            self.running = False
            print("[KeepAlive] 세션 유지 스레드 중단됨")

keepalive = KeepAliveThread(interval=300)
keepalive.start()

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
# ============================================================
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
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

# val/test 50%씩 나누기
val_batches = int(0.5 * tf.data.experimental.cardinality(val_test_dataset).numpy())
val_dataset = val_test_dataset.take(val_batches)
test_dataset = val_test_dataset.skip(val_batches)

print("✅ Dataset 준비 완료")
print(f"Train batches: {tf.data.experimental.cardinality(train_dataset).numpy()}")
print(f"Val batches: {tf.data.experimental.cardinality(val_dataset).numpy()}")
print(f"Test batches: {tf.data.experimental.cardinality(test_dataset).numpy()}")

# ============================================================
# 2-1. 클래스 이름 저장 및 Class Weight 계산
# ============================================================
class_names = train_dataset.class_names
NUM_CLASSES = len(class_names)
with open('class_names.json', 'w', encoding='utf-8') as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)
print(f"총 클래스 수: {NUM_CLASSES}, 클래스 이름: {class_names}")
print("✅ class_names.json 파일 생성 완료")

# Class weight 계산
labels = np.concatenate([y.numpy() for x, y in train_dataset], axis=0)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(NUM_CLASSES),
    y=labels
)
class_weight_dict = dict(enumerate(class_weights))
print("✅ Class weight 계산 완료:", class_weight_dict)

# ============================================================
# 3. 데이터 전처리 + 증강
# ============================================================
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

def preprocess_dataset(dataset, training=False):
    dataset = dataset.map(lambda x, y: (tf.keras.applications.efficientnet_v2.preprocess_input(x), y))
    if training:
        dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y))
    return dataset.prefetch(tf.data.AUTOTUNE)

train_dataset = preprocess_dataset(train_dataset, training=True)
val_dataset = preprocess_dataset(val_dataset)
test_dataset = preprocess_dataset(test_dataset)

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
    x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = Dropout(0.4)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model

model, base_model = build_efficientnet_model(IMAGE_SIZE + (3,), NUM_CLASSES)
model.summary()

# ============================================================
# 4-1. 커스텀 콜백
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
# 5. Step1: Head layer 학습
# ============================================================
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint(f'{MODEL_NAME}_best_step1.keras', monitor='val_accuracy', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    EarlyStopOnStableMetrics(monitor_acc='val_accuracy', monitor_loss='val_loss', patience=3)
]

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_STEP1),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

try:
    history_step1 = model.fit(
        train_dataset,
        epochs=30,
        validation_data=val_dataset,
        callbacks=callbacks,
        class_weight=class_weight_dict
    )

    model.load_weights(f'{MODEL_NAME}_best_step1.keras')

    # ============================================================
    # Step2: Fine-tuning
    # ============================================================
    num_layers_to_train = int(len(base_model.layers) * 0.3)
    for layer in base_model.layers[-num_layers_to_train:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_STEP2),
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
        callbacks=callbacks_ft,
        class_weight=class_weight_dict
    )

    model.load_weights(f'{MODEL_NAME}_best_final.keras')
    print(f"\n✅ 최종 모델 가중치가 '{MODEL_NAME}_best_final.keras'에 저장되었습니다.")

    # ============================================================
    # 6. Test 데이터 평가
    # ============================================================
    test_loss, test_acc = model.evaluate(test_dataset)
    print(f"테스트 정확도: {test_acc*100:.2f}%")

    # ============================================================
    # 7. 학습 곡선 시각화
    # ============================================================
    def plot_history(history1, history2=None):
        acc = history1.history['accuracy']
        val_acc = history1.history['val_accuracy']
        loss = history1.history['loss']
        val_loss = history1.history['val_loss']

        if history2:
            acc += history2.history['accuracy']
            val_acc += history2.history['val_accuracy']
            loss += history2.history['loss']
            val_loss += history2.history['val_loss']

        epochs = range(1, len(acc) + 1)

        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, acc, 'b-', label='Training acc')
        plt.plot(epochs, val_acc, 'r-', label='Validation acc')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, loss, 'b-', label='Training loss')
        plt.plot(epochs, val_loss, 'r-', label='Validation loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()

    plot_history(history_step1, history_step2)

finally:
    keepalive.stop()
    print("✅ 학습 종료 및 Colab 세션 유지 스레드 중단 완료")
