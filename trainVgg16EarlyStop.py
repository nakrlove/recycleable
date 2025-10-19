import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback

# ============================================================
# 0. CPU 설정 (M1/M2 Mac 안정화)
# ============================================================
USE_GPU = False  # CPU만 사용
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
print("CPU 모드로 학습합니다.")

# ============================================================
# 1. 환경 설정 및 하이퍼파라미터
# ============================================================
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
DATASET_PATH = 'trash_dataset_path'
MODEL_NAME = 'trash_classifier_vgg16'

LEARNING_RATE_STEP1 = 1e-3
LEARNING_RATE_STEP2 = 1e-5

# ============================================================
# 2. 데이터 준비 및 전처리
# ============================================================
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input
)

val_test_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'train'),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

validation_generator = val_test_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'val'),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

class_names = list(train_generator.class_indices.keys())
NUM_CLASSES = len(class_names)
print(f"총 클래스 수: {NUM_CLASSES}, 클래스 이름: {class_names}")

# ============================================================
# 3. 모델 구축 (VGG16)
# ============================================================
def build_vgg16_model(input_shape, num_classes):
    base_model = VGG16(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Step1: Head Layer만 학습

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model

model, base_model = build_vgg16_model(IMAGE_SIZE + (3,), NUM_CLASSES)
# model.summary()

# ============================================================
# 4. 커스텀 콜백: 연속 반복 정확도 변화 없으면 조기 종료
# ============================================================
class RepeatEarlyStopping(Callback):
    def __init__(self, monitor='val_accuracy', patience=3):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.wait = 0
        self.prev_value = None

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if self.prev_value is not None and current == self.prev_value:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"\n🔹 {self.monitor} 값이 {self.patience}번 연속 변화 없으므로 학습 종료")
                self.model.stop_training = True
        else:
            self.wait = 0
        self.prev_value = current

# ============================================================
# 5. 학습 절차
# ============================================================

# ---------------- Step1: Head Layer 학습 ----------------
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    RepeatEarlyStopping(monitor='val_accuracy', patience=3),  # 추가
    ModelCheckpoint(f'{MODEL_NAME}_best_step1.keras', monitor='val_accuracy', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
]

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_STEP1, amsgrad=False),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_step1 = model.fit(
    train_generator,
    epochs=1,
    validation_data=validation_generator,
    callbacks=callbacks
)

model.load_weights(f'{MODEL_NAME}_best_step1.keras')

# ---------------- Step2: Fine-tuning ----------------
num_layers_to_train = int(len(base_model.layers) * 0.3)
for layer in base_model.layers[-num_layers_to_train:]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_STEP2, amsgrad=False),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_ft = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    RepeatEarlyStopping(monitor='val_accuracy', patience=3),  # 추가
    ModelCheckpoint(f'{MODEL_NAME}_best_final.keras', monitor='val_accuracy', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, min_lr=1e-7, verbose=1)
]

history_step2 = model.fit(
    train_generator,
    epochs=1,
    validation_data=validation_generator,
    callbacks=callbacks_ft
)

model.load_weights(f'{MODEL_NAME}_best_final.keras')
print(f"\n 최종 모델 가중치가 '{MODEL_NAME}_best_final.keras'에 저장되었습니다.")

# ============================================================
# 6. Inference 및 분리수거 가이드
# ============================================================
def predict_trash_type(model, image_path, class_names, threshold=0.9):
    img = tf.keras.utils.load_img(image_path, target_size=IMAGE_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = tf.keras.applications.vgg16.preprocess_input(img_array)

    predictions = model.predict(preprocessed_img, verbose=0)[0]
    max_prob = np.max(predictions)
    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]

    if max_prob >= threshold:
        result_message = f"[확정]: {predicted_class}로 분류되었습니다."
        confidence_level = "높음"
    else:
        result_message = f"[불확실]: {predicted_class}로 예측되지만, 신뢰도({max_prob*100:.2f}%)가 낮아 재확인이 필요합니다."
        confidence_level = "낮음"

    top_3_indices = np.argsort(predictions)[::-1][:3]
    top_3_results = [f"{class_names[i]} ({predictions[i]*100:.2f}%)" for i in top_3_indices]

    return predicted_class, max_prob, result_message, confidence_level, top_3_results

TRASH_GUIDE_MAP = {
    'steel_can1': {'category': '캔류', 'action': '내용물 비우고 압착 후 배출'},
    'aluminum_can1': {'category': '캔류', 'action': '내용물 비우고 압착 후 배출'},
    'paper1': {'category': '종이류', 'action': '물기에 젖지 않도록 펴서 묶어 배출'},
    'pet_clear_single1': {'category': '플라스틱', 'action': '내용물 비우고 라벨 제거 후 압착 배출'},
    'battery': {'category': '특수 폐기물', 'action': '폐건전지 수거함에 배출 (분리수거 아님)'},
    'fluorescent_lamp': {'category': '특수 폐기물', 'action': '전용 수거함에 배출 (분리수거 아님)'},
}

def get_recycling_guidance(predicted_class, mapping_table):
    if predicted_class in mapping_table:
        guide = mapping_table[predicted_class]
        return f"분리수거 카테고리: {guide['category']}, 배출 방법: {guide['action']}"
    return "해당 쓰레기에 대한 구체적인 분리수거 가이드가 없습니다."
