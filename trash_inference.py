# ============================================================
# trash_inference.py
# ============================================================

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import EfficientNetV2M

# ============================================================
# 환경 설정
# ============================================================
IMAGE_SIZE = (224, 224)
MODEL_PATH = 'trash_classifier_efficientnetv2_best_final.keras'
CLASS_NAMES_PATH = 'class_names.json'

# ============================================================
# 1. 학습 완료 모델 로드
# ============================================================
# EfficientNetV2M 기반 모델 구조 정의
def build_efficientnet_model(input_shape, num_classes):
    base_model = EfficientNetV2M(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = True  # fine-tuned 상태
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
    return model

# 클래스 이름 로드
with open(CLASS_NAMES_PATH, 'r', encoding='utf-8') as f:
    class_names = json.load(f)

NUM_CLASSES = len(class_names)

# 모델 구조 정의 후 학습 완료 가중치 로드
model = build_efficientnet_model(IMAGE_SIZE + (3,), NUM_CLASSES)
model.load_weights(MODEL_PATH)
print("✅ 학습 완료 모델 로드 완료")

# ============================================================
# 2. 분리수거 가이드 맵
# ============================================================
TRASH_GUIDE_MAP = {
    'steel_can1': {'category': '캔류', 'action': '내용물 비우고 압착 후 배출'},
    'aluminum_can1': {'category': '캔류', 'action': '내용물 비우고 압착 후 배출'},
    'paper1': {'category': '종이류', 'action': '물기에 젖지 않도록 펴서 묶어 배출'},
    'pet_clear_single1': {'category': '플라스틱', 'action': '내용물 비우고 라벨 제거 후 압착 배출'},
    'battery': {'category': '특수 폐기물', 'action': '폐건전지 수거함에 배출 (분리수거 아님)'},
    'fluorescent_lamp': {'category': '특수 폐기물', 'action': '전용 수거함에 배출 (분리수거 아님)'},
}

# ============================================================
# 3. Django에서 호출할 예측 함수
# ============================================================
def predict_trash(image_path, threshold=0.9):
    """
    단일 이미지 예측 및 분리수거 안내 반환
    """
    img = tf.keras.utils.load_img(image_path, target_size=IMAGE_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)

    predictions = model.predict(preprocessed_img, verbose=0)[0]
    max_prob = float(np.max(predictions))
    predicted_index = int(np.argmax(predictions))
    predicted_class = class_names[predicted_index]

    # 예측 메시지
    if max_prob >= threshold:
        result_message = f"🟢 [확정]: {predicted_class}로 분류되었습니다."
        confidence_level = "높음"
    else:
        result_message = f"🟡 [불확실]: {predicted_class}로 예측되지만, 신뢰도({max_prob*100:.2f}%)가 낮아 재확인이 필요합니다."
        confidence_level = "낮음"

    # Top-3 예측
    top_3_indices = np.argsort(predictions)[::-1][:3]
    top_3_results = [f"{class_names[i]} ({predictions[i]*100:.2f}%)" for i in top_3_indices]

    # 분리수거 안내
    if predicted_class in TRASH_GUIDE_MAP:
        guide = TRASH_GUIDE_MAP[predicted_class]
        guide_msg = f"👉 분리수거 카테고리: {guide['category']}, 📝 배출 방법: {guide['action']}"
    else:
        guide_msg = "해당 쓰레기에 대한 구체적인 분리수거 가이드가 없습니다."

    return {
        "predicted_class": predicted_class,
        "max_prob": max_prob,
        "result_message": result_message,
        "confidence_level": confidence_level,
        "top_3_results": top_3_results,
        "guide_message": guide_msg
    }
