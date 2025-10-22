from keras.applications.efficientnet_v2 import preprocess_input
import keras
import os
import json

MODEL_PATH = "/Users/nakrlove/Desktop/dev/aix/aix_final_prj/keras/trash_classifier_efficientnetv2_best_final.keras"
CLASS_NAMES_PATH = "/Users/nakrlove/Desktop/dev/aix/aix_final_prj/keras/class_names.json"

_MODEL = None
_CLASS_NAMES = None

def load_model_and_classes():
    global _MODEL, _CLASS_NAMES

    if _MODEL is None:
        try:
            keras.config.enable_unsafe_deserialization()

            # preprocess_input을 Keras에 시리얼라이즈 가능한 함수로 등록
            @keras.saving.register_keras_serializable()
            def preprocess_input_serializable(x):
                return preprocess_input(x)

            # 등록된 함수를 custom_objects에 넣어야 Lambda 레이어가 deserialize됨
            _MODEL = keras.models.load_model(
                MODEL_PATH,
                compile=False,
                custom_objects={"preprocess_input": preprocess_input_serializable}
            )
            print("✅ 모델 로드 완료")
        except Exception as e:
            print("❌ 모델 로드 실패:", e)
            _MODEL = None

    if _CLASS_NAMES is None:
        if os.path.exists(CLASS_NAMES_PATH):
            with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
                _CLASS_NAMES = json.load(f)
            print("✅ Class names 로드 완료")
        else:
            _CLASS_NAMES = [f"class_{i}" for i in range(_MODEL.output_shape[-1])]
            print("⚠️ class_names.json 없음, 인덱스로 대체")

    return _MODEL, _CLASS_NAMES

# --------------------------
# 테스트 실행
# --------------------------
if __name__ == "__main__":
    print("main called -=====")
    model, class_names = load_model_and_classes()
    if model is not None:
        print("모델 summary:")
        model.summary()
    if class_names is not None:
        print(f"\n클래스 수: {len(class_names)}")
        print("첫 10개 클래스:", class_names[:10])
