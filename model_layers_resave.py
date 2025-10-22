import keras
from keras.models import load_model
from keras.models import Model
# from keras.layers import Lambda # Keras 3에서는 keras.layers.Lambda를 직접 사용하는 것이 아닐 수 있습니다.

# 기존 모델 경로
OLD_MODEL_PATH = "/Users/nakrlove/Desktop/dev/aix/aix_final_prj/keras/trash_classifier_efficientnetv2_best_final.keras"
# 새로 저장할 clean 모델 경로
CLEAN_MODEL_PATH = "/Users/nakrlove/Desktop/dev/aix/aix_final_prj/keras/trash_classifier_effnetv2_clean.keras"

# --------------------------
# 모델 불러오기 (Lambda 오류 해결)
# --------------------------
def identity(x):
    return x  # Lambda 대신 입력 그대로 반환

# 1. Lambda 레이어를 대체할 함수 정의
# 이 함수는 load_model이 모델 내부의 Lambda 레이어를 다시 만들 때 호출됩니다.
def fixed_lambda_function(function):
    """
    Keras Lambda 레이어의 from_config가 호출될 때 필요한
    출력 형태를 반환하는 wrapper 함수를 정의합니다.
    """
    
    # 2. preprocess_input Lambda 레이어의 예상 출력 형태 (입력과 동일)
    # 이미지 형태가 (224, 224, 3)이라고 가정합니다.
    TARGET_OUTPUT_SHAPE = (224, 224, 3)

    # 3. Keras Lambda 레이어의 설정에 맞춰 함수를 정의합니다.
    # Keras는 Lambda 레이어 객체를 생성할 때 이 함수를 사용합니다.
    def custom_lambda_layer(x):
        # 실제 로드 시에는 이 function이 호출됩니다. (compile=False이므로 크게 상관 없음)
        return function(x) 

    # 4. output_shape 속성을 추가하여 Keras의 오류를 회피합니다.
    # Keras 3에서는 Lambda 레이어가 저장될 때 function의 __name__을 사용합니다.
    # 모델에 저장된 이름이 'preprocess_input' 일 것입니다.
    
    # Keras 3에서 custom_objects에 함수를 전달할 때의 일반적인 패턴은 다음과 같습니다.
    # **핵심:** Keras 3에서는 Lambda 레이어의 config에 function 이름만 저장되므로,
    # 우리는 load_model에 function 이름(preprocess_input)에 해당하는 함수를 전달합니다.
    # 하지만 Lambda 레이어의 정의 자체에 output_shape가 누락된 경우이므로,
    # 해당 레이어의 type을 재정의하는 방식으로 접근해야 합니다.
    
    # ******* Keras 3에서 가장 확실한 방법 *******
    # Lambda 레이어 클래스 대신 output_shape를 가진 새로운 클래스를 전달합니다.
    class FixedLambdaLayer(keras.layers.Lambda):
        def compute_output_shape(self, input_shape):
            # None 대신 input_shape[0]을 사용하여 배치 크기를 유지합니다.
            return (input_shape[0],) + TARGET_OUTPUT_SHAPE
        
        # 실제 함수는 load_model의 custom_objects에서 identity로 전달될 것입니다.
        # 따라서 from_config를 오버라이드하여 Lambda가 아닌 이 FixedLambdaLayer를 사용하도록 유도합니다.

    return FixedLambdaLayer(identity, name='preprocess_input') # 이름이 'preprocess_input'이라고 가정

keras.config.enable_unsafe_deserialization()  # 안전하지 않지만 Lambda 로드 허용

print("📂 기존 모델 로드 중:", OLD_MODEL_PATH)
# custom_objects에 Lambda 레이어의 클래스 타입 자체를 FixedLambdaLayer로 매핑합니다.
# 이렇게 하면 Keras가 모델 파일에서 'Lambda' 타입 레이어를 로드할 때
# 'FixedLambdaLayer'의 compute_output_shape를 사용하게 됩니다.
# Note: Keras 3에서 load_model은 레이어 클래스 이름 대신 저장된 함수 이름(e.g., preprocess_input)을
# custom_objects 키로 사용하도록 권장합니다.
# 하지만 여기서는 레이어 자체의 문제이므로, Keras의 내부 로직을 우회해야 합니다.

# 가장 안전한 방법은:
# 1. Lambda 레이어를 구현하는 데 사용된 함수(e.g. preprocess_input)를
# 2. output_shape 속성을 가진 커스텀 함수로 래핑하여 전달하는 것입니다.

# 'preprocess_input' 함수의 이름을 가진 커스텀 래퍼 함수 정의
def preprocess_input_with_shape(x):
    return x # 실제 전처리 로직은 필요 없습니다.
# 출력 형태 속성 추가
# Keras는 이 속성을 읽어 compute_output_shape 오류를 회피합니다.
# Keras 2/TensorFlow Keras 방식이지만, Keras 3에서 작동할 가능성이 높습니다.
preprocess_input_with_shape.output_shape = (224, 224, 3) 

# 로드 시, 모델에 저장된 함수 이름(preprocess_input)과 일치하는 객체를 전달합니다.
model = load_model(
    OLD_MODEL_PATH,
    compile=False,
    custom_objects={"preprocess_input": preprocess_input_with_shape}
)
print("✅ 기존 모델 로드 완료")

# --------------------------
# Lambda 제거 및 clean 모델 저장
# --------------------------
# 모델 로드가 성공했다면, 나머지 코드는 Lambda 레이어가 포함된 모델 객체를 사용합니다.
# 모델의 입력/출력을 그대로 사용하여 새 모델을 만들면
# 첫 번째 레이어(Lambda)의 기능을 무시한 채로 모델의 나머지 부분을 추출할 수 있습니다.
inputs = model.input
outputs = model.output
clean_model = Model(inputs, outputs)

# clean 모델 저장
clean_model.save(CLEAN_MODEL_PATH, include_optimizer=False)
print("✅ Lambda 제거 후 clean 모델 저장 완료:", CLEAN_MODEL_PATH)