import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import efficientnet_v2

# 모델 로드
model = load_model("EffNetV2S_final.keras", compile=False)
class_names = [...]  # 학습 시 train_ds.class_names 그대로 입력

def predict_image(img_path, top_k=3):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = efficientnet_v2.preprocess_input(img_array)

    preds = model.predict(img_array)
    top_indices = preds[0].argsort()[-top_k:][::-1]
    top_probs = preds[0][top_indices]
    top_classes = [class_names[i] for i in top_indices]

    return list(zip(top_classes, top_probs))

# 사용 예시
result = predict_image("sample.jpg")
print("Top predictions:", result)
