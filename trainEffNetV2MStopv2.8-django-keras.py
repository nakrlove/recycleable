# app/ml_utils/model_loader.py
import tensorflow as tf

MODEL_PATH = "trained_model_EffNetV2M_v3.0.keras"
CLASS_PATH = "class_names.txt"

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, "r", encoding="utf-8") as f:
    CLASS_NAMES = [line.strip() for line in f.readlines()]



from django.http import JsonResponse
from .ml_utils.model_loader import model, CLASS_NAMES
import tensorflow as tf

def predict_image(request):
    img_path = request.GET.get("img")
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # batch dimension

    preds = model.predict(img_array)
    predicted_class = CLASS_NAMES[tf.argmax(preds[0])]
    confidence = tf.reduce_max(preds[0]).numpy()

    return JsonResponse({
        "predicted_class": predicted_class,
        "confidence": float(confidence)
    })

