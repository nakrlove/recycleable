# =====================================
# 🧠 Colab 세션 유지 코드 (업그레이드 버전)
# =====================================
from IPython.display import Javascript, display

def keep_colab_alive(interval_min=15):
    js_code = f"""
    async function ClickConnect() {{
        console.log("[KeepAlive] Colab 세션 유지 중...");
        const btn = document.querySelector("colab-connect-button") 
                 || document.querySelector("#connect") 
                 || document.querySelector("colab-toolbar-button#connect") 
                 || document.querySelector("colab-sessions-button") 
                 || document.querySelector("#top-toolbar colab-connect-button");
        if (btn) {{
            btn.click();
            console.log("✅ 연결 유지 버튼 클릭 완료");
        }} else {{
            console.log("⚠️ Colab 연결 버튼을 찾을 수 없습니다. (UI 변경 가능성)");
        }}
    }}
    setInterval(ClickConnect, {interval_min} * 60 * 1000);
    console.log("✅ Colab Keep-Alive 활성화됨 — {interval_min}분 간격으로 클릭 중");
    """
    try:
        display(Javascript(js_code))
    except Exception as e:
        print("⚠️ Colab 환경이 아닙니다. 세션 유지 스크립트는 무시됩니다.")
        print(e)

keep_colab_alive(30)

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2M
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ================================
# 설정
# ================================
split_dir = "dataset_split"  # train/val/test 포함 폴더
img_size = (224, 224)        # 이미 224x224라 resize 불필요
batch_size = 32
epochs = 25
model_save_path = "recycle_classifier_v2m_finetuned.keras"

# ================================
# 데이터 디렉토리
# ================================
train_dir = Path(split_dir) / "train"
val_dir   = Path(split_dir) / "val"
test_dir  = Path(split_dir) / "test"

# ================================
# 클래스 탐색
# ================================
classes = [d.name for d in train_dir.iterdir() if d.is_dir()]
class_indices = {cls: i for i, cls in enumerate(classes)}

# ================================
# 클래스별 이미지 경로 수집
# ================================
class_files = {}
for cls in classes:
    files = list((train_dir / cls).glob("*"))
    class_files[cls] = [str(f) for f in files]

num_classes = len(classes)
max_count = max(len(v) for v in class_files.values())

# ================================
# 학습 시점 oversampling (tf.data)
# ================================
AUTOTUNE = tf.data.AUTOTUNE

def preprocess_image(file_path, label):
    # 이미 224x224로 되어 있어 resize/전처리 제거
    img = tf.io.read_file(file_path)
    img = tf.image.decode_image(img, channels=3)
    img.set_shape([224, 224, 3])
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

per_class_datasets = []
for i, cls in enumerate(classes):
    paths = class_files[cls]
    ds = tf.data.Dataset.from_tensor_slices((paths, [i]*len(paths)))
    ds = ds.shuffle(len(paths)).repeat() \
           .map(preprocess_image, num_parallel_calls=AUTOTUNE)
    per_class_datasets.append(ds)

# sample_from_datasets로 균형 샘플링
train_ds = tf.data.experimental.sample_from_datasets(per_class_datasets, weights=[1/num_classes]*num_classes)
train_ds = train_ds.batch(batch_size).prefetch(AUTOTUNE)

# ================================
# validation/test dataset (deterministic)
# ================================
def make_eval_ds(dir_path):
    images, labels = [], []
    for i, cls in enumerate(classes):
        for p in (dir_path / cls).glob("*"):
            images.append(str(p))
            labels.append(i)
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    ds = ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
    return ds.batch(batch_size).prefetch(AUTOTUNE)

val_ds = make_eval_ds(val_dir)
test_ds = make_eval_ds(test_dir)

# ================================
# 모델 구성
# ================================
base_model = EfficientNetV2M(weights="imagenet", include_top=False, input_shape=(224,224,3))
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.3)(x)
output = Dense(num_classes, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer=Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# ================================
# 콜백
# ================================
callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
    ModelCheckpoint(model_save_path, monitor="val_accuracy", save_best_only=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-6)
]

# ================================
# 1단계: Feature Extractor 학습
# ================================
history_stage1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    steps_per_epoch=(max_count*num_classes)//batch_size,
    callbacks=callbacks
)

# ================================
# 2단계: Fine-Tuning
# ================================
for layer in base_model.layers[-40:]:
    layer.trainable = True

model.compile(optimizer=Adam(1e-5), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history_stage2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    steps_per_epoch=(max_count*num_classes)//batch_size,
    callbacks=callbacks
)

# ================================
# 학습 결과 시각화
# ================================
def plot_history(hist1, hist2):
    acc = hist1.history["accuracy"] + hist2.history["accuracy"]
    val_acc = hist1.history["val_accuracy"] + hist2.history["val_accuracy"]
    loss = hist1.history["loss"] + hist2.history["loss"]
    val_loss = hist1.history["val_loss"] + hist2.history["val_loss"]

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(acc, label="train_acc")
    plt.plot(val_acc, label="val_acc")
    plt.title("Accuracy")
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(loss, label="train_loss")
    plt.plot(val_loss, label="val_loss")
    plt.title("Loss")
    plt.legend()
    plt.show()

plot_history(history_stage1, history_stage2)

# ================================
# 모델 저장
# ================================
model.save(model_save_path, save_format="keras")
print("✅ Fine-Tuned 모델 저장 완료:", model_save_path)


# ================================
# 분류 함수
# ================================
# def classify_image(model, image_path, threshold=0.5):
#     img = tf.io.read_file(image_path)
#     img = tf.image.decode_jpeg(img, channels=3)
#     img = tf.cast(img, tf.float32) / 255.0
#     img = tf.expand_dims(img, axis=0)

#     preds = model.predict(img)
#     class_id = np.argmax(preds[0])
#     confidence = preds[0][class_id]

#     if confidence < threshold:
#         return "일반쓰레기"
#     else:
#         return classes[class_id]

def classify_image(model, image_path, classes, threshold=0.5):
    """
    학습용 코드(옵션1) 기반으로 안전하게 이미지 분류
    Args:
        model: 학습된 Keras 모델
        image_path: 분류할 이미지 경로
        classes: 학습 시 사용된 클래스 리스트
        threshold: confidence 임계값
    Returns:
        class_name 또는 "일반쓰레기"
    """
    # 이미지 읽기 및 디코딩 (JPEG, PNG 모두 가능)
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    
    # 학습용 코드 기준: 이미 224x224이므로 resize 불필요
    img.set_shape([224, 224, 3])
    
    # 정규화
    img = tf.cast(img, tf.float32) / 255.0
    
    # 배치 차원 추가
    img = tf.expand_dims(img, axis=0)
    
    # 예측
    preds = model.predict(img)
    class_id = np.argmax(preds[0])
    confidence = preds[0][class_id]
    
    # threshold 기준 판정
    if confidence < threshold:
        return "일반쓰레기"
    else:
        return classes[class_id], float(confidence)

# 예시
# result = classify_image(model, "sample.jpg")
# print("분류 결과:", result)
