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


# =====================================
# ⚙️ EfficientNetV2M Fine-Tuning 학습 파이프라인
# =====================================
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2M
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os

# ================================
# 설정
# ================================
split_dir = "dataset_sp"  # train/val/test 포함 폴더
img_size = (224, 224)  # 입력 크기 정의만, resize는 하지 않음
batch_size = 32
epochs = 25
base_drive = Path("/content/drive/MyDrive")
model_save_path = "recycle_classifier_v2m_finetuned.keras"

# ================================
# 데이터 디렉토리 설정
# ================================
train_dir = base_drive / split_dir / "train"
val_dir   = base_drive / split_dir / "val"
test_dir  = base_drive / split_dir / "test"

# ================================
# 클래스 탐색
# ================================
classes = [d.name for d in train_dir.iterdir() if d.is_dir()]
class_indices = {cls: i for i, cls in enumerate(classes)}

# ================================
# 클래스별 이미지 개수
# ================================
class_counts = {cls: len(list((train_dir / cls).glob("*"))) for cls in classes}
max_count = max(class_counts.values())
print("클래스별 개수:", class_counts)

# ================================
# train oversampling
# ================================
train_images, train_labels = [], []
for cls in classes:
    cls_path = train_dir / cls
    imgs = list(cls_path.glob("*"))
    if len(imgs) < max_count:
        imgs = imgs + np.random.choice(imgs, max_count - len(imgs)).tolist()
    labels = [class_indices[cls]] * len(imgs)
    train_images.extend([str(p) for p in imgs])  # 문자열 변환
    train_labels.extend(labels)

# 셔플
combined = list(zip(train_images, train_labels))
np.random.shuffle(combined)
train_images, train_labels = zip(*combined)

# ================================
# tf.data Dataset 구성
# ================================
def preprocess_image(file_path, label):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)  # JPEG으로 명시 (속도, 안정성 ↑)
    img = tf.cast(img, tf.float32) / 255.0       # 정규화
    return img, label

train_ds = tf.data.Dataset.from_tensor_slices((list(train_images), list(train_labels)))
train_ds = train_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

def prepare_dataset(directory):
    images, labels = [], []
    for cls in classes:
        cls_path = Path(directory) / cls
        imgs = list(cls_path.glob("*"))
        images.extend([str(p) for p in imgs])  # 문자열 변환
        labels.extend([class_indices[cls]] * len(imgs))
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    ds = ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

val_ds = prepare_dataset(val_dir)
test_ds = prepare_dataset(test_dir)

# ================================
# 클래스 가중치 계산
# ================================
total = sum(class_counts.values())
class_weights = {i: total / count for i, (cls, count) in enumerate(class_counts.items())}
print("클래스 가중치:", class_weights)

# ================================
# 모델 구성 (EfficientNetV2M)
# ================================
base_model = EfficientNetV2M(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # 1단계: Transfer Learning

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.3)(x)
output = Dense(len(classes), activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ================================
# 콜백 설정
# ================================
callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
    ModelCheckpoint(model_save_path, monitor="val_accuracy", save_best_only=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-6)
]

# ================================
# 1단계: Feature Extractor 학습
# ================================
print("\n===== [1단계] Transfer Learning 학습 시작 =====")
history_stage1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    class_weight=class_weights,
    callbacks=callbacks
)

# ================================
# 2단계: Fine-Tuning
# ================================
print("\n===== [2단계] Fine-Tuning 학습 시작 =====")
for layer in base_model.layers[-40:]:  # 마지막 40개 layer만 학습 허용
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_stage2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    class_weight=class_weights,
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

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label="train_acc")
    plt.plot(val_acc, label="val_acc")
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
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
def classify_image(model, image_path, threshold=0.5):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.expand_dims(img, axis=0)

    preds = model.predict(img)
    class_id = np.argmax(preds[0])
    confidence = preds[0][class_id]

    if confidence < threshold:
        return "일반쓰레기"
    else:
        return classes[class_id]


# 예시
# result = classify_image(model, "sample.jpg")
# print("분류 결과:", result)
