import tensorflow as tf
import tensorflow.keras as datasets,models,layers
import numpy as np

# x = np.random.rand(10, 224, 224, 3).astype(np.float32)
# y = np.random.randint(0, 2, size=(10,))

# model = tf.keras.applications.EfficientNetV2M(
#     input_shape=(224,224,3),
#     include_top=True,
#     weights=None,
#     classes=2
# )

# model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
# model.fit(x, y, epochs=1)

X_train, Y_train , x_test,y_text = datasets.mnist.load_datas()