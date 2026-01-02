import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


# -----------------------------------------------------------
# 1) خواندن و پیش‌پردازش دیتاست
# -----------------------------------------------------------
def load_dataset(dataset_path):
    X = []
    y = []
    classes = ["Car", "Airplane"]   # 0 = not_cat , 1 = cat, 2 = dog

    for label, folder in enumerate(classes):
        folder_path = os.path.join(dataset_path, folder)
        
        if not os.path.exists(folder_path):
            print(f" پوشه '{folder_path}' یافت نشد!")
            continue
            
        print(f" در حال پردازش پوشه: {folder_path}")

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)

            # خواندن تصویر
            img = cv2.imread(img_path)
            if img is None:
                continue

            # پیش پردازش
            img = cv2.resize(img, (128, 128))
            img = img.astype("float32") / 255.0

            X.append(img)
            y.append(label)

    return np.array(X), np.array(y)


# لود داده‌ها
X, y = load_dataset("D:/uni/Term3/az/9/dataset")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)



base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(128,128,3)
)


for layer in base_model.layers[-20:]:
    layer.trainable = True

x = base_model.output
x = Flatten()(x)
x = Dense(64, activation="relu")(x)
output = Dense(2, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)


model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=8,
    validation_split=0.2
)



plt.plot(history.history['accuracy'], label="train acc")
plt.plot(history.history['val_accuracy'], label="val acc")
plt.legend()
plt.title("Accuracy")
plt.show()

plt.plot(history.history['loss'], label="train loss")
plt.plot(history.history['val_loss'], label="val loss")
plt.legend()
plt.title("Loss")
plt.show()

def predict_image(path):

    img = cv2.imread(path)
    img_resized = cv2.resize(img, (128,128))
    img_resized = img_resized.astype("float32") / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)

    pred = model.predict(img_resized)[0]
    class_id = np.argmax(pred)

    label = "AirPlane" if class_id == 1 else "Car"
    cv2.imshow(label, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

predict_image("D:/uni/Term3/az/8/1.jpg")