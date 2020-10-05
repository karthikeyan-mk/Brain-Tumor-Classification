import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

model = load_model("custom_cnn.h5")
print(model.summary())

IMG_SIZE = 224
classes = ["Glioma", "Meningioma", "Pituitary tumor", "Negative"]

def predict(image_path):
    d = []
    img = load_img(image_path, color_mode='rgb', target_size=(IMG_SIZE,IMG_SIZE))
    img = img_to_array(img)
    img = img/255.
    d.append(img)
    d = np.asarray(d)
    pred = np.argmax(model.predict(d))
    return classes[pred]

# prediction using sample image
data = load_image("brain_img.jpg")
print(data)