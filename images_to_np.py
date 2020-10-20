from glob import glob
import csv
import cv2
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
from tqdm import tqdm
import os

IMG_SIZE = 224
train = glob("images/train/*/*.jpg")
x_train = []
for img in tqdm(train):
    img = load_img(img, target_size=(IMG_SIZE, IMG_SIZE), color_mode='rgb')
    img_arr = image.img_to_array(img)
    img_arr = img_arr/255.0
    x_train.append(img_arr)

x_train = np.asarray(x_train)
np.save("train_data.npy", x_train)


label_path = "images/train"
label_list = []
for path in os.listdir(label_path):
    label_list.append(path)
print(label_list)


Data = glob("images/train/*/*.jpg")
Data_output=list()
Data_output.append(["Classes"])

for file_name in tqdm(Data):
    if file_name[13] == 'G':
        Data_output.append([file_name[13:19]])
    elif file_name[13] == 'M':
        Data_output.append([file_name[13:23]])
    elif file_name[13] == 'N':
        Data_output.append([file_name[13:21]])
    elif file_name[13] == 'P':
        Data_output.append([file_name[13:22]])



with open("label.csv", "w") as f:
    writer = csv.writer(f)
    for val in Data_output:
        writer.writerows([val])

test = glob("images/test/*/*.jpg")
Data_1 = glob("images/test/*/*.jpg")
Data_output_1=list()
Data_output_1.append(["Classes"])

for file_name in tqdm(Data_1):
    if file_name[12] == 'G':
        Data_output_1.append([file_name[12:18]])
    elif file_name[12] == 'M':
        Data_output_1.append([file_name[12:22]])
    elif file_name[12] == 'N':
        Data_output_1.append([file_name[12:20]])
    elif file_name[12] == 'P':
        Data_output_1.append([file_name[12:21]])

with open("test_label.csv", "w") as f:
    writer = csv.writer(f)
    for val in Data_output_1:
        writer.writerows([val])

x_test = []
for img in test:
    img = load_img(img, target_size=(IMG_SIZE, IMG_SIZE), color_mode='rgb')
    img_arr = image.img_to_array(img)
    img_arr = img_arr/255.0
    x_test.append(img_arr)



x_test = np.asarray(x_test)
print(x_test.shape)
# saves the image array in npy file
np.save("test_data.npy", x_test)

print(x_train.shape)
print(x_test.shape)

import matplotlib.pyplot as plt
plt.imshow(x_train[0])