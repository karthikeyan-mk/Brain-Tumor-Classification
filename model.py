# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# %%
print(tf.test.is_gpu_available())


# %%
x_train = np.load("train_data.npy")


# %%
x_test = np.load("test_data.npy")


# %%
print(x_train.shape)
print(x_test.shape)


# %%
y_train = pd.read_csv("train_label.csv")
y_test = pd.read_csv("test_label.csv")


# %%
y_train=y_train.replace(to_replace="Negative",value=3)
y_train=y_train.replace(to_replace="Meningioma",value=1)
y_train=y_train.replace(to_replace="Glioma",value=0)
y_train=y_train.replace(to_replace="Pituitary",value=2)


# %%
from tensorflow.keras.utils import to_categorical
y_train =to_categorical(y_train, num_classes=4)
print(y_train.shape)


# %%
y_test=y_test.replace(to_replace="Negative",value=3)
y_test=y_test.replace(to_replace="Meningioma",value=1)
y_test=y_test.replace(to_replace="Glioma",value=0)
y_test=y_test.replace(to_replace="Pituitary",value=2)


# %%
y_test = to_categorical(y_test, num_classes=4)


# %%
from tensorflow.keras.layers import Input, Activation, Dense, Conv2D, BatchNormalization, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import optimizers
import matplotlib.image  as mpimg
from random import randint
from sklearn.utils import shuffle


# %%
from tensorflow.keras.models import load_model


# %%
from tensorflow.keras.layers import Dropout
model = Sequential()


model.add(Conv2D(16, kernel_size=(3,3), padding='same', input_shape=(224, 224, 3), activation='relu'))
model.add(Conv2D(16, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(32, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(256, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(units=256, activation='relu', kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(units=4, activation='sigmoid'))


# %%
print(model.summary())


# %%
model.compile(optimizer="RMSprop",
              loss='categorical_crossentropy', metrics=["accuracy"])


# %%
BATCH_SIZE = 16
EPOCHS = 100

earlystop = EarlyStopping(monitor="val_accuracy",
                          patience=50,
                          verbose=1,
                          mode='max',
                          restore_best_weights=True)


# %%
history = model.fit(x_train, y_train, 
                    batch_size=BATCH_SIZE, 
                    shuffle=True, 
                    verbose=1, 
                    validation_data=(x_test, y_test),
                    epochs=EPOCHS, callbacks=[earlystop])


# %%
model.save("custom_cnn.h5")
#model.save_weights("custom_cnn_weights.h5")


# %%
print(history.history.keys())

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# %%
