import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from random import randint
from sklearn.utils import shuffle
import tensorflow as tf

get_ipython().run_line_magic('matplotlib', 'inline')
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
print(tf.test.is_gpu_available())

# ## Loading the data
x_train = np.load("train_data.npy")
x_test = np.load("test_data.npy")
y_train = pd.read_csv("train_label.csv")
y_test = pd.read_csv("test_label.csv")

print(y_train.shape)
print(y_test.shape)
print(x_train.shape)
print(x_test.shape)

y_train=y_train.replace(to_replace="Negative",value=3)
y_train=y_train.replace(to_replace="Meningioma",value=1)
y_train=y_train.replace(to_replace="Glioma",value=0)
y_train=y_train.replace(to_replace="Pituitary",value=2)

print(y_train["Classes"].value_counts())


from keras.utils import to_categorical
y_train =to_categorical(y_train, num_classes=4)

y_test=y_test.replace(to_replace="Negative",value=3)
y_test=y_test.replace(to_replace="Meningioma",value=1)
y_test=y_test.replace(to_replace="Glioma",value=0)
y_test=y_test.replace(to_replace="Pituitary",value=2)

y_test = to_categorical(y_test, num_classes=4)


from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
from tensorflow.keras.layers import Dropout


def build_model(hp):
    model = Sequential()
    model.add(Conv2D(hp.Int('filters', min_value=16, max_value=64, step=16), 
                     kernel_size=(3,3), padding='same', input_shape=(224, 224, 3), 
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(hp.Float('first_drop', min_value=0.2, max_value=0.3, step=0.05)))
    
    for i in range(hp.Int('num_layers', 1, 4)):
        model.add(Conv2D(hp.Int('conv_{i}', min_value=32, max_value=512, step=32), 
                         kernel_size=(3,3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(hp.Float('conv_drop', min_value=0.2, max_value=0.4, step=0.05)))
    
    model.add(Conv2D(512, 
                     kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(hp.Float('dropout', min_value=0.25, max_value=0.5, step=0.1)))
    
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(units=256, activation='relu', 
                    kernel_initializer='he_uniform'))
    model.add(Dropout(hp.Float('final_drop', min_value=0.2, max_value=0.5, step=0.1)))
    model.add(Dense(units=4, activation='sigmoid'))
    
    model.compile(optimizer=hp.Choice('opt', values=["Adam", "RMSprop", "AdaDelta"]),
                  loss='categorical_crossentropy', metrics=["accuracy"])
    
    return model

SEED = 10
direc='logs'

from kerastuner.tuners import Hyperband

turner = RandomSearch(build_model,
                      objective="val_accuracy",
                      max_trials=3,
                      seed=SEED,
                      directory=direc,
                      executions_per_trial=3)

print(turner.search_space_summary())

turner.search(x_train, y_train,
             epochs=50,
             batch_size=24,
             validation_data=(x_test, y_test))

print(turner.results_summary())

model = turner.get_best_models()[0]
print(model.summary())
model.save("HP_model.h5")

print(turner.get_best_hyperparameters()[0].values)
hps = turner.oracle.get_best_trials(num_trials=1)[0].hyperparameters

model = build_model(hps)
print(model.summary())