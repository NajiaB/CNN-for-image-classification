# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 20:51:14 2022

@author: BOUADDOUCH Najia
"""

import tensorflow as tf
import pathlib
import pandas as pd
import tqdm 

from tensorflow import keras
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras import callbacks
import sklearn
import seaborn as sns

import keras_tuner as kt
from keras import layers
from keras import models
data = pd.read_csv('C:/Users/BOUADDOUCH Najia/Documents/feux/AllLabels.csv',sep=',')    # reading the csv file
images = []

for i in tqdm.tqdm(range(data.shape[0])):
    img = tf.keras.utils.load_img('C:/Users/BOUADDOUCH Najia/Documents/feux/data/'+data['file'][i]
                         ,target_size=(256,256,3)
                         )
    
    img = tf.keras.utils.img_to_array(img)
    img = img/255
    images.append(img)

myimages = np.array(images)

labels=np.array(data['label'])
plt.imshow(myimages[5]) 
plt.bar(data['label'].value_counts().index, data['label'].value_counts().values)

x_train, x_test, y_train, y_test = train_test_split(myimages, labels, random_state=42, test_size=0.2)

encoder=LabelEncoder()
encoder.fit(y_train)

y_train = encoder.transform(y_train)
y_test = encoder.transform(y_test)


def model_builder(hp) :
    
    
    model1=models.Sequential()
    hp_filters = hp.Int('filters_hp', min_value = 32, max_value = 64, step = 32)
    model1.add(layers.Conv2D(hp_filters,(3,3),activation='relu',input_shape=(256,256,3)))
    model1.add(layers.MaxPooling2D((2,2)))
    model1.add(layers.Conv2D(64,(3,3),activation='relu'))
    model1.add(layers.MaxPooling2D((2,2)))
    model1.add(layers.Conv2D(64,(3,3),activation='relu'))
    
    
    
    model1.add(layers.Flatten())
    model1.add(layers.Dense(units=hp.Int('units1', min_value=32, max_value=128, step=32),activation='relu'))
    #model1.add(layers.Dense(units=hp.Int('units2', min_value=32, max_value=128, step=32),activation='relu'))
    model1.add(layers.Dense(3,activation='softmax'))
    
    model1.summary()
    
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3])

    model1.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    
    return model1
tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     )

earlystopping = callbacks.EarlyStopping(monitor ="val_loss", 
                                        mode ="min", patience = 5, 
                                        restore_best_weights = True)

tuner.search(x_train, y_train, epochs=10, validation_split=0.2, 
             callbacks=[earlystopping]
             )


           