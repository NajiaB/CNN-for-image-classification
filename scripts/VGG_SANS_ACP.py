# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 21:27:08 2022

@author: BOUADDOUCH Najia
"""
###----------- EXTRACTION DE FEATURES AVEC VGG + PASSAGE DANS COUCHE FULLY CONNECTED SANS REDUCTION DE DIMENSION

#IMPORTAITION DES PACKAGES 

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
from keras import layers
from keras import models
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input
from keras.layers import BatchNormalization
import time


##-- IMPORTATION DU CSV ET DES IMAGES ET PREPROCESSING

data = pd.read_csv('C:/Users/BOUADDOUCH Najia/Documents/feux/AllLabels.csv',sep=',')    # reading the csv file
images = []

for i in tqdm.tqdm(range(data.shape[0])):
    img = tf.keras.utils.load_img('C:/Users/BOUADDOUCH Najia/Documents/feux/toutes_images_rogne/'+data['file'][i]
                         ,target_size=(255,255,3)
                         )
    
    img = tf.keras.utils.img_to_array(img)
    img = img/255   #normalisation des pixels
    images.append(img)

myimages = np.array(images)
labels=np.array(data['label'])
plt.imshow(myimages[5]) 
plt.bar(data['label'].value_counts().index, data['label'].value_counts().values)

# SPLIT ENTRE TEST ET TRAIN

x_train, x_test, y_train, y_test = train_test_split(myimages, labels, random_state=42, test_size=0.2)

#RECODAGE DES LABels en 0 1 2

encoder=LabelEncoder()
encoder.fit(y_train)

y_train = encoder.transform(y_train)
y_test = encoder.transform(y_test)

from keras.utils import to_categorical
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

#IMPORTATION DU MODELE PRE ENTRAINE VGG (uniquement les couches de convolution)

from keras.applications.vgg16 import VGG16

VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(255, 255, 3))

for layer in VGG_model.layers:
	layer.trainable = False
    
VGG_model.summary()

#EXCTRACTION DES FEATURES 

train_feature_extractor=VGG_model.predict(x_train)
train_features = train_feature_extractor.reshape(train_feature_extractor.shape[0], -1)

#test features
test_feature_extractor=VGG_model.predict(x_test)
test_features = test_feature_extractor.reshape(test_feature_extractor.shape[0], -1)   

# COUCHE FULLY CONNECTED DE SORTIE

model = Sequential()
inputs = Input(shape=(train_features.shape[1],)) #Shape = n_components
hidden = Dense(256, activation='relu')(inputs)
#hidden1 = Dense(512, activation='relu')(inputs)
#hidden2 = Dense(256, activation='relu')(hidden1)
output = Dense(3, activation='softmax')(hidden)
model = Model(inputs=inputs, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'])

#FIT DU MODELES SUR LES FEATURES EXTRAITS ET CHRONOMETRAGE (dans le but de comparer avec et sans ACP)

import datetime
start = datetime.datetime.now()
model.fit(train_features, y_train_one_hot, epochs=20, verbose=1)
end = datetime.datetime.now()
print("Temps d'execution sans ACP': ", end-start)  # plus lent qu'avec ACP 

#PREDICTIONS
predict_test = model.predict(test_features)  

test_loss, test_acc = model.evaluate(test_features, y_test_one_hot)   #même accuracy à peu près que avec ACP
