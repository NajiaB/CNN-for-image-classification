# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 23:02:07 2022

@author: BOUADDOUCH Najia
"""
###----------- EXTRACTION DE FEATURES AVEC VGG + ACP SUR FEATURES + PASSAGE DES COMPOSANTES PRINCIPALES DANS COUCHE FULLY CONNECTED

#IMPORTATION DES PACKAGES

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
   

#----- REALISATION DE L'ACP

from sklearn.decomposition import PCA

#CHOISIR LE NOMBRE DE COMPOSANTES QUI PERMETTENT DE GARDER UN MAX D'INFORMATION

pca_test = PCA(n_components=400) #
pca_test.fit(train_features)
plt.plot(np.cumsum(pca_test.explained_variance_ratio_))
plt.xlabel("Nombre de composantes")
plt.ylabel("Information récupérée")


#ON CHOISI n=300 car on recupère environ 90%, mais on peut jouer sur ce paramètre (hyperparamètre du coup)

n_PCA_components = 300
pca = PCA(n_components=n_PCA_components)
train_PCA = pca.fit_transform(train_features)
test_PCA = pca.transform(test_features)

#COUCHE FULLY CONNECTED

model = Sequential()
inputs = Input(shape=(n_PCA_components,)) #Shape = n_components
hidden = Dense(256, activation='relu')(inputs)
#hidden1 = Dense(512, activation='relu')(inputs)
#hidden2 = Dense(256, activation='relu')(hidden1)
output = Dense(3, activation='softmax')(hidden)
model = Model(inputs=inputs, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'])

import datetime
start = datetime.datetime.now()
model.fit(train_PCA, y_train_one_hot, epochs=20, verbose=1)

end = datetime.datetime.now()
print("Temps d'execution après ACP': ", end-start)  # CA VA BEAUCOUP PLUS VITE (environ 15 fois plus rapide)


predict_test = model.predict(test_PCA)
predict_test = np.argmax(predict_test, axis=1)

test_loss, test_acc = model.evaluate(test_PCA, y_test_one_hot)

#matrice de confusion

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, predict_test)
#print(cm)
sns.heatmap(cm, annot=True)

#fonction pour transformer le encoding en une couleur
def reverse(x):
    if int(x)==0 :
        return('green')
    elif int(x)==1 :
        return('orange')
    else:
        return('red')

#vérifier la prédiction pour qlq échantillons

n=np.random.randint(0, x_test.shape[0])
img = x_test[n]
plt.imshow(img)

input_img = np.expand_dims(img, axis=0) #Expand dims so the input is (num images, x, y, c)
input_img_feature=VGG_model.predict(input_img)
input_img_features=input_img_feature.reshape(input_img_feature.shape[0], -1)
input_img_PCA = pca.transform(input_img_features)
prediction_img = model.predict(input_img_PCA)
prediction_img = np.argmax(prediction_img, axis=1)
prediction_img = reverse(prediction_img)  #Reverse the label encoder to original name
print("The prediction for this image is: ", prediction_img)
print("The actual label for this image is: ", reverse(y_test[n]))

