# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 19:30:04 2022

@author: BOUADDOUCH Najia
"""
import tensorflow as tf
import pathlib
import pandas as pd
import tqdm 
import time
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

###---------------------- Notre CNN après optimisation + ajout de dropout()

#IMPORTATION DU JEU DE DONNEES et preprocessing

data = pd.read_csv('C:/Users/BOUADDOUCH Najia/Documents/feux/AllLabels.csv',sep=',')    # reading the csv file
images = []

for i in tqdm.tqdm(range(data.shape[0])):
    img = tf.keras.utils.load_img('C:/Users/BOUADDOUCH Najia/Documents/feux/toutes_images_rogne/'+data['file'][i]
                         ,target_size=(255,255,3)
                         )
    
    img = tf.keras.utils.img_to_array(img)
    img = img/255
    images.append(img)

myimages = np.array(images)

labels=np.array(data['label'])
plt.imshow(myimages[5]) 
plt.bar(data['label'].value_counts().index, data['label'].value_counts().values)

x_train, x_test, y_train, y_test = train_test_split(myimages, labels, random_state=42, test_size=0.2)

#ENCODAGE DES LABELS EN 0 1 2


encoder=LabelEncoder()
encoder.fit(y_train)

y_train = encoder.transform(y_train)
y_test = encoder.transform(y_test)

#--- Augmentation de données

#couche d'augmentation à intégrer au modèele
data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
  #♠layers.RandomBrightness(factor=0.2)

])

image = tf.expand_dims(x_train[5], 0)
plt.figure(figsize=(10, 10))
for i in range(9):
  augmented_image = data_augmentation(image,training=True)
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(augmented_image[0])
  plt.axis("off")

# =============================================================================



#CONSTRUCTION DU MODELE  EN PRENANT EN COMPTE LES SORTIES DU TUNER + en rajoutant dropout, ce qui a augmenté l'accuracy


model1=models.Sequential()
model1.add(data_augmentation)
model1.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(255,255,3)))
#model1.add(layers.BatchNormalization())  #n'a pas fonctionné ==> diminue l'accuracy à environ 30-40 %
model1.add(layers.MaxPooling2D((2,2)))
model1.add(layers.Conv2D(64,(3,3),activation='relu'))
#model1.add(layers.BatchNormalization())

model1.add(layers.MaxPooling2D((2,2)))
model1.add(layers.Conv2D(64,(3,3),activation='relu'))
#model1.add(layers.BatchNormalization())


model1.add(layers.Dropout(0.3))   #COUCHE qui regularise et prévient l'overfitting en desactivant d'une manière aléatoire (ici 30% des unités ou neuronnes)

model1.add(layers.Flatten())
model1.add(layers.Dense(128,activation='relu'))
#model1.add(layers.BatchNormalization())
model1.add(layers.Dropout(0.2))


model1.add(layers.Dense(96,activation='relu'))
#model1.add(layers.BatchNormalization())
#model1.add(layers.Dropout(0.4))


model1.add(layers.Dense(3,activation='softmax'))


#fonction qui arrete l'apprentissage dès lors que le réseau n'apprend plus rien
earlystopping = callbacks.EarlyStopping(monitor ="val_loss", 
                                        mode ="min", patience = 5, 
                                        restore_best_weights = True)

model1.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy']) #on utilise sparse car nos labels sont encodée et ne sont pas passées au to_categorial()

import datetime
start = datetime.datetime.now()
history=model1.fit(x_train,y_train, epochs=10,batch_size=32,validation_split=0.2
           , callbacks=[earlystopping]
           )
end = datetime.datetime.now()

print("Temps d'execution de notre CNN : ", end-start)

#evaluation du modele
test_loss, test_acc = model1.evaluate(x_test, y_test)


#♣tres satisfaite du modele ==> je l'enregistre
model1.save('tuned_model_prise_rogne')
#a=keras.models.load_model('1er_model')



plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.xlim(0 , 10)
plt.ylim(0 , 1)
plt.title('Model Accuracy')

plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'val'], loc='upper left')
plt.savefig('Model_Accuracy.png')
plt.show()


test_acc


#MATRICE DE CONFUSION
model1.predict(x_train)
predictions=model1.predict(x_test)
classes = np.argmax(predictions, axis = 1)
classes
#matrice de confusion
cm=sklearn.metrics.confusion_matrix(y_test,classes)
sns.heatmap(cm,annot=True,xticklabels=['green','orange','red'],yticklabels=['green','orange','red'])

#plt.imshow(x_train[0])  #==> 2 = rouge
#plt.imshow(x_train[4])
#plt.imshow(x_train[1])   # ==> 1 = oreange
#plt.imshow(x_train[3])
#plt.imshow(x_train[2]) #==> 0 =  vert



# def reverse(x):
#     if x==0 :
#         return('green')
#     elif x==1 :
#         return('orange')
#     else:
#         return('red')
    