# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 20:16:21 2022

@author: BOUADDOUCH Najia
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # statistical data visualization
import tqdm 
import tensorflow as tf
import pathlib
import pandas as pd
import tqdm 

from tensorflow import keras
from keras.preprocessing import image
from keras import layers
from keras import models
from keras import callbacks


data = pd.read_csv('C:/Users/BOUADDOUCH Najia/Documents/feux/AllLabels.csv',sep=',')    # reading the csv file
images = []

for i in tqdm.tqdm(range(data.shape[0])):
    img = tf.keras.utils.load_img('C:/Users/BOUADDOUCH Najia/Documents/feux/toutes_images_rogne/'+data['file'][i]
                         ,target_size=(255,255,3)
                         )
    
    img = tf.keras.utils.img_to_array(img)
    #img = img/255
    images.append(img)

myimages = np.array(images)


labels=np.array(data['label'])



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(myimages, labels, test_size = 0.33, random_state = 42)

# check the shape of X_train and X_test

X=myimages
X.shape
nsamples, nx, ny, nrgb = X.shape
X = X.reshape((nsamples,nx*ny*nrgb))

Y=labels


X_train.shape, X_test.shape
nsamples, nx, ny, nrgb = X_train.shape
x_train2 = X_train.reshape((nsamples,nx*ny*nrgb))

nsamples, nx, ny, nrgb = X_test.shape
x_test2 = X_test.reshape((nsamples,nx*ny*nrgb))

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X)
X=scaler.transform(X)

from sklearn.decomposition import PCA

pca_test=PCA(n_components=418)
pca_test.fit(X)
plt.plot(np.cumsum(pca_test.explained_variance_ratio_))
plt.xlabel('number of component')
plt.ylabel('cum variance')

n_PCA_components=300
pca=PCA(n_components=n_PCA_components)
principalComponents = pca.fit_transform(X)
pca_std = np.std(principalComponents)




import pandas as pd

columns=[]
for i in range(1,301):
    
    columns1 = ['PC',str(i)]
    columns.append(''.join(columns1))
    
principalDf=pd.DataFrame(data=principalComponents,columns=columns)

finalDf=pd.concat([principalDf,data['label']],axis=1)

fig=plt.figure(figsize=(8,8))
ax=fig.add_subplot(1,1,1)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
labels=['green','orange','red']
colors=['g','y','r']
for label,color in zip(labels,colors):
    indicesToKeep=finalDf['label']==label
    ax.scatter(finalDf.loc[indicesToKeep,'PC1'],
               finalDf.loc[indicesToKeep,'PC2'],
               c=color,
               s=50)
 
final_X=finalDf.drop(labels=['label'],axis=1)
y=finalDf['label'].values

from sklean.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
final_Y=labelencoder.fit_transform(y)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(final_X,final_Y,test_size=0.2)


from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix 


model = models.Sequential()
model.add(Dense(128,input_dim=n_PCA_components,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(128,activation='relu'))
model.add(BatchNormalization())

model.add(Dropout(0.2))

model.add(Dense(64,activation='relu'))

model.add(Dense(3))
model.add(Activation('softmax'))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
earlystopping = callbacks.EarlyStopping(monitor ="val_loss", 
                                        mode ="min", patience = 5, 
                                        restore_best_weights = True)
history=model.fit(X_train,y_train,epochs=1000,validation_data=(X_test,y_test),batch_size=64,callbacks=[earlystopping])

_, acc=model.evaluate(X_test,y_test)

from keras.layers import Dense, GaussianNoise, Dropout, Activation, BatchNormalization

model = models.Sequential()
layers = 1
units = 64

model.add(Dense(units, input_dim=n_PCA_components, activation='relu'))
model.add(GaussianNoise(pca_std))
for i in range(layers):
    model.add(Dense(units, activation='relu'))
    model.add(GaussianNoise(pca_std))
    model.add(Dropout(0.1))
model.add(Dense(3, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['categorical_accuracy'])

model.fit(X_train, y_train, epochs=1000, batch_size=256, validation_split=0.15, verbose=2)
model.evaluate(X_test,y_test)
