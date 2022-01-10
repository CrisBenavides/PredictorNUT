import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
import numpy as np
import pandas as pd
from joblib import dump
from joblib import load



X_train_L_20=np.load('/home/userbda/Documents/X_train_L_20.npy')
y_train_L_20=np.load('/home/userbda/Documents/y_train_L_20.npy')
X_train_M_20=np.load('/home/userbda/Documents/X_train_M_20.npy')
y_train_M_20=np.load('/home/userbda/Documents/y_train_M_20.npy')
X_train_P_20=np.load('/home/userbda/Documents/X_train_P_20.npy')
y_train_P_20=np.load('/home/userbda/Documents/y_train_P_20.npy')

X_train_L_60=np.load('/home/userbda/Documents/X_train_L_60.npy')
y_train_L_60=np.load('/home/userbda/Documents/y_train_L_60.npy')
X_train_M_60=np.load('/home/userbda/Documents/X_train_M_60.npy')
y_train_M_60=np.load('/home/userbda/Documents/y_train_M_60.npy')
X_train_P_60=np.load('/home/userbda/Documents/X_train_P_60.npy')
y_train_P_60=np.load('/home/userbda/Documents/y_train_P_60.npy')

X_train_L_100=np.load('/home/userbda/Documents/X_train_L_100.npy')
y_train_L_100=np.load('/home/userbda/Documents/y_train_L_100.npy')
X_train_M_100=np.load('/home/userbda/Documents/X_train_M_100.npy')
y_train_M_100=np.load('/home/userbda/Documents/y_train_M_100.npy')
X_train_P_100=np.load('/home/userbda/Documents/X_train_P_100.npy')
y_train_P_100=np.load('/home/userbda/Documents/y_train_P_100.npy')


#Modelo

def Entrenamiento(X_train,y_train):
    #Consiste en una red compuesta por una capa de unidades LSTM y otra capa convolucional
    n_output=19
    model = keras.Sequential()
    model.add( 
        keras.layers.LSTM( 
        units=64, 
        input_shape=(X_train.shape[1], X_train.shape[2]) , 
        return_sequences=False
        
    )
    )

    model.add(keras.layers.Dense(units=n_output,activation='linear'))
    model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.Dense(units=n_output,activation='linear'))
    model.add(Reshape([n_output]))
    model.compile(loss='mean_squared_error', optimizer='adam')  


    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=2,mode='min')

    #Entrenamiento
    

    history = model.fit(X_train,y_train, epochs=50,validation_split=0.15, 
    callbacks=[early_stopping],shuffle=False, batch_size=400)
    return model

#Entrenando los 9 submodelos
modelo_L_20=Entrenamiento(X_train_L_20,y_train_L_20)
modelo_M_20=Entrenamiento(X_train_M_20,y_train_M_20)   
modelo_P_20=Entrenamiento(X_train_P_20,y_train_P_20)   

modelo_L_60=Entrenamiento(X_train_L_60,y_train_L_60)
modelo_M_60=Entrenamiento(X_train_M_60,y_train_M_60)   
modelo_P_60=Entrenamiento(X_train_P_60,y_train_P_60)

modelo_L_100=Entrenamiento(X_train_L_100,y_train_L_100)
modelo_M_100=Entrenamiento(X_train_M_100,y_train_M_100)   
modelo_P_100=Entrenamiento(X_train_P_100,y_train_P_100)

#Guardar modelos
path_modelo_L_20= '/home/userbda/Documents/modelo_L_20.h5'
path_modelo_M_20= '/home/userbda/Documents/modelo_M_20.h5'
path_modelo_P_20= '/home/userbda/Documents/modelo_P_20.h5'

path_modelo_L_60= '/home/userbda/Documents/modelo_L_60.h5'
path_modelo_M_60= '/home/userbda/Documents/modelo_M_60.h5'
path_modelo_P_60= '/home/userbda/Documents/modelo_P_60.h5'

path_modelo_L_100= '/home/userbda/Documents/modelo_L_100.h5'
path_modelo_M_100= '/home/userbda/Documents/modelo_M_100.h5'
path_modelo_P_100= '/home/userbda/Documents/modelo_P_100.h5'

modelo_L_20.save(path_modelo_L_20)
modelo_M_20.save(path_modelo_M_20)
modelo_P_20.save(path_modelo_P_20)

modelo_L_60.save(path_modelo_L_60)
modelo_M_60.save(path_modelo_M_60)
modelo_P_60.save(path_modelo_P_60)

modelo_L_100.save(path_modelo_L_100)
modelo_M_100.save(path_modelo_M_100)
modelo_P_100.save(path_modelo_P_100)

print("Entrenamiento finalizado")