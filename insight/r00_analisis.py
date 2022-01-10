#Script de prueba para obtener la regresion del throughtput para un conjunto de celdas

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
from math import floor
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler



#Cargar modelos
modelo_L_20=keras.models.load_model("/home/userbda/Documents/modelo_L_20.h5")
modelo_L_60=keras.models.load_model("/home/userbda/Documents/modelo_L_60.h5")
modelo_L_100=keras.models.load_model("/home/userbda/Documents/modelo_L_100.h5")

modelo_M_20=keras.models.load_model("/home/userbda/Documents/modelo_M_20.h5")
modelo_M_60=keras.models.load_model("/home/userbda/Documents/modelo_M_60.h5")
modelo_M_100=keras.models.load_model("/home/userbda/Documents/modelo_M_100.h5")

modelo_P_20=keras.models.load_model("/home/userbda/Documents/modelo_P_20.h5")
modelo_P_60=keras.models.load_model("/home/userbda/Documents/modelo_P_60.h5")
modelo_P_100=keras.models.load_model("/home/userbda/Documents/modelo_P_100.h5")

Datos_L=pd.read_csv('/home/userbda/Documents/celdas_L.csv')
Datos_M=pd.read_csv('/home/userbda/Documents/celdas_M.csv')
Datos_P=pd.read_csv('/home/userbda/Documents/celdas_P.csv')

#Creacion del tiempo como indice del dataframe
Datos_L["DATETIME_ID2"]=pd.to_datetime(Datos_L["DATETIME_ID2"])
Datos_L=Datos_L.set_index('DATETIME_ID2')

Datos_P["DATETIME_ID2"]=pd.to_datetime(Datos_P["DATETIME_ID2"])
Datos_P=Datos_P.set_index('DATETIME_ID2')

Datos_M["DATETIME_ID2"]=pd.to_datetime(Datos_M["DATETIME_ID2"])
Datos_M=Datos_M.set_index('DATETIME_ID2')

Datos=Datos_L
lista=Datos["EutrancellFDD"].unique().tolist() 
celdas=[lista[1]]


print("Datos cargados")
#Prueba para celdas la primera celda de la banda L
def regresion(Datos,lista_celdas,prb_20,prb_60,prb_100):
    largo_dia=19  #Horas del dia
    n_dias_input=7  #Cantidad de dias (semana)
    for cell in lista_celdas:
        #Se obtienen los datos de una de las celdas
        datos_celda=(Datos[Datos["EutrancellFDD"]==cell])
        #Se filtra el horario de 6:00 am hasta 12:00 pm
        datos_celda=datos_celda.reindex(pd.date_range(min(datos_celda.index), max(datos_celda.index),freq='H')).fillna(np.nan)
        datos_celda=datos_celda.between_time(start_time='6:00', end_time='00:00')[1:]
        datos_celda=datos_celda.fillna(method='ffill',axis=0).fillna(method='bfill',axis=0)
        datos_celda=datos_celda.drop(columns=["EutrancellFDD"])
        
        #En este dataset desde el dia 84 se consideró conjunto test (no se usó para entrenar)
        #Se obtiene la regresion para el dia 85
        for numero_dia in list(range(84,85)):  
        
            X=largo_dia*numero_dia
            entrada=datos_celda.iloc[X-largo_dia*n_dias_input:X]
            salida=datos_celda.iloc[X:X+largo_dia,0]
            variable=entrada.rolling(19).max()
            hora_v = variable.index.hour
            max_prb_mean=(variable.iloc[(hora_v == 0)]).iloc[:,8].mean()          

            #Se selecciona modelo segun el uso de prb de los ultimos 7 dias
            if (max_prb_mean>0) &( max_prb_mean<20):
                model=prb_20
            if (max_prb_mean>20) &( max_prb_mean<60):
                model=prb_60           
            if (max_prb_mean>60) &( max_prb_mean<100):
                model=prb_100          

            #Normalizacion
            sca =RobustScaler(quantile_range=(0,95))
            sca.fit(entrada)
            entrada=sca.transform(entrada)

            #Prediccion
            predicciones=model.predict(np.array([entrada]))
            #Variable auxiliar
            a = np.zeros(shape=(largo_dia,datos_celda.shape[1]))
            prediccion = pd.DataFrame(a,columns=datos_celda.columns)
            prediccion["DL_User_Thp"]=predicciones[0].reshape(-1,1)
            prediccion_normed=prediccion.to_numpy()
            prediccion_normed=prediccion_normed[:,0] 
            prediccion=sca.inverse_transform(prediccion)
            prediccion=prediccion[:,0]


               

            #Graficos de prediccion contra valor real
            #plt.plot(tiempo, prediccion/1000, label="predicción")
            #plt.plot(tiempo, salida/1000, label="valor real")
            #plt.xticks(rotation='vertical')
            #plt.ylabel("DL_User_Thp [Mbps]")
            #plt.xlabel("Hora")
            #plt.title(cell)
            #plt.legend()
            #plt.show()
    return salida/1000, prediccion/1000       
print("Ejemplo de celdas banda 2600")
x_real,x_pred=regresion(Datos,celdas,modelo_L_20,modelo_L_60,modelo_L_100)

tiempo=np.arange(6,25,1)
plt.plot(tiempo, x_pred, label="predicción")
plt.plot(tiempo, x_real, label="valor real")
plt.xticks(rotation='vertical')
plt.ylabel("DL_User_Thp [Mbps]")
plt.xlabel("Hora")
plt.title(celdas)
plt.show()
print(" NUT prediccion ")
print(np.array(x_pred))
print(" NUT real")
print(np.array(x_real))