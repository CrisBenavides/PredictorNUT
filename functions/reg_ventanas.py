#Creacion de ventanas con la estructura del regresor

import numpy as np
import pandas as pd
from math import floor
from sklearn.preprocessing import RobustScaler

def Ventanas_regresion(Datos_banda,prb):
    if prb==20:
        prb_min=0
    if prb==60:
        prb_min=20
    if prb==100:
        prb_min=60        


    X_train=[]
    y_train=[]



    lista_nula=[]

    lista_celdas=Datos_banda["EutrancellFDD"].unique().tolist() 


    #Filtro por hora


    for celda in lista_celdas:


        aux=Datos_banda[Datos_banda["EutrancellFDD"]==celda]
        aux2=aux.reindex(pd.date_range(min(aux.index), max(aux.index),freq='H')).fillna(np.nan)
        aux3=aux2.between_time(start_time='6:00', end_time='00:00')[1:]
        aux4=aux3.fillna(method='ffill',axis=0).fillna(method='bfill',axis=0)


        if aux4.isna().sum().sum()>0:
            lista_nula.append(celda)

        #Largo total
        L=len(aux4)

        #Cantidad de puntos que representa un dia
        largo_dia=18+1  #ref las 7:00 por eso el "18" 

        #Particion en training y test  (validacion esa dentro de train)
        train=aux4.iloc[:int(L*0.9),1:]
        test=aux4.iloc[int(L*0.9):,1:]

        #Normalizar datos
        scaler =RobustScaler(quantile_range=(0,95)) 
        scaler.fit(train)
        train_n=scaler.transform(train)
        test_n=scaler.transform(test)

        #Asegurar que los datos sean numeros
        train_n = train_n.astype(float)
        test_n = test_n.astype(float)



        #Creacion de ventanas
        n_dias_input=7

        L_train=len(train_n)



        for i in list(range(n_dias_input, floor(L_train/largo_dia)-(n_dias_input+1))):
            variable=(train.iloc[i*largo_dia:(i+n_dias_input)*largo_dia,:].rolling(19).max())
            hora_v = variable.index.hour
            max_prb_mean=(variable.iloc[(hora_v == 0)]).iloc[:,8].mean()
            if (max_prb_mean>prb_min) & (max_prb_mean<prb):
                X_train.append(train_n[i*largo_dia:(i+n_dias_input)*largo_dia,:])
                y_train.append(train_n[(i+n_dias_input)*largo_dia:(i+n_dias_input)*largo_dia+largo_dia,0])


    X_train, y_train=np.array(X_train), np.array(y_train)            
    return  X_train, y_train