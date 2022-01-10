#Creacion de ventanas con la estructura del clasificador
from math import floor
import numpy as np
import pandas as pd
import sys
sys.path.append('/home/userbda/Documents/predictornut/functions')
from funciones import entropia
from funciones import MCR


def Ventanas_clasificador(Datos_banda):
    X_train=[]
    y_train=[]



    lista_celdas=Datos_banda["EutrancellFDD"].unique().tolist() 

    for celda in lista_celdas:
        
        aux=Datos_banda[Datos_banda["EutrancellFDD"]==celda]
        aux2=aux.reindex(pd.date_range(min(aux.index), max(aux.index),freq='H')).fillna(np.nan)
        aux3=aux2.between_time(start_time='6:00', end_time='00:00')[1:]
        aux4=aux3.fillna(method='ffill',axis=0).fillna(method='bfill',axis=0)

        
        #Particion en training y test  (validacion esa dentro de train)
        L=len(aux4)
        train=aux4.iloc[:int(L*1.00),1:]

        
    
        
        #Asegurar que los datos sean numeros
        train_a = np.array(train.astype(float))


        #Creacion de ventanas
        largo_dia=19
        n_dias_input=7
        n_input=largo_dia*n_dias_input    #Largo de ventana input  (de 1 dias)
       

        L_train=len(train)


     #Calculo de ventanas   
        for i in list(range(n_dias_input, floor(L_train/largo_dia)-(n_dias_input+1))):
            variable=(train.iloc[i*largo_dia:(i+n_dias_input)*largo_dia,:].rolling(19).max())
            hora_v = variable.index.hour
            max_prb_mean=(variable.iloc[(hora_v == 0)]).iloc[:,8].mean()
            if (max_prb_mean>0) & (max_prb_mean<101):

                X_train.append(train_a[i*largo_dia:(i+n_dias_input)*largo_dia,:])
                
                ventana_output=train.iloc[(i+n_dias_input)*largo_dia:(i+n_dias_input)*largo_dia+largo_dia,0]
                Valores_bajo_umbral=ventana_output.lt(3300).sum()
                if Valores_bajo_umbral>=2:
                    y_train.append(1)
                else:
                    y_train.append(0)
    #Calculo de estadisticos sobre las ventanas
    X=[]
    y=y_train
    for ventana in X_train:
        aux_kpi=[]  
        for dia in [0,1,2,3,4,5,6]:   
            for kpi in [0,1,2,3,4,5,6,7,8,9]:
                promedio=ventana[19*dia:19*(dia+1),kpi].mean()
                maximo=ventana[19*dia:19*(dia+1),kpi].max()
                minimo=ventana[19*dia:19*(dia+1),kpi].min()
                rango=maximo-minimo
                varianza=ventana[19*dia:19*(dia+1),kpi].var()
                MC_rate=MCR(ventana[19*dia:19*(dia+1),kpi])
                entropia_promedio=entropia(ventana[19*dia:19*(dia+1),kpi])/len(ventana[19*dia:19*(dia+1),kpi])
                skewness=pd.Series(ventana[19*dia:19*(dia+1),kpi]).skew()
                kurt=pd.Series(ventana[19*dia:19*(dia+1),kpi]).kurtosis()
                RMS=np.sqrt(np.mean(ventana[19*dia:19*(dia+1),kpi]**2))
                aux_kpi.append([promedio, maximo, minimo, rango, varianza,MC_rate, entropia_promedio, skewness, kurt, RMS])
        X.append([val for sublist in aux_kpi for val in sublist])      
      
    X, y = np.array(X), np.array(y)
    return X, y                
