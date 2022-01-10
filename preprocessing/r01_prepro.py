#Script para crear datos de prueba del regresor

import numpy as np
import pandas as pd
import sys
sys.path.append('/home/userbda/Documents/predictornut/functions')
from math import floor
from sklearn.preprocessing import RobustScaler
from reg_ventanas import Ventanas_regresion




#Cargar datos iniciales
Datos=pd.read_csv('/home/userbda/Documents/Datos_hr1_pol4.csv')

#Creacion del tiempo como indice del dataframe
Datos["DATETIME_ID"]=pd.to_datetime(Datos["DATE_ID"])+pd.to_timedelta(Datos["HOUR_ID"], unit='h')
Datos=Datos.drop(columns=["DATE_ID","HOUR_ID"])
Datos["DATETIME_ID2"]=Datos["DATETIME_ID"]
Datos=Datos.set_index('DATETIME_ID')

#Filtrar KPI seleccionados (seleccion de features fue hecho anteriormente)
cols=['DATETIME_ID2',
    'EutrancellFDD',
 'DL_User_Thp',
 'UL_User_Thp',
 'DL_Cell_THROUGHPUT',
 'Active_Users_DL',
 'Active_Users_UL',
 'DL_Latency',
 'DL_TRAFFIC_VOLUME_GB',
 'UL_TRAFFIC_VOLUME_GB',
 'uso_prb',
 'Exito_paquetesDL']


Datos=Datos[cols]


#Se agrega el dia de la semana y la hora como inputs 
Datos["dia"]=Datos.index.dayofweek
Datos["hora"]=Datos.index.hour



#Se crean 3 dataframes, uno por cada banda de frecuencia
Datos_L=Datos[Datos['EutrancellFDD'].astype(str).str[0] == 'L']
Datos_M=Datos[Datos['EutrancellFDD'].astype(str).str[0] == 'M']
Datos_P=Datos[Datos['EutrancellFDD'].astype(str).str[0] == 'P']


#Se guardan las celdas por banda
Datos_L.to_csv('/home/userbda/Documents/celdas_L.csv',header=True,index=False)
Datos_M.to_csv('/home/userbda/Documents/celdas_M.csv',header=True,index=False)
Datos_P.to_csv('/home/userbda/Documents/celdas_P.csv',header=True,index=False)