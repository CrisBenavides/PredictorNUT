import numpy as np
import pandas as pd
import sys
sys.path.append('/home/userbda/Documents/predictornut/functions')
from cla_ventanas import Ventanas_clasificador

#Cargar datos iniciales
Datos=pd.read_csv('/home/userbda/Documents/Datos_hr1_pol4.csv')


#Creacion del tiempo como indice del dataframe
Datos["DATETIME_ID"]=pd.to_datetime(Datos["DATE_ID"])+pd.to_timedelta(Datos["HOUR_ID"], unit='h')
Datos=Datos.drop(columns=["DATE_ID","HOUR_ID"])
Datos=Datos.set_index('DATETIME_ID') 


#Filtrar KPI seleccionados
cols=['EutrancellFDD',
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


#Separacion por banda de frecuencia
Datos_L=Datos[Datos['EutrancellFDD'].astype(str).str[0] == 'L']
Datos_M=Datos[Datos['EutrancellFDD'].astype(str).str[0] == 'M']
Datos_P=Datos[Datos['EutrancellFDD'].astype(str).str[0] == 'P']

#Datos se transforman en inputs de 7 dias y output de 1 dia
X_L ,y_L = Ventanas_clasificador(Datos_L)
X_M ,y_M = Ventanas_clasificador(Datos_M)
X_P ,y_P = Ventanas_clasificador(Datos_P)



print('X_train_L shape == {}.'.format(X_L.shape))
print('y_train_L shape == {}.'.format(y_L.shape))

print('X_train_M shape == {}.'.format(X_M.shape))
print('y_train_M shape == {}.'.format(y_M.shape))

print('X_train_P shape == {}.'.format(X_P.shape))
print('y_train_P shape == {}.'.format(y_P.shape))

#Se guardan los arreglos procesados
np.save('/home/userbda/Documents/X_train_L_c.npy',X_L)
np.save('/home/userbda/Documents/y_train_L_c.npy',y_L)
np.save('/home/userbda/Documents/X_train_M_c.npy',X_M)
np.save('/home/userbda/Documents/y_train_M_c.npy',y_M)
np.save('/home/userbda/Documents/X_train_P_c.npy',X_P)
np.save('/home/userbda/Documents/y_train_P_c.npy',y_P)

#Se guardan las celdas por banda
Datos_L['EutrancellFDD'].to_csv('/home/userbda/Documents/celdas_L.csv',header=True,index=False)
Datos_M['EutrancellFDD'].to_csv('/home/userbda/Documents/celdas_M.csv',header=True,index=False)
Datos_P['EutrancellFDD'].to_csv('/home/userbda/Documents/celdas_P.csv',header=True,index=False)