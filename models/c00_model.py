#Cargar
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from joblib import load
from math import sqrt
from math import floor
from sklearn.ensemble import GradientBoostingClassifier


X_L=np.load('/home/userbda/Documents/X_train_L_c.npy')
y_L=np.load('/home/userbda/Documents/y_train_L_c.npy')
X_M=np.load('/home/userbda/Documents/X_train_M_c.npy')
y_M=np.load('/home/userbda/Documents/y_train_M_c.npy')
X_P=np.load('/home/userbda/Documents/X_train_P_c.npy')
y_P=np.load('/home/userbda/Documents/y_train_P_c.npy')



#Entrenar clasificadores
print("Entrenando clasificadores...")
clf_L=GradientBoostingClassifier(random_state=0,max_features=700, n_iter_no_change=1, validation_fraction=0.15)#'sqrt')
clf_L.fit(X_L,  y_L)


clf_M=GradientBoostingClassifier(random_state=0,max_features=700, n_iter_no_change=1, validation_fraction=0.15)#'sqrt')
clf_M.fit(X_M,  y_M)


clf_P=GradientBoostingClassifier(random_state=0,max_features=700, n_iter_no_change=1, validation_fraction=0.15)#'sqrt')
clf_P.fit(X_P,  y_P)


#Guardar clasificadores
dump(clf_L, '/home/userbda/Documents/GB_L.pkl') 
dump(clf_M, '/home/userbda/Documents/GB_M.pkl') 
dump(clf_P, '/home/userbda/Documents/GB_P.pkl') 

print("Entrenamiento finalizado...")