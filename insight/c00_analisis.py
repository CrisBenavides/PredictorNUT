import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from joblib import load
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_validate

#Cargar datos
X_L=np.load('/home/userbda/Documents/X_train_L_c.npy')
y_L=np.load('/home/userbda/Documents/y_train_L_c.npy')

X_M=np.load('/home/userbda/Documents/X_train_M_c.npy')
y_M=np.load('/home/userbda/Documents/y_train_M_c.npy')

X_P=np.load('/home/userbda/Documents/X_train_P_c.npy')
y_P=np.load('/home/userbda/Documents/y_train_P_c.npy')

#Cargar modelos
clf_L=load( '/home/userbda/Documents/GB_L.pkl') 
clf_M=load( '/home/userbda/Documents/GB_M.pkl') 
clf_P=load( '/home/userbda/Documents/GB_P.pkl') 

scoring= ['precision_macro','recall_macro','accuracy','f1_macro']

#Ejemplo de metricas para banda 2600 (L)
cv_results = cross_validate(clf_L,X_L,y_L,cv=5, scoring= scoring, return_train_score= False)
print("Metricas con 5 folds: ")
print("Accuracy: %0.2f (+/- %0.2f)" % (np.mean(cv_results['test_accuracy']), np.std(cv_results['test_accuracy']) * 2))
print("Precision: %0.2f (+/- %0.2f)" % (np.mean(cv_results['test_precision_macro']), np.std(cv_results['test_precision_macro']) * 2))
print("Recall: %0.2f (+/- %0.2f)" % (np.mean(cv_results['test_recall_macro']), np.std(cv_results['test_recall_macro']) * 2))
print("F1: %0.2f (+/- %0.2f)" % (np.mean(cv_results['test_f1_macro']), np.std(cv_results['test_f1_macro']) * 2))