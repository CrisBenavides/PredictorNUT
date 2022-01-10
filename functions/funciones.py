import numpy as np

#Calculo de la entropia
def entropia(x):
    # Crea una lista con cantidad de veces que ocurre cada valor
    data_set = list(set(x))
    lista_frec = []
    for valor in data_set:
        contador = 0.
        for i in x:
            if i == valor:
                contador += 1
        lista_frec.append(float(contador) / len(x))

    # calculo de la entropia (sumatoria)
    entropia = 0.0
    for frec in lista_frec:
        entropia += frec * np.log2(frec)
    entropia = -entropia
    return entropia
#Calculo del mean crossing rate   
def MCR(arreglo):
    cambios=0
    for i in list(range(len(arreglo))):
        if i==0:
            pass
        else:
            auxiliar=np.abs(np.sign(arreglo[i-1]-np.mean(arreglo))-np.sign(arreglo[i]-np.mean(arreglo)))
            if auxiliar==2:
                cambios=cambios+1
            else:
                pass
    return cambios/(len(arreglo)-1)       