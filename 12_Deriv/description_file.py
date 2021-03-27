import pandas as pd
import numpy as np
import wfdb
import matplotlib.pyplot as plt 

#Lee la señal; wfdb en particular es capaz de leer toda la info que contiene el dat. 
#Descripción del .dat: 
#   Es una tupla de dos elementos: (data, labels)
#   A su vez, la data es un array de 1000x12 en float. 
#   -> Cada fila es un vector con los datos de las 12 derivaciones (orden en labels) y son 1000 filas. Es decir, cada columna es una derivación, cada fila es un dato del sampleo 

#   Por su parte el labels es un diccionario que contiene: 
#       fs          -> Freq Sampleo
#       sig_len     -> Cantidad de datos (en este caso 1000 x señal)
#       n_sig       -> Cantidad de seañles (en este caso 12)
#       base_date   -> ? (aparece como none)
#       base_time   -> ? (aparece como none)
#       units       -> Unidad de medida de los valores de la señal
#       sig_name    -> Lista que contiene -en orden correspondiente- los nombres de las derivaciones
#       comments    -> Comentarios. 
ecg = wfdb.rdsamp('00003_lr',)

print(type(ecg[1]["sig_name"]))

#signal = ecg[0]
#signal = signal.transpose()

#I = signal[0]

#plt.plot(I)
#plt.show()