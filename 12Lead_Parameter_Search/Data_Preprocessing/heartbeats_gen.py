import numpy as np
import wfdb
import matplotlib.pyplot as plt 
from biosppy.signals import ecg
from tachogram_detect import extract_heartbeats_RRadapted
from pathfiles import files #Toma todas las direcciones de los archivos originales. 
import time 

#Este código toma todas los archivos de señales de la carpeta de BD (señales de 10seg y 12 canales. 1 por derivación) 
# y la transforma en 12 archivos. Cada archivo es una base de datos que contiene todos los pulsos individuales correspondientes a cada derivación. 

sampling_rate = 500.

TRAINING_DERIVATION = 'I'
derivations = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

ST = time.time() #Mide el tiempo. 

for i in derivations:                                                                                                   #Por cada Derivación

    l_st = time.time()

    #CONSTRUCCIÓN DE LA BD

    TRAINING_DERIVATION = i   #Abstracción de la derivación. 
    deriv_index = derivations.index(TRAINING_DERIVATION) #Obtención del índice correspondiente

    heartbeats = []                                                                                                     #Se crea una lista que contendrá TODOS los pulsos correspondientes a la derivación    
    ch_name = []                                                                                                        #Lo mismo, pero para el metadato nombre correspondiente a cada uno de esos pulsos

    for f in files:                                                                                                     #Se lee cada uno de los registros de la BD original        

        ecg_signal = wfdb.rdsamp(f)                                 
        signal = ecg_signal[0].transpose()[deriv_index]                                                                 #De este archivo se rescata sólo la señal correspondiente a la derivación

        if ecg_signal[1]['sig_name'][deriv_index] in ['AVR', 'AVL','V1', 'V2']:                                         #Doy vuelta la señal, así efectivamente toma el R, que en estas derivaciones es negativo
            signal *= -1    
        
        out = ecg.ecg(signal= signal, sampling_rate=sampling_rate, show=False)                                          #Procesa la señal rescatada de varias maneras:

        filtered = out['filtered']                                                                                      #una de ellas es filtrar la señal 
        rpeaks = out['rpeaks']                                                                                          #Y otra es detectar las posiciones de los rpeaks

        if ecg_signal[1]['sig_name'][deriv_index] in ['AVR', 'AVL','V1', 'V2']:                                         #Una vez efectuado el rpocesamiento la desdoy vuelta, para devolverla a su estado original. 
            filtered *= -1

        t = extract_heartbeats_RRadapted(out['filtered'], out['rpeaks'], sampling_rate,before=0.5, after=0.5)           #Se extraen los pulsos individuales de la señal ya filtrada

        ch = [ f[-8:-3] + '-' + str(i) for i in range(len(t))]                                                          #Se construye el metadato nombre para cada uno de los pulsos
        
        heartbeats.extend(t)                                                                                            #Se extiende la lista con los pulsos extraídos
        ch_name.extend(ch)                                                                                              #Se extiende la lista con los metadatos construídos

    heartbeats = np.array(heartbeats)                                                                                   #Se transpone la matriz para guardarla según las especificaciones de wfdb                                                        
    h = heartbeats                                                                                                      #Creación de una copia independiente para posterior constraste de la recuperación
    heartbeats = heartbeats.transpose()                         
    units = ['mV' for i in range(heartbeats.shape[1])]                                                                  #Construcción de otros metadatos genéricos necesarios
    ch_name = [TRAINING_DERIVATION + '-' + ch_name[i] for i in range(len(ch_name))]
    sig_name = 'Derivations_BDs/'+ 'BD_'+TRAINING_DERIVATION+'_signal'
    fmt = ['32'for i in range(heartbeats.shape[1]) ]



    wfdb.wrsamp(sig_name, fs=sampling_rate,p_signal=heartbeats, units = units, sig_name=ch_name, fmt=fmt)               #Creación del archivo BD correspondinte a la Derivación. Contiene TODOS los pulsos correspondientes. 

    #RECUPERACIÓN DE LA BD CREADA Y CONSTRASTE 

    ecg_recover = wfdb.rdsamp(sig_name)                                                                                 #Lectura de la BD recién creada
    sign_recover = ecg_recover[0].transpose()

    err = sum(sum((h - sign_recover)**2))                                                                               #Medición de la diferencia entre la BD leída, y la copia independiente (el original)
    base = sum(sum(h**2))                                                                                               #Se toma el valor base de la BD completa

    l_ed = time.time()

    #REPORTE DE RESULTADOS
    
    print()                                                                                                             
    print('->', TRAINING_DERIVATION)
    print(err)
    print(base)
    print(100*err/base)
    print('t: ', l_ed - l_st)
    print("========")

ED = time.time()

print()
print('t: ', ED-ST)