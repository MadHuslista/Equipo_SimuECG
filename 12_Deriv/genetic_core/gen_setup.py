
import numpy as np 
import matplotlib.pyplot as plt 
import wfdb
import random as rnd 
import time


def create_subsets(signal, retain_pctg=0, batch_size=10): 

        """
        Controla la creación de nuevos Subsets. Puede tomar dos comportamientos:
        1.- Si sólo se entrega una BD, asume que es una BD original con n pulsos individuales, y retorna un objeto con la misma BD reorganizada en m batches de tamaño batch_size (m = n // batch_size + 1)
        2.- De de indicarse un retain_pctg junto con la  BD, asume que recibe una BD organizada en m batches, y efectúa una randomización de toda la BD generando nuevos batches. 
            - Cada uno de estos nuevos batches mantiene un porcentaje retain_pctg del 100% de los pulsos contenidos en el batch correspondiente original, 
              enviando el % restante a un pool para su randomización. 
            - Efectuado esto, los nuevos batch completan su 100% = batch_size, tomando pulsos al azar del pool recién randomizado. 
        
        """
    
        if not(retain_pctg):                                    #Comportamiento n° 1

            signal = np.array(signal)                           #Asegura que sea un array numpy

            sig_cant = signal.shape[0]                          #Determina el n de pulsos individuales

            indexs = np.arange(0,sig_cant,1)                    #Genera un guía para tomar los pulsos de manera random
            rnd.shuffle(indexs)

            subsets = []                                        #Objeto que contendrá la BD reorganizada que se retornará
            batch = []                                          #Objeto intermediario para la creación de cada batch
            batch_quantity = 0                                  #Registro de la cantidad de batchs credaos

            for i in indexs:                                    #Aprovechando el azar contenido en la lista indexs, se crea cada batch.

                if len(batch) < batch_size:                     #Agrega pulsos al objeto intermediario
                    batch.append(signal[i])
                else:                                           #Si el batch intermediario alcanza el batch_size
                    b_copy = list(batch)                        #Crea una copia independiente del batch creado 
                    subsets.append(b_copy)                      #Y lo agrega al subsets
                    batch = []                                  #Vacía el objeto intermediario
                    batch.append(signal[i])                     #Y guarda el pulso actual.
                    batch_quantity += 1                         #Registra la creación del batch
            else: 
                subsets.append(list(batch))                     #Una vez terminado con los pulsos, adiciona el batch con los pulsos restantes
                batch_quantity += 1                             #Y lo registra. 

        else:                                                   #Comportamiento n° 2

            torandom_pool = []                                  #Objeto que contendrá el pool a randomizar
            remain_subset = []                                  #Objeto que contendrá los batches retenidos

            for batch in signal:                                #Por cada batch en la BD
                retain_pos = int((len(batch) * retain_pctg))    #Determina la cantidad de pulsos que se mantendrán en el batch

                rnd.shuffle(batch)                              #Randomiza el batch, para que los pulsos mantenidos no sean siempre los mismos
                torandom_pool.extend(batch[retain_pos:])        #Envía los pulsos que no se mantendrán, al pool a randomizar
                remain_subset.append(batch[:retain_pos])        #Envía el batch con los pulsos mantenidos.
            
            indexs = np.arange(0,len(torandom_pool),1)          #Ejecuta la randomización del pool. 
            rnd.shuffle(indexs)
            indexs = iter(indexs)

            for batch in remain_subset:                         #Por cada batch retenido

                try: 
                    while len(batch) < batch_size:              #Y mientras el batch no alcance el batch_size
                    
                        i = next(indexs)                        #Toma una posición previamente randomizada 
                        batch.append(torandom_pool[i])          #Y agrega el pulso correspondiente al batch    
                except: 
                    break                                       #Este except controla la finalización de la completación de los batches. Se encuentra a la escucha del error que ocurrirá cuando el interador 'indexs' se termine, y al detectarlo, lo captura y envía la señal de finalzación del for.
                
            
            subsets = remain_subset                             #Renombra la BD construida. 
        
        return subsets


if __name__ == "__main__":

    ecg_recover = wfdb.rdsamp("../Derivations_Data/BD_II_signal")
    sign_recover = ecg_recover[0].transpose()

    s = create_subsets(sign_recover)

    for i in s[0]: 
        print(sum(i))
    f = create_subsets(s, 0.6)

    print()
    for i in f[0]: 
        print(sum(i))


