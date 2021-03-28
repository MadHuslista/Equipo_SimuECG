
import wfdb
import random
import math as m 
import matplotlib.pyplot as plt 
import Initial_Parameters as init_params 
from Derivative_Core import Derivative_Core 
import numpy as np 



class Learner(): 

    """
    Este objeto desarrolla sólo dos cosas: 
        1.- Para su inicialización construye un pulso ECG en base al modelo de McSharry y los parámetros entregados, 
            e incorpora como atributo en 'batch' de pulsos entregado desde la BD
        2.- Posee un Método para el cálculo del error acumulado entre el 'batch' entregado y la señal pulso construida. 
    """

    def __init__(self, batch, params=0): 

        if not(params):                                             #Si no recibe parámetros, los genera al azar.
            t = [random.uniform(-m.pi, m.pi) for i in range(6)]
            a = [random.uniform(-30, 30) for i in range(6)]
            b = [random.uniform(0, 1) for i in range(6)]
            y0 = [random.uniform(0, 1) for i in range(3)]
            self.params = t + a + b + y0
        else:
            self.params = params                                


        self.error = 0                                              #Registro del Error detectado
        self.batch = batch                                          #Incorporación del batch entregado
        self.base = sum(sum(np.array(batch)**2))                    #Cálculo del valor base del batch
        self.Sig_Core = Derivative_Core()                           #Crea una instancia del modelo de McSharry
        self.signal = self.Sig_Core.calc_model(self.params)         #Utiliza el modelo de McSharry para crear una señal en base a los parámetros entregados.

    def calc_error(self):                                           #Por temas de eficiencia, esta función se construye como iterador.

        for b_sig in self.batch:                                    #Por cada señal b_sig en el batch entregado

            self.error += sum((b_sig - self.signal[1])**2)          #Adiciona el error entre la señal creada en base a parámetros y b_sig
            
            yield self.error                                        #Aquí retorna el error actual con un yield, porque la función que busca el mejor 'Learner', descarta este cálculo una vez el error superó el actual mínimo error. 

    
    

if __name__ == "__main__": 
    ecg_recover = wfdb.rdsamp("Derivations_BDs/BD_II_signal")
    s = ecg_recover[0].transpose()

    s = s[0:10]


    a = Learner(s,p)

    