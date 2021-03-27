#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 02:50:55 2019

@author: madhuslista
"""

# MANEJO DEL SISTEMA DIFERENCIAL 


"""
Argumentos func para ODE. 
Función Python que retorna una lista de valores correspondientes a las n 
funciones correspondientes al modelo dinámico, a un tiempo t. 
"""

import math as m 
import numpy as np 
from scipy.integrate import solve_ivp as slv 
from scipy.integrate import ode
from biosppy.signals import ecg
import gen_variabs as gv

import matplotlib.pyplot as plt 

from pprint import pprint



#En resumen mi modelo tiene thetha(6) + a(6) + b(6) + y0(3) = 21 parámetros. 

class Derivative_Core: 

    def __init__(self, sampling_rate=500,rr=0.9): 

        self.dt = 1/sampling_rate 
        self.rr = rr

        self.hr_factor = np.sqrt((60/rr)/60) #En teoría es np.sqrt(hrmean/60) pero el hrmean se expresa directamente como 60/rr
        
    def derivs_calc(self, t,y, params):    

        #Desempaquetamiento de variables
        self.t_P = params[0]    * np.sqrt(self.hr_factor)   
        self.t_Q = params[1]    * self.hr_factor            
        self.t_R = params[2]                                
        self.t_S = params[3]    * self.hr_factor            
        self.t_Td = params[4]   * np.sqrt(self.hr_factor)       
        self.t_Tu = params[5]   * np.sqrt(self.hr_factor)   

        self.a_P = params[6]
        self.a_Q = params[7]
        self.a_R = params[8]
        self.a_S = params[9]
        self.a_Td = params[10]  * (self.hr_factor **(2.5))
        self.a_Tu = params[11]  * (self.hr_factor **(2.5))

        self.b_P = params[12]   * self.hr_factor
        self.b_Q = params[13]   * self.hr_factor
        self.b_R = params[14]   * self.hr_factor
        self.b_S = params[15]   * self.hr_factor
        self.b_Td = params[16]  * 1/self.hr_factor
        self.b_Tu = params[17]  * self.hr_factor

        


        #Reasignación para comodidad
        X,Y,Z = y
        t_P, t_Q, t_R, t_S, t_Td, t_Tu = self.t_P, self.t_Q, self.t_R, self.t_S, self.t_Td, self.t_Tu 
        a_P, a_Q, a_R, a_S, a_Td, a_Tu = self.a_P, self.a_Q, self.a_R, self.a_S, self.a_Td, self.a_Tu 
        b_P, b_Q, b_R, b_S, b_Td, b_Tu = self.b_P, self.b_Q, self.b_R, self.b_S, self.b_Td, self.b_Tu 

        #Variables utilitarias para el cálculo
        alfa = 1 - m.sqrt(X**2 + Y**2)
        w = (2*m.pi)/self.rr
        theta = m.atan2(Y, X)

        dt_P = np.fmod((theta - t_P),(2*m.pi))
        dt_Q = np.fmod((theta - t_Q),(2*m.pi))
        dt_R = np.fmod((theta - t_R),(2*m.pi))
        dt_S = np.fmod((theta - t_S),(2*m.pi))
        dt_Td = np.fmod((theta - t_Td),(2*m.pi))
        dt_Tu = np.fmod((theta - t_Tu),(2*m.pi))

        sum_P = a_P * dt_P * m.exp(     ( - (dt_P**2)/(2 * b_P**2)   )  ) 
        sum_Q = a_Q * dt_Q * m.exp(     ( - (dt_Q**2)/(2 * b_Q**2)   )  ) 
        sum_R = a_R * dt_R * m.exp(     ( - (dt_R**2)/(2 * b_R**2)   )  ) 
        sum_S = a_S * dt_S * m.exp(     ( - (dt_S**2)/(2 * b_S**2)   )  ) 
        sum_Td = a_Td * dt_Td * m.exp(  ( - (dt_Td**2)/(2 * b_Td**2)   )  ) 
        sum_Tu = a_Tu * dt_Tu * m.exp(  ( - (dt_Tu**2)/(2 * b_Tu**2)   )  )      
        
        derivs = [

            alfa*X - w*Y,
            alfa*Y + w*X,
            -(sum_P + sum_Q + sum_R + sum_S + sum_Td + sum_Tu) - Z
        ]        
        return derivs
        
    def calc_model(self, p):

        #Cálculo de la Señal
            #Como el modelo inicia en ángulo 0, pero el heartbeat se describe durante el rango [-pi:+pi]
            #Ocurre que de calcular sólo 1 heartbeat se obtiene la mitad posterior del primero + la mitad anterior del segundo. 
            #Por tanto, se calculan 2 heartbeats para así obtener un heartbeat completo. 

        params  = p[:18]
        y0      = p[18:21]
        amp_ecg = abs(p[21])
    
        ti = 0
        tf = self.rr *2 #Determinación de los dos heartbeats de igual RR 

        samples = int(tf/self.dt)
        self.t = np.linspace(ti,tf,samples)
        
        self.sol = slv(self.derivs_calc, (ti,tf),y0, args=[params], dense_output=True, method='LSODA')    
        solz = self.sol.sol(self.t)[2]

        #Extracción del HeartBeat
            #El modelo está pensado para ejecutar un beat de la señal en una circunferencia de 2PI. 
            #Por tanto, con un RR constante, los 2PI se expresan directamente en self.rr*/self.dt
            #Luego, como la señal inicia en ángulo 0, mientras que las posiciones se expresan en el rango -pi a +pi
            #La señal inicia en la mitad de un heartbeat quedando los dos heartbeat como [0, pi, 0, pi, lim -> 0 ] (el último queda como lim -> 0, porque el punto efectivamente correspondiente al 0 pertenece al siguiente heartbeat por lo que aquí no aparece.)
            #Por tanto el heartbeat de interés se encuentra entre pi y pi. 
            #Por tanto, basta transformar pi a samples y calcular señal[pi:-pi]
            #Tal que pi_samples = samples_beat * pi/(2pi) = samples_beat * 1/2

        pi_samples = int((self.rr/self.dt)/2)
        
        z = solz[pi_samples:-pi_samples]
        t = self.t[pi_samples:-pi_samples]

        zmin = min(z)
        zmax = max(z)
        zrange = zmax - zmin
        z = z*(amp_ecg)/zrange                
        #print(amp_ecg)
        return t, z     


if __name__ == "__main__":

    import train_variabs as tv 

    p = tv.theta_vals + tv.a_vals + tv.b_vals + tv.y0
    sampling_rate = 1/tv.dt
    rr = tv.RR

    p2 = list(p)
    p2[-1] = 0


    sig = Derivative_Core(sampling_rate, rr)
    t, z = sig.calc_model(p)

    sig2 = Derivative_Core(sampling_rate, rr)
    t2, z2 = sig2.calc_model(p2)

    print(len(z))
    


    

    plt.scatter(t,z, s=2)
    plt.plot(t,z)  
    #plt.plot(t2,z2) 
    plt.show()




    





