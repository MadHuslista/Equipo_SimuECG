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

def dinamic_function(t, y, params):
    
    X, Y, Z = y     #Desempaquetamiento de los valores de las funciones involucradas en un tiempo 't'
    theta_P, theta_Q, theta_R, theta_S, theta_Td, theta_Tu, a_P, a_Q, a_R, a_S, \
    a_Td, a_Tu, b_P, b_Q, b_R, b_S, b_Td, b_Tu, RR, fresp = params #Desempaquetamiento de los parámetros involucrados en cada función derivada
    
    
    #Variables utilitarias para mantener el código legible y ordenado
    alfa = 1 - m.sqrt(X**2 + Y**2)
    w = (2*m.pi)/RR      # w = 2pi/RR . Se asume RR = 1 seg
    #print(w, RR)
    theta = m.atan2(Y, X)
    #print(theta)
    
    delta_theta_P = np.fmod((theta - theta_P),(2*m.pi))
    delta_theta_Q = np.fmod((theta - theta_Q),(2*m.pi))
    delta_theta_R = np.fmod((theta - theta_R),(2*m.pi))
    delta_theta_S = np.fmod((theta - theta_S),(2*m.pi))
    delta_theta_Td = np.fmod((theta - theta_Td),(2*m.pi))
    delta_theta_Tu = np.fmod((theta - theta_Tu),(2*m.pi))


    Zo = 0.005*m.sin(2*m.pi*(fresp)*t)
    #Zo = 0
    
    sumatoria_P = a_P * delta_theta_P * m.exp(  ( - (delta_theta_P**2)/(2 * b_P**2)   )  ) 
    sumatoria_Q = a_Q * delta_theta_Q * m.exp(  ( - (delta_theta_Q**2)/(2 * b_Q**2)   )  ) 
    sumatoria_R = a_R * delta_theta_R * m.exp(  ( - (delta_theta_R**2)/(2 * b_R**2)   )  ) 
    sumatoria_S = a_S * delta_theta_S * m.exp(  ( - (delta_theta_S**2)/(2 * b_S**2)   )  ) 
    sumatoria_Td = a_Td * delta_theta_Td * m.exp(  ( - (delta_theta_Td**2)/(2 * b_Td**2)   )  ) 
    sumatoria_Tu = a_Tu * delta_theta_Tu * m.exp(  ( - (delta_theta_Tu**2)/(2 * b_Tu**2)   )  )
    
    derivs = [
            
          alfa*X - w*Y,                                                                         #Eq para X'
          alfa*Y + w*X,                                                                         #Eq para Y'
          -(sumatoria_P + sumatoria_Q + sumatoria_R + sumatoria_S + sumatoria_Td + sumatoria_Tu) - (Z - Zo)     #Eq para Z'
            
            ]
    
    #print('z ', derivs[2])
    
    return derivs 