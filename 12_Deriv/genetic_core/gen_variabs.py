#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 02:41:17 2019

@author: meyerhof
"""
 
import math as m 
import numpy as np

"""
####################### 0.- PARÁMETROS DE CONFIGURACIÓN ####################################
"""

#FS y RR
dt = 0.002                          # En segundos 500Hz
RR = 0.9                            # RR normalizado a 900ms
Amp_ECG = 0.8
#Posición angular de cada Peak

delay = 0#-m.pi*1/3

theta_P = -(1/3)*m.pi           -0.1
theta_Q = -(1/12)*m.pi          +0.1
theta_R = 0                     
theta_S = (1/12)*m.pi           -0.1
theta_Td = ((5/9)-(1/60))*m.pi  
theta_Tu = (5/9)*m.pi           

#Determina el alto de cada peak
a_P = 0.8
a_Q = -5
a_R = 30
a_S = -7.5
a_Td = 0.5
a_Tu = 0.75


#Determina la duración de cada peak 
b_P = 0.2 
b_Q = 0.1
b_R = 0.1
b_S = 0.1 
b_Td = 0.4 
b_Tu = 0.2 


#Valores Iniciales y empaquetamiento                                     
X0 = 1
Y0 = 0
Z0 = 0.04

#Empaquetamiento 
theta_vals  = [theta_P, theta_Q, theta_R, theta_S, theta_Td, theta_Tu]
a_vals      = [a_P, a_Q, a_R, a_S, a_Td, a_Tu]
b_vals      = [b_P, b_Q, b_R, b_S, b_Td, b_Tu]
y0 = [X0, Y0, Z0] 

#En resumen mi modelo tiene thetha(6) + a(6) + b(6) + y0(3) = 21 parámetros. 