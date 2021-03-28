#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 02:41:17 2019

@author: meyerhof
"""

import math as m 

"""
####################### 0.- PARÁMETROS DE CONFIGURACIÓN ####################################
"""

#Parámetros Generales

hrmean = 60                         #Frecuencia Cardíaca
Resp_by_min = 15                    #Frecuencia Respiratoria
Amp_ECG = 1.7                       #Amplitud Máxima ECG
n = 30                              #Cantidad de Pulsaciones simuladas
dt = 0.001                           # En segundos
FPS = 40

#Control de Artefactos
Anoise = 0.15                       #Amplitud del Ruido Aleatorio
Hz_Noise = 50                       #Frecuencia de la Interferencia
Hz_Anoise = 0.05                    #Amplitud de la Interferencia


#Variabilidad del Pulso Cardíaco
hrstd = 0                           #Desviación Estándar de la Frecuencia Cardíaca
c1 = 2*m.pi*0.01                    #Desviación Estándar Onda Mayer
c2 = 2*m.pi*0.01                    #Desviación Estándar Onda RSA
f1 = 0.1*2*m.pi                     #Frecuencia Central Onda Mayer
f2 = 0.25*2*m.pi                    #Frecuencia Central Onda RSA


#Posición angular de cada Peak
theta_P = -(1/3)*m.pi 
theta_Q = -(1/12)*m.pi 
theta_R = 0
theta_S = (1/12)*m.pi 
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
param_gener = [hrmean, Resp_by_min, Amp_ECG, n, dt, FPS]
param_Artf  = [Anoise, Hz_Noise, Hz_Anoise]
param_HVR   = [hrstd, c1, c2, f1, f2]

theta_vals  = [theta_P, theta_Q, theta_R, theta_S, theta_Td, theta_Tu]
a_vals      = [a_P, a_Q, a_R, a_S, a_Td, a_Tu]
b_vals      = [b_P, b_Q, b_R, b_S, b_Td, b_Tu]
y0 = [X0, Y0, Z0] 