#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 19:53:03 2019

@author: meyerhof
"""
 

from scipy.integrate import ode
from scipy.integrate import solve_ivp as slv 
import math as m
import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3

import matplotlib.pyplot as plt 
from matplotlib import animation

from rr import RR_gen
from deriv_core import dinamic_function
import variab

import time 


plt.close("all")





def model(param_g, param_A, param_H, theta, a, b, y0): 
    
    """#### Desempaquetado ### """
    
    hrmean, Resp_by_min, Amp_ECG, n, dt, FPS    = param_g
    Anoise, Hz_Noise, Hz_Anoise                 = param_A
    hrstd, c1, c2, f1, f2                       = param_H
    
    theta_P, theta_Q, theta_R, theta_S, theta_Td, theta_Tu  = theta
    a_P, a_Q, a_R, a_S, a_Td, a_Tu                          = a
    b_P, b_Q, b_R, b_S, b_Td, b_Tu                          = b
    
    
    """#### CÁLCULO DE VALORES ### """
    #Los cálculos de valores se hace según modelado del paper
    
    hr_factor = np.sqrt(hrmean/60)            #Factor que permite la adaptabilidad de las posiciones al ritmo cardíaco 
    print(hr_factor)
    fresp = Resp_by_min/60      

    #Posición angular de cada Peak
    theta_P = theta_P   * np.sqrt(hr_factor)
    theta_Q = theta_Q   * hr_factor
    theta_R = theta_R
    theta_S = theta_S   * hr_factor
    theta_Td = theta_Td * np.sqrt(hr_factor)
    theta_Tu = theta_Tu * np.sqrt(hr_factor)
    
    #Determina el alto de cada peak
    a_P = a_P #Estos son redundantes, pero como no generan coste en procesamiento, están sólo para mantener uniformidad en la codificación 
    a_Q = a_Q
    a_R = a_R
    a_S = a_S
    a_Td = a_Td *(hr_factor**(2.5))
    a_Tu = a_Tu  *(hr_factor**(2.5))
    
    
    #Determina la duración de cada peak 
    b_P = b_P * hr_factor
    b_Q = b_Q * hr_factor
    b_R = b_R * hr_factor
    b_S = b_S * hr_factor
    b_Td = b_Td * (1/hr_factor)
    b_Tu = b_Tu * hr_factor
    
    """#### Creación Tacograma ### """
    
    #Generación de intervalos rr. rr_times guarda la duración de cada pulsación
    rr_times = RR_gen(f1, f2, c1, c2, hrmean, hrstd, n)

    #De duraciones se transforma a unicaciones temporales en una linea de tiempo completa
    rr_axis = []
    cumulative_time = 0

    for i in rr_times:
        cumulative_time += i 
        rr_axis.append(cumulative_time)
    
    t = np.arange(0, rr_axis[-1], dt)       #rr_axis[-1] representa al último elemento de rr_axis

    RR = rr_times[0]                        #Esta definición está aquí para poder iniciar el empaquetamiento para el ODE. 
                                            #Si bien 'RR' actúa como constante, como a lo largo del tiempo debe ser actualizada, aquí sería como especificar otro valor inicial 

    #print(rr_times)

    params = [theta_P, theta_Q, theta_R, theta_S, theta_Td, theta_Tu, a_P, a_Q, a_R, a_S, a_Td, a_Tu, b_P, b_Q, b_R, b_S, b_Td, b_Tu, RR, fresp]
    
    
    """#### Utilización del solver ### """

    ode_st = time.time()
     
    solver = ode(dinamic_function)          #Creación de una instancia del ODE, con la 'dinamic_function' como función llamable
    solver.set_integrator("lsoda")          #Se setea el método de integración
    solver.set_initial_value(y0)            #Se setean los valores iniciales para X, Y, Z
    solver.set_f_params(params)             #Se setean los parametros para la función llamable
    
    
    pos = 0
    psoln = []                                                  
    for i in range(len(t)):         #Este for permite que cada vez que el parámetro RR, se actualice cada vez que el t alcance al siguiente intervalo RR. Para esto, la serie de intervalos de rr_time se pasaron a una escala temporal en rr_axis. Y cada vez que t alcanza al siguiente rr_axis, rr_time[pos] se actualiza al siguiente valor. 
        if t[i] > rr_axis[pos]:
            params[-2] = rr_times[pos+1]
            solver.set_f_params(params)        
            pos = pos+1    
            
        solver.integrate(solver.t+dt)
        psoln.append(solver.y)

    ode_ed = time.time()
    print(ode_ed-ode_st)

    """### Otra alternativa ####"""

    slv_st = time.time()

    #print(y0)
    sol = slv(dinamic_function, (t[0],t[-1]),y0,args=[params], dense_output=True, method='LSODA')

    #print(sol.sol(t).shape)

    solz = sol.sol(t)[2]     

    slv_ed = time.time()

    print(slv_ed-slv_st)

    #print(type(sol.y),sol.y.shape)
    #print(sol.sol(t))
    

    solzmin = min(solz)
    solzmax = max(solz)
    solzrange = solzmax - solzmin
    solz = (solz - solzmin) * (Amp_ECG)/solzrange

    """#### Escalamiento y Ruido ### """
        
    psoln_transp = np.array(psoln).T
    z = psoln_transp[2]
    
    
    zmin = min(z)
    zmax = max(z)
    zrange = zmax - zmin              #Aquí se obtiene el rango máximo de z
    z = (z - zmin)*(Amp_ECG)/zrange #-0.4    #Aquí cada dato, es escalado en proporción zrange:1.6 con una regla de 3 simple =>  Zrange/(Z- zmin) = 1.6 / x ; donde x es el nuevo valor de z
    
    white_noise = 2*np.random.rand(len(z), 1) -1    #Aquí el np.random.rand() genera ruido aleatorio de distribución uniforme, entre [0,1]. Luego al multiplicar por 2, el rango queda en [0,2], y finalmente al restar en uno, queda [-1,1] => Conclusión: Ruido aleatorio entre -1 y 1
    for i in range(len(z)):
        z[i] = z[i] + Anoise*white_noise[i]         #Aquí el ruido aleatorio entre [-1,1] se escala a la magnitud deseada del ruido (Anoise) y se suma a cada valor de z[i]
        
    noise = np.sin(2*np.pi*t*Hz_Noise)
    z = z + Hz_Anoise*noise
    
    
    x_values = np.array(psoln).T[0]
    y_values = np.array(psoln).T[1]
    z_values = z
    
    return x_values, y_values, z_values, t, solz
    
if __name__ == "__main__":
      
    #Esto está sólo para probar el modelo o  probarlo de manera independiente
    param_gener = variab.param_gener
    param_Artf = variab.param_Artf
    param_HVR = variab.param_HVR
    theta_vals = variab.theta_vals
    a_vals = variab.a_vals
    b_vals = variab.b_vals
    y0 = variab.y0
    
    x, y, z, t, z_s = model(param_gener, param_Artf, param_HVR, theta_vals, a_vals, b_vals, y0)
    
    
    plt.figure()
    plt.plot(t, z, label='orig')
    plt.plot(t,z_s, label='z_s')

    plt.legend()

    print(len(z), len(z_s))
    
    print("======")
    print(sum((np.zeros_like(z) - z)**2))
    print(sum((z_s - z)**2))
    print(100*sum((z_s - z)**2)/sum((np.zeros_like(z) - z)**2))
    plt.show()