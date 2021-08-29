
from torchdiffeq import odeint
import torch
import torch.nn as nn
import numpy as np 
import math as m
import matplotlib.pyplot as plt


import Initial_Parameters as ip

theta_P, theta_Q, theta_R, theta_S, theta_Td, theta_Tu = ip.theta_vals
a_P, a_Q, a_R, a_S, a_Td, a_Tu = ip.a_vals
b_P, b_Q, b_R, b_S, b_Td, b_Tu = ip.b_vals
sampling_rate = 1/ip.dt
rr = ip.RR
hr_factor = np.sqrt((60/rr)/60)

t_P = theta_P * np.sqrt(hr_factor)
t_Q = theta_Q * hr_factor
t_R = theta_R 
t_S = theta_S * hr_factor
t_Td = theta_Td * np.sqrt(hr_factor)
t_Tu = theta_Tu * np.sqrt(hr_factor)

a_Td = a_Td * (hr_factor**2.5)
a_Tu = a_Tu * (hr_factor**2.5)

b_P = b_P * hr_factor
b_Q = b_Q * hr_factor
b_R = b_R * hr_factor
b_S = b_S * hr_factor
b_Td = b_Td * 1/hr_factor
b_Tu = b_Tu * hr_factor


def derivs_calc(t, y): 
    # Interesante: 
    # Dentro de esta función da lo mismo lo que ocurra, siempre que sea capaz de recibir como argumentos Tensores, y retornar Tensores. 
    # Ahora, si estos tensores son descompuestos y utilizados como numpys, o floats en bruto, da lo mismo con tal que se respete la tensoridad del argumento y la salida. 
    X, Y, Z = y
    alfa = 1 - np.sqrt(X**2 + Y**2)
    w = (2*m.pi)/rr
    theta = m.atan2(Y,X)

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
    derivs = torch.tensor(derivs)
    return derivs #OJO La salida TIENE que ser un Tensor, no una lista de tensores. 


### USO DEL ODEINT DEL PYTORCH PARA LA CREACIÓN DE LA SEÑAL 
# El ode de Pytorch no acepta parámetros, por lo que estos tienen que quedar harcoded de antes. 

# Declaro parámetros iniciales y vector de tiempo. 
ti = 0
tf = rr * 2
samples = int( (tf-ti)/ip.dt )

y0 = torch.tensor(ip.y0)
t = torch.linspace(ti,tf, samples)

# Utilizo el odeint. 
# Este me retorna un Tensor de t.shape filas y 3 columnas. (cada columna registra una dimensión X,Y,Z)
a = odeint(derivs_calc, y0,t) 

# Transposición del Tensor, para obtener los vectores de coordenadas
X, Y, Z = a.T

# Ploteo de la Señal Z
plt.plot(Z)
plt.show()
