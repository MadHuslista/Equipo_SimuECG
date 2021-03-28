#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 13:43:59 2019

@author: meyerhof
"""

import math as m
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import animation

from model_func import model
import variables_func as varfun #Aquí están las variables de inicialización
import Slider_Interfaz as slid

plt.close("all")
slid.plt.close("all")


"""
####################### 0.- Inicialización de Parámetros ####################################
#######################       (Variables Globales)       ####################################
"""

#Se va al variables_func.py y se recuperan las listas en las que están las variables con sus valores default
#De allí se traspasan sus valores a nuevas listas y se independiza su referencia 
#de las listas de 'variables_func.py' a través del método .copy(); 
#De esta manera se puede trabajar libremente en la modificación de estos valores, sin generar alteraciones inesperadas.
 
param_gener = varfun.param_gener.copy()
param_Artf = varfun.param_Artf.copy()
param_HVR = varfun.param_HVR.copy()
theta_vals = varfun.theta_vals.copy()
a_vals = varfun.a_vals.copy()
b_vals = varfun.b_vals.copy()
y0 = varfun.y0.copy()



"""
####################### 2.2.- Elementos Slider ####################################
"""

#En general las slider tienen el método 'on.changed'; pero esto dispara un procesamiento completo de
#para cada movimiento mínimo del slider. Lo cual es demasiada carga para el procesamiento y pega el sistema. 
#Entonces se utilizan los botones de 'Simular' 


                                
def update_Artf(event):                     #Función para actualizar los parámetros una vez llamado el botón simular.         

    global Flag                             #Tiene que estar definida como global, sino asume que es una Flag local, distinta de la Flag local usada para señalizar. 
    param_Artf[0] = slid.s_Anoise.val       #Este no necesita definida como global porque ya se inicializó como global (el scope default de variables fuera de función)
    param_Artf[1] = slid.s_Hznoise.val
    param_Artf[2] = slid.s_AHznoise.val
    Flag = True


slid.fig_Artf.show()                        #Se muestra la ventana de control 
slid.sim_Artf.on_clicked(update_Artf)       #Se conecta el botón de simulación con la función de actualización. Se hace aquí para acceder a las referencias de los parámetros. 



def update_gen(event):                      #Función para actualizar los parámetros una vez llamado el botón simular. 
    global Flag
                                            # necesita generación completa de la señal <- Ni idea del por qué de este comentario. 
    param_gener[0] = slid.s_hrmean.val
    param_gener[1] = slid.s_resp.val
    param_gener[2] = slid.s_Amp_ECG.val  
    param_gener[3] = slid.s_n.val

                                            # Revisar FPs abajo. Genera problema con el FPS si no se actualiza hacia abajo también
    param_gener[4] = 1/(10**slid.s_dt.val)
    param_gener[5] = slid.s_FPS.val
    Flag = True


slid.fig_gen.show()                         #Se muestra la ventana de control 
slid.sim_gen.on_clicked(update_gen)         #Se conecta el botón de simulación con la función de actualización. Se hace aquí para acceder a las referencias de los parámetros. 



def update_HVR(event):                      #Función para actualizar los parámetros una vez llamado el botón simular. 
    global Flag
    param_HVR[0] = slid.s_hrstd.val
    param_HVR[1] = 2*m.pi*slid.s_c1.val
    param_HVR[2] = 2*m.pi*slid.s_c2.val
    param_HVR[3] = 2*m.pi*slid.s_f1.val
    param_HVR[4] = 2*m.pi*slid.s_f2.val
    Flag = True


slid.fig_HVR.show()                         #Se muestra la ventana de control 
slid.sim_HVR.on_clicked(update_HVR)         #Se conecta el botón de simulación con la función de actualización. Se hace aquí para acceder a las referencias de los parámetros. 



def update_theta(event):                    #Función para actualizar los parámetros una vez llamado el botón simular. 
    global Flag
    theta_vals[0] = slid.s_tP.val * m.pi
    theta_vals[1] = slid.s_tQ.val * m.pi
    theta_vals[2] = slid.s_tR.val * m.pi
    theta_vals[3] = slid.s_tS.val * m.pi
    theta_vals[4] = slid.s_tTd.val * m.pi
    theta_vals[5] = slid.s_tTu.val * m.pi
    Flag = True


slid.fig_theta.show()                       #Se muestra la ventana de control 
slid.sim_th.on_clicked(update_theta)        #Se conecta el botón de simulación con la función de actualización. Se hace aquí para acceder a las referencias de los parámetros. 



def update_gauss(event):                    #Función para actualizar los parámetros una vez llamado el botón simular. 
    global Flag
    a_vals[0] = slid.s_gaP.val
    a_vals[1] = slid.s_gaQ.val
    a_vals[2] = slid.s_gaR.val
    a_vals[3] = slid.s_gaS.val
    a_vals[4] = slid.s_gaTd.val
    a_vals[5] = slid.s_gaTu.val

    b_vals[0] = slid.s_gbP.val
    b_vals[1] = slid.s_gbQ.val
    b_vals[2] = slid.s_gbR.val
    b_vals[3] = slid.s_gbS.val
    b_vals[4] = slid.s_gbTd.val
    b_vals[5] = slid.s_gbTu.val
    Flag = True


slid.fig_gauss.show()                       #Se muestra la ventana de control 
slid.sim_g.on_clicked(update_gauss)         #Se conecta el botón de simulación con la función de actualización. Se hace aquí para acceder a las referencias de los parámetros. 


"""
####################### 2.1- Generador para Animación ####################################
"""
#Para entender el proceso de animación completo: 
#   1°:     Ir al final del doc y entender FuncAnimation 
#   2°:     Ir a  "3.- Animacióin 2D" para conocer el esqueleto de la figure() de animación 
#   3°:     Entender la función 'dibujadora': ecg_beat() 
#   4°:     Entender la función 'generadora de datos': generator()


Flag = False                                        #Trigger para la Actualización de la Data el Algo de Visualización


def generator(dpf):                                         #Función generadora de Data. 

    """
    Por ahora el modelo siempre genera la secuencia completa de datos para toda la animación. 
    No genera un sólo frame. (posibilidad de mejora aquí)
    Entonces lo que hace esta función es tomar esta nueva secuencia completa, identificar la posición del dato 
    de la lista, correspondiente al frame "i" en el que ocurrió el cambio de parámetros (si es que hubo,
    lo que está determinado por "Flag") 
    Y construir una nueva lista de datos tal que sea un frankenstein entre los datos antiguos desde el origen 
    hasta la posición de cambio -1, unidos con los datos nuevos desde la posición de cambio hasta el fin de la lista. 
    
    Ahora, como por limitaciones del FuncAnimation, esta función no puede tomar un argumento variable
    Es necesario resolver el avance frame a frame (necesario para la ubicación del punto de cambio 
    y la actualización de los datos a entregar a partir de los nuevos parámetros) de manera local, asumiendo que 
    esta función resetea cualquier parámetro inicializado dentro de esta. 
    
    La solución se aplica cambiando precisamente este comportamiento a través de 'FUNCIONES GENERADORAS'
    Gracias al comando 'yield' cada vez que se llama la función, esta se ejecuta de manera normal y retorna 
    lo que indique el comando 'yield' como si un 'return' se tratara. 
    La diferencia está en que el 'yield' guarda el estado de la función al momento de retornar y recontinua 
    > desde allí < al ser llamada nuevamente. De esta característica se aprovecha para mantener el contador 
    i += 1 que permite el avance de frames. 
    """
    i = 0                                               #Frame actual que se está animando.  
    data = []                                           #Definición de la variable que guarda la data. 

    global Flag                                         #Definición global (para que el intérprete entienda que es la bandera global de cambio y no una distinta local)

    x_val, y_val, z_val, t = model( param_gener,        #Generación de la data inicial antes de cualquier manipulación
                                    param_Artf, 
                                    param_HVR, 
                                    theta_vals, 
                                    a_vals, 
                                    b_vals, 
                                    y0)

    while True:                                         #Este loop permite evitar las reinicializaciones anteriores

        actual_point = int(dpf*i)                       #Calcula la posición actual en la que se encuentra. (Por cada frame es un avance discreto de 1 DPF)
        
        if Flag:                                        #Se analiza si se levantó la señal de cambio 
            
        
            x_val, y_val, z_m, t = model(param_gener,   #Si fue así, se llama al modelo para la regeneración de la data (ojo que aquí queda el set completo)
                                         param_Artf, 
                                         param_HVR, 
                                         theta_vals, 
                                         a_vals, 
                                         b_vals, 
                                         y0)
            
        
            a = z_val[:actual_point]                    #Genera el segmento de data antigua. Del inicio al punto actual -1
            b = z_m[actual_point:]                      #Genera el segmento de data nueva. Del punto actual al fin de la señal
            
            z_val = np.concatenate((a, b))              #Se concatenan ambas señales armando el frankenstein
            

            fi = 1 / param_gener[4]                     # Se actualiza el frame interval 
            dpf = fi / param_gener[5]                   # Y el DPF en caso que se hayan manipulado los FPS o el dt

            Flag = False                                #Se baja la bandera, reconocienndo que el cambio ya se aplicó


        data = [x_val, y_val, z_val, t, i, param_gener[4], dpf] #Se empaqueta la data a enviar a la función dibujadora 
                                                                #(Si no hubo cambio, se pasa inmediatamente aquí, lo que mantiene la data antigua)
        
        yield data                                      #Se retorna la data empaquetada y se guarda el estado actual de la función. 
                                                            
                                                        #Luego, cuando la función es vuelta a llamar (lo que ocurre cuando el gestor de animación intenta construir el siguiente frame)    
        
        i += 1                                          #Se avanza el Frame animado al siguiente
        n_frames = round(len(t)/dpf)                    #Se calcula el número total de Frames que compone la señal; calculado según: #datos/dpf = (round(len(t)/dpf))
        if i+1 >= n_frames:                             #Y si el número de frames obtenidos supera el total de frames a generar
            i = 0                                       #Vuelve al primer frame 
            z_val = z_m                                 #Y mantiene los datos nuevos como la señal a ser generada. 


"""
####################### 3.- Animacióin 2D ####################################
"""


fig_2d, ax_2d = plt.subplots()                          #Crea la figura para la animación de la señal

mtr = 8                                                 # Monitor Time Range (max lim X)
#hrmean = param_gener[0]                                 
#Amp_ECG = param_gener[2]
FPS = param_gener[5]                                    #Frames per Second
dt = param_gener[4]                                     #dt entre punto y punto 

ylim_s = 5.5                                            #Límites mV
ylim_i = -0.5                                   
                                                        #ax.plot() retorna una tupla con un objeto. La coma post el signo le dice al intérprete que desempaquete esa tupla y asigne el elemento [0] (el único) a la variable asignada. Se deriva de la lógica: "x1, x2, x3 = 1, 2, 3"
                                                        #Por tanto aquí se construye el esqueleto de las señales que luego contendrán los datos a graficar
sign, = ax_2d.plot([], [], 'g')                         #Ésta corresponde a la nueva señal que se va dibujando
signr, = ax_2d.plot([], [], 'g')                        #Ésta corresponde al rastro de la señal antigua que se va sobrescribiendo (pensar en como se ve la actualización de la señal ECG en eun monitor)

ax_2d.set_xlim([0, mtr])                                #Límites eje 
ax_2d.set_ylim(ylim_i, ylim_s)                  

ax_2d.set_xlabel('Tiempo [s]')                          #Títulos eje. 
ax_2d.set_ylabel('Voltaje [mV]')

ax_2d.set_aspect(0.4)                                   #Permite que los cuadrados de ECG se vean cuadrados. 


ax_2d.set_yticks(np.arange(ylim_i, ylim_s, step=0.5), minor=False)  #Define la ubicación de las horizontales del cuadriculado. 
ax_2d.set_yticks(np.arange(ylim_i, ylim_s, step=0.1), minor=True)


ax_2d.xaxis.grid(True, which='major', lw=1.5)           #Crea el cuadriculado
ax_2d.xaxis.grid(True, which='minor', lw=0.5)
ax_2d.yaxis.grid(True, which='major', lw=1.5)
ax_2d.yaxis.grid(True, which='minor', lw=0.5)


                                                        #Crea los esqueletos de lista que guardarán las señales. 
xdata1, ydata1 = [], []                                 #Listas Señal nueva. 
xdata2, ydata2 = [], []                                 #Listas Señal rastro. 

FI = 1 / FPS                                            # Frame Interval
DpF = FI / dt                                           # Datos por frame


def init():                                             # [OLD] Sin esta función también funciona. Documentación sugiere que es más eficiente. No lo sé
                                                        # Update: Dibuja el Frame 0. Lo que se estructura como no modificable, y utilizado por el blit para la comparación. 

    ax_2d.set_ylim(ylim_i, ylim_s)
    ax_2d.set_xlim(0, mtr)
    del xdata1[:]                                       #ni idea por qué elimino las instancia de las listas de señal ._. #REVISAR QUITÁNDOLAS.
    del ydata1[:]
    del xdata2[:]
    del ydata2[:]
    sign, = ax_2d.plot([], [])
    signr, = ax_2d.plot([], [])
    return sign, signr                                  #Aquí sólo está retornando las señales vacías y tampoco el ax_2d. Revisar si es necesario un init. 


def ecg_beat(data, sign, signr, mtr):                   #Esta función dibuja el frame to frame. 
    """    
    # La animación se establece sobre una señal (sign) ploteada entre dos puntos: un punto ancla fijado 
    # en xlim inferior, y un punto 'cursor de crecimiento' que avanza una cantidad de datos frame a frame. 
    # Esta cantidad de datos corresponde a los DPF (datos por frame) y es calculada como: 
    #                                                                  DPF = Tiempo_entre_frames/dt = (1/FPS)/dt

    # Luego la animación se reduce a avanzar el 'cursor de crecimiento' en 1 DPF por cada frame hasta llegar
    # al xlim superior, en donde:
    
    #   - este xlim superior se convierte en el nuevo xlim inferior y actualizando el punto ancla de la señal, 
    #   - se actualiza el nuevo valor para xlim superior según xlim_sup = xlim_sup + mtr ; donde:
    #                                                               mtr = Monitor Time Range = (xlim_sup - xlim_inf)
    #   - el 'cursor de crecimiento' se mantiene en el punto al que llegó, pero como se movieron los xlim,
    #       ahora se ubica en el actual xlim inferior. 
    
    # Y luego se continua el proceso de avanzar el cursor en 1 DPF por cada frame, hasta que se reinicie
    # al llegar al nuevo xlim superior. 

    # En la práctica esto se observa como que la señal crece desde xlim inf hasta llegar a xlim superior, 
    # en donde la visualización se corre hacia la derecha en 1 mtr, dejando el cursor ahora en el nuevo 
    # xlim inferior y permitiendo observar el siguiente crecimiento hasta repetir el ciclo. 

    # Además, para emular el comportamiento de 'rastro' de un monitor de ECG clásico (i.e. la nueva señal 
    # va sobreescribiendo el rastro que (ella misma) dejó en una anterior pasada por el monitor); se crea 
    # una segunda señal (signr) animada con la misma estrategia que la señal anterior; tan sólo que:
    
    #   - el punto ancla se fija en el xlim superoir,
    #   - el cursor es un 'cursor de decrecimiento' que retrocede de xlim_inf hacia xlim_sup, a igual 
    #       velocidad que el 'cursor de crecimiento' (1 DPF/frame), con cierta separación entre ambos y dando la 
    #       impresión de que el cursor de crecimiento va sobreescribiendo a la señal rastro (signr), y que
    #   - la data que está siendo ploteada por signr, corresponde al segmento de data ploteado por sing en el 
    #       ciclo inmediatamente anterior. 

    # En conjunto esto da la impresión de que una vez que sign alcanza el xlim superior, esta señal no desaparece
    # sino que se mantiene en pantalla mientras va siendo reescribiendo por la nueva señal sign. 
    # En la práctica no se 'mantiene' sino que esta señal antigua está siendo simultáneamente ploteada 
    # en 'reversa' por signr. 
    """
    
    #Desempaquetamiento de datos entregados por el generator
    z = data[2]                                         #Datos [mV]
    t = data[3]                                         #Datos [s]
    num = data[4]                                       #Frame Actual
    dt = data[5]                                        #Delta Tiempo entre punto y punto                
    DpF_n = data[6]                                     #Cantidad de Datos por cada frame. 
    # t = data[0]
    # z = data[1]
    # Posible mejora: Usar el argumento 'Frames' para pasar la data. Ahora, para cada frame, le paso la lista completa de datos. Mucho

    time_gap = 0.01                                     # Separación entre la nueva señal y la anterior. En [s]
    data_gap = time_gap/dt                              # Separación entre la nueva señal y la anterior. En posiciones de datos (en puntos)
    growth_cursor = int(round(num*DpF_n - int(data_gap/2)))     #Cursor de crecimiento
    decrease_cursor = int(round(num*DpF_n + int(data_gap/2)))   #Cursor de decrecimiento

    # print()

    xmin, xmax = ax_2d.get_xlim()                       #Obtención de los límites actuales
    ymin, ymax = ax_2d.get_ylim()

    pos_inf = int(xmin/dt)                              #A partir  de los límites actuales,calcular la posición correspondiente a los límites en la lista de datos. 
    pos_sup = int(xmax/dt)

    if pos_sup > len(t)-1:                              #Evitar que el límite máximo quede por sobre el último dato de la lista. 
        pos_sup = len(t)-1
                                                        
                                                        #Asignación de los datos a la señal creciente. 
    xdata1 = t[pos_inf:growth_cursor]                       #Asignación de ubicación temporal presente.
    ydata1 = z[pos_inf:growth_cursor]                       #Asignación de datos de mV correspondeintes. 


    #Lógica para dibujar el rastro sólo después de haber superado mtr por primera vez. 
    if growth_cursor*dt < mtr:                          
        xdata2 = []                                                     #Si no lo ha superado, componentes de signr = vacío
        ydata2 = []
    else:                                                               #De lo contrario se le asignan los datos a la señal decreciente
        xdata2 = t[decrease_cursor:pos_sup]                                     #Asignación de ubicación temporal presente.
        ydata2 = z[decrease_cursor-int(mtr/dt):pos_sup-int(mtr/dt)]             #Asignación de datos de mV correspondientes al ciclo pasado. (así se hace el efecto del rastro)


    #Lógica para determinar los límites y la cuadrícula en un inicio, y luego en actualizaciones posteriores.  
    
    if num <= 1:                                                        #Si es el primer ciclo
        ax_2d.set_xlim(0, mtr)                                          #Se setea xlim inf = 0, xlim sup = mtr ; sólo se está mostrando sign

        ax_2d.set_xticks(np.arange(0, mtr, step=0.2), minor=False)      #Se setea la cuadrícula acorde. 
        ax_2d.set_xticks(np.arange(0, mtr, step=0.04), minor=True)

        ax_2d.figure.canvas.draw()                                      #Se dibujan estos cambios en la figure. No se pasa al algoritmo de animación, porque no es necesario actualizar frame a frame. 

    elif growth_cursor*dt > xmax:                                       #Si son los ciclos después de haber superado xlim superior por primera vez
        
        ax_2d.set_xlim(xmin+mtr, xmax+mtr)                              #Se actualizan los límites al siguiente mtr

        ax_2d.set_xticks(np.arange(xmin+mtr, xmax+mtr, step=0.2), minor=False)  #Se actualiza la cuadrícula acorde
        ax_2d.set_xticks(np.arange(xmin+mtr, xmax+mtr, step=0.04), minor=True)

        ax_2d.figure.canvas.draw()                                      #Se redibujan los cambios. 

    #Asignación de la data a plotear.                                   
    sign.set_data(xdata1, ydata1)                                       #Se asigna el resultado de las asignaciones anteriores. => Es decir, se dibujan. 
    signr.set_data(xdata2, ydata2)

    return sign, signr #, ax_2d (?)


ani_2d = animation.FuncAnimation(fig_2d,                        #Figura sobre la cual se anima
                                 ecg_beat,                      #La función que dibuja el frame to frame. Retorna sólo los elementos modificados para la comparación del blit. 
                                 frames=generator(DpF),         #Crea la data que luego es pasada a la función que dibuja.
                                 init_func=init,                #Crea el frame cero. La estructura que no se actualiza. Utilizado por blit para comparación    
                                 fargs=(sign, signr, mtr),      #Argumentos accesorios para la función dibujadora
                                 interval=FI*1000,              #Delay entre frame y frame en ms    
                                 blit=1)                        #Compara frame y frame y dibuja sólo lo nuevo 

plt.show()                                                      #Inicia la animación. 
