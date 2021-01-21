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

#Se va al variables_func.py y se recuperan las listas en las que están las variables con sus valores 
#de inicialización en la práctica aquí sólo se renombran para facilitar manipulación para su actualización; 
#en todo momento están haciendo referencia a las variables creadas en variables_func

param_gener = varfun.param_gener
param_Artf = varfun.param_Artf
param_HVR = varfun.param_HVR
theta_vals = varfun.theta_vals
a_vals = varfun.a_vals
b_vals = varfun.b_vals
y0 = varfun.y0


"""
####################### 2.2.- Elementos Slider ####################################
"""


def update_Artf(event):
    global param_Artf
    global Flag
    param_Artf[0] = slid.s_Anoise.val
    param_Artf[1] = slid.s_Hznoise.val
    param_Artf[2] = slid.s_AHznoise.val
    Flag = True


slid.fig_Artf.show()
slid.sim_Artf.on_clicked(update_Artf)


def update_gen(event):
    global param_gener
    global Flag
    # necesita generación completa de la señal
    param_gener[0] = slid.s_hrmean.val
    param_gener[1] = slid.s_resp.val
    param_gener[2] = slid.s_Amp_ECG.val  # Revisar máximos graph - CHECK
    param_gener[3] = slid.s_n.val
    # Revisar FPs abajo. Genera problema con el FPS si no se actualiza hacia abajo también
    param_gener[4] = 1/(10**slid.s_dt.val)
    param_gener[5] = slid.s_FPS.val
    Flag = True


slid.fig_gen.show()
slid.sim_gen.on_clicked(update_gen)


def update_HVR(event):
    global param_HVR
    global Flag
    param_HVR[0] = slid.s_hrstd.val
    param_HVR[1] = 2*m.pi*slid.s_c1.val
    param_HVR[2] = 2*m.pi*slid.s_c2.val
    param_HVR[3] = 2*m.pi*slid.s_f1.val
    param_HVR[4] = 2*m.pi*slid.s_f2.val
    Flag = True


slid.fig_HVR.show()
slid.sim_HVR.on_clicked(update_HVR)


def update_theta(event):
    global theta_vals
    global Flag
    theta_vals[0] = slid.s_tP.val * m.pi
    theta_vals[1] = slid.s_tQ.val * m.pi
    theta_vals[2] = slid.s_tR.val * m.pi
    theta_vals[3] = slid.s_tS.val * m.pi
    theta_vals[4] = slid.s_tTd.val * m.pi
    theta_vals[5] = slid.s_tTu.val * m.pi
    Flag = True


slid.fig_theta.show()
slid.sim_th.on_clicked(update_theta)


def update_gauss(event):
    global a_vals
    global b_vals
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


slid.fig_gauss.show()
slid.sim_g.on_clicked(update_gauss)


"""
####################### 2.1- Generador para Animación ####################################
"""

Flag = False


def generator(dpf):
    i = 0
    data = []

    global Flag

    x_val, y_val, z_val, t = model(
        param_gener, param_Artf, param_HVR, theta_vals, a_vals, b_vals, y0)
    z_m = z_val
    while True:
        actual_point = int(dpf*i)
        if Flag:
            print("yas1")
            x_val, y_val, z_m, t = model(
                param_gener, param_Artf, param_HVR, theta_vals, a_vals, b_vals, y0)
            print("yas2")
            print(len(z_val), len(z_m))
            a = z_val[:actual_point]
            b = z_m[actual_point:]

            print(len(a), len(b))
            print("yas3")
            z_val = np.concatenate((a, b))
            print("yas4")

            fi = 1 / param_gener[4]  # Frame Interval
            dpf = fi / param_gener[5]  # Datos por frame

            Flag = False
        data = [x_val, y_val, z_val, t, i, param_gener[4], dpf]
        yield data

        i += 1
        n_frames = round(len(t)/dpf)
        if i+1 >= n_frames:
            i = 0
            z_val = z_m


"""
####################### 3.- Animacióin 2D ####################################
"""

fig_2d, ax_2d = plt.subplots()
# Agregar grid reglamentaria del papel al gráfico

mtr = 8  # Monitor Time Range
hrmean = param_gener[0]
Amp_ECG = param_gener[2]
FPS = param_gener[5]
dt = param_gener[4]

ylim_s = 5.5
ylim_i = -0.5

sign, = ax_2d.plot([], [], 'g')
signr, = ax_2d.plot([], [], 'g')

ax_2d.set_xlim([0, mtr])
ax_2d.set_xlabel('Tiempo [s]')

ax_2d.set_aspect(0.4)

ax_2d.xaxis.grid(True, which='major', lw=1.5)
ax_2d.xaxis.grid(True, which='minor', lw=0.5)

ax_2d.set_ylim(ylim_i, ylim_s)
ax_2d.set_ylabel('Voltaje [mV]')

ax_2d.set_yticks(np.arange(ylim_i, ylim_s, step=0.5), minor=False)
ax_2d.set_yticks(np.arange(ylim_i, ylim_s, step=0.1), minor=True)
ax_2d.yaxis.grid(True, which='major', lw=1.5)
ax_2d.yaxis.grid(True, which='minor', lw=0.5)


xdata1, ydata1 = [], []
xdata2, ydata2 = [], []

FI = 1 / FPS  # Frame Interval
DpF = FI / dt  # Datos por frame


def init():  # Sin esta función también funciona. Documentación sugiere que es más eficiente. No lo sé
    ax_2d.set_ylim(ylim_i, ylim_s)
    ax_2d.set_xlim(0, mtr)
    del xdata1[:]
    del ydata1[:]
    del xdata2[:]
    del ydata2[:]
    sign, = ax_2d.plot([], [])
    signr, = ax_2d.plot([], [])
    return sign, signr


def ecg_beat(data, sign, signr, mtr):

    z = data[2]
    t = data[3]
    num = data[4]
    dt = data[5]
    DpF_n = data[6]
    # t = data[0]
    # z = data[1]
    # Posible mejora: Usar el argumento 'Frames' para pasar la data. Ahora, para cada frame, le paso la lista completa de datos. Mucho

    time_gap = 0.01  # Separación entre la nueva señal y la anterior. En [s]

    data_gap = time_gap/dt    #
    growth_cursor = int(round(num*DpF_n - int(data_gap/2)))
    decrease_cursor = int(round(num*DpF_n + int(data_gap/2)))

    # print()

    xmin, xmax = ax_2d.get_xlim()
    ymin, ymax = ax_2d.get_ylim()
    pos_inf = int(xmin/dt)
    pos_sup = int(xmax/dt)

    if pos_sup > len(t)-1:
        pos_sup = len(t)-1

    xdata1 = t[pos_inf:growth_cursor]
    ydata1 = z[pos_inf:growth_cursor]

    if growth_cursor*dt < mtr:
        xdata2 = []
        ydata2 = []
    else:
        xdata2 = t[decrease_cursor:pos_sup]
        ydata2 = z[decrease_cursor-int(mtr/dt):pos_sup-int(mtr/dt)]

    if num <= 1:
        ax_2d.set_xlim(0, mtr)

        ax_2d.set_xticks(np.arange(0, mtr, step=0.2), minor=False)
        ax_2d.set_xticks(np.arange(0, mtr, step=0.04), minor=True)

        ax_2d.figure.canvas.draw()

    elif growth_cursor*dt > xmax:
        ax_2d.set_xlim(xmin+mtr, xmax+mtr)

        ax_2d.set_xticks(np.arange(xmin+mtr, xmax+mtr, step=0.2), minor=False)
        ax_2d.set_xticks(np.arange(xmin+mtr, xmax+mtr, step=0.04), minor=True)

        ax_2d.figure.canvas.draw()

    sign.set_data(xdata1, ydata1)
    signr.set_data(xdata2, ydata2)

    return sign, signr


ani_2d = animation.FuncAnimation(fig_2d, ecg_beat, frames=generator(
    DpF), init_func=init, fargs=(sign, signr, mtr), interval=FI*1000, blit=1)
plt.show()
