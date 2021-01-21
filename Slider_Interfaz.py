#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 15:00:05 2019

@author: meyerhof
"""

from scipy import signal
import math as m
import numpy as np

import matplotlib.pyplot as plt 
from matplotlib.widgets import Slider, Button
import unicodedata

plt.close("all")





"""
####################### 0.- PARÁMETROS GENERALES ####################################
"""
axcolor = 'lightgoldenrodyellow'

#Ventana para control de Parámetros Generales

fig_gen = plt.figure(figsize=[5,2])
fig_gen.suptitle("Control de Parámetros Generales", y = 0.9)

ax_hrmean       = fig_gen.add_axes([0.5, 0.65, 0.3, 0.05], facecolor=axcolor)
ax_Resp_by_min  = fig_gen.add_axes([0.5, 0.55, 0.3, 0.05], facecolor=axcolor)
ax_Amp_ECG      = fig_gen.add_axes([0.5, 0.45, 0.3, 0.05], facecolor=axcolor)
ax_n            = fig_gen.add_axes([0.5, 0.35, 0.3, 0.05], facecolor=axcolor)
ax_dt           = fig_gen.add_axes([0.5, 0.25, 0.3, 0.05], facecolor=axcolor)
ax_FPS          = fig_gen.add_axes([0.5, 0.15, 0.3, 0.05], facecolor=axcolor)
ax_sim_gen      = fig_gen.add_axes([0.69, 0.020, 0.11, 0.1], facecolor=axcolor)
ax_rst_gen      = fig_gen.add_axes([0.5, 0.020, 0.11, 0.1], facecolor=axcolor)


s_hrmean    = Slider(ax_hrmean, 'Frecuencia Cardíaca Promedio', 20, 200, valinit=60, valstep=1, valfmt = "%3d [Hz]")
s_resp      = Slider(ax_Resp_by_min, 'Frecuencia Respiratoria Promedio', 0, 70, valinit=15, valstep=1, valfmt = "%2d  [Hz]")
s_Amp_ECG   = Slider(ax_Amp_ECG, 'Amplitud Máxima ECG', 0, 5, valinit=1.7, valstep=0.1 ,valfmt = "%2.1f [mV]")
s_n         = Slider(ax_n, 'Pulsaciones Simuladas', 2, 100, valinit=30, valstep=1,valfmt = "%3d")
s_dt        = Slider(ax_dt, 'Frecuencia de Muestreo 10^x',1,4, valinit=3, valstep=1,valfmt = "%1d")
s_FPS       = Slider(ax_FPS, 'Cuadros por Segundo', 25, 50, valinit=30, valstep=5, valfmt = "%2d  [FPS]")
sim_gen     = Button(ax_sim_gen, 'Simular', color=axcolor, hovercolor='0.975')
rst_gen     = Button(ax_rst_gen, 'Reset', color=axcolor, hovercolor='0.975')

def reset_gen(event):
    s_hrmean.reset()
    s_resp.reset()
    s_Amp_ECG.reset()
    s_n.reset()
    s_dt.reset()
    s_FPS.reset()
    
rst_gen.on_clicked(reset_gen)

"""
####################### 0.- PARÁMETROS ARTEFACTOS ####################################
"""

#Ventana para control de Parámetros Artefactos
fig_Artf = plt.figure(figsize=[5,1.5])
fig_Artf.suptitle("Control de Parámetros de Artefactos", y = 0.9)
#fig_Artf = fig_gen

ax_Anoise       = fig_Artf.add_axes([0.4, 0.44, 0.4, 0.1], facecolor=axcolor)
ax_Hz_noise     = fig_Artf.add_axes([0.4, 0.6, 0.4, 0.1], facecolor=axcolor)
ax_AHznoise     = fig_Artf.add_axes([0.4, 0.28, 0.4, 0.1], facecolor=axcolor)
ax_sim_Artf     = fig_Artf.add_axes([0.69, 0.020, 0.11, 0.1], facecolor=axcolor)
ax_rst_Artf     = fig_Artf.add_axes([0.4, 0.020, 0.11, 0.1], facecolor=axcolor)

s_Anoise    = Slider(ax_Anoise, 'Amplitud Ruido Aleatorio', 0, 1, valinit=0.15, valstep=0.01, valfmt = "%4.2f [mV]")
s_Hznoise   = Slider(ax_Hz_noise, 'Frecuencia Interferencia', 0, 100, valinit=50, valstep=1, valfmt = "%3d  [Hz]")
s_AHznoise  = Slider(ax_AHznoise, 'Amplitud Interferencia', 0, 1, valinit=0.15, valstep=0.01, valfmt = "%4.2f [mV]")
sim_Artf    = Button(ax_sim_Artf, 'Simular', color=axcolor, hovercolor='0.975')
rst_Artf    = Button(ax_rst_Artf, 'Reset', color=axcolor, hovercolor='0.975')

def reset_Artf(event):
    s_Anoise.reset()
    s_Hznoise.reset()
    s_AHznoise.reset()
    
rst_Artf.on_clicked(reset_Artf)

"""
####################### 0.- PARÁMETROS HVR ####################################
"""
#Ventana para control de Parámetros HVR
fig_HVR = plt.figure(figsize=[5,2])
fig_HVR.suptitle("Control de Parámetros HVR", y = 0.9)

ax_hrstd    = fig_HVR.add_axes([0.5, 0.7, 0.3, 0.07], facecolor=axcolor)
ax_c1       = fig_HVR.add_axes([0.5, 0.575, 0.3, 0.07], facecolor=axcolor)
ax_c2       = fig_HVR.add_axes([0.5, 0.45, 0.3, 0.07], facecolor=axcolor)
ax_f1       = fig_HVR.add_axes([0.5, 0.325, 0.3, 0.07], facecolor=axcolor)
ax_f2       = fig_HVR.add_axes([0.5, 0.20, 0.3, 0.07], facecolor=axcolor)
ax_sim_HVR  = fig_HVR.add_axes([0.8, 0.020, 0.1, 0.1], facecolor=axcolor)
ax_rst_HVR  = fig_HVR.add_axes([0.5, 0.020, 0.1, 0.1], facecolor=axcolor)

s_hrstd     = Slider(ax_hrstd, 'Desviación Estándar F. Cardíaca', 0, 10, valinit=0., valstep=0.1, valfmt = "%4.1f [Hz]")
s_c1        = Slider(ax_c1, 'Desviación Estándar Onda Mayer', 0, 0.5, valinit=0.01, valstep=0.01, valfmt = "%4.2f [Hz]")
s_c2        = Slider(ax_c2, 'Desviación Estándar Onda RSA', 0, 0.5, valinit=0.15, valstep=0.01, valfmt = "%4.2f [Hz]")
s_f1        = Slider(ax_f1, 'Frecuencia Central Onda Mayer', 0, 0.5, valinit=0.1, valstep=0.01, valfmt = "%4.2f [Hz]")
s_f2        = Slider(ax_f2, 'Frecuencia Central Onda RSA', 0, 0.5, valinit=0.25, valstep=0.01, valfmt = "%4.2f [Hz]")
sim_HVR     = Button(ax_sim_HVR, 'Simular', color=axcolor, hovercolor='0.975')
rst_HVR     = Button(ax_rst_HVR, 'Reset', color=axcolor, hovercolor='0.975')

def reset_HVR(event):
    s_hrstd.reset()
    s_c1.reset()
    s_c2.reset()
    s_f1.reset()
    s_f2.reset()
    
rst_HVR.on_clicked(reset_HVR)

"""
####################### 0.- PARÁMETROS THETA ####################################
"""
#Ventana para control de Parámetros Theta
fig_theta = plt.figure(figsize=[8,4])
fig_theta.suptitle("Control de la Posición Angular de las Ondas PQRST", x = 0.3, y = 0.9)
ax_circle = plt.subplot(frame_on=False)
plt.axis('off')
circle = plt.Circle((0,0), radius = 1, fill = False)
ax_circle.add_patch(circle)
ax_circle.axis('scaled')
ax_circle.set_position([0.3,0.02,0.9,0.9])
pi_symb = unicodedata.lookup("GREEK SMALL LETTER PI")
max_min = unicodedata.lookup("plus-minus sign")

ax_tP       = fig_theta.add_axes([0.05, 0.7, 0.4, 0.05], facecolor=axcolor)
ax_tQ       = fig_theta.add_axes([0.05, 0.6, 0.4, 0.05], facecolor=axcolor)
ax_tR       = fig_theta.add_axes([0.05, 0.5, 0.4, 0.05], facecolor=axcolor)
ax_tS       = fig_theta.add_axes([0.05, 0.4, 0.4, 0.05], facecolor=axcolor)
ax_tTd      = fig_theta.add_axes([0.05, 0.3, 0.4, 0.05], facecolor=axcolor)  
ax_tTu      = fig_theta.add_axes([0.05, 0.2, 0.4, 0.05], facecolor=axcolor)
ax_sim_th   = fig_theta.add_axes([0.35, 0.1, 0.1, 0.05], facecolor=axcolor)
ax_rst_th   = fig_theta.add_axes([0.05, 0.1, 0.1, 0.05], facecolor=axcolor)

s_tP   = Slider(ax_tP, 'P',     -1, 1, valinit=-1/3, valstep=0.01, valfmt = "%5.2f"+pi_symb)
s_tQ   = Slider(ax_tQ, 'Q',     -1, 1, valinit=-1/12, valstep=0.01, valfmt = "%5.2f"+pi_symb)
s_tR   = Slider(ax_tR, 'R',     -1, 1, valinit=0.00, valstep=0.01, valfmt = "%5.2f"+pi_symb)
s_tS   = Slider(ax_tS, 'S',     -1, 1, valinit=1/12, valstep=0.01, valfmt = "%5.2f"+pi_symb)
s_tTd  = Slider(ax_tTd, 'Td',   -1, 1, valinit=(5/9 - 1/60), valstep=0.01, valfmt = "%5.2f"+pi_symb)
s_tTu  = Slider(ax_tTu, 'Tu',   -1, 1, valinit=5/9, valstep=0.01, valfmt = "%5.2f"+pi_symb)
sim_th = Button(ax_sim_th, 'Simular', color=axcolor, hovercolor='0.975')
rst_th = Button(ax_rst_th, 'Reset', color=axcolor, hovercolor='0.975')


p_P  = ax_circle.scatter(np.cos(-1/3*m.pi), np.sin(-1/3*m.pi))
p_Q  = ax_circle.scatter(np.cos(-1/12*m.pi), np.sin(-1/12*m.pi))
p_R  = ax_circle.scatter(np.cos(0), np.sin(0))
p_S  = ax_circle.scatter(np.cos(1/12*m.pi), np.sin(1/12*m.pi))
p_Td = ax_circle.scatter(np.cos((5/9 - 1/60)*m.pi), np.sin((5/9 - 1/60)*m.pi))
p_Tu = ax_circle.scatter(np.cos(5/9*m.pi), np.sin(5/9*m.pi))

an_P  = ax_circle.annotate('P', (1.1*np.cos(-1/3*m.pi), 1.1*np.sin(-1/3*m.pi)))
an_Q  = ax_circle.annotate('Q', (1.1*np.cos(-1/12*m.pi), 1.1*np.sin(-1/12*m.pi)))
an_R  = ax_circle.annotate('R', (1.1*np.cos(0), 1.1*np.sin(0)))
an_S  = ax_circle.annotate('S', (1.1*np.cos(1/12*m.pi), 1.1*np.sin(1/12*m.pi)))
an_Td = ax_circle.annotate('Td', (1.1*np.cos((5/9 - 1/60)*m.pi), 1.1*np.sin((5/9 - 1/60)*m.pi)))
an_Tu = ax_circle.annotate('Tu', (1.1*np.cos(5/9*m.pi), 1.1*np.sin(5/9*m.pi)))

exte = 1.01
inte = 0.99
offset = [0.08, 0.04]
ang_v = [0,1/4, 1/2, 3/4, 1,-1/4, -1/2, -3/4]
ang_t = ['0','1/4'+pi_symb, '1/2'+pi_symb, '3/4'+pi_symb, max_min+pi_symb,'-1/4'+pi_symb, '-1/2'+pi_symb, '-3/4'+pi_symb ]
for i, j in zip(ang_v, ang_t): 
    x = np.cos(i*np.pi)*np.array([inte, exte])
    y = np.sin(i*np.pi)*np.array([inte, exte])
    ax_circle.plot(x, y, 'k')
    ax_circle.annotate(j, (0.85*np.cos(i*np.pi)-offset[0],0.85*np.sin(i*np.pi)-offset[1]))

def update_circle(val):
    hr_factor = np.sqrt(s_hrmean.val/60)
    tP      = s_tP.val*m.pi      *np.sqrt(hr_factor)
    tQ      = s_tQ.val*m.pi      *hr_factor
    tR      = s_tR.val*m.pi
    tS      = s_tS.val*m.pi      *hr_factor
    tTd     = s_tTd.val*m.pi     *np.sqrt(hr_factor)
    tTu     = s_tTu.val*m.pi     *np.sqrt(hr_factor)
    
    p_P.set_offsets( [np.cos(tP),np.sin(tP)] )
    p_Q.set_offsets( [np.cos(tQ),np.sin(tQ)] )
    p_R.set_offsets( [np.cos(tR),np.sin(tR)] )
    p_S.set_offsets( [np.cos(tS),np.sin(tS)] )
    p_Td.set_offsets( [np.cos(tTd),np.sin(tTd)] )
    p_Tu.set_offsets( [np.cos(tTu),np.sin(tTu)] )
    
    an_P.set_position((1.1*np.cos(tP), 1.1*np.sin(tP)))
    an_Q.set_position((1.1*np.cos(tQ), 1.1*np.sin(tQ)))
    an_R.set_position((1.1*np.cos(tR), 1.1*np.sin(tR)))
    an_S.set_position((1.1*np.cos(tS), 1.1*np.sin(tS)))
    an_Td.set_position((1.12*np.cos(tTd), 1.12*np.sin(tTd)))
    an_Tu.set_position((1.12*np.cos(tTu), 1.12*np.sin(tTu)))
    fig_theta.canvas.draw_idle()

s_hrmean.on_changed(update_circle)
s_tP.on_changed(update_circle)
s_tQ.on_changed(update_circle)
s_tR.on_changed(update_circle)
s_tS.on_changed(update_circle)
s_tTd.on_changed(update_circle)
s_tTu.on_changed(update_circle)

def reset_th(event):
    s_tP.reset()
    s_tQ.reset()
    s_tR.reset()
    s_tS.reset()
    s_tTd.reset()
    s_tTu.reset()
    
rst_th.on_clicked(reset_th)

"""
####################### 0.- PARÁMETROS GAUSS ####################################
"""
#Ventana para control de Parámetros a y b
fig_gauss = plt.figure(figsize=[8,4])
fig_gauss.suptitle("Control de la Amplitud y Duración de las Ondas PQRST",  y = 0.98,fontsize=15)

ax_gauss = fig_gauss.subplots(1,6, sharex = True, sharey = True)
fig_gauss.subplots_adjust(0.05,0.625, 0.95, 0.825)

ondas = ('P','Q','R','S','Td','Tu')
for i in range(len(ondas)) :
    ax_gauss[i].set_title(ondas[i])

def gaussian(a, b, hr=1, exp_a=0, exp_b=1):
    mean = b*0.4        *hr**exp_b
    desv = b*(0.8/6)    *hr**exp_b
    a = a*(hr)**exp_a
    x = np.linspace(0, b*0.8,121)
    y = a * np.exp(-((x - mean)*(x - mean)) / (2 * desv*desv))
    return x,y
    
sig_P = gaussian(0.8,0.2)
sig_Q = gaussian(-5,0.1)
sig_R = gaussian(30,0.1)
sig_S = gaussian(-7.5,0.1)
sig_Td = gaussian(0.5,0.4)
sig_Tu = gaussian(0.75,0.2)

g_P, = ax_gauss[0].plot(sig_P[0],sig_P[1])
g_Q, = ax_gauss[1].plot(sig_Q[0],sig_Q[1])
g_R, = ax_gauss[2].plot(sig_R[0],sig_R[1])
g_S, = ax_gauss[3].plot(sig_S[0],sig_S[1])
g_Td, = ax_gauss[4].plot(sig_Td[0],sig_Td[1])
g_Tu, = ax_gauss[5].plot(sig_Tu[0],sig_Tu[1])


width_g = 0.045
ax_gaP  = fig_gauss.add_axes([0.05, 0.45, 0.4, width_g], facecolor=axcolor)
ax_gaQ  = fig_gauss.add_axes([0.05, 0.38, 0.4, width_g], facecolor=axcolor)
ax_gaR  = fig_gauss.add_axes([0.05, 0.31, 0.4, width_g], facecolor=axcolor)
ax_gaS  = fig_gauss.add_axes([0.05, 0.24, 0.4, width_g], facecolor=axcolor)
ax_gaTd = fig_gauss.add_axes([0.05, 0.17, 0.4, width_g], facecolor=axcolor)
ax_gaTu = fig_gauss.add_axes([0.05, 0.1, 0.4, width_g], facecolor=axcolor)

ax_gbP  = fig_gauss.add_axes([0.55, 0.45, 0.4, width_g], facecolor=axcolor)
ax_gbQ  = fig_gauss.add_axes([0.55, 0.38, 0.4, width_g], facecolor=axcolor)
ax_gbR  = fig_gauss.add_axes([0.55, 0.31, 0.4, width_g], facecolor=axcolor)
ax_gbS  = fig_gauss.add_axes([0.55, 0.24, 0.4, width_g], facecolor=axcolor)
ax_gbTd = fig_gauss.add_axes([0.55, 0.17, 0.4, width_g], facecolor=axcolor)
ax_gbTu = fig_gauss.add_axes([0.55, 0.1, 0.4, width_g], facecolor=axcolor)

ax_gaP.set_title("Amplitud de Cada Onda")
ax_gbP.set_title("Duración de Cada Onda")
ax_sim_g   = fig_gauss.add_axes([0.85, 0.02, 0.1, 0.05], facecolor=axcolor)
ax_rst_g   = fig_gauss.add_axes([0.05, 0.02, 0.1, 0.05], facecolor=axcolor)

s_gaP     = Slider(ax_gaP, 'P', -20, 40, valinit=0.8, valstep=0.1)
s_gaQ     = Slider(ax_gaQ, 'Q', -20, 40, valinit=-5, valstep=0.1)
s_gaR     = Slider(ax_gaR, 'R', -20, 40, valinit=30, valstep=0.1)
s_gaS     = Slider(ax_gaS, 'S', -20, 40, valinit=-7.5, valstep=0.1)
s_gaTd     = Slider(ax_gaTd, 'Td', -20, 40, valinit=0.5, valstep=0.1)
s_gaTu     = Slider(ax_gaTu, 'Tu', -20, 40, valinit=0.75, valstep=0.1)

s_gbP     = Slider(ax_gbP, 'P', 0.1, 1.5, valinit=0.2, valstep=0.1)
s_gbQ     = Slider(ax_gbQ, 'Q', 0.1, 1.5, valinit=0.1, valstep=0.1)
s_gbR     = Slider(ax_gbR, 'R', 0.1, 1.5, valinit=0.1, valstep=0.1)
s_gbS     = Slider(ax_gbS, 'S', 0.1, 1.5, valinit=0.1, valstep=0.1)
s_gbTd     = Slider(ax_gbTd, 'Td', 0.1, 1.5, valinit=0.4, valstep=0.1)
s_gbTu     = Slider(ax_gbTu, 'Tu', 0.1, 1.5, valinit=0.2, valstep=0.1)

sim_g = Button(ax_sim_g, 'Simular', color=axcolor, hovercolor='0.975')
rst_g = Button(ax_rst_g, 'Reset', color=axcolor, hovercolor='0.975')

def update_gauss(val):
    hr_factor = np.sqrt(s_hrmean.val/60)
    a_P, a_Q, a_R, a_S, a_Td, a_Tu = s_gaP.val, s_gaQ.val, s_gaR.val, s_gaS.val, s_gaTd.val*(hr_factor**(2.5)), s_gaTu.val*(hr_factor**(2.5))
    b_P, b_Q, b_R, b_S, b_Td, b_Tu = s_gbP.val*hr_factor, s_gbQ.val*hr_factor, s_gbR.val*hr_factor, s_gbS.val*hr_factor, s_gbTd.val*(hr_factor**(-1)), s_gbTu.val*hr_factor
    s_P = gaussian(a_P, b_P, hr_factor,0,1)
    s_Q = gaussian(a_Q, b_Q, hr_factor,0,1)
    s_R = gaussian(a_R, b_R, hr_factor,0,1)
    s_S = gaussian(a_S, b_S, hr_factor,0,1)
    s_Td = gaussian(a_Td, b_Td, hr_factor,2.5,-1)
    s_Tu = gaussian(a_Tu, b_Tu, hr_factor,2.5,1)
    
    g_P.set_data(s_P[0],s_P[1])
    g_Q.set_data(s_Q[0],s_Q[1])
    g_R.set_data(s_R[0],s_R[1])
    g_S.set_data(s_S[0],s_S[1])
    g_Td.set_data(s_Td[0],s_Td[1])
    g_Tu.set_data(s_Tu[0],s_Tu[1])
    for i in ax_gauss:
        i.relim()
        i.autoscale_view(0,1,1)
    fig_gauss.canvas.draw_idle()

s_hrmean.on_changed(update_gauss)
s_gaP.on_changed(update_gauss)
s_gaQ.on_changed(update_gauss)
s_gaR.on_changed(update_gauss)
s_gaS.on_changed(update_gauss)
s_gaTd.on_changed(update_gauss)
s_gaTu.on_changed(update_gauss)

s_gbP.on_changed(update_gauss)
s_gbQ.on_changed(update_gauss)
s_gbR.on_changed(update_gauss)
s_gbS.on_changed(update_gauss)
s_gbTd.on_changed(update_gauss)
s_gbTu.on_changed(update_gauss)

def reset_g(event):
    s_gaP.reset()
    s_gaQ.reset()
    s_gaR.reset()
    s_gaS.reset()
    s_gaTd.reset()
    s_gaTu.reset()
    
    s_gbP.reset()
    s_gbQ.reset()
    s_gbR.reset()
    s_gbS.reset()
    s_gbTd.reset()
    s_gbTu.reset()
    
rst_g.on_clicked(reset_g)