#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 01:47:25 2019

@author: Pablo Vega
"""
 

#CREACIÓN DEL TACOGRAMA



import math as m
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt 


def RR_gen(f1, f2, c1, c2, hrmean, hrstd, n):
    
    #def param
    rrstd = 60*(hrstd)/(hrmean*hrmean)
    rrmean = 60/hrmean
    sfrr = 1
    theta1 = 0.5
    theta2 = 1
    df = sfrr/n
    w = np.arange(0,n,1)*2*m.pi*df
    
    
    def s1(x):
        return(((theta1)*m.exp(-0.5*((x-f1)/c1)**2))/m.sqrt(2*m.pi*c1**2))
    def s2(y):
        return(((theta2)*m.exp(-0.5*((y-f2)/c2)**2))/m.sqrt(2*m.pi*c2**2))
    sf = []
    for i in w:
        suma = s1(i)+s2(i)
        sf.append(suma)
    #plt.plot(sf)
    sf0 = []
    for i in range(0,int(n/2)):
        piv = sf[i]
        sf0.append(piv)
    for i in range(int((n/2)-1),-1,-1):
        piv = sf[i]
        sf0.append(piv)
    sf1 = []
    for i in sf0:
        piv = (sfrr/2)*m.sqrt(i)
        sf1.append(piv)
        
    var = np.random.rand(int(n/2+1))
    ph0 = []
    for i in range(int((n/2)-1)):
        piv= 1*var[i]*2*m.pi
        ph0.append(piv)
    
    ph = [0]
    for i in range(len(ph0)):
        piv = ph0[i]
        ph.append(piv)
    ph.append(0)
    
    for i in range(len(ph0)-1,-1,-1):
        piv = -ph0[i]
        ph.append(piv)
    
    sfw = []
    for i in range(len(ph)):
        piv = sf1[i]*np.exp(1j*ph[i])
        sfw.append(piv)
    x = (1/n)*sp.real(sp.ifft(sfw))
    xstd = np.std(x)
    ratio = rrstd / xstd
    rr = []
    for i in range(len(x)):
        piv = rrmean + x[i]*ratio
        rr.append(piv)
    
    rr_times = rr       #SERIE DE INTERVALOS RR
    
    #Time Tool
        
    return rr_times

def main():

    #Esto está sólo para probar el modelo o  probarlo de manera independiente
    hrmean = 60
    hrstd = 1
    sfrr = 1
    c1 = 2*m.pi*0.01
    c2 = 2*m.pi*0.01
    f1 = 0.1*2*m.pi
    f2 = 0.25*2*m.pi
    n = 250

    rr_times = RR_gen(f1, f2, c1, c2, hrmean, hrstd, n)
    
if __name__ == "__main__":
    main()
