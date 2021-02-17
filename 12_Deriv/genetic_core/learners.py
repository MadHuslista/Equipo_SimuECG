
import wfdb
import random
import math as m 
import matplotlib.pyplot as plt 
import gen_variabs as gv 
from genderiv_core import Derivative_Core
import numpy as np 

class Learner(): 

    def __init__(self, batch, params=0): 

        if not(params):
            t = [random.uniform(-m.pi, m.pi) for i in range(6)]
            a = [random.uniform(-30, 30) for i in range(6)]
            b = [random.uniform(0, 1) for i in range(6)]
            y0 = [random.uniform(0, 1) for i in range(3)]
            self.params = t + a + b + y0
        else:
            self.params = params


        self.error = 0
        self.batch = batch
        self.base = sum(sum(np.array(batch)**2))
        self.Sig_Core = Derivative_Core()
        self.signal = self.Sig_Core.calc_model(self.params)

    def calc_error(self): 

        for b_sig in self.batch: 

            self.error += sum((b_sig - self.signal[1])**2)
            
            yield self.error

        


        
        #entra self. batch -> f() -> self.error

        #return self.error

    

if __name__ == "__main__": 
    ecg_recover = wfdb.rdsamp("Derivations_Data/BD_II_signal")
    s = ecg_recover[0].transpose()

    p = gv.theta_vals + gv.a_vals + gv.b_vals + gv.y0
    a = Learner(s,p)
    
    for i in s: 
        plt.plot(i,c='g')
    plt.plot(a.signal[1],c='r')
    plt.show()
    