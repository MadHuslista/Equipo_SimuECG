from generation import Generation
from learners import Learner
import wfdb
import numpy as np
import random as rnd
from pprint import pprint
import matplotlib.pyplot as plt 
import gen_variabs as gv 

class Training(): 

    def __init__(self, signal, generations, params=0): 

        print("Inicio Entrenamiento")
        self.subsets = self.create_subsets(signal)
        self.generations = generations
        self.best_genes = []
        self.err_history = []
        self.p_history = []

        if not(params):
            self.Gen = Generation(self.subsets,aleat_params=True)
        else: 
            self.Gen = Generation(self.subsets,params)
            

        print("Generación Inicial creada")


    def evolution(self): 

        print("-> ", self.Gen.gen)
        print("Evolution Gen: ", self.Gen.gen)

        mut_boom = 0
        for i in range(self.generations): 

            best_params = self.Gen.find_BestGenes()
            new_gen = self.Gen.gen + 1

            best_err = self.Gen.best_childs[0].error
            self.err_history.append(best_err)
            self.p_history.append(best_params)

            change_rate = 1
            if i > 5:
                change_rate = abs((best_err - self.err_history[-2])/self.err_history[-2])
                if change_rate < 0.2 : 
                    mut_boom += 1
                else: 
                    mut_boom = 0

            print("Best Err: ", best_err)
            print("M Boom: ", mut_boom,"Ch_Rate: ", change_rate)
            
            print("End Gen: ", self.Gen.gen -1 )
            print("=====")
            print()

            print("-> ", self.Gen.gen)
            print("Evolution Gen: ", self.Gen.gen)

            self.subsets = self.create_subsets(self.subsets, 0.6)

            if mut_boom >= 10: 
                print("                                        MUT BOOM!")
                self.Gen = Generation(self.subsets, best_params, new_gen, mut_prob=0.005 )    
                mut_boom = 0
            else: 
                self.Gen = Generation(self.subsets, best_params, new_gen)    


    def create_subsets(self, signal, retain_pctg=0, batch_size=10): 
    
        if not(retain_pctg):

            signal = np.array(signal)

            sig_cant = signal.shape[0]

            indexs = np.arange(0,sig_cant,1)
            rnd.shuffle(indexs)

            subsets = []
            batch = []
            batch_quantity = 0

            for i in indexs: 

                if len(batch) < batch_size:
                    batch.append(signal[i])
                else: 
                    b_copy = list(batch)
                    subsets.append(b_copy)
                    batch = []
                    batch.append(signal[i])
                    batch_quantity += 1
            else: 
                subsets.append(list(batch))
                batch_quantity += 1

        else: 

            torandom_pool = []
            remain_subset = []

            for batch in signal:
                retain_pos = int((len(batch) * retain_pctg))

                torandom_pool.extend(batch[retain_pos:])
                remain_subset.append(batch[:retain_pos])
            
            indexs = np.arange(0,len(torandom_pool),1)
            rnd.shuffle(indexs)
            indexs = iter(indexs)

        
            
            for batch in remain_subset: 

                try: 
                    while len(batch) < batch_size: 
                    
                        i = next(indexs)
                        batch.append(torandom_pool[i])
                except: 
                    break
                
            
            subsets = remain_subset
        
        return subsets


if __name__ == "__main__": 


    ecg_recover = wfdb.rdsamp("Derivations_Data/BD_II_signal")
    print('Signal Readed')
    s = ecg_recover[0].transpose()
    p = gv.theta_vals + gv.a_vals + gv.b_vals + gv.y0

    #s = s[:100]



    T = Training(s,100,p)
    T.evolution()

    p = T.best_genes
    g = Learner(1,p)

    plt.plot(T.err_history)
    plt.show()

    plt.figure()
    for i in s: 
        plt.plot(i, c='g')
    plt.plot(g.signal[1], c = 'r', label = "Last_Best")

    min_err = min(T.err_history)
    min_i = T.err_history.index(min_err)

    b_p = T.p_history[min_i]

    h = Learner(1,p)

    plt.plot(h.signal[1], c = 'b', label = "Hist_Best")
    plt.legend()



    plt.show()