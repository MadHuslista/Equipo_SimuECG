
from learners import Learner
import math as m
import wfdb
import numpy as np
import random
import matplotlib.pyplot as plt 
import gen_variabs as gv 

class Generation(): 

    def __init__(self, signal, popu_size=100, params=0,gen =1, mut_prob=0.002, aleat_params = False): 

        self.gen = gen
        self.mut_prob = mut_prob

        self.childs = []

        self.best_childs = [-1,-1]        
        self.best_genes = []

        print("  Creating Childs gen: ", self.gen)


        if not(aleat_params): 

            child = Learner(signal, params)        
            self.childs.append(child)

            for person in range(popu_size-1): 
                child = self.get_mutatedChild(signal, params)
                self.childs.append(child)

        else: 
            for person in range(popu_size-1): 
                child = Learner(signal)
                self.childs.append(child)

        print("  Childs created. Gen: ", self.gen)


    def get_mutatedChild(self,batch, best_params): 

        #t = [random.uniform(-m.pi, m.pi) for i in range(6)]
        #a = [random.uniform(-30, 30) for i in range(6)]
        #b = [random.uniform(0, 1) for i in range(6)]
        #y0 = [random.uniform(0, 1) for i in range(3)]
        #new_params = t + a + b + y0

        for i in range(len(best_params)): 
            
            if random.random() < self.mut_prob:
                print("    Mutation!")
                k = random.uniform(-0.5,1.5)
                best_params[i] = best_params[i] * k

        mutated_Child = Learner(batch, best_params)

        return mutated_Child                

    def find_BestGenes(self): 

        self.find_BestChilds()

        self.best_genes = self.crossover()
        #self.best_genes = self.best_childs[1].params

        return self.best_genes

        
    def find_BestChilds(self): 

        print("  Searching for the best childs. Gen: ", self.gen)

        fir_best = m.inf
        sec_best = m.inf

        #print(len(self.childs))
        i = 0
        for child in self.childs: 
            print(i)
            for err in child.calc_error(): 

                if err >= sec_best: 
                    print("   x")
                    break

            else: 
                if err < fir_best: 
                    self.best_childs[0] = child
                    fir_best = err
                elif fir_best < err < sec_best: 
                    self.best_childs[1] = child 
                    sec_best = err        
        #    print(fir_best, sec_best)
            i += 1
        else: 
            if self.best_childs[1] == -1:
                self.best_childs[1] = self.best_childs[0]
                print('gotcha')

        print("  Search ended. Gen: ", self.gen)
        
        return self.best_childs

            
    def crossover(self): 
        child1 = self.best_childs[0].params
        child2 = self.best_childs[1].params

        best_genes = []

        cross_pos = random.randrange(len(child1))

        if cross_pos > len(child1)/2: 
            #De esta manera siempre la mayor cantidad queda en el que fue mejor evaluado
            best_genes = child1[:cross_pos] + child2[cross_pos:]
        else: 
            best_genes = child2[:cross_pos] + child1[cross_pos:]

        print("  Crossover Terminado")

        return best_genes




if __name__ == "__main__": 

    ecg_recover = wfdb.rdsamp("Derivations_Data/BD_II_signal")
    s = ecg_recover[0].transpose()

    
    p = gv.theta_vals + gv.a_vals + gv.b_vals + gv.y0

    print('done')

    b = Generation(s,5,p)
    print(len(b.childs))
    a = b.find_BestChilds()

    print(a[0].error,a[1].error)
    
    for i in b.childs: 
        plt.plot(i.signal[1])

    plt.show()

        






