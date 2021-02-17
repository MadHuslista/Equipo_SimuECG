
from learners import Learner
import math as m
import wfdb
import numpy as np
from gen_setup import create_subsets
import random
import matplotlib.pyplot as plt 
import gen_variabs as gv 

class Generation(): 

    def __init__(self, subset, params=0,gen =1, mut_prob=0.002, aleat_params = False): 

        self.gen = gen
        self.mut_prob = mut_prob

        self.childs = []

        self.best_childs = [-1,-1]        
        self.best_genes = []

        print("  Creating Childs gen: ", self.gen)


        if not(aleat_params): 

            child = Learner(subset[0], params)        
            self.childs.append(child)

            for batch in subset[1:]: 
                child = self.get_mutatedChild(batch, params)
                self.childs.append(child)

        else: 
            for batch in subset: 
                child = Learner(batch)
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

        for child in self.childs: 
            
            for err in child.calc_error(): 

                if err >= sec_best: 
                    break
            else: 
                if err < fir_best: 
                    self.best_childs[0] = child
                    fir_best = err
                elif fir_best < err < sec_best: 
                    self.best_childs[1] = child 
                    sec_best = err
        
        #    print(fir_best, sec_best)
        #print(self.best_childs)

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

    s = create_subsets(s)        
    p = gv.theta_vals + gv.a_vals + gv.b_vals + gv.y0

    print('done')

    b = Generation(s,p)
    
    for i in b.childs: 
        plt.plot(i.signal[1])

    plt.show()

        






