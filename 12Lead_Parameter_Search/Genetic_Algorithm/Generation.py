
from Learners import Learner
import math as m
import wfdb
import numpy as np
from SubSets_Setup import create_subsets
import random
import matplotlib.pyplot as plt 
import Initial_Parameters as init_params 

class Generation(): 
    """
    Este objeto controla los procesos de selección 'natural' asociados a la evolución: 
        1.- En su inicialización recibe la BD que viene ppreviamente organizada en batches, y crea un nuevo 'Learner' por cada batch. 
        2.- Dentro de sus métodos implementa los procesos de selección del mejor 'Learner', mutación y crossover, 
    """
    def __init__(self, subset, params=0,gen =1, mut_prob=0.002, aleat_params = False): 

        self.gen = gen
        self.mut_prob = mut_prob

        self.childs = []

        self.best_childs = [-1,-1]        
        self.best_genes = []

        print("  Creating {} Childs gen: {}".format(len(subset), self.gen))


        if not(aleat_params):                                   #En el caso que sí recibe un set de parámetros base

            child = Learner(subset[0], params)                  #Crea el primer 'Learner' en base a estos parámetros
            self.childs.append(child)

            for batch in subset[1:]:                            #Y luego crea el resto de Learners en base a versiones mutadas de estos parámetros. 
                child = self.get_mutatedChild(batch, params)
                self.childs.append(child)

        else:                                                   #De no recibir parámetros, crea Learner en base a parámetros aleatorios. 
            for batch in subset: 
                child = Learner(batch)
                self.childs.append(child)

        print("  Childs created. Gen: ", self.gen)


    def get_mutatedChild(self,batch, best_params): 
        """
        Genera Learnes en base a versiones mutadas de los parámetros best_params recibidos.
        """

        #t = [random.uniform(-m.pi, m.pi) for i in range(6)]
        #a = [random.uniform(-30, 30) for i in range(6)]
        #b = [random.uniform(0, 1) for i in range(6)]
        #y0 = [random.uniform(0, 1) for i in range(3)]
        #new_params = t + a + b + y0

        for i in range(6,len(best_params)-4):           #Aquí se limita los parámetros sujetos a posible randomización, dejando fuera las posiciones theta y los valores iniciales y0
            
            if random.random() < self.mut_prob:         
                print("    Mutation! {}".format(i))
                k = random.uniform(-0.01,0.01)
                best_params[i] = best_params[i] + k
                break

        mutated_Child = Learner(batch, best_params)

        return mutated_Child                

    def find_BestGenes(self): 
        """
        Identifica el set de parámetros que provocó el menor error. 
        """

        self.find_BestChilds()                          #Selecciona los mejores Learners

        self.best_genes = self.crossover()              #Y obtiene el resultado del crossover de los parámetros de ambos.
        #self.best_genes = self.best_childs[1].params

        return self.best_genes

        
    def find_BestChilds(self): 
        """
        Identifica los mejores Learners
        """

        print("  Searching for the best childs. Gen: ", self.gen)

        fir_best = m.inf                                    #Identifica el primer y segundo mejor
        sec_best = m.inf

        #print(len(self.childs))

        for child in self.childs:                           #Por cada Learner creado
            
            for err in child.calc_error():                  #Solicita el valor del error acumulado, pulso a pulso de cada batch (revisar Learner.calc_error(). Recordar que por eficiencia es un iterador)

                if err >= sec_best:                         #Si el error acumulado supera al segundo mejor actual, se descarta  el actual Learner y se pasa al siguiente. 
                    break
            else:                                           #Si el error acumulado no supera el segundo mejor, se selecciona como nuevo primer o segundo mejor. 
                if err < fir_best: 
                    self.best_childs[0] = child
                    fir_best = err
                elif fir_best < err < sec_best: 
                    self.best_childs[1] = child 
                    sec_best = err
        
        #    print(fir_best, sec_best)
        #print(self.best_childs)
        else:                                               #Una vez finalizado la medición, en el caso de que no haya segundo mejor, se duplica el primer mejor. 
            if self.best_childs[1] == -1:
                self.best_childs[1] = self.best_childs[0]
                print('gotcha')


        print("  Search ended. Gen: ", self.gen)
        
        return self.best_childs                             

            
    def crossover(self): 

        """
        Combinación de los genes.
        Se selecciona una posición arbitraria y se efectúa el cruce mantiendo siempre la mayor porción del mejor Learner. 
        """

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
    p = init_params.theta_vals + init_params.a_vals + init_params.b_vals + init_params.y0 + [-0.98765]

    print('done')

    b = Generation(s,p)
    
    for i in b.childs: 
        plt.plot(i.signal[1])

    plt.show()

        






