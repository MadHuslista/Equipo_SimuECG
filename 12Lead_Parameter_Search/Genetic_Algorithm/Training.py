from Generation import Generation
from Learners import Learner
import wfdb
import numpy as np
import random as rnd
from pprint import pprint
import matplotlib.pyplot as plt 
import Initial_Parameters as init_params 
import time 


"""
La evolución está coordinada en torno al objeto 'Learner'
    Este objeto desarrolla sólo dos cosas: 
        1.- Para su inicialización construye un pulso ECG en base al modelo de McSharry y los parámetros entregados, e incorpora como atributo en 'batch' de pulsos entregado desde la BD
        2.- Posee un Método para el cálculo del error acumulado entre el 'batch' entregado y la señal pulso construida. 

Para controlar los procesos selección 'natural' asociados a la evolución, está el objeto 'Generation'
    Este objeto desarrolla lo siguiente: 
        1.- En su inicialización simplemente recibe la BD que viene ppreviamente organizada en batches, y crea un 'Learner' por cada batch. 
        2.- Y dentro de sus métodos implementa los procesos de selección del mejor 'Learner', mutación y crossover, 

Por último, para configurar y dirigir la evolución se crea un Entorno de Entrenamiento en base a la clase 'Training' 
    Este objeto desarrolla lo siguiente: 
        1.- En su inicialización: 
            - configura el número de generaciones a entrenar, 
            - toma la BD de pulsos individuales reordenandola en subsets con n pulsos cada subset; 
            - e instancia la clase 'Generation' en el atributo self.Gen en base a la BD reordenada en subsets y, de existirlos, parámetros base

        2.- Para dirigir el entrenamiento posee el método 'evolution' que: 
            - controla el Avance de Generaciones 
            - Dirige los procesos de evolución, solicitándolos al objeto self.Gen
            - registra los menores valores de error por cada generación 
            - y ejecuta acciones en el caso de que la mejora sea demasiado lenta, o tienda a retroceder. 

"""

#Clase que controla el entrenamiento completo
class Training(): 

    def __init__(self, signal, generations, params=0): 
        """
        Construcción del Entorno de Entrenamiento
                signal      = BD con n pulsos individuales
                generations = el número de generaciones que entrenará
                params      = parametros de partida. (genes) 
        """

        print("Inicio Entrenamiento")
        self.subsets = self.create_subsets(signal)      #Reordena la BD en 'batches'
        self.generations = generations                  #Incorpora la generación
        self.best_genes = []                            #Guardará los parámetros que generen menos error
        
        #Atributos de Registro
        self.err_history = []   
        self.p_history = []
        self.pcg_history = []

        #Creación del Atributo que se encargará de controlar a los Individuos. 
        if not(params):
            self.Gen = Generation(self.subsets,aleat_params=True)
        else: 
            self.Gen = Generation(self.subsets,params)
            

        print("Generación Inicial creada")


    def evolution(self): 

        """
        Control de la Evolución.
        """

        print("-> ", self.Gen.gen)
        print("Evolution Gen: ", self.Gen.gen)

        mut_boom = 0     #Mide cantidad de generaciones con poco cambio, desencadenando un leve aumento temporal en la probabilidad de mutación
        stuckness = 0    #Mide cantidad de generaciones con cero cambio, desencadenando un gran aumento temporal en la probabilidad de mutación

        for i in range(self.generations):                                                               #Regula el avance de generación

            best_params = self.Gen.find_BestGenes()                                                     #Obtiene los mejores genes (implícitamente se determinan los dos mejores 'Learners' y se efectúa un crossover entre sus parámetros)
        
            best_err = self.Gen.best_childs[0].error                                                    #Se obtiene el menor error y se calcula su % de error respecto a la sumatoria del batch sobre el que trabajó el Learner. 
            base = self.Gen.best_childs[0].base                                                         
            pct_err = 100*best_err/base

            if i > 1:                                                                                   #Evita el retroceso en el aprendizaje
                if pct_err > self.pcg_history[-1]:
                    best_params = self.p_history[-1]
                    best_err = self.err_history[-1]
                    pct_err = self.pcg_history[-1]
                    print('same_param')

            self.err_history.append(best_err)                                                           #Registro
            self.p_history.append(best_params)
            self.pcg_history.append(pct_err)

            change_rate = 1 
            if i > 5:                                                                                   #Estimula la mutación en caso de que la mejora sea poca o nula         
                #change_rate = abs((best_err - self.err_history[-2])/self.err_history[-2])
                change_rate = abs((pct_err - self.pcg_history[-2])/self.pcg_history[-2])
                if change_rate < 0.2 : 
                    mut_boom += 1
                else: 
                    mut_boom = 0
                    stuckness = 0

                if change_rate == 0: 
                    stuckness += 1
            
            if stuckness > 100:                                                                         #Termina el aprendizaje si no hay mejora por más de 100 generaciones
                #pass
                break

            print("Best Err: ", best_err)                                                               #Logs
            print("Base Val: ", base)
            print("Perc Err: ", pct_err)
            print("M Boom: ", mut_boom, "Stuck: ", stuckness, "Ch_Rate: ", change_rate)
            
            print("End Gen: ", self.Gen.gen -1 )
            print("=====")
            print()

            print("-> ", self.Gen.gen)
            print("Evolution Gen: ", self.Gen.gen)

            new_gen = self.Gen.gen + 1                                                                  #Creación de la nueva generación                
            self.subsets = self.create_subsets(self.subsets, 0.6)
        
            if mut_boom >= 10: 
                print("                                        MUT BOOM!")
                self.Gen = Generation(self.subsets, best_params, new_gen, mut_prob=0.005 )    
                mut_boom = 0
            else: 
                self.Gen = Generation(self.subsets, best_params, new_gen)    
        
        else: 
            self.best_genes = best_params                                                               #Una vez terminado el entrenamiento, devuelve los mejores parámetros obtejidos. 


    def create_subsets(self, signal, retain_pctg=0, batch_size=10): 

        """
        Controla la creación de nuevos Subsets. Puede tomar dos comportamientos:
        1.- Si sólo se entrega una BD, asume que es una BD original con n pulsos individuales, y retorna un objeto con la misma BD reorganizada en m batches de tamaño batch_size (m = n // batch_size + 1)
        2.- De de indicarse un retain_pctg junto con la  BD, asume que recibe una BD organizada en m batches, y efectúa una randomización de toda la BD generando nuevos batches. 
            - Cada uno de estos nuevos batches mantiene un porcentaje retain_pctg del 100% de los pulsos contenidos en el batch correspondiente original, 
              enviando el % restante a un pool para su randomización. 
            - Efectuado esto, los nuevos batch completan su 100% = batch_size, tomando pulsos al azar del pool recién randomizado. 
        
        """
    
        if not(retain_pctg):                                    #Comportamiento n° 1

            signal = np.array(signal)                           #Asegura que sea un array numpy

            sig_cant = signal.shape[0]                          #Determina el n de pulsos individuales

            indexs = np.arange(0,sig_cant,1)                    #Genera un guía para tomar los pulsos de manera random
            rnd.shuffle(indexs)

            subsets = []                                        #Objeto que contendrá la BD reorganizada que se retornará
            batch = []                                          #Objeto intermediario para la creación de cada batch
            batch_quantity = 0                                  #Registro de la cantidad de batchs credaos

            for i in indexs:                                    #Aprovechando el azar contenido en la lista indexs, se crea cada batch.

                if len(batch) < batch_size:                     #Agrega pulsos al objeto intermediario
                    batch.append(signal[i])
                else:                                           #Si el batch intermediario alcanza el batch_size
                    b_copy = list(batch)                        #Crea una copia independiente del batch creado 
                    subsets.append(b_copy)                      #Y lo agrega al subsets
                    batch = []                                  #Vacía el objeto intermediario
                    batch.append(signal[i])                     #Y guarda el pulso actual.
                    batch_quantity += 1                         #Registra la creación del batch
            else: 
                subsets.append(list(batch))                     #Una vez terminado con los pulsos, adiciona el batch con los pulsos restantes
                batch_quantity += 1                             #Y lo registra. 

        else:                                                   #Comportamiento n° 2

            torandom_pool = []                                  #Objeto que contendrá el pool a randomizar
            remain_subset = []                                  #Objeto que contendrá los batches retenidos

            for batch in signal:                                #Por cada batch en la BD
                retain_pos = int((len(batch) * retain_pctg))    #Determina la cantidad de pulsos que se mantendrán en el batch

                rnd.shuffle(batch)                              #Randomiza el batch, para que los pulsos mantenidos no sean siempre los mismos
                torandom_pool.extend(batch[retain_pos:])        #Envía los pulsos que no se mantendrán, al pool a randomizar
                remain_subset.append(batch[:retain_pos])        #Envía el batch con los pulsos mantenidos.
            
            indexs = np.arange(0,len(torandom_pool),1)          #Ejecuta la randomización del pool. 
            rnd.shuffle(indexs)
            indexs = iter(indexs)

            for batch in remain_subset:                         #Por cada batch retenido

                try: 
                    while len(batch) < batch_size:              #Y mientras el batch no alcance el batch_size
                    
                        i = next(indexs)                        #Toma una posición previamente randomizada 
                        batch.append(torandom_pool[i])          #Y agrega el pulso correspondiente al batch    
                except: 
                    break                                       #Este except controla la finalización de la completación de los batches. Se encuentra a la escucha del error que ocurrirá cuando el interador 'indexs' se termine, y al detectarlo, lo captura y envía la señal de finalzación del for.
                
            
            subsets = remain_subset                             #Renombra la BD construida. 
        
        return subsets


if __name__ == "__main__": 

    #Definición de la Derivación
    TRAINING_DERIVATION = 'I'
    derivations = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    #Definición de los archivos de log y resultados.

    result_file = 'Results/21-02-18_results'
    log_file    = 'Results/21-02-18_log'

    last_params = [
        [-1.1471975511965977, -0.1617993877991494, 0, 0.1617993877991494, 1.6929693744344996, 1.7453292519943295, 0.8120812342152769, -5.001791134962331, 30.003910180961153, -7.486495296984126, 0.5173193840936107, 0.7604877414164366, 0.19742542321927406, 0.10758288788728022, 0.08092341583951007, 0.09683605807604678, 0.3980991693063881, 0.1930326695545934, 1.002547973318158, 0, 0.04],
        [-1.1471975511965977, -0.1617993877991494, 0, 0.1617993877991494, 1.6929693744344996, 1.7453292519943295, 0.803336145785575, -4.98888690419002, 29.990870833659436, -7.487720019245078, 0.48432914196268484, 0.7464775656164093, 0.20267126366722837, 0.11786401832849114, 0.09138486472746055, 0.1100453880502475, 0.40778037440924697, 0.1890055434259555, 0.9912449019542849, 0, 0.04],
        [-1.1471975511965977, -0.1617993877991494, 0, 0.1617993877991494, 1.6929693744344996, 1.7453292519943295, 0.7986306044435985, -4.991929228923516, 29.980740168390163, -7.516603148269003, 0.49856844767918995, 0.7431705941031669, 0.2, 0.1022692389672059, 0.09358764339321739, 0.10052299387749382, 0.3933423716069886, 0.21566411175531766, 1.0063775705943017, 0, 0.04],
        [-1.1471975511965977, -0.1617993877991494, 0, 0.1617993877991494, 1.6929693744344996, 1.7453292519943295, 0.7885978307389331, -4.995780974247471, 30.006354625306958, -7.452083667154117, 0.5088735071148706, 0.7620341394759355, 0.231251594644881, 0.08835984818839855, 0.10382195645753846, 0.10852482093251595, 0.3939040028712723, 0.2082495497939626, 1.0111153327770503, 0, 0.04],
        [-1.1471975511965977, -0.1617993877991494, 0, 0.1617993877991494, 1.6929693744344996, 1.7453292519943295, 0.7993239518932396, -4.9973491520076445, 29.992514095267126, -7.493972128339478, 0.5046108161243993, 0.7604314469102952, 0.1895284217912473, 0.09872979298074845, 0.10827933226229507, 0.08822297409698254, 0.3927801405416687, 0.21555170399893664, 0.9896045285893472, 0, 0.04],
        [-1.1471975511965977, -0.1617993877991494, 0, 0.1617993877991494, 1.6929693744344996, 1.7453292519943295, 0.7864524750981536, -4.991294386989476, 29.98955410469975, -7.50299489488064, 0.4956428850773395, 0.75, 0.19280475576540393, 0.08498481173099595, 0.09952767058466454, 0.10373177080932976, 0.40886312817258513, 0.21177688643902767, 0.9902134202405258, 0, 0.04],
        [-1.1471975511965977, -0.1617993877991494, 0, 0.1617993877991494, 1.6929693744344996, 1.7453292519943295, 0.7925388645705539, -5.003646015751189, 29.985743438372257, -7.475236951142573, 0.49418016000498044, 0.7642494472540099, 0.19052894714079102, 0.08580764539788163, 0.09231889679319251, 0.09282889309529505, 0.40838135030875555, 0.21308173182169954, 1.0000455728087396, 0, 0.04],
        [-1.1471975511965977, -0.1617993877991494, 0, 0.1617993877991494, 1.6929693744344996, 1.7453292519943295, 0.7932787268792627, -4.978221643796146, 30.002686591299724, -7.508040991739306, 0.5243642530166646, 0.7535906982194406, 0.2, 0.10614871530133807, 0.09395980571205154, 0.09912618934151649, 0.3988730089314773, 0.21151065972875313, 1.001307026012343, 0, 0.04],
        [-1.1471975511965977, -0.1617993877991494, 0, 0.1617993877991494, 1.6929693744344996, 1.7453292519943295, 0.8106027316894047, -5.001578062223189, 30, -7.500456021867569, 0.5, 0.7464313043372288, 0.18474195320127895, 0.09090224341278612, 0.1189761530610252, 0.08348621132316038, 0.39275563164692173, 0.19862723696440424, 1.015534842632823, 0, 0.04],
        [-1.1471975511965977, -0.1617993877991494, 0, 0.1617993877991494, 1.6929693744344996, 1.7453292519943295, 0.7889707225582293, -5.00081104047417, 30.01177321104575, -7.511812060399266, 0.5, 0.7521158271319092, 0.2036954402037835, 0.1, 0.08931500096286002, 0.10323212314127429, 0.4046229653374198, 0.2170596148772306, 1.0044379462716915, 0, 0.04],
        [-1.1471975511965977, -0.1617993877991494, 0, 0.1617993877991494, 1.6929693744344996, 1.7453292519943295, 0.788384055867279, -4.992931433560074, 29.97952803515596, -7.4957186807997775, 0.48269703091307714, 0.7438446786183907, 0.18645805123773834, 0.08640146140915925, 0.1127334718995953, 0.11648172076994563, 0.4169858071110732, 0.18660005249786563, 0.9953849537373345, 0, 0.04],
        [-1.1471975511965977, -0.1617993877991494, 0, 0.1617993877991494, 1.6929693744344996, 1.7453292519943295, 0.8, -5.02050073678753, 30.015127327750832, -7.5008524009467505, 0.5196529988070715, 0.7548625125925562, 0.19208757425852133, 0.1, 0.1053345985192225, 0.08128590056975422, 0.40747288843459073, 0.18283976725097326, 1.0031501015320001, 0, 0.04],
    ]

    mt = time.time()

    for d in range(len(derivations)):
        ST = time.time()
        TRAINING_DERIVATION = derivations[d]
        #Lectura de la BD de Derivación
        file_name = "Derivations_BDs/BD_"+  TRAINING_DERIVATION + "_signal"         #Ojo que el Derivations_BDs/ es un Acceso Directo
        ecg_recover = wfdb.rdsamp(file_name)
        print('Signal Readed: {}'.format(TRAINING_DERIVATION))
        s = ecg_recover[0].transpose()

        #Cálculo de la amplitud promedio de la señal 
        mean = 0
        for i in s: 
            if TRAINING_DERIVATION in ['AVR', 'AVL','V1', 'V2']:
                mean += min(i)
            else: 
                mean += max(i)
        else: 
            mean = mean/len(s)
            print(mean)
    
        #Obtención de Parámetros de Partida
        p = init_params.theta_vals + init_params.a_vals + init_params.b_vals + init_params.y0 + [mean]
        #p = last_params[d]
        Orig = Learner(s[:2],p)
        
        #Reducción opcional de la señal. Para pruebas rápidas. 
        #s = s[:50]

        #Cálculo del error original de la señal
        err = 0
        for i in s : 
            err += sum((Orig.signal[1] - i)**2)
        print(err)

        #Construcción del Ambiente de Entrenamiento
        T = Training(s,generations = 100,params = p)
        T.evolution()

        #Obtención del último mejor set de parámetros
        LB_params = T.best_genes
        L_min_pct = T.pcg_history[-1]
        L_min_err = T.err_history[-1]
        Last_Best = Learner(s[:2],LB_params)

        #Obtención del históricamente mejor set de parámetros
        H_min_err = min(T.err_history)
        H_min_i = T.err_history.index(H_min_err)
        H_min_pct = T.pcg_history[H_min_i]
        HB_params = T.p_history[H_min_i]
        Hist_Best = Learner(s[:2],HB_params)

        #Obtención del históricamente pct mejor set de parámetros
        P_min_pct = min(T.pcg_history)
        P_min_i = T.pcg_history.index(P_min_pct)
        PT_params = T.p_history[P_min_i]
        Pct_Best = Learner(s[:2],PT_params)

        ED = time.time()

        f = open(result_file,'a')
        f.write("\n\n -> {}\n".format(TRAINING_DERIVATION))
        f.write("t: {}\n\n".format(ED - ST))
        f.write("LB_Params: {}\nLErr: {}\nLPct: {}%\nPos: {}\n\n".format(LB_params,  L_min_err,  L_min_pct,  -1))
        f.write("HB_Params: {}\nHErr: {}\nHPct: {}%\nPos: {}\n\n".format(HB_params,  H_min_err,  H_min_pct,  H_min_i))
        f.write("PT_Params: {}\nPErr: {}\nPPct: {}%\nPos: {}\n\n".format(PT_params,  P_min_pct,  H_min_pct,  P_min_i))
        f.write("Pct_data: {}\n".format(T.pcg_history))
        f.write("Err_data: {}\n\n".format(T.err_history))
        f.write("==========\n\n")
        f.close()

    f = open(result_file,'a')
    
    f.write("t: {}".format(ED- mt))
    f.close()

#"""         #Ploteo de la evolución del error
#        plt.plot(T.err_history, label='Err')
#        plt.plot(T.pcg_history, label='Pct')
#        plt.legend()
#        plt.show()

#        #Ploteo de las señales originales. 
#        plt.figure()
#        for i in s: 
#            plt.plot(i, c='g')

#        #Ploteo del Último mejor y el Históricamente Mejor
#        plt.plot(Orig.signal[1], c = 'y', label = "Orig")
#        plt.plot(Last_Best.signal[1], c = 'r', label = "Last_Best")
#        plt.plot(Hist_Best.signal[1], c = 'b', label = "Hist_Best {}".format(H_min_i))
#        plt.plot(Pct_Best.signal[1], c = 'k', label = "PCT_Best {}".format(min_pct))
        
#        plt.legend()

#        plt.show() """