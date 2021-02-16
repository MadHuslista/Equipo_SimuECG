
import numpy as np 
import matplotlib.pyplot as plt 
import wfdb
import random as rnd 
import time

ecg_recover = wfdb.rdsamp("Derivations_Data/BD_II_signal")
sign_recover = ecg_recover[0].transpose()

def create_subsets(signal, retain_pctg=0, batch_size=10): 
    
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




s = create_subsets(sign_recover)
f = create_subsets(s, 0.6)


