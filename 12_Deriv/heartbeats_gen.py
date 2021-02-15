import pandas as pd
import numpy as np
import wfdb
import matplotlib.pyplot as plt 
from biosppy.signals import ecg
from tachogram_detect import extract_heartbeats_RRadapted
from norm_data import files
import time 

sampling_rate = 500.

TRAINING_DERIVATION = 'I'
derivations = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

ST = time.time()

for i in derivations: 

    l_st = time.time()

    TRAINING_DERIVATION = i

    deriv_index = derivations.index(TRAINING_DERIVATION)

    heartbeats = []
    ch_name = []
    for f in files: 

        ecg_signal = wfdb.rdsamp(f)

        signal = ecg_signal[0].transpose()[deriv_index]

        if ecg_signal[1]['sig_name'][deriv_index] in ['AVR', 'AVL','V1', 'V2']:
            #Doy vuelta la señal, así efectivamente toma el R, que en estas derivaciones es negativo
            signal *= -1    
        
        out = ecg.ecg(signal= signal, sampling_rate=sampling_rate, show=False)

        filtered = out['filtered']
        rpeaks = out['rpeaks']

        if ecg_signal[1]['sig_name'][deriv_index] in ['AVR', 'AVL','V1', 'V2']:
            #Una vez efectuado el rpocesamiento la desdoy vuelta, para devolverla a su estado original. 
            filtered *= -1

        t = extract_heartbeats_RRadapted(out['filtered'], out['rpeaks'], sampling_rate)

        ch = [ f[-8:-3] + '-' + str(i) for i in range(len(t))]
        
        ch_name.extend(ch)
        heartbeats.extend(t)

    heartbeats = np.array(heartbeats)
    h = heartbeats
    heartbeats = heartbeats.transpose()

    units = ['mV' for i in range(heartbeats.shape[1])]
    ch_name = [TRAINING_DERIVATION + '-' + ch_name[i] for i in range(len(ch_name))]
    sig_name = 'Derivations_Data/'+ 'BD_'+TRAINING_DERIVATION+'_signal'
    fmt = ['32'for i in range(heartbeats.shape[1]) ]



    wfdb.wrsamp(sig_name, fs=sampling_rate,p_signal=heartbeats, units = units, sig_name=ch_name, fmt=fmt)



    ecg_recover = wfdb.rdsamp(sig_name)
    sign_recover = ecg_recover[0].transpose()

    err = sum(sum((h - sign_recover)**2))
    base = sum(sum(h**2))

    l_ed = time.time()

    print()
    print('->', TRAINING_DERIVATION)
    print(err)
    print(base)
    print(100*err/base)
    print('t: ', l_ed - l_st)
    print("========")

ED = time.time()

print()
print('t: ', ED-ST)