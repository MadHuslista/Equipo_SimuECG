 
import pandas as pd
import numpy as np
import wfdb
import matplotlib.pyplot as plt 
import joblib
from biosppy import storage
from biosppy.signals import ecg
from scipy import signal as scp_sig

#Dervations order: ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
#                  [ 0  ...                       ...                              ...    11 ]

#ecg.ecg outputs: ['ts', 'filtered', 'rpeaks', 'templates_ts', 'templates', 'heart_rate_ts', 'heart_rate']

def extract_heartbeats_RRadapted(signal=None, rpeaks=None, sampling_rate=0, before=0.2, after=0.4, rr_norm_interval=0.9):


    #Convierto los Delimitadores Referenciales de inicio y término a samples. 
    before = before * sampling_rate
    after = after * sampling_rate
    ref_rr = before + after 

    #Get RR intervals. 
    RR_int_sig = rpeaks[1:] - rpeaks[:-1]
    length = len(signal)
    templates = []


    max_bef = int(before * (float(max(RR_int_sig))/float(ref_rr)))
    #print(max_bef)

    for i in range(len(RR_int_sig)): 
        #print(rpeaks[i], " ", RR_int_sig[i], end='    ')

        #Obtengo los Delimitadores Proporcionales, en referencia al intervalo propio de la pulsación particular. 
        bef_adapt = int(before * (float(RR_int_sig[i])/float(ref_rr)))
        aft_adapt = int(after * (float(RR_int_sig[i])/float(ref_rr)))

        #Extraigo el heartbeat
        #Se extrae de manera diferenciada en respecto al punto R para mantener su referencia y permitir el calce de todos los R. 
        a = rpeaks[i] - bef_adapt
        if a < 0: 
            continue
        b = rpeaks[i] + aft_adapt
        if b > length: 
            break
        
        #sig = signal[a:b]
        a_sig = signal[a:rpeaks[i]]
        b_sig = signal[rpeaks[i]:b]

        #Resampling to 900ms. At 500Hz it would be before:150 + after:300 samples (total:450 samples = 500Hz*0.9s)
        #Calculo los Delimitadores de Resampleo
        norm_rr = rr_norm_interval * sampling_rate
        norm_bef =  int(before* float(norm_rr)/float(ref_rr))
        norm_aft =  int(after* float(norm_rr)/float(ref_rr))

        #Ejecuto el resampleo. Se efectúa aún por parte para mantener la posición coordinada del R en todos los HR. 
        a_sig = scp_sig.resample(a_sig, norm_bef)
        b_sig = scp_sig.resample(b_sig, norm_aft)


        #Se termina de construir la señal completa ya resampleada. 
        sig = np.concatenate((a_sig, b_sig))

        #if bef_adapt < max_bef:                    #Calce a mano por técnica de 'zero padding' 
        #    padd = np.zeros((max_bef - bef_adapt))
        #    sig = np.insert(sig, 0,padd)

        #Se reunen y devuelven todos los heartbeats independizados de la señal recibida. 
        templates.append(sig)
        
    return templates



files = ['03','04','05','06','09','10','13','14']    
l = []
for i in files: 
    l.append(('000'+i+'_hr'))
    

for f in l: 
    ecg_l = wfdb.rdsamp(f)

    for i in range(12):
        signal_l = ecg_l[0].transpose()[i] 

        if ecg_l[1]['sig_name'][i] in ['AVR', 'AVL','V1', 'V2']:
            #Doy vuelta la señal, así efectivamente toma el R, que en estas derivaciones es negativo
            signal_l *= -1

        out = ecg.ecg(signal= signal_l, sampling_rate=500., show=True)

        filtered = out['filtered']
        rpeaks = out['rpeaks']

        if ecg_l[1]['sig_name'][i] in ['AVR', 'AVL','V1', 'V2']:
            #Una vez efectuado el rpocesamiento la desdoy vuelta, para devolverla a su estado original. 
            filtered *= -1

        t = extract_heartbeats_RRadapted(filtered, rpeaks, 500)

        for j in t: 
            plt.plot(j)
            
            #print(len(i))
        plt.title((ecg_l[1]['sig_name'][i] +' - '+ f))
        plt.show()

