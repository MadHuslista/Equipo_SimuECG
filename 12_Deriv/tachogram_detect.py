 
import pandas as pd
import numpy as np
import wfdb
import matplotlib.pyplot as plt 
import joblib
from biosppy import storage
from biosppy.signals import ecg

#Dervations order: ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
#                  [ 0  ...                       ...                              ...    11 ]


ecg_s = wfdb.rdsamp('00003_lr')




for i in range(12):
    signal = ecg_s[0].transpose()[i]
    out = ecg.ecg(signal= signal, sampling_rate=100., show=True)

