# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 15:19:10 2022

@author: willi
"""

import pandas   
from IPython.display import Image, display
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import os
import tqdm.notebook as tq
import PIL
import ipywidgets as widgets
from ipywidgets import interact
#import imread
#import cv2
#import skimage.io as io
from matplotlib.colors import ListedColormap
from matplotlib.colors import LogNorm

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix

print(os.getcwd())
from mc_core import multiplexing_core 
from apply_style import apply_style
import numpy as np, scipy.stats as st

multiplexing_df = pd.read_csv('./P300_KDM5B_24000s_Same_intensity_gaussian_14scale/ki_ke_sweep_same_int_KDM5B_KDM5B_0.014139262990455991_5.33333_0.csv')
snrs_0 = []
for i in range(int(np.max(multiplexing_df['particle']))):
    pdf = multiplexing_df[multiplexing_df['particle'] == i]
    snr = np.mean(pdf['green_int_mean'])/np.std(pdf['blue_int_mean'])
    snrs_0.append(snr)

print(st.t.interval(0.95, len(snrs_0)-1, loc=np.mean(snrs_0), scale=st.sem(snrs_0)))


multiplexing_df = pd.read_csv('./P300_KDM5B_24000s_Same_intensity_gaussian_14scale/ki_ke_sweep_same_int_P300_P300_0.009675852685050798_5.33333_0.csv')
snrs_0 = []
for i in range(int(np.max(multiplexing_df['particle']))):
    pdf = multiplexing_df[multiplexing_df['particle'] == i]
    snr = np.mean(pdf['green_int_mean'])/np.std(pdf['blue_int_mean'])
    snrs_0.append(snr)

print(st.t.interval(0.95, len(snrs_0)-1, loc=np.mean(snrs_0), scale=st.sem(snrs_0)))

print('___________')
multiplexing_df = pd.read_csv('./P300_KDM5B_24000s_similar_intensity_gaussian_14scale/ki_ke_sweep_same_int_KDM5B_KDM5B_0.0186_5.33333_0.csv')
snrs_0 = []
for i in range(int(np.max(multiplexing_df['particle']))):
    pdf = multiplexing_df[multiplexing_df['particle'] == i]
    snr = np.mean(pdf['green_int_mean'])/np.std(pdf['blue_int_mean'])
    snrs_0.append(snr)

print(st.t.interval(0.95, len(snrs_0)-1, loc=np.mean(snrs_0), scale=st.sem(snrs_0)))


multiplexing_df = pd.read_csv('./datasets/P300_KDM5B_24000s_similar_intensity_gaussian_14scale/ki_ke_sweep_same_int_P300_P300_0.009675858127721334_5.33333_0.csv')
snrs_0 = []
for i in range(int(np.max(multiplexing_df['particle']))):
    pdf = multiplexing_df[multiplexing_df['particle'] == i]
    snr = np.mean(pdf['green_int_mean'])/np.std(pdf['blue_int_mean'])
    snrs_0.append(snr)

print(st.t.interval(0.95, len(snrs_0)-1, loc=np.mean(snrs_0), scale=st.sem(snrs_0)))

print('___________')
multiplexing_df = pd.read_csv('./construct_length_dataset_larger_range_14scale/construct_lengths_RRAGC_RRAGC.csv')
snrs_0 = []
for i in range(int(np.max(multiplexing_df['particle']))):
    pdf = multiplexing_df[multiplexing_df['particle'] == i]
    snr = np.mean(pdf['green_int_mean'])/np.std(pdf['blue_int_mean'])
    snrs_0.append(snr)

print(st.t.interval(0.95, len(snrs_0)-1, loc=np.mean(snrs_0), scale=st.sem(snrs_0)))


multiplexing_df = pd.read_csv('./construct_length_dataset_larger_range_14scale/construct_lengths_P300_P300.csv')
snrs_0 = []
for i in range(int(np.max(multiplexing_df['particle']))):
    pdf = multiplexing_df[multiplexing_df['particle'] == i]
    snr = np.mean(pdf['green_int_mean'])/np.std(pdf['blue_int_mean'])
    snrs_0.append(snr)

print(st.t.interval(0.95, len(snrs_0)-1, loc=np.mean(snrs_0), scale=st.sem(snrs_0)))


print('___________')
multiplexing_df = pd.read_csv('./datasets/par_sweep_5000/ki_ke_sweep_5000spots_0.06000000000000001_5.333333333333334.csv')
snrs_0 = []
snrs_1 = []
for i in range(int(np.max(multiplexing_df['particle']))):
    pdf = multiplexing_df[multiplexing_df['particle'] == i]
    if pdf['Classification'].iloc[0] == 0:
        snr = np.mean(pdf['green_int_mean'])/np.std(pdf['blue_int_mean'])
        snrs_0.append(snr)
    else:
        snr = np.mean(pdf['green_int_mean'])/np.std(pdf['blue_int_mean'])
        snrs_1.append(snr)        

print(st.t.interval(0.95, len(snrs_0)-1, loc=np.mean(snrs_0), scale=st.sem(snrs_0)))
print(st.t.interval(0.95, len(snrs_1)-1, loc=np.mean(snrs_1), scale=st.sem(snrs_1)))

print('___________')

multiplexing_df = pd.read_csv('./par_sweep_5000/ki_ke_sweep_5000spots_0.1_2.0.csv')
snrs_0 = []
snrs_1 = []
for i in range(int(np.max(multiplexing_df['particle']))):
    pdf = multiplexing_df[multiplexing_df['particle'] == i]
    if pdf['Classification'].iloc[0] == 0:
        snr = np.mean(pdf['green_int_mean'])/np.std(pdf['blue_int_mean'])
        snrs_0.append(snr)
    else:
        snr = np.mean(pdf['green_int_mean'])/np.std(pdf['blue_int_mean'])
        snrs_1.append(snr)        

print(st.t.interval(0.95, len(snrs_0)-1, loc=np.mean(snrs_0), scale=st.sem(snrs_0)))
print(st.t.interval(0.95, len(snrs_1)-1, loc=np.mean(snrs_1), scale=st.sem(snrs_1)))

print('___________')
multiplexing_df = pd.read_csv('./par_sweep_kes/parsweep_kes_p300_2.0.csv')
snrs_0 = []
for i in range(int(np.max(multiplexing_df['particle']))):
    pdf = multiplexing_df[multiplexing_df['particle'] == i]
    snr = np.mean(pdf['green_int_mean'])/np.std(pdf['blue_int_mean'])
    snrs_0.append(snr)

print(st.t.interval(0.95, len(snrs_0)-1, loc=np.mean(snrs_0), scale=st.sem(snrs_0)))


multiplexing_df = pd.read_csv('./datasets/par_sweep_kes/parsweep_kes_p300_12.0.csv')
snrs_0 = []
for i in range(int(np.max(multiplexing_df['particle']))):
    pdf = multiplexing_df[multiplexing_df['particle'] == i]
    snr = np.mean(pdf['green_int_mean'])/np.std(pdf['blue_int_mean'])
    snrs_0.append(snr)

print(st.t.interval(0.95, len(snrs_0)-1, loc=np.mean(snrs_0), scale=st.sem(snrs_0)))


print('___________')
multiplexing_df = pd.read_csv('./datasets/par_sweep_kes/parsweep_kes_kdm5b_2.0.csv')
snrs_0 = []
for i in range(int(np.max(multiplexing_df['particle']))):
    pdf = multiplexing_df[multiplexing_df['particle'] == i]
    snr = np.mean(pdf['green_int_mean'])/np.std(pdf['blue_int_mean'])
    snrs_0.append(snr)

print(st.t.interval(0.95, len(snrs_0)-1, loc=np.mean(snrs_0), scale=st.sem(snrs_0)))


multiplexing_df = pd.read_csv('./par_sweep_kes/parsweep_kes_kdm5b_12.0.csv')
snrs_0 = []
for i in range(int(np.max(multiplexing_df['particle']))):
    pdf = multiplexing_df[multiplexing_df['particle'] == i]
    snr = np.mean(pdf['green_int_mean'])/np.std(pdf['blue_int_mean'])
    snrs_0.append(snr)

print(st.t.interval(0.95, len(snrs_0)-1, loc=np.mean(snrs_0), scale=st.sem(snrs_0)))



print('___________')
multiplexing_df = pd.read_csv('./par_sweep_kis/parsweep_kis_p300_0.01.csv')
snrs_0 = []
for i in range(int(np.max(multiplexing_df['particle']))):
    pdf = multiplexing_df[multiplexing_df['particle'] == i]
    snr = np.mean(pdf['green_int_mean'])/np.std(pdf['blue_int_mean'])
    snrs_0.append(snr)

print(st.t.interval(0.95, len(snrs_0)-1, loc=np.mean(snrs_0), scale=st.sem(snrs_0)))


multiplexing_df = pd.read_csv('./par_sweep_kis/parsweep_kis_p300_0.1.csv')
snrs_0 = []
for i in range(int(np.max(multiplexing_df['particle']))):
    pdf = multiplexing_df[multiplexing_df['particle'] == i]
    snr = np.mean(pdf['green_int_mean'])/np.std(pdf['blue_int_mean'])
    snrs_0.append(snr)

print(st.t.interval(0.95, len(snrs_0)-1, loc=np.mean(snrs_0), scale=st.sem(snrs_0)))


print('___________')
multiplexing_df = pd.read_csv('./par_sweep_kis/parsweep_kis_kdm5b_0.01.csv')
snrs_0 = []
for i in range(int(np.max(multiplexing_df['particle']))):
    pdf = multiplexing_df[multiplexing_df['particle'] == i]
    snr = np.mean(pdf['green_int_mean'])/np.std(pdf['blue_int_mean'])
    snrs_0.append(snr)

print(st.t.interval(0.95, len(snrs_0)-1, loc=np.mean(snrs_0), scale=st.sem(snrs_0)))


multiplexing_df = pd.read_csv('./par_sweep_kis/parsweep_kis_kdm5b_0.1.csv')
snrs_0 = []
for i in range(int(np.max(multiplexing_df['particle']))):
    pdf = multiplexing_df[multiplexing_df['particle'] == i]
    snr = np.mean(pdf['green_int_mean'])/np.std(pdf['blue_int_mean'])
    snrs_0.append(snr)

print(st.t.interval(0.95, len(snrs_0)-1, loc=np.mean(snrs_0), scale=st.sem(snrs_0)))


print('___________')
multiplexing_df = pd.read_csv('./construct_length_dataset_larger_range_14scale/construct_lengths_RRAGC_RRAGC.csv')
snrs_0 = []
for i in range(int(np.max(multiplexing_df['particle']))):
    pdf = multiplexing_df[multiplexing_df['particle'] == i]
    snr = np.mean(pdf['green_int_mean'])/np.std(pdf['blue_int_mean'])
    snrs_0.append(snr)

print(st.t.interval(0.95, len(snrs_0)-1, loc=np.mean(snrs_0), scale=st.sem(snrs_0)))

multiplexing_df = pd.read_csv('./construct_length_dataset_larger_range_14scale/construct_lengths_DOCK8_DOCK8.csv')
snrs_0 = []
for i in range(int(np.max(multiplexing_df['particle']))):
    pdf = multiplexing_df[multiplexing_df['particle'] == i]
    snr = np.mean(pdf['green_int_mean'])/np.std(pdf['blue_int_mean'])
    snrs_0.append(snr)

print(st.t.interval(0.95, len(snrs_0)-1, loc=np.mean(snrs_0), scale=st.sem(snrs_0)))


multiplexing_df = pd.read_csv('./construct_length_dataset_larger_range_14scale/construct_lengths_ORC2_ORC2.csv')
snrs_0 = []
for i in range(int(np.max(multiplexing_df['particle']))):
    pdf = multiplexing_df[multiplexing_df['particle'] == i]
    snr = np.mean(pdf['green_int_mean'])/np.std(pdf['blue_int_mean'])
    snrs_0.append(snr)

print(st.t.interval(0.95, len(snrs_0)-1, loc=np.mean(snrs_0), scale=st.sem(snrs_0)))

multiplexing_df = pd.read_csv('./construct_length_dataset_larger_range_14scale/construct_lengths_PHIP_PHIP.csv')
snrs_0 = []
for i in range(int(np.max(multiplexing_df['particle']))):
    pdf = multiplexing_df[multiplexing_df['particle'] == i]
    snr = np.mean(pdf['green_int_mean'])/np.std(pdf['blue_int_mean'])
    snrs_0.append(snr)

print(st.t.interval(0.95, len(snrs_0)-1, loc=np.mean(snrs_0), scale=st.sem(snrs_0)))

