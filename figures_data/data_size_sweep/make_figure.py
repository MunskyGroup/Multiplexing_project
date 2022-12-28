# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 18:15:19 2022

@author: willi
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

import os
cwd = os.getcwd()
os.chdir('../../')
import apply_style  as aps#apply custom matplotlib style
import mc_core as multiplexing_core
os.chdir(cwd)

aps.apply_style()

accs = np.load('D:/multiplexing_ML/finalized_plots_gaussians/figures_data/data_size_sweep/accs_data_size.npy')
accs_2 = np.load('D:/multiplexing_ML/finalized_plots_gaussians/figures_data/data_size_sweep/accs_2_data_size.npy')

a = np.vstack([accs,accs_2[3:]])

plt.figure(dpi=300, figsize=(5,3))
data_size = [50,100,150,200,250,300,350,400,500,600,700,800,900,1000,1250] + np.linspace(100,4000,10).astype(int).tolist()[3:]
plt.errorbar(data_size, np.mean(a,axis=1), yerr= np.std(a,axis=1), capsize=2, linestyle='', marker='o')
plt.xlabel('Training data size')
plt.ylabel('Test Accuracy')

plt.savefig('training_data_size.svg')
#plt.xlim([0,000])