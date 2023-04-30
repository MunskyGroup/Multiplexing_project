# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 15:14:22 2022

@author: willi
"""
# imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


##################################################################################
# This code makes the data pipeline figure for the paper from example trajectories
##################################################################################

import os
cwd = os.getcwd()
os.chdir('../../')
import apply_style #apply custom matplotlib style
os.chdir(cwd)


apply_style.apply_style(dark=True)

ssas = np.load('./single_traj_example.csv_ssas.npy')
observed = pd.read_csv('./single_traj_example.csv')

plt.figure()
plt.plot(ssas[0,0,0],  color='#00ff00ff'); plt.ylim([100,260]); plt.xlim([0,500])
plt.xlabel('Time'); plt.ylabel('Intensity')
plt.title('KDM5B Spot Trajectory (model)')

plt.figure()
plt.plot(ssas[0,1,0],  color='#00ff00ff'); plt.ylim([100,260]); plt.xlim([0,500])
plt.xlabel('Time'); plt.ylabel('Intensity')
plt.title('P300 Spot Trajectory (model)')


plt.figure()
plt.plot(observed['green_int_mean'][:500],  color='#00ff00ff'); plt.ylim([500,2000]); plt.xlim([0,500])
plt.xlabel('Time'); plt.ylabel('Intensity')
plt.title('KDM5B Observed Intensity')


plt.figure()
plt.plot(np.linspace(0,499,500), observed['green_int_mean'][1000:1500],  color='#00ff00ff'); plt.ylim([500,2000]); plt.xlim([0,500])
plt.xlabel('Time'); plt.ylabel('Intensity')
plt.title('P300 Observed Intensity')
