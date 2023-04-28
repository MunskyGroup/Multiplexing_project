# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 11:46:44 2022

@author: willi
"""
import pandas as pd
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

import apply_style  as aps#apply custom matplotlib style
import mc_core as multiplexing_core


aps.apply_style()

mc = multiplexing_core.multiplexing_core()
aps.apply_style()
files = ['./par_sweep_5000/ki_ke_sweep_5000spots_0.06000000000000001_5.333333333333334.csv']

ntraj = 5000
ntimes = 3000

green_intensities = []
all_labels = []
xs = []
ys = []

msds = []

def compute_msd(trajectory):
  '''
  This function is intended to calculate the mean square displacement of a given trajectory.
  msd(τ)  = <[r(t+τ) - r(t)]^2>
  Inputs:
    trajectory: list of temporal evolution of a centers of mass .  [Y_val_particle_i_tp_0, X_val_particle_i_tp_0]   , ... , [Y_val_particle_i_tp_n, X_val_particle_i_tp_n] ]

  Returns
    msd: float, mean square displacement
    rmsd: float, root mean square displacement
  '''
  total_length_trajectory=len(trajectory)
  msd=[]
  for i in range(total_length_trajectory-1):
      tau=i+1
      # Distance that a particle moves for each time point (tau) divided by time
      # msd(τ)                 = <[r(t+τ)  -    r(t)]^2>
      msd.append(np.sum((trajectory[0:-tau]-trajectory[tau::])**2)/(total_length_trajectory-tau)) # Reverse Indexing 
  # Converting list to np.array
  msd=np.array(msd)   # array with shape Nspots vs time_points
  rmsd = np.sqrt(msd)
  return msd, rmsd 


for path1 in files:
    multiplexing_df = pd.read_csv(path1)
    green_intensities.append(multiplexing_df['green_int_mean'].values.reshape([ntraj,ntimes]))
    all_labels.append(multiplexing_df['Classification'].values.reshape([ntraj,ntimes])[:,0])
    x, y = multiplexing_df['x'].values.reshape([ntraj,ntimes]) , multiplexing_df['y'].values.reshape([ntraj,ntimes]) 
    xy_arr = np.moveaxis(np.array([x,y]),0,1)
    
    xs.append( multiplexing_df['x'].values.reshape([ntraj,ntimes]) )
    ys.append( multiplexing_df['y'].values.reshape([ntraj,ntimes]) )
    
    
    msd_c1 = np.array([compute_msd(xy_arr[i].T)[0] for i in range(len(xy_arr))])
    diff_t = np.mean(msd_c1,axis=0)/(2*2*np.linspace(1,2999,2999))


linear_fit_model = np.polyfit(x = np.linspace(1,2999,2999), y = diff_t, deg = 1)
predict = np.poly1d(linear_fit_model)
x_lin_reg = np.linspace(1,2999,2999)
y_lin_reg = predict(x_lin_reg)
linear_fit_slope = np.round(linear_fit_model[0],10)
print ('The slope is: ', str(linear_fit_slope), 'px^2 / sec')
print ('Calulated D = slope / 2*n = ', str(linear_fit_slope/4), 'px^2 / sec')







number_spots_per_cell = 50

t=np.arange(1,3000,1)
msd_trajectories = np.zeros((100,50,3000-1))
rmsd_trajectories = np.zeros((100,50,3000-1))
for j in range(0,100):
    for i in range(0,50):
      # extracting the particle positions from the dataframe.
      particle_trajectory = multiplexing_df[['y','x']] [multiplexing_df['particle'] == i] [multiplexing_df['cell_number'] == j].values
      msd_trajectories[j,i,:], rmsd_trajectories[j,i,:] = compute_msd(particle_trajectory)


# MSD Statistics (mu, sigma) for all trajectories.
msd_trajectories_all_mu = np.mean(np.mean(msd_trajectories,axis=1),axis=0)
#msd_trajectories_all_mu = np.mean(np.squeeze(msd_trajectories_all_mu),axis=1)
#msd_trajectories_all_sem = np.std(msd_trajectories,axis=1) /np.sqrt(number_spots_per_cell)



number_dimenssions = 2
k_diff_calulated = msd_trajectories_all_mu/(2*number_dimenssions*t)


#@title ####Plotting the MSD vs Time
downsampling = 20


# using a linear fit to calculate the slope and D.
linear_fit_model = np.polyfit(x = t, y = msd_trajectories_all_mu, deg = 1)
predict = np.poly1d(linear_fit_model)
x_lin_reg = t
y_lin_reg = predict(x_lin_reg)
linear_fit_slope = np.round(linear_fit_model[0],3)
print ('The slope is: ', str(linear_fit_slope), 'px^2 / sec')
print ('Calulated D = slope / 2*n = ', str(linear_fit_slope/4), 'px^2 / sec')

fig, ax = plt.subplots(1,2, figsize=(10, 3))

for i in range(0,number_spots_per_cell):
  ax[0].plot(msd_trajectories[i,:],'-',color=[0.8,0.8,0.8])
#ax[0].errorbar(t[::downsampling], msd_trajectories_all_mu[::downsampling],  yerr=msd_trajectories_all_sem[::downsampling],ecolor='dodgerblue',linestyle='')
ax[0].plot(t[::downsampling], msd_trajectories_all_mu[::downsampling], marker='o', markersize=5, linestyle='-',color='dodgerblue',label ='mean' )
ax[0].set_title('Mean square displacement')
ax[0].set_ylabel('MSD ($px^2$)')
ax[0].set_xlabel('time (s)')
ax[0].set_ylim((0,10000))
ax[0].plot(x_lin_reg, y_lin_reg, c = 'orangered',lw =1,label ='Calulated D = '+str(linear_fit_slope/4))
ax[0].legend()

ax[1].plot(t[::downsampling], k_diff_calulated[::downsampling], marker='o', markersize=5, linestyle='-',color='dodgerblue',label ='mean' )
ax[1].legend()
ax[1].set_title('Diffusion coefficient')
ax[1].set_ylabel('D ($px^{2}$/sec)')
ax[1].set_xlabel('time (s)')

plt.subplots_adjust(wspace=0.3, hspace=0)

plt.show()


D_pixels = float(str(linear_fit_slope/4))

pixel_um = 130*0.001
D_um = D_pixels *pixel_um**2




