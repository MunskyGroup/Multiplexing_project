# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 11:06:06 2021

@author: willi
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


test_file = 'D:/multiplexing_ML/par_sweep_5000/par_sweep_5000/ki_ke_sweep_5000spots_0.08_10.88888888888889.csv'

def df_to_array(dataframe_simulated_cell, total_traj=5000, spot_nums=50):
  
    labels2 = np.zeros([total_traj,1])
    label = 1
    for i in range(total_traj):
        if i%spot_nums == 0:
            label = (label+1)%2
        labels2[i] = label
      
    # get the total number of particles in all cells
    total_particles = 0
    for cell in set(dataframe_simulated_cell['cell_number']):
        total_particles += len(set(dataframe_simulated_cell[dataframe_simulated_cell['cell_number'] == 0]['particle'] ))
      
    #preallocate numpy array sof n_particles by nframes
    I_g = np.zeros([total_particles, np.max(dataframe_simulated_cell['frame'])+1] )  #intensity green
    I_g_std = np.zeros([total_particles, np.max(dataframe_simulated_cell['frame'])+1] ) #intensity green std
    x_loc = np.zeros([total_particles, np.max(dataframe_simulated_cell['frame'])+1] ) #x loc
    y_loc = np.zeros([total_particles, np.max(dataframe_simulated_cell['frame'])+1] ) #y_loc
    I_r_std   = np.zeros([total_particles, (np.max(dataframe_simulated_cell['frame'])+1)] ) #intensity red
    I_r = np.zeros([total_particles, (np.max(dataframe_simulated_cell['frame'])+1) ] ) #intensity red std
    labels = np.zeros([total_particles])
    label_list = list(set(np.unique(dataframe_simulated_cell['Classification'])))
    k = 0
    
    for cell in set(dataframe_simulated_cell['cell_number']):  #for every cell 
        for particle in set(dataframe_simulated_cell[dataframe_simulated_cell['cell_number'] == 0]['particle'] ): #for every particle
            tmpdf = dataframe_simulated_cell[(dataframe_simulated_cell['cell_number'] == cell) & (dataframe_simulated_cell['particle'] == particle)]  #slice the dataframe
            maxframe = np.max(tmpdf['frame'])
            minframe = np.min(tmpdf['frame'])
            I_g[k, 0:(maxframe+1-minframe)] = tmpdf['green_int_mean']  #fill the arrays to return out
            x_loc[k, 0:(maxframe+1-minframe)] = tmpdf['x']
            y_loc[k, 0:(maxframe+1-minframe)] = tmpdf['y']
            I_g_std[k, 0:(maxframe+1-minframe)] = tmpdf['green_int_std']
            #I_r[k, 0:(maxframe+1-minframe)] = tmpdf['red_int_mean']
            #I_r_std[k, 0:(maxframe+1-minframe)] = tmpdf['red_int_std']
            labels[k] = label_list.index(list(set(np.unique(tmpdf['Classification'])))[0])
              
            k+=1 #iterate over k (total particles)
    return I_g, I_g_std, labels, x_loc,y_loc, labels2 #I_r, #I_r_std,    #return everything backout


st = time.time()

I_g, I_g_std, labels, x_loc,y_loc, labels2 = df_to_array(pd.read_csv(test_file))


print(time.time()-st)




st = time.time()
data_df = pd.read_csv(test_file)
int_g_fast = data_df['green_int_mean'].values.reshape([5000,3000])

labels_fast = data_df['Classification'].values.reshape([5000,3000])[:,0]

print(time.time()-st)

print(np.sum(int_g_fast - I_g))

print(np.sum(labels - labels_fast))


