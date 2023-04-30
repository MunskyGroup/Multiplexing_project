# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 11:24:42 2023

@author: willi
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
cur_dir = os.getcwd()
os.chdir('..')
from multiplex_core import multiplexing_core 
os.chdir(cur_dir)
from skimage.io import imread


##############################################################################
# This code matches particles from the tracked particles photobleaching
# and links the particles together and recovers the spots that
# last longer than 300s and match the real simulation particles to a given error
##############################################################################


mc = multiplexing_core()
ntimes = 350
ntraj = 1250
seed = 42
Nsamples = 50
Fr = 5
Nframes = 128
ncells = 25
nspots=50

target_dir = './P300_KDM5B_350s_base_pb2'
save = False
def quantile_norm(movie, q ): # normalize a numpy array of a movie
    norm_movie = np.zeros(movie.shape)
    for i in range(movie.shape[-1]):
        max_val = np.quantile(movie[:,:,:,i], q)
        min_val = np.quantile(movie[:,:,:,i], .005)
        norm_movie[:,:,:,i] = (movie[:,:,:,i] - min_val)/(max_val - min_val)
        norm_movie[norm_movie > 1] = 1
        norm_movie[norm_movie < 0] = 0
    return norm_movie


# TRACKING MATCHING FOR KDM5B
# no photobleach correction
for i in range(11): # for each photobleaching rate

    # KDM5B w_correction
    
    df_real ='%s/kdm5b_base_pb_KDM5B_KDM5B_0.06_5.33333_%i.csv'%(target_dir,i)
    df_correction = '%s/kdm5b_base_pb_KDM5B_KDM5B_0.06_5.33333_%i.csv_tracking_w_correction'%(target_dir,i)
    df_no_correction = '%s/kdm5b_base_pb_KDM5B_KDM5B_0.06_5.33333_%i.csv_tracking_wo_correction'%(target_dir,i)

    multiplexing_df1 = pd.read_csv(df_real)
    m_df1_wo_correction = pd.read_csv(df_no_correction)
    m_df1_w_correction = pd.read_csv(df_correction)
    
    m_df1_wo_correction['include'] = [0,]*m_df1_wo_correction.shape[0]   
    m_df1_wo_correction['real'] = [0,]*m_df1_wo_correction.shape[0]   
    m_df1_w_correction['include'] = [0,]*m_df1_w_correction.shape[0]   
    m_df1_w_correction['real'] = [0,]*m_df1_w_correction.shape[0]   
    
    
    int_g = multiplexing_df1['green_int_mean'].values.reshape([ntraj,ntimes])    
    
    t = np.linspace(0,len(int_g) - 1,len(int_g))  #time vector in seconds
    labels = multiplexing_df1['Classification'].values.reshape([ntraj,ntimes])[:,0]
    

    
    xys_df1_wc = []
    
    best_rsmes = []
    
    includes = []
    reals = []
    
    int_g_wo_correction = []
    
    for cell_number in range(ncells): #for every cell
        # get the tracked particles trajectories
        cell_traj = m_df1_wo_correction[m_df1_wo_correction['cell_number'] == cell_number]
        real_x = multiplexing_df1[multiplexing_df1['cell_number'] == cell_number]['x'] # get the real particle x's
        real_y = multiplexing_df1[multiplexing_df1['cell_number'] == cell_number]['y'] # get the real particle y's
        real_xy = np.array([real_x.values, real_y.values])
        real_xy = real_xy.reshape(2,nspots,ntimes) # xy, cell, frame
        
        for n_particle in range(np.max(cell_traj['particle'])): # for each tracked particle
            particle_traj = cell_traj[cell_traj['particle'] == n_particle]
            min_rsme = 1e6
            valid_frames = particle_traj['frame'] # get how long this particle was tracked
    
            if len(valid_frames) > 0: # if it was tracked for more than one frame
                tracked_x =  particle_traj['x'] 
                tracked_y =  particle_traj['y']
                tracked_xy = np.expand_dims(np.array([tracked_x, tracked_y]),-1)
                matched_real = np.moveaxis(real_xy[:,:,valid_frames], 1,2)
                xydiff = (matched_real - tracked_xy) 
                lenxydiff = xydiff.shape[1]
                rmse = np.sum(np.sum(xydiff**2, axis=0), axis=0)/lenxydiff #calculate MSE to real spots
                potential_match = np.argmin(rmse) # get which particle matches the best
                include = 0
                real = 0
                starting_frame = valid_frames.values[0] # what frame did tracking start
                if rmse[potential_match] < 3: #if the matching error is below 3 its a real particle
                    real = 1
                if real:
                    if len(valid_frames) >= 300: # if it was tracked for 300 seconds keep this spot
                        include = 1
                        int_g_wo_correction.append(particle_traj['green_int_mean'].values[:300])
                 
                # matching array: cell, N particle, matched particle, length of trajectory, error, real, include       
                best_rsmes.append([cell_number,n_particle,potential_match, len(valid_frames), rmse[potential_match], include, real, starting_frame])
                  

            
    int_g_wo_correction = np.array(int_g_wo_correction)
    matching_array = np.array(best_rsmes)  
    if save:
        np.save('./%s/kdm5b_base_pb_KDM5B_KDM5B_0.06_5.33333_%i_tracking_wo_correction_intensities.npy'%(target_dir,i), int_g_wo_correction)
        np.save('./%s/kdm5b_base_pb_KDM5B_KDM5B_0.06_5.33333_%i_tracking_wo_correction_matching.npy'%(target_dir,i), matching_array)
    
      
    
    xys_df1_wc = []
    
    best_rsmes = []
    
    includes = []
    reals = []
    
    int_g_w_correction = []
    
    # w/ photobleach correction
    for cell_number in range(ncells):
        cell_traj = m_df1_w_correction[m_df1_w_correction['cell_number'] == cell_number]
        
        real_x = multiplexing_df1[multiplexing_df1['cell_number'] == cell_number]['x']
        real_y = multiplexing_df1[multiplexing_df1['cell_number'] == cell_number]['y']
        real_xy = np.array([real_x.values, real_y.values])
        real_xy = real_xy.reshape(2,nspots,ntimes) # xy, cell, frame
        
        for n_particle in range(np.max(cell_traj['particle'])):
            particle_traj = cell_traj[cell_traj['particle'] == n_particle]

            
            min_rsme = 1e6
            
            valid_frames = particle_traj['frame']
    
            if len(valid_frames) > 0:
                tracked_x =  particle_traj['x']
                tracked_y =  particle_traj['y']
                tracked_xy = np.expand_dims(np.array([tracked_x, tracked_y]),-1)
                matched_real = np.moveaxis(real_xy[:,:,valid_frames], 1,2)
                xydiff = (matched_real - tracked_xy)
                lenxydiff = xydiff.shape[1]
                rmse = np.sum(np.sum(xydiff**2, axis=0), axis=0)/lenxydiff
                potential_match = np.argmin(rmse)
                include = 0
                real = 0
                if rmse[potential_match] < 3:
                    real = 1
                if real:
                    if len(valid_frames) >= 300:
                        include = 1
                        int_g_w_correction.append(particle_traj['green_int_mean'].values[:300])
                starting_frame = valid_frames.values[0]
                best_rsmes.append([cell_number,n_particle,potential_match, len(valid_frames), rmse[potential_match], include, real, starting_frame])

            
    int_g_w_correction = np.array(int_g_w_correction)
    matching_array = np.array(best_rsmes)  
    
    if save:
        np.save('./%s/kdm5b_base_pb_KDM5B_KDM5B_0.06_5.33333_%i_tracking_w_correction_intensities.npy'%(target_dir,i), int_g_w_correction)
        np.save('./%s/kdm5b_base_pb_KDM5B_KDM5B_0.06_5.33333_%i_tracking_w_correction_matching.npy'%(target_dir,i), matching_array)
    

# TRACKING MATCHING FOR P300
# w/o photobleach correction
for i in range(11):
    
    
    df_real ='%s/p300_base_pb_P300_P300_0.06_5.33333_%i.csv'%(target_dir,i)
    df_correction = '%s/p300_base_pb_P300_P300_0.06_5.33333_%i.csv_tracking_w_correction'%(target_dir,i)
    df_no_correction = '%s/p300_base_pb_P300_P300_0.06_5.33333_%i.csv_tracking_wo_correction'%(target_dir,i)

    multiplexing_df1 = pd.read_csv(df_real)
    m_df1_wo_correction = pd.read_csv(df_no_correction)
    m_df1_w_correction = pd.read_csv(df_correction)
    
    m_df1_wo_correction['include'] = [0,]*m_df1_wo_correction.shape[0]   
    m_df1_wo_correction['real'] = [0,]*m_df1_wo_correction.shape[0]   
    m_df1_w_correction['include'] = [0,]*m_df1_w_correction.shape[0]   
    m_df1_w_correction['real'] = [0,]*m_df1_w_correction.shape[0]   
    
    
    int_g = multiplexing_df1['green_int_mean'].values.reshape([ntraj,ntimes])    
  
    t = np.linspace(0,len(int_g) - 1,len(int_g))  
    labels = multiplexing_df1['Classification'].values.reshape([ntraj,ntimes])[:,0]
    
    
    xys_df1_wc = []
    
    best_rsmes = []
    
    includes = []
    reals = []
    
    int_g_wo_correction = []
    
    for cell_number in range(ncells):
        cell_traj = m_df1_wo_correction[m_df1_wo_correction['cell_number'] == cell_number]
        
        real_x = multiplexing_df1[multiplexing_df1['cell_number'] == cell_number]['x']
        real_y = multiplexing_df1[multiplexing_df1['cell_number'] == cell_number]['y']
        real_xy = np.array([real_x.values, real_y.values])
        real_xy = real_xy.reshape(2,nspots,ntimes) # xy, cell, frame
        
        for n_particle in range(np.max(cell_traj['particle'])):
            particle_traj = cell_traj[cell_traj['particle'] == n_particle]
            #particle_traj = particle_traj.iloc[:int(len(particle_traj)/2),:] #data is accidently doubled, delete half
            
            min_rsme = 1e6
            
            valid_frames = particle_traj['frame']
    
            if len(valid_frames) > 0:
                tracked_x =  particle_traj['x']
                tracked_y =  particle_traj['y']
                tracked_xy = np.expand_dims(np.array([tracked_x, tracked_y]),-1)
                matched_real = np.moveaxis(real_xy[:,:,valid_frames], 1,2)
                xydiff = (matched_real - tracked_xy)
                lenxydiff = xydiff.shape[1]
                rmse = np.sum(np.sum(xydiff**2, axis=0), axis=0)/lenxydiff
                potential_match = np.argmin(rmse)
                include = 0
                real = 0
                if rmse[potential_match] < 3:
                    real = 1
                if real:
                    if len(valid_frames) >= 300:
                        include = 1
                        int_g_wo_correction.append(particle_traj['green_int_mean'].values[:300])
                starting_frame = valid_frames.values[0]       
                best_rsmes.append([cell_number,n_particle,potential_match, len(valid_frames), rmse[potential_match], include, real, starting_frame])
    
    int_g_wo_correction = np.array(int_g_wo_correction)
    matching_array = np.array(best_rsmes)  
    if save:
        np.save('./%s/p300_base_pb_P300_P300_0.06_5.33333_%i_tracking_wo_correction_intensities.npy'%(target_dir,i), int_g_wo_correction)
        np.save('./%s/p300_base_pb_P300_P300_0.06_5.33333_%i_tracking_wo_correction_matching.npy'%(target_dir,i), matching_array)
    


    
    xys_df1_wc = []
    
    best_rsmes = []
    
    includes = []
    reals = []
    
    int_g_w_correction = []
    
    # w/ photobleach correction
    for cell_number in range(ncells):
        cell_traj = m_df1_w_correction[m_df1_w_correction['cell_number'] == cell_number]
        
        real_x = multiplexing_df1[multiplexing_df1['cell_number'] == cell_number]['x']
        real_y = multiplexing_df1[multiplexing_df1['cell_number'] == cell_number]['y']
        real_xy = np.array([real_x.values, real_y.values])
        real_xy = real_xy.reshape(2,nspots,ntimes) # xy, cell, frame
        
        for n_particle in range(np.max(cell_traj['particle'])):
            particle_traj = cell_traj[cell_traj['particle'] == n_particle]

            
            min_rsme = 1e6
            
            valid_frames = particle_traj['frame']
    
            if len(valid_frames) > 0:
                tracked_x =  particle_traj['x']
                tracked_y =  particle_traj['y']
                tracked_xy = np.expand_dims(np.array([tracked_x, tracked_y]),-1)
                matched_real = np.moveaxis(real_xy[:,:,valid_frames], 1,2)
                xydiff = (matched_real - tracked_xy)
                lenxydiff = xydiff.shape[1]
                rmse = np.sum(np.sum(xydiff**2, axis=0), axis=0)/lenxydiff
                potential_match = np.argmin(rmse)
                include = 0
                real = 0
                if rmse[potential_match] < 3:
                    real = 1
                if real:
                    if len(valid_frames) >= 300:
                        include = 1
                        int_g_w_correction.append(particle_traj['green_int_mean'].values[:300])
                starting_frame = valid_frames.values[0]        
                best_rsmes.append([cell_number,n_particle,potential_match, len(valid_frames), rmse[potential_match], include, real, starting_frame])

            
    int_g_w_correction = np.array(int_g_w_correction)
    matching_array = np.array(best_rsmes)  
    if save:
        np.save('./%s/p300_base_pb_P300_P300_0.06_5.33333_%i_tracking_w_correction_intensities.npy'%(target_dir,i), int_g_w_correction)
        np.save('./%s/p300_base_pb_P300_P300_0.06_5.33333_%i_tracking_w_correction_matching.npy'%(target_dir,i), matching_array)
    



