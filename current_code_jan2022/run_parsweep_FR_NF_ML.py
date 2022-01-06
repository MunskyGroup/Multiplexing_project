# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 14:17:40 2021

@author: willi
"""

import numpy as np
import subprocess
import os
import tqdm


#####################################################################
# run the ML cnn training for gene at ke1 and ke2
# store the accuracy within acc_mat_par.npy
# data_dir = ./parsweep5000_ML
#####################################################################

meta_data = {}


FRs = [1,2,5,7,10,12,15,20,25,30]
total_time = [3000,2500,2000,1500,1000,750,500,250,200,150][::-1]
nfs = []



ki = '.06'
kes = '5.33'


data_dir = 'par_sweep_kis'
save_dir = 'parsweep_fr_ML'

if not os.path.exists(os.path.join('.', data_dir)):
    os.makedirs(os.path.join('.', data_dir))
    
if not os.path.exists(os.path.join('.', save_dir)):
    os.makedirs(os.path.join('.', save_dir))

pairs_already_used = []


datafile1 = 'parsweep_kes_kdm5b_' +'0.06000000000000001' +'.csv'
datafile2 = 'parsweep_kes_p300_' + '0.06000000000000001' +'.csv'

tmp = np.zeros([1,3000])

n = 10

valid_mat = np.zeros([10,10])
FR_mat = np.zeros([10,10])
NF_mat = np.zeros([10,10])
time_mat = np.zeros([10,10])
with tqdm.tqdm(n**2) as pbar:
    for i in range(0,n):
        for j in range(0,n):
             
            nframes = int(np.floor(total_time[j] / FRs[i]))
            available_frames = int(tmp[:,::FRs[i]].shape[1])
            
            
            
            if nframes <= available_frames:
    
                
                tmp[:,::FRs[i]][:nframes] #check if its a valid combo, if so run
                
                valid_mat[i,j] = 1
                FR_mat[i,j] = FRs[i]
                NF_mat[i,j] = nframes
                time_mat[i,j] = nframes*FRs[i]
                '''
                subprocess.run(['python', 'run_cnn_commandline_2file.py',
                                '--i=%s'%str(i),
                                '--j=%s'%str(j),
                                '--data_file1=%s'% os.path.join('.',data_dir,datafile1),
                                '--data_file2=%s'% os.path.join('.',data_dir,datafile2),
                                '--save_model=%s'%str(1),
                                '--save_dir=%s'%os.path.join('.',save_dir),
                                '--acc_file=%s'%'acc_mat_par.npy',
                                '--retrain=%s'%str(1),
                                '--model_file=%s'%'unused',
                                '--model_name=%s'%'cnn_par',
                                '--verbose=%s'%str(0),
                                '--Fr=%s'%str(FRs[i]),
                                '--NFr=%s'%str(nframes)
                                ])
                '''
                
    
                pbar.update(1)


            