# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 14:02:45 2021

@author: willi
"""

import numpy as np
import subprocess
import os
import tqdm


#####################################################################
# run the ML cnn training for each construct length difference
# store the accuracy within acc_mat_cl.npy
# data_dir = ./construct_L_ML
#####################################################################

names1 = ['GOLT1A','CDC42','MYOZ3','GPN2','CD46','LRRC42','SPATA6','SCP2','BBS5','CAMK2B']
names2 = ['RRAGC','ORC2','LONRF2', 'EDEM3','TRIM33','MAP3K6','COL3A1','KDM6B','PHIP','DOCK8' ]
meta_data = {}

data_dir = 'construct_length_dataset'
save_dir = 'construct_L_ML'

if not os.path.exists(os.path.join('.', data_dir)):
    os.makedirs(os.path.join('.', data_dir))
    
if not os.path.exists(os.path.join('.', save_dir)):
    os.makedirs(os.path.join('.', save_dir))


pairs_already_used = []

n = 10
with tqdm.tqdm(n**2) as pbar:
    for i in range(0,n):
        for j in range(0,n):
            datafile1 = 'construct_lengths_' + names2[i] + '_' + names2[i]+'.csv'
            datafile2 = 'construct_lengths_' + names2[j] + '_' + names2[j]+'.csv'
            if set([names1[i], names1[j]]) not in pairs_already_used:
                subprocess.run(['python', 'run_cnn_commandline_2file.py',
                                '--i=%s'%str(i),
                                '--j=%s'%str(j),
                                '--data_file1=%s'% os.path.join('.',data_dir,datafile1),
                                '--data_file2=%s'% os.path.join('.',data_dir,datafile2),
                                '--save_model=%s'%str(1),
                                '--save_dir=%s'%os.path.join('.',save_dir),
                                '--acc_file=%s'%'acc_mat_cl_smaller.npy',
                                '--retrain=%s'%str(1),
                                '--model_file=%s'%'unused',
                                '--model_name=%s'%'cnn_cl_smaller',
                                '--verbose=%s'%str(0)
                                ])
                
                pairs_already_used.append(set([names1[i], names1[j]]) )
            pbar.update(1)
