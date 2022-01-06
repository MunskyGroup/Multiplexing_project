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
# run transfer learning specifying a set model and a dataset
# 
# 
#####################################################################

names1 = ['GOLT1A','CDC42','MYOZ3','GPN2','CD46','LRRC42','SPATA6','SCP2','BBS5','CAMK2B']
names2 = ['RRAGC','ORC2','LONRF2', 'EDEM3','TRIM33','MAP3K6','COL3A1','KDM6B','PHIP','DOCK8' ]
meta_data = {}

data_dir = 'construct_length_dataset'
save_dir = 'construct_L_ML_smaller'

if not os.path.exists(os.path.join('.', data_dir)):
    os.makedirs(os.path.join('.', data_dir))
    
if not os.path.exists(os.path.join('.', save_dir)):
    os.makedirs(os.path.join('.', save_dir))


pairs_already_used = []
mi,mj = 5,5 #model indexes, the one to use as the base model.
n = 10
datafile1 = 'construct_lengths_' + names1[mi] + '_' + names1[mi]+'.csv'
datafile2 = 'construct_lengths_' + names1[mj] + '_' + names1[mj]+'.csv'
subprocess.run(['python', 'run_cnn_commandline_2file_transferlearn.py',
                '--data_dir=%s'%data_dir,
                '--mi=%s'%str(mi),
                '--mj=%s'%str(mj),
                '--data_file1=%s'% os.path.join('.',data_dir,datafile1),
                '--data_file2=%s'% os.path.join('.',data_dir,datafile2),
                '--save_model=%s'%str(1),
                '--save_dir=%s'%os.path.join('.',save_dir),
                '--acc_file=%s'%'acc_mat_cl_smaller.npy',
                '--model_file=%s'%'unused',
                '--model_name=%s'%'cnn_cl_smaller',
                '--verbose=%s'%str(1)
                ])


