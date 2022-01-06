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
# run the ML cnn training for gene at ki1 and ki2
# store the accuracy within acc_mat_par.npy
# data_dir = ./parsweep5000_ML
#####################################################################

meta_data = {}

kis = ['0.1','0.09000000000000001','0.08','0.07','0.06000000000000001','0.05000000000000001',
       '0.04000000000000001','0.030000000000000006','0.020000000000000004','0.01']
ke = '5.33'
FR = 1
Nframes = 3000


data_dir = 'par_sweep_kis'
save_dir = 'parsweep_kis_ML'

if not os.path.exists(os.path.join('.', data_dir)):
    os.makedirs(os.path.join('.', data_dir))
    
if not os.path.exists(os.path.join('.', save_dir)):
    os.makedirs(os.path.join('.', save_dir))

pairs_already_used = []

n = 10
with tqdm.tqdm(n**2) as pbar:
    for i in range(0,n):
        for j in range(0,n):
            datafile1 = 'parsweep_kis_kdm5b_' + kis[i] +'.csv'
            datafile2 = 'parsweep_kis_p300_' + kis[j] +'.csv'
            
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
                            '--Fr=%s'%str(FR),
                            '--NFr=%s'%str(Nframes)
                            ])
            

            pbar.update(1)
