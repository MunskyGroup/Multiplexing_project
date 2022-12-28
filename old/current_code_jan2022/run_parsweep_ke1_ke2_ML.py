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

ki = '.06'
kes = ['2.0','3.111111111111111','4.222222222222222','5.333333333333334',
       '6.444444444444445','7.555555555555555','8.666666666666668','9.777777777777779',
       '10.88888888888889','12.0']
FR = 1
Nframes = 3000

data_dir = 'par_sweep_kes'
save_dir = 'parsweep_kes_ML'

if not os.path.exists(os.path.join('.', data_dir)):
    os.makedirs(os.path.join('.', data_dir))
    
if not os.path.exists(os.path.join('.', save_dir)):
    os.makedirs(os.path.join('.', save_dir))

pairs_already_used = []

n = 10
with tqdm.tqdm(n**2) as pbar:
    for i in range(0,n):
        for j in range(0,n):
            datafile1 = 'parsweep_kes_kdm5b_' + kes[i] +'.csv'
            datafile2 = 'parsweep_kes_p300_' + kes[j] +'.csv'
            
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
