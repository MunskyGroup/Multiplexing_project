# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 12:28:48 2021

@author: willi
"""
import numpy as np
import subprocess
import os
import tqdm
import datetime
import time
import yaml
import pandas as pd
import argparse


meta_data = {}

# metadata and naming

data_root = '.'
model_base_name = 'cnn_par'
start_time = time.time()
start_iso = datetime.datetime.fromtimestamp(time.time()).isoformat()

# global data directories
dataset_dir = 'datasets'
CL_data_dir = 'construct_length_dataset_larger_range'
img_data_dir = 'P300_KDM5B_3000s_base_pb'
ke_ki_data_dir = 'par_sweep_5000'
ke_data_dir = 'par_sweep_kes'
ki_data_dir = 'par_sweep_kis'

# global save directories
CL_save_dir = 'parsweep_cl_ML'
img_save_dir = 'parsweep_pb_ML'
ke_ki_save_dir = 'parsweep_keki_ML'
ke_save_dir = 'parsweep_kes_ML'
ki_save_dir = 'parsweep_kis_ML'
pb_save_dir = 'parsweep_pb_ML'

parser = argparse.ArgumentParser(description='get paths and save dir and acc_file_name')

parser.add_argument('--base_dir', dest='base_dir', type=str,)
parser.add_argument('--base_command', dest='base_command', type=str,)
parser.add_argument('--global_samples', dest='global_samples', type=int,)
parser.add_argument('--test_size', dest='test_size', type=float,)
parser.add_argument('--witheld', dest='witheld', type=int,)
parser.add_argument('--test_type', dest='test_type', type=str,)

args = parser.parse_args()

base_command = args.base_command
base_dir = args.base_dir
retrain = 1
global_samples = args.global_samples
witheld = args.witheld
test_size = args.test_size
test_type = args.test_type


meta_data['base_dir'] = base_dir
meta_data['timestamp'] = start_iso
meta_data['base_model_name'] = model_base_name
meta_data['base_command'] = base_command
meta_data['retrain'] = 1
meta_data['samples']=global_samples


global_n = 10

#          cl img  keki  kes kis
runwho =  [0,  1,   0,   0,   0]


print(os.getcwd())
##############################################################################
# run the ML cnn training for P300 and Kdm5b at different imaging conditions
# store the accuracy within acc_mat_img.npy
# data_dir = ./parsweep5000_ML
##############################################################################





pbs = [0,1,2,3,4,5,6,7,8,9,10]
test_types = ['base','w_correction','wo_correction']

if not os.path.exists(os.path.join('.',base_dir, pb_save_dir)):
    os.makedirs(os.path.join('.', base_dir,pb_save_dir))

pairs_already_used = []

n = 11


if test_type == 'base':
    with tqdm.tqdm(len(pbs)) as pbar:
        for i in range(0,1):
            for j in range(0,len(pbs)):
                
                datafile1 = 'p300_base_pb_P300_P300_0.06_5.33333_%i.csv'%j
                datafile2 = 'kdm5b_base_pb_KDM5B_KDM5B_0.06_5.33333_%i.csv'%j
    
    
    
    
                subprocess.run(['python', base_command,
                                '--i=%s'%str(i),
                                '--j=%s'%str(j),
                                '--data_file1=%s'% os.path.join('.',dataset_dir,img_data_dir,datafile1),
                                '--data_file2=%s'% os.path.join('.',dataset_dir,img_data_dir,datafile2),
                                '--save_model=%s'%str(1),
                                '--save_dir=%s'%os.path.join('.',base_dir,pb_save_dir),
                                '--acc_file=%s'%'acc_mat_pb.npy',
                                '--retrain=%s'%str(retrain),
                                '--model_file=%s'%'unused',
                                '--model_name=%s'%(model_base_name + '_pb'),
                                '--verbose=%s'%str(0),
                                '--Fr=%s'%str(5),
                                '--NFr=%s'%str(60),
                                
                                '--two_files=%s'%(str(1)),
                                '--witheld=%s'%(str(witheld)),
                                '--ntimes=350',
                                '--test_size=%s'%(str(test_size)),
                                '--Nsamples=%s'%(str(2500)),
                                '--ntraj=%s'%(str(1250)),    
                                ])
                
    
                pbar.update(1)
      
        
        fr_key = []
        acc_mat_file = np.load( os.path.join('.', base_dir,pb_save_dir,'acc_mat_pb.npy' )  )
        for i in range(0,n):
            sub_key = []
            for j in range(0,n):
                sub_key.append((i,j, 0, pbs[j], acc_mat_file[i,j] ) )
            fr_key.append(sub_key)
        
        key_csv = pd.DataFrame(data=fr_key, index=[0], columns=pbs)
        key_csv.to_csv(os.path.join('.',base_dir, pb_save_dir, 'pb_key.csv'))

    
if test_type == 'wo_correction':
    with tqdm.tqdm(len(pbs)) as pbar:
        for i in range(0,1):
            for j in range(0,len(pbs)):
                
                datafile1 = 'p300_base_pb_P300_P300_0.06_5.33333_%i_tracking_wo_correction_intensities.npy'%j
                datafile2 = 'kdm5b_base_pb_KDM5B_KDM5B_0.06_5.33333_%i_tracking_wo_correction_intensities.npy'%j
    
    
    
    
                subprocess.run(['python', base_command,
                                '--i=%s'%str(i),
                                '--j=%s'%str(j),
                                '--data_file1=%s'% os.path.join('.',dataset_dir,img_data_dir,datafile1),
                                '--data_file2=%s'% os.path.join('.',dataset_dir,img_data_dir,datafile2),
                                '--save_model=%s'%str(1),
                                '--save_dir=%s'%os.path.join('.',base_dir,pb_save_dir),
                                '--acc_file=%s'%'acc_mat_pb_wo.npy',
                                '--retrain=%s'%str(retrain),
                                '--model_file=%s'%'unused',
                                '--model_name=%s'%(model_base_name + '_pb_wo'),
                                '--verbose=%s'%str(0),
                                '--Fr=%s'%str(5),
                                '--NFr=%s'%str(60),
                                '--Nsamples=%s'%str(global_samples),
                                '--two_files=%s'%(str(1)),
                                '--witheld=%s'%(str(witheld)),
                                '--ntimes=300',
                                '--test_size=%s'%(str(test_size)),
                                '--Nsamples=%s'%(str(2500)),
                                '--ntraj=%s'%(str(1250)),
    
                                ])
                
    
                pbar.update(1)
      
        
        fr_key = []
        acc_mat_file = np.load( os.path.join('.', base_dir,pb_save_dir,'acc_mat_pb_wo.npy' )  )
        for i in range(0,n):
            sub_key = []
            for j in range(0,n):
                sub_key.append((i,j, 0, pbs[j], acc_mat_file[i,j] ) )
            fr_key.append(sub_key)
        
        key_csv = pd.DataFrame(data=fr_key, index=[0], columns=pbs)
        key_csv.to_csv(os.path.join('.',base_dir, pb_save_dir, 'pb_wo_key.csv'))    

