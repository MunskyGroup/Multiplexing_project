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

data_root = '..'
model_base_name = 'cnn_par'
start_time = time.time()
start_iso = datetime.datetime.fromtimestamp(time.time()).isoformat()

# global data directories
dataset_dir = 'datasets'
CL_data_dir = 'construct_length_dataset_larger_range'
img_data_dir = 'P300_KDM5B_24000s_Same_intensity_gaussian_14scale'
ke_ki_data_dir = 'par_sweep_5000'
ke_data_dir = 'par_sweep_kes'
ki_data_dir = 'par_sweep_kis'

# global save directories
CL_save_dir = 'parsweep_cl_ML'
img_save_dir = 'parsweep_img_ML'
ke_ki_save_dir = 'parsweep_keki_ML'
ke_save_dir = 'parsweep_kes_ML'
ki_save_dir = 'parsweep_kis_ML'

parser = argparse.ArgumentParser(description='get paths and save dir and acc_file_name')

parser.add_argument('--base_dir', dest='base_dir', type=str,)
parser.add_argument('--base_command', dest='base_command', type=str,)
parser.add_argument('--global_samples', dest='global_samples', type=int,)
parser.add_argument('--test_size', dest='test_size', type=float,)
parser.add_argument('--witheld', dest='witheld', type=int,)

args = parser.parse_args()

base_command = args.base_command
base_dir = args.base_dir
retrain = 1
global_samples = args.global_samples
witheld = args.witheld
test_size = args.test_size


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



    
FRs = [1,5,10,20,30,45,60,120,180,240]
#total_time = [3000,2500,2000,1500,1000,750,500,250,200,150][::-1]
total_time = [6000,5000,4000,3000,2000,1000,750,500,250,150, ][::-1]
nframes = [10,20,30,40,50,60,70,80,90,100, 150, 250, 400, 600, 800, 1000, 1500, 2000]
nfs = []

ki = '.06'
kes = '5.33'




if not os.path.exists(os.path.join(data_root, img_data_dir)):
    x=1 ##add exception
    
if not os.path.exists(os.path.join(data_root,base_dir, img_save_dir)):
    os.makedirs(os.path.join(data_root,base_dir, img_save_dir))

pairs_already_used = []


datafile1 = 'ki_ke_sweep_same_int_P300_P300_0.009675852685050798_5.33333_0.csv'
datafile2 = 'ki_ke_sweep_same_int_KDM5B_KDM5B_0.014139262990455991_5.33333_0.csv'

tmp = np.zeros([1,12000])

n = len(FRs)*len(nframes)

acc_path =  os.path.join(data_root,base_dir,img_save_dir, 'acc_mat_img.npy')
if not os.path.exists(acc_path):
    acc_mat = np.zeros([len(FRs),len(nframes)])
    np.save(acc_path, acc_mat)
    
    
    
acc_mat = np.zeros([len(FRs),len(nframes)])
valid_mat = np.zeros([10,10])
FR_mat = np.zeros([10,10])
NF_mat = np.zeros([10,10])
time_mat = np.zeros([10,10])
with tqdm.tqdm(n**2) as pbar:
    for i in range(0,len(FRs)):
        for j in range(0,len(nframes)):
             tt = nframes[j]*FRs[i]
             print(i,j, nframes[j], FRs[i], tt)
             if tt <= 24000:
                print('ran')

                subprocess.run(['python', base_command,
                                '--i=%s'%str(i),
                                '--j=%s'%str(j),
                                '--data_file1=%s'% os.path.join(data_root,dataset_dir,img_data_dir,datafile1),
                                '--data_file2=%s'% os.path.join(data_root,dataset_dir,img_data_dir,datafile2),
                                '--save_model=%s'%str(1),
                                '--save_dir=%s'%os.path.join(data_root,base_dir,img_save_dir),
                                '--acc_file=%s'%'acc_mat_img.npy',
                                '--retrain=%s'%str(retrain),
                                '--model_file=%s'%'unused',
                                '--model_name=%s'%(model_base_name + '_img'),
                                '--verbose=%s'%str(0),
                                '--Fr=%s'%str(FRs[i]),
                                '--NFr=%s'%str(nframes[j]),
                                '--Nsamples=%s'%str(global_samples),
                                '--two_files=%s'%(str(1)),
                                '--witheld=%s'%(str(witheld)),
                                '--ntimes=%s'%(str(24000)),
                                '--test_size=%s'%(str(test_size)),
                                ])
    
                pbar.update(1)                
                
                
                
                pbar.update(1)
                
acc_mat_file = np.load( os.path.join(data_root, base_dir,img_save_dir,'acc_mat_img.npy' )  )
fr_key = []
for i in range(0,len(FRs)):
    sub_key = []
    for j in range(0,len(nframes)):
        sub_key.append((i,j, FRs[i], nframes[j], acc_mat_file[i,j]))
    fr_key.append(sub_key)

key_csv = pd.DataFrame(data=fr_key, index=FRs, columns=nframes)
key_csv.to_csv(os.path.join(data_root,base_dir, img_save_dir, 'img_key.csv'))
