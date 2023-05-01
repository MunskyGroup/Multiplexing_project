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
img_save_dir = 'parsweep_img_ML'
ke_ki_save_dir = 'parsweep_keki_ML'
ke_save_dir = 'parsweep_kes_ML'
ki_save_dir = 'parsweep_kis_ML'




parser = argparse.ArgumentParser(description='get paths and save dir and acc_file_name')

parser.add_argument('--base_dir', dest='base_dir', type=str,)
parser.add_argument('--global_samples', dest='global_samples', type=int,)
parser.add_argument('--test_size', dest='test_size', type=float,)
parser.add_argument('--witheld', dest='witheld', type=int,)
parser.add_argument('--base_command', dest='base_command', type=str,)
parser.add_argument('--datafile1', dest='datafile1', type=str,)
parser.add_argument('--datafile2', dest='datafile2', type=str,)
parser.add_argument('--test_type', dest='test_type', type=str)
parser.add_argument('--debug', dest='debug', type=int, default=0)


args = parser.parse_args()

base_command = args.base_command
base_dir = args.base_dir
retrain = 1
global_samples = args.global_samples
witheld = args.witheld
test_size = args.test_size
d1 = args.datafile1
d2 = args.datafile2

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

#####################################################################
# Setup file tree and storage names for each:
# data_root----|__par_sweep_kis
#              |__par_sweep_kes
#              |__par_sweep_5000
#              |__construct_length_dataset
#
# base_dir-----|__ML_run_1 ---|__meta.yaml
#                             |__parsweep_cl_ML   --- |__acc_mat_cl.npy
#                                                     |__acc_mat_cl.csv
#                                                     |__cnn_par*.h5
#
#                             |__parsweep_img_ML  --- |__acc_mat_img.npy
#                                                     |__acc_mat_img.csv
#                                                     |__cnn_par*.h5
#
#                             |__parsweep_ke_ki_ML ---|__acc_mat_ke_ki.npy
#                                                     |__acc_mat_ke_ki.csv
#                                                     |__cnn_par*.h5
#
#                             |__parsweep_kes_ML   ---|__acc_mat_kes.npy
#                                                     |__acc_mat_kes.csv
#                                                     |__cnn_par*.h5
#
#                             |__parsweep_kis_ML   ---|__acc_mat_kis.npy
#                                                     |__acc_mat_kis.csv
#                                                     |__cnn_par*.h5
#
#####################################################################


if not os.path.exists(os.path.join('.', base_dir)):
    os.makedirs(os.path.join('.', base_dir))


##############################################################################
# run the ML cnn training for P300 and Kdm5b at different imaging conditions
# store the accuracy within acc_mat_img.npy
# data_dir = ./parsweep5000_ML
##############################################################################


if runwho[1]:
        
    FRs = [1]
    #FRs = [30]
    #total_time = [3000,2500,2000,1500,1000,750,500,250,200,150][::-1]
    total_time = [1000][::-1]
    nframes = [5]
    #nframes=[40,70,100]
    nfs = []
    
    ki = '.06'
    kes = '5.33'
    
    
    if not os.path.exists(os.path.join('.', img_data_dir)):
        x=1 ##add exception
        
    if not os.path.exists(os.path.join('.',base_dir, img_save_dir)):
        os.makedirs(os.path.join('.',base_dir, img_save_dir))
    
    pairs_already_used = []
    
    
    datafile1 = d1
    datafile2 = d2
    
    tmp = np.zeros([1,12000])
    
    n = len(FRs)*len(nframes)

    acc_path =  os.path.join('.',base_dir,img_save_dir, 'acc_mat_img.npy')
    if not os.path.exists(acc_path):
        acc_mat = np.zeros([len(FRs),len(nframes)])
        np.save(acc_path, acc_mat)
    acc_mat = np.zeros([len(FRs),len(nframes)])
    valid_mat = np.zeros([1,1])
    FR_mat = np.zeros([1,1])
    NF_mat = np.zeros([1,1])
    time_mat = np.zeros([1,1])
    with tqdm.tqdm(n**2) as pbar:
        for i in range(0,len(FRs)):
            for j in range(0,len(nframes)):
                 tt = nframes[j]*FRs[i]
                 print(i,j, nframes[j], FRs[i], tt)
                 if tt <= 12000:
                    print('ran')
                    subprocess.run(['python', base_command,
                                    '--i=%s'%str(i),
                                    '--j=%s'%str(j),
                                    '--data_file1=%s'% os.path.join('.',dataset_dir,img_data_dir,datafile1),
                                    '--data_file2=%s'% os.path.join('.',dataset_dir,img_data_dir,datafile2),
                                    '--save_model=%s'%str(1),
                                    '--save_dir=%s'%os.path.join('.',base_dir,img_save_dir),
                                    '--acc_file=%s'%'acc_mat_img.npy',
                                    '--retrain=%s'%str(retrain),
                                    '--model_file=%s'%'unused',
                                    '--model_name=%s'%(model_base_name + '_img'),
                                    '--verbose=%s'%str(0),
                                    '--Fr=%s'%str(FRs[i]),
                                    '--NFr=%s'%str(nframes[j]),
                                    '--ntimes=%s'%(str(3000)),
                                    '--two_files=%s'%(str(1)),                                    
                                    '--witheld=%s'%(str(witheld)),
                                    '--test_size=%s'%(str(test_size)),
                                    '--test_type=%s'%(test_type),
                                    '--debug=%s'%str(debug),

                                    ])
                    
                    
                    pbar.update(1)
                    
    acc_mat_file = np.load( os.path.join('.', base_dir,img_save_dir,'acc_mat_img.npy' )  )
    fr_key = []
    for i in range(0,len(FRs)):
        sub_key = []
        for j in range(0,len(nframes)):
            sub_key.append((i,j, FRs[i], nframes[j], acc_mat_file[i,j]))
        fr_key.append(sub_key)
    
    key_csv = pd.DataFrame(data=fr_key, index=FRs, columns=nframes)
    key_csv.to_csv(os.path.join('.',base_dir, img_save_dir, 'img_key.csv'))



meta_data['runtime'] = time.time() - start_time

with open(os.path.join('.',base_dir, 'metadata.yaml'), 'w') as outfile:
    yaml.dump(meta_data, outfile)
