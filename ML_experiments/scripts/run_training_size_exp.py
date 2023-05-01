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
CL_data_dir = 'construct_length_dataset_larger_range_14scale'
img_data_dir = 'par_sweep_kis'
ke_ki_data_dir = 'par_sweep_training_size'
ke_data_dir = 'par_sweep_kes'
ki_data_dir = 'par_sweep_kis'

# global save directories
CL_save_dir = 'parsweep_cl_ML'
img_save_dir = 'parsweep_img_ML'
ke_ki_save_dir = 'parsweep_keki_ML'
ke_save_dir = 'parsweep_kes_ML'
ki_save_dir = 'parsweep_kis_ML'




parser = argparse.ArgumentParser(description='get paths and save dir and acc_file_name')
parser.add_argument('--global_fr', dest='global_fr', type=int,)
parser.add_argument('--global_nf', dest='global_nf', type=int,)
parser.add_argument('--base_dir', dest='base_dir', type=str,)
parser.add_argument('--global_samples', dest='global_samples', type=int, default=5000)
parser.add_argument('--test_size', dest='test_size', type=float,)
parser.add_argument('--witheld', dest='witheld', type=int,)
parser.add_argument('--base_command', dest='base_command', type=str,)

parser.add_argument('--cl', dest='cl', type=int,)
parser.add_argument('--img', dest='img', type=int,)
parser.add_argument('--keki', dest='keki', type=int,)
parser.add_argument('--kes', dest='kes', type=int,)
parser.add_argument('--sizes', dest='sizes', type=str,)
parser.add_argument('--kis', dest='kis', type=int,)
parser.add_argument('--debug', dest='debug', type=int, default=0)


args = parser.parse_args()

base_command = args.base_command
base_dir = args.base_dir
global_fr = args.global_fr
global_nf = args.global_nf
retrain = 1
global_samples = 5000
witheld = 1000
test_size = 0
debug = args.debug

cl = args.cl
img = args.img
keki = args.keki
kes = args.kes
kis = args.kis


meta_data['global_framerate'] = global_fr
meta_data['global_N_frames'] = global_nf
meta_data['base_dir'] = base_dir
meta_data['timestamp'] = start_iso
meta_data['base_model_name'] = model_base_name
meta_data['base_command'] = base_command
meta_data['retrain'] = 1
meta_data['samples']=global_samples


global_n = 10


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


if not os.path.exists(os.path.join(data_root, base_dir)):
    os.makedirs(os.path.join(data_root, base_dir))


#####################################################################
#####################################################################


datafile1 = 'parsweep_kis_kdm5b_' +'0.06000000000000001' +'.csv'
datafile2 = 'parsweep_kis_p300_' + '0.06000000000000001' +'.csv'

FR = 5
Nframes = 64
sizes = [int(x) for x in args.sizes[1:-1].split(',')]
ki = '.06'
kes = '5.33'

    
if not os.path.exists(os.path.join(data_root,base_dir, img_save_dir)):
    os.makedirs(os.path.join(data_root,base_dir, img_save_dir))


tmp = np.zeros([1,3000])

n = global_n

valid_mat = np.zeros([10,10])
FR_mat = np.zeros([10,10])
NF_mat = np.zeros([10,10])
time_mat = np.zeros([10,10])
with tqdm.tqdm(180) as pbar:
    for i in range(0,len(sizes)):
        subfold = int(min(4000/sizes[i],5))
        for j in range(0,subfold):
             

                
            subprocess.run(['python', base_command,
                            '--i=%s'%str(i),
                            '--j=%s'%str(j),
                            '--data_file1=%s'% os.path.join(data_root,dataset_dir,img_data_dir,datafile1),
                            '--data_file2=%s'% os.path.join(data_root,dataset_dir,img_data_dir,datafile2),
                            '--save_model=%s'%str(1),
                            '--save_dir=%s'%os.path.join(data_root,base_dir,img_save_dir),
                            '--acc_file=%s'%'acc_mat_training.npy',
                            '--retrain=%s'%str(retrain),
                            '--model_file=%s'%'unused',
                            '--model_name=%s'%(model_base_name + '_training'),
                            '--verbose=%s'%str(0),
                            '--Fr=%s'%str(FR),
                            '--NFr=%s'%str(Nframes),
                            '--Nsamples=%s'%str(global_samples),
                            '--two_files=%s'%(str(1)),
                            '--witheld=%s'%(str(witheld)),
                            '--test_size=%s'%(str(test_size)),
                            '--data_size=%s'%str(sizes[i]),
                            '--subfold=%s'%str(j),
                            '--debug=%s'%(str(debug)),
                            ])
            pbar.update(1)
                
acc_mat_file = np.load( os.path.join(data_root, base_dir,img_save_dir,'acc_mat_training.npy' )  )
fr_key = []
for i in range(0,len(sizes)):
    sub_key = []
    subfold = int(min(4000/sizes[i],5))
    for j in range(0,subfold):
        sub_key.append((i,j, sizes[i], j, acc_mat_file[i,j]))
    fr_key.append(sub_key)

key_csv = pd.DataFrame(data=fr_key, index=sizes, columns=[0,1,2,3,4])
key_csv.to_csv(os.path.join(data_root,base_dir, img_save_dir, 'training_key.csv'))


meta_data['runtime'] = time.time() - start_time

with open(os.path.join(data_root,base_dir, 'metadata.yaml'), 'w') as outfile:
    yaml.dump(meta_data, outfile)
