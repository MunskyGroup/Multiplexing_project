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

meta_data = {}

# metadata and naming
base_dir = 'ML_run1'
data_root = '.'
model_base_name = 'cnn_par'
start_time = time.time()
start_iso = datetime.datetime.fromtimestamp(time.time()).isoformat()

# global data directories
CL_data_dir = 'construct_length_dataset_larger_range'
img_data_dir = 'par_sweep_kis'
ke_ki_data_dir = 'par_sweep_5000'
ke_data_dir = 'par_sweep_kes'
ki_data_dir = 'par_sweep_kis'

# global save directories
CL_save_dir = 'parsweep_cl_ML'
img_save_dir = 'parsweep_img_ML'
ke_ki_save_dir = 'parsweep_ke_ki_ML'
ke_save_dir = 'parsweep_kes_ML'
ki_save_dir = 'parsweep_kis_ML'

# global data setup
global_fr = 10
global_nf = 200
retrain = 1

meta_data['global_framerate'] = global_fr
meta_data['global_N_frames'] = global_nf
meta_data['base_dir'] = base_dir
meta_data['timestamp'] = start_iso
meta_data['base_model_name'] = model_base_name
meta_data['base_command'] = 'run_cnn_commandline_2file.py'
meta_data['retrain'] = 1

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


#####################################################################
# run the ML cnn training for each construct length difference
# store the accuracy within acc_mat_cl.npy
# data_dir = ./construct_L_ML
#####################################################################

if runwho[0]:

    names1 = ['GOLT1A','CDC42','MYOZ3','GPN2','CD46','LRRC42','SPATA6','SCP2','BBS5','CAMK2B']
    names2 = ['RRAGC','ORC2','LONRF2', 'EDEM3','TRIM33','MAP3K6','COL3A1','KDM6B','PHIP','DOCK8' ]
    
    
    
    if not os.path.exists(os.path.join('.', CL_data_dir)):
        x=1 ##Add exception
        
    if not os.path.exists(os.path.join('.',base_dir, CL_save_dir)):
        os.makedirs(os.path.join('.',base_dir, CL_save_dir))
    
    
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
                                    '--data_file1=%s'% os.path.join('.',CL_data_dir,datafile1),
                                    '--data_file2=%s'% os.path.join('.',CL_data_dir,datafile2),
                                    '--save_model=%s'%str(1),
                                    '--save_dir=%s'%os.path.join('.',base_dir,CL_save_dir),
                                    '--acc_file=%s'%'acc_mat_cl.npy',
                                    '--retrain=%s'%str(retrain),
                                    '--model_file=%s'%'unused',
                                    '--model_name=%s'%(model_base_name + '_cl'),
                                    '--verbose=%s'%str(0),
                                    '--Fr=%s'%str(global_fr),
                                    '--NFr=%s'%str(global_nf)  
                                    ])
                    
                    pairs_already_used.append(set([names1[i], names1[j]]) )
                pbar.update(1)
                
    CL_key = []
    for i in range(0,n):
        CL_keyi = []
        for j in range(0,n):
            CL_keyi.append((i,j, names2[i], names2[j]))
        CL_key.append(CL_keyi)
    
    CL_key_csv = pd.DataFrame(data=CL_key, index=names2, columns=names2)
    CL_key_csv.to_csv(os.path.join('.',base_dir, CL_save_dir, 'cl_key.csv'))

##############################################################################
# run the ML cnn training for P300 and Kdm5b at different imaging conditions
# store the accuracy within acc_mat_img.npy
# data_dir = ./parsweep5000_ML
##############################################################################


if runwho[1]:
        
    
    FRs = np.linspace(1,global_fr,10).astype(int)
    max_time = global_fr*global_nf
    total_time = np.linspace(20,max_time,10).astype(int)

        
    nfs = []
    
    meta_data['img_frs'] = FRs
    meta_data['img_times'] = total_time
    
    ki = '.06'
    kes = '5.33'
    
    
    
    
    if not os.path.exists(os.path.join('.', img_data_dir)):
        x=1 ##add exception
        
    if not os.path.exists(os.path.join('.',base_dir, img_save_dir)):
        os.makedirs(os.path.join('.',base_dir, img_save_dir))
    
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
                    
                    subprocess.run(['python', 'run_cnn_commandline_2file.py',
                                    '--i=%s'%str(i),
                                    '--j=%s'%str(j),
                                    '--data_file1=%s'% os.path.join('.',img_data_dir,datafile1),
                                    '--data_file2=%s'% os.path.join('.',img_data_dir,datafile2),
                                    '--save_model=%s'%str(1),
                                    '--save_dir=%s'%os.path.join('.',base_dir,img_save_dir),
                                    '--acc_file=%s'%'acc_mat_img.npy',
                                    '--retrain=%s'%str(retrain),
                                    '--model_file=%s'%'unused',
                                    '--model_name=%s'%(model_base_name + '_img'),
                                    '--verbose=%s'%str(0),
                                    '--Fr=%s'%str(FRs[i]),
                                    '--NFr=%s'%str(nframes)
                                    ])
                    pbar.update(1)
    
    fr_key = []
    for i in range(0,n):
        sub_key = []
        for j in range(0,n):
            sub_key.append((i,j, FRs[i], total_time[j]))
        fr_key.append(sub_key)
    
    key_csv = pd.DataFrame(data=fr_key, index=FRs, columns=total_time)
    key_csv.to_csv(os.path.join('.',base_dir, img_save_dir, 'img_key.csv'))

#####################################################################
# run the ML cnn training for each ki/ke pair difference
# store the accuracy within acc_mat_par.npy
# data_dir = ./parsweep5000_ML
#####################################################################

if runwho[2]:
    
    kis = ['0.1','0.09000000000000001','0.08','0.07','0.06000000000000001','0.05000000000000001',
           '0.04000000000000001','0.030000000000000006','0.020000000000000004','0.01'][::-1]
    kes = ['2.0','3.111111111111111','4.222222222222222','5.333333333333334',
           '6.444444444444445','7.555555555555555','8.666666666666668','9.777777777777779',
           '10.88888888888889','12.0']
    
    
    
    if not os.path.exists(os.path.join('.', ke_ki_data_dir)):
        x=1 ##add exception
        
    if not os.path.exists(os.path.join('.',base_dir, ke_ki_save_dir)):
        os.makedirs(os.path.join('.', base_dir,ke_ki_save_dir))
    
    pairs_already_used = []
    
    n = 10
    with tqdm.tqdm(n**2) as pbar:
        for i in range(0,n):
            for j in range(0,n):
                datafile = 'ki_ke_sweep_5000spots_' + kis[i] + '_' + kes[j]+'.csv'
    
    
    
    
                subprocess.run(['python', 'run_cnn_commandline.py',
                                '--i=%s'%str(i),
                                '--j=%s'%str(j),
                                '--data_file=%s'% os.path.join('.',ke_ki_data_dir,datafile),
                                '--save_model=%s'%str(1),
                                '--save_dir=%s'%os.path.join('.',base_dir,ke_ki_save_dir),
                                '--acc_file=%s'%'acc_mat_keki.npy',
                                '--retrain=%s'%str(retrain),
                                '--model_file=%s'%'unused',
                                '--model_name=%s'%(model_base_name + '_keki'),
                                '--verbose=%s'%str(0),
                                '--Fr=%s'%str(global_fr),
                                '--NFr=%s'%str(global_nf)                            
                                ])
                
    
                pbar.update(1)
    
    fr_key = []
    for i in range(0,n):
        sub_key = []
        for j in range(0,n):
            sub_key.append((i,j, kes[i], kis[j]))
        fr_key.append(sub_key)
    
    key_csv = pd.DataFrame(data=fr_key, index=kes, columns=kis)
    key_csv.to_csv(os.path.join('.',base_dir, ke_ki_save_dir, 'keki_key.csv'))

#####################################################################
# run the ML cnn training for gene at ke1 and ke2
# store the accuracy within acc_mat_par.npy
# data_dir = ./parsweep5000_ML
#####################################################################

if runwho[3]:
    ki = '.06'
    kes = ['2.0','3.111111111111111','4.222222222222222','5.333333333333334',
           '6.444444444444445','7.555555555555555','8.666666666666668','9.777777777777779',
           '10.88888888888889','12.0']
    FR = 1
    #Nframes = 3000
    
    
    
    if not os.path.exists(os.path.join('.', ke_data_dir)):
        x=1##add exception
        
    if not os.path.exists(os.path.join('.',base_dir, ke_save_dir)):
        os.makedirs(os.path.join('.', base_dir,ke_save_dir))
    
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
                                '--data_file1=%s'% os.path.join('.',ke_data_dir,datafile1),
                                '--data_file2=%s'% os.path.join('.',ke_data_dir,datafile2),
                                '--save_model=%s'%str(1),
                                '--save_dir=%s'%os.path.join('.',base_dir,ke_save_dir),
                                '--acc_file=%s'%'acc_mat_kes.npy',
                                '--retrain=%s'%str(retrain),
                                '--model_file=%s'%'unused',
                                '--model_name=%s'%(model_base_name + '_kes'),
                                '--verbose=%s'%str(0),
                                '--Fr=%s'%str(global_fr),
                                '--NFr=%s'%str(global_nf)
                                ])
                
    
                pbar.update(1)
    
    fr_key = []
    for i in range(0,n):
        sub_key = []
        for j in range(0,n):
            sub_key.append((i,j, kes[i], kes[j]))
        fr_key.append(sub_key)
    
    key_csv = pd.DataFrame(data=fr_key, index=kes, columns=kes)
    key_csv.to_csv(os.path.join('.',base_dir, ke_save_dir, 'kes_key.csv'))

#####################################################################
# run the ML cnn training for gene at ki1 and ki2
# store the accuracy within acc_mat_par.npy
# data_dir = ./parsweep5000_ML
#####################################################################

if runwho[4]:
    kis = ['0.1','0.09000000000000001','0.08','0.07','0.06000000000000001','0.05000000000000001',
           '0.04000000000000001','0.030000000000000006','0.020000000000000004','0.01'][::-1]
    ke = '5.33'
    
    
    if not os.path.exists(os.path.join('.', ki_data_dir)):
        x=1 ##add exception
        
    if not os.path.exists(os.path.join('.',base_dir, ki_save_dir)):
        os.makedirs(os.path.join('.', base_dir,ki_save_dir))
    
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
                                '--data_file1=%s'% os.path.join('.',ki_data_dir,datafile1),
                                '--data_file2=%s'% os.path.join('.',ki_data_dir,datafile2),
                                '--save_model=%s'%str(1),
                                '--save_dir=%s'%os.path.join('.',base_dir,ki_save_dir),
                                '--acc_file=%s'%'acc_mat_kis.npy',
                                '--retrain=%s'%str(retrain),
                                '--model_file=%s'%'unused',
                                '--model_name=%s'%(model_base_name + '_kis'),
                                '--verbose=%s'%str(0),
                                '--Fr=%s'%str(global_fr),
                                '--NFr=%s'%str(global_nf)
                                ])
                
    
                pbar.update(1)
                
                
    fr_key = []
    for i in range(0,n):
        sub_key = []
        for j in range(0,n):
            sub_key.append((i,j, kis[i], kis[j]))
        fr_key.append(sub_key)
    
    key_csv = pd.DataFrame(data=fr_key, index=kis, columns=kis)
    key_csv.to_csv(os.path.join('.',base_dir, ki_save_dir, 'kis_key.csv'))


meta_data['runtime'] = time.time() - start_time


def get_size(start_path = '.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size

run_size = get_size(os.path.join('.', base_dir))
meta_data['directory_size'] = run_size

with open(os.path.join('.',base_dir, 'metadata.yaml'), 'w') as outfile:
    yaml.dump(meta_data, outfile)