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
img_data_dir = 'par_sweep_kis'
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
parser.add_argument('--global_fr', dest='global_fr', type=int,)
parser.add_argument('--global_nf', dest='global_nf', type=int,)
parser.add_argument('--base_dir', dest='base_dir', type=str,)
parser.add_argument('--global_samples', dest='global_samples', type=int,)
parser.add_argument('--test_size', dest='test_size', type=float,)
parser.add_argument('--witheld', dest='witheld', type=int,)
parser.add_argument('--base_command', dest='base_command', type=str,)

parser.add_argument('--cl', dest='cl', type=int,)
parser.add_argument('--img', dest='img', type=int,)
parser.add_argument('--keki', dest='keki', type=int,)
parser.add_argument('--kes', dest='kes', type=int,)
parser.add_argument('--kis', dest='kis', type=int,)
parser.add_argument('--debug', dest='debug', type=int, default=0)


args = parser.parse_args()

base_command = args.base_command
base_dir = args.base_dir
global_fr = args.global_fr
global_nf = args.global_nf
retrain = 1
global_samples = args.global_samples
witheld = args.witheld
test_size = args.test_size
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

#          cl img  keki  kes kis
runwho =  [cl,  img,   keki,   kes,   kis]


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
# run the ML cnn training for each construct length difference
# store the accuracy within acc_mat_cl.npy
# data_dir = ./construct_L_ML
#####################################################################

if runwho[0]:

    names1 = ['GOLT1A','CDC42','MYOZ3','GPN2','CD46','LRRC42','SPATA6','SCP2','BBS5','CAMK2B']
    names2 = ['RRAGC','ORC2','LONRF2', 'EDEM3','TRIM33','MAP3K6','COL3A1','KDM5B','KDM6B','PHIP','DOCK8','P300']
    
    
    
    if not os.path.exists(os.path.join(data_root, CL_data_dir)):
        x=1 ##Add exception
        
    if not os.path.exists(os.path.join(data_root,base_dir, CL_save_dir)):
        os.makedirs(os.path.join(data_root,base_dir, CL_save_dir))
    


    pairs_already_used = []
    
    n = len(names2)
    
    acc_path = os.path.join(data_root,base_dir,CL_save_dir,'acc_mat_cl.npy')
    print(acc_path)
    print(os.path.exists(acc_path))
    if not os.path.exists(acc_path):
        acc_mat = np.zeros([n,n])
        print('made acc_mat')
        print(acc_mat.shape)
        np.save(acc_path, acc_mat)
        
    with tqdm.tqdm(n**2) as pbar:
        for i in range(0,n):
            
            for j in range(0,n):
                
                if j >= i:
                    
                    datafile1 = 'construct_lengths_' + names2[i] + '_' + names2[i]+'.csv'
                    datafile2 = 'construct_lengths_' + names2[j] + '_' + names2[j]+'.csv'
    
                    subprocess.run(['python', base_command,
                                    '--i=%s'%str(i),
                                    '--j=%s'%str(j),
                                    '--data_file1=%s'% os.path.join(data_root,dataset_dir,CL_data_dir,datafile1),
                                    '--data_file2=%s'% os.path.join(data_root,dataset_dir,CL_data_dir,datafile2),
                                    '--save_model=%s'%str(1),
                                    '--save_dir=%s'%os.path.join(data_root,base_dir,CL_save_dir),
                                    '--acc_file=%s'%'acc_mat_cl.npy',
                                    '--retrain=%s'%str(retrain),
                                    '--model_file=%s'%'unused',
                                    '--model_name=%s'%(model_base_name + '_cl'),
                                    '--verbose=%s'%str(0),
                                    '--Fr=%s'%str(global_fr),
                                    '--NFr=%s'%str(global_nf),
                                    '--Nsamples=%s'%str(global_samples),
                                    '--two_files=%s'%(str(1)),
                                    '--witheld=%s'%(str(witheld)),
                                    '--test_size=%s'%(str(test_size)),
                                    '--debug=%s'%(str(debug)),
    
                                    ])
                    
    
                    pbar.update(1)
        
    acc_mat_file = np.load( os.path.join(data_root, base_dir,CL_save_dir,'acc_mat_cl.npy' )  )
    CL_key = []
    for i in range(0,n):
        CL_keyi = []
        for j in range(0,n):
            CL_keyi.append((i,j, names2[i], names2[j], acc_mat_file[i,j]))
        CL_key.append(CL_keyi)
    
    CL_key_csv = pd.DataFrame(data=CL_key, index=names2, columns=names2)
    CL_key_csv.to_csv(os.path.join(data_root,base_dir, CL_save_dir, 'cl_key.csv'))

##############################################################################
# run the ML cnn training for P300 and Kdm5b at different imaging conditions
# store the accuracy within acc_mat_img.npy
# data_dir = ./parsweep5000_ML
##############################################################################


if runwho[1]:
        
    FRs = [1,2,5,7,10,12,15,20,25,30]
    total_time = [3000,2500,2000,1500,1000,750,500,250,200,150][::-1]
    nframes = [10,20,30,40,50,60,70,80,90,100]
    nfs = []
    
    ki = '.06'
    kes = '5.33'
    
    
    
    
    if not os.path.exists(os.path.join(data_root, img_data_dir)):
        x=1 ##add exception
        
    if not os.path.exists(os.path.join(data_root,base_dir, img_save_dir)):
        os.makedirs(os.path.join(data_root,base_dir, img_save_dir))
    
    pairs_already_used = []
    
    
    datafile1 = 'parsweep_kis_kdm5b_' +'0.06000000000000001' +'.csv'
    datafile2 = 'parsweep_kis_p300_' + '0.06000000000000001' +'.csv'
    
    tmp = np.zeros([1,3000])
    
    n = global_n
    
    valid_mat = np.zeros([10,10])
    FR_mat = np.zeros([10,10])
    NF_mat = np.zeros([10,10])
    time_mat = np.zeros([10,10])
    with tqdm.tqdm(n**2) as pbar:
        for i in range(0,n):
            for j in range(0,n):
                 
                #nframes = int(np.floor(total_time[j] / FRs[i]))
                #available_frames = int(tmp[:,::FRs[i]].shape[1])
                #if nframes <= available_frames:
                if 1:
        
                    #tmp[:,::FRs[i]][:nframes] #check if its a valid combo, if so run
                    
                    #valid_mat[i,j] = 1
                    #FR_mat[i,j] = FRs[i]
                    #NF_mat[i,j] = nframes
                    #time_mat[i,j] = nframes*FRs[i]
                    
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
                                    '--test_size=%s'%(str(test_size)),
                                    '--debug=%s'%(str(debug)),
                                    ])
                    pbar.update(1)
                    
    acc_mat_file = np.load( os.path.join(data_root, base_dir,img_save_dir,'acc_mat_img.npy' )  )
    fr_key = []
    for i in range(0,n):
        sub_key = []
        for j in range(0,n):
            sub_key.append((i,j, FRs[i], nframes[j], acc_mat_file[i,j]))
        fr_key.append(sub_key)
    
    key_csv = pd.DataFrame(data=fr_key, index=FRs, columns=total_time)
    key_csv.to_csv(os.path.join(data_root,base_dir, img_save_dir, 'img_key.csv'))

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
    
    
    
    if not os.path.exists(os.path.join(data_root, ke_ki_data_dir)):
        x=1 ##add exception
        
    if not os.path.exists(os.path.join(data_root,base_dir, ke_ki_save_dir)):
        os.makedirs(os.path.join(data_root, base_dir,ke_ki_save_dir))
    
    pairs_already_used = []
    
    n = global_n
  
    with tqdm.tqdm(n**2) as pbar:
        for i in range(0,n):
            for j in range(0,n):
                datafile = 'ki_ke_sweep_5000spots_' + kis[i] + '_' + kes[j]+'.csv'
    
    
    
    
                subprocess.run(['python', base_command,
                                '--i=%s'%str(i),
                                '--j=%s'%str(j),
                                '--data_file1=%s'% os.path.join(data_root,dataset_dir,ke_ki_data_dir,datafile),
                                '--save_model=%s'%str(1),
                                '--save_dir=%s'%os.path.join(data_root,base_dir,ke_ki_save_dir),
                                '--acc_file=%s'%'acc_mat_keki.npy',
                                '--retrain=%s'%str(retrain),
                                '--model_file=%s'%'unused',
                                '--model_name=%s'%(model_base_name + '_keki'),
                                '--verbose=%s'%str(0),
                                '--Fr=%s'%str(global_fr),
                                '--NFr=%s'%str(global_nf),
                                '--Nsamples=%s'%str(global_samples),
                                '--two_files=%s'%(str(0)),
                                '--ntraj=%s'%(str(5000)),
                                '--witheld=%s'%(str(witheld)),
                                '--test_size=%s'%(str(test_size)),
                                '--debug=%s'%(str(debug)),

                                ])
                
    
                pbar.update(1)
  
    
    fr_key = []
    acc_mat_file = np.load( os.path.join(data_root, base_dir,ke_ki_save_dir,'acc_mat_keki.npy' )  )
    for i in range(0,n):
        sub_key = []
        for j in range(0,n):
            sub_key.append((i,j, kis[i], kes[j], acc_mat_file[i,j] ) )
        fr_key.append(sub_key)
    
    key_csv = pd.DataFrame(data=fr_key, index=kes, columns=kis)
    key_csv.to_csv(os.path.join(data_root,base_dir, ke_ki_save_dir, 'keki_key.csv'))

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
    
    
    
    if not os.path.exists(os.path.join(data_root, ke_data_dir)):
        x=1##add exception
        
    if not os.path.exists(os.path.join(data_root,base_dir, ke_save_dir)):
        os.makedirs(os.path.join(data_root, base_dir,ke_save_dir))
    
    pairs_already_used = []
    
    n = global_n
    with tqdm.tqdm(n**2) as pbar:
        for i in range(0,n):
            for j in range(0,n):
                datafile1 = 'parsweep_kes_kdm5b_' + kes[i] +'.csv'
                datafile2 = 'parsweep_kes_p300_' + kes[j] +'.csv'
                
                subprocess.run(['python', base_command,
                                '--i=%s'%str(i),
                                '--j=%s'%str(j),
                                '--data_file1=%s'% os.path.join(data_root,dataset_dir,ke_data_dir,datafile1),
                                '--data_file2=%s'% os.path.join(data_root,dataset_dir,ke_data_dir,datafile2),
                                '--save_model=%s'%str(1),
                                '--save_dir=%s'%os.path.join(data_root,base_dir,ke_save_dir),
                                '--acc_file=%s'%'acc_mat_kes.npy',
                                '--retrain=%s'%str(retrain),
                                '--model_file=%s'%'unused',
                                '--model_name=%s'%(model_base_name + '_kes'),
                                '--verbose=%s'%str(0),
                                '--Fr=%s'%str(global_fr),
                                '--NFr=%s'%str(global_nf),
                                '--Nsamples=%s'%str(global_samples),
                                '--two_files=%s'%(str(1)),
                                '--witheld=%s'%(str(witheld)),
                                '--test_size=%s'%(str(test_size)),
                                '--debug=%s'%(str(debug)),

                                ])
                
    
                pbar.update(1)
                
    acc_mat_file = np.load( os.path.join(data_root, base_dir,ke_save_dir,'acc_mat_kes.npy' )  )
    fr_key = []
    for i in range(0,n):
        sub_key = []
        for j in range(0,n):
            sub_key.append((i,j, kes[i], kes[j], acc_mat_file[i,j]))
        fr_key.append(sub_key)
    
    key_csv = pd.DataFrame(data=fr_key, index=kes, columns=kes)
    key_csv.to_csv(os.path.join(data_root,base_dir, ke_save_dir, 'kes_key.csv'))

#####################################################################
# run the ML cnn training for gene at ki1 and ki2
# store the accuracy within acc_mat_par.npy
# data_dir = ./parsweep5000_ML
#####################################################################

if runwho[4]:
    kis = ['0.1','0.09000000000000001','0.08','0.07','0.06000000000000001','0.05000000000000001',
           '0.04000000000000001','0.030000000000000006','0.020000000000000004','0.01'][::-1]
    ke = '5.33'
    
    
    if not os.path.exists(os.path.join(data_root, ki_data_dir)):
        x=1 ##add exception
        
    if not os.path.exists(os.path.join(data_root,base_dir, ki_save_dir)):
        os.makedirs(os.path.join(data_root, base_dir,ki_save_dir))
    
    pairs_already_used = []
    
    n = global_n
    with tqdm.tqdm(n**2) as pbar:
        for i in range(0,n):
            for j in range(0,n):
                datafile1 = 'parsweep_kis_kdm5b_' + kis[i] +'.csv'
                datafile2 = 'parsweep_kis_p300_' + kis[j] +'.csv'
                
                subprocess.run(['python', base_command,
                                '--i=%s'%str(i),
                                '--j=%s'%str(j),
                                '--data_file1=%s'% os.path.join(data_root,dataset_dir,ki_data_dir,datafile1),
                                '--data_file2=%s'% os.path.join(data_root,dataset_dir,ki_data_dir,datafile2),
                                '--save_model=%s'%str(1),
                                '--save_dir=%s'%os.path.join(data_root,base_dir,ki_save_dir),
                                '--acc_file=%s'%'acc_mat_kis.npy',
                                '--retrain=%s'%str(retrain),
                                '--model_file=%s'%'unused',
                                '--model_name=%s'%(model_base_name + '_kis'),
                                '--verbose=%s'%str(0),
                                '--Fr=%s'%str(global_fr),
                                '--NFr=%s'%str(global_nf),
                                '--Nsamples=%s'%str(global_samples),
                                '--two_files=%s'%(str(1)),
                                '--witheld=%s'%(str(witheld)),
                                '--test_size=%s'%(str(test_size)),
                                '--debug=%s'%(str(debug)),
                                ])
                
    
                pbar.update(1)
                
    acc_mat_file = np.load( os.path.join(data_root, base_dir,ki_save_dir,'acc_mat_kis.npy' )  )
    fr_key = []
    for i in range(0,n):
        sub_key = []
        for j in range(0,n):
            sub_key.append((i,j, kis[i], kis[j], acc_mat_file[i,j]))
        fr_key.append(sub_key)
    
    key_csv = pd.DataFrame(data=fr_key, index=kis, columns=kis)
    key_csv.to_csv(os.path.join(data_root,base_dir, ki_save_dir, 'kis_key.csv'))


meta_data['runtime'] = time.time() - start_time

with open(os.path.join(data_root,base_dir, 'metadata.yaml'), 'w') as outfile:
    yaml.dump(meta_data, outfile)
