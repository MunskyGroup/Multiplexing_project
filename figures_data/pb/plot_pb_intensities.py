# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 17:44:02 2023

@author: willi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


target_dir_data = '../../datasets/P300_KDM5B_350s_base_pb'
target_dir_ml = '../../ML_PB/parsweep_pb_ML'


base = np.load('%s/acc_mat_pb.npy'%target_dir_ml)
wocorr = np.load('%s/acc_mat_pb_wo.npy'%target_dir_ml)
plt.plot(base[0,:8])
plt.plot(wocorr[0,:8])


    
for i in range(11):
    plt.figure()
    int_g1 = np.load('%s1/p300_base_pb_P300_P300_0.06_5.33333_%i_tracking_wo_correction_intensities.npy'%(target_dir_data,i))
    int_g2 = np.load('%s1/kdm5b_base_pb_KDM5B_KDM5B_0.06_5.33333_%i_tracking_wo_correction_intensities.npy'%(target_dir_data,i))
    plt.plot(np.mean(int_g1,axis=0))
    plt.plot(np.mean(int_g2,axis=0))

    plt.title('After thresholding tracked average intensites pb_rate = %i'%i)
    plt.xlabel('time')
    plt.ylabel('intensity')

ntimes = 350
ntraj = 1250
for i in range(11):
    plt.figure()
    df1 = pd.read_csv('%s1/p300_base_pb_P300_P300_0.06_5.33333_%i.csv'%(target_dir_data,i))
    df2 =  pd.read_csv('%s1/kdm5b_base_pb_KDM5B_KDM5B_0.06_5.33333_%i.csv'%(target_dir_data,i))
    int_g1 = df1['green_int_mean'].values.reshape([ntraj,ntimes])  
    int_g2 = df1['red_int_mean'].values.reshape([ntraj,ntimes])  
    plt.plot(np.mean(int_g1,axis=0),'g')
    plt.plot(np.mean(int_g2,axis=0),'r')
    
    plt.title('True average intensites pb_rate = %i'%i)
    plt.xlabel('time')
    plt.ylabel('intensity')
        



for i in range(11):
    plt.figure()
    
    df_real ='%s1/kdm5b_base_pb_KDM5B_KDM5B_0.06_5.33333_%i.csv'%(target_dir_data,i)
    df_no_correction = '%s1/kdm5b_base_pb_KDM5B_KDM5B_0.06_5.33333_%i.csv_tracking_wo_correction'%(target_dir_data,i)

    
    multiplexing_df1 = pd.read_csv(df_real)
    m_df1_wo_correction = pd.read_csv(df_no_correction)
    
    
    av_int_r = np.zeros(350)
    av_int_g = np.zeros(350)
        
    for j in range(0,350):
        av_int_r[j] = np.mean(m_df1_wo_correction[m_df1_wo_correction['frame'] == j]['red_int_mean'])
        av_int_g[j] = np.mean(m_df1_wo_correction[m_df1_wo_correction['frame'] == j]['green_int_mean'])
    
    plt.plot(av_int_r,'r')
    plt.plot(av_int_g,'g')
    plt.title('All tracked particle intensites (not corrected) pb_rate = %i'%i)
    plt.xlabel('time')
    plt.ylabel('intensity')
    