# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 10:38:37 2022

@author: willi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

import os
cwd = os.getcwd()
os.chdir('../../')
import apply_style  as aps#apply custom matplotlib style
import mc_core as multiplexing_core
os.chdir(cwd)

target_dir = '../../ML_imaging_long_same_int_24000_14scale'
aps.make_heatmaps_from_keyfile_big(target_dir)

1/0
target_dir = '../../ML_IF_kisdiff'
aps.make_heatmaps_from_keyfile_big(target_dir)

1/0
target_dir = '../../ML_imaging_long_different_int_12000_14scale'
aps.make_heatmaps_from_keyfile_big(target_dir)


mc = multiplexing_core.multiplexing_core()
aps.apply_style()

similar_kdm5b = 'D:/multiplexing_ML/finalized_plots_gaussians/datasets/P300_KDM5B_24000s_similar_intensity_gaussian_14scale/ki_ke_sweep_same_int_P300_P300_0.009675858127721334_5.33333_0.csv'
similar_p300 = 'D:/multiplexing_ML/finalized_plots_gaussians/datasets/P300_KDM5B_24000s_similar_intensity_gaussian_14scale/ki_ke_sweep_same_int_KDM5B_KDM5B_0.0186_5.33333_0.csv'

same_kdm5b = 'D:/multiplexing_ML/finalized_plots_gaussians/datasets/P300_KDM5B_24000s_Same_intensity_gaussian_14scale/ki_ke_sweep_same_int_KDM5B_KDM5B_0.014139262990455991_5.33333_0.csv'
same_p300 = 'D:/multiplexing_ML/finalized_plots_gaussians/datasets/P300_KDM5B_24000s_Same_intensity_gaussian_14scale/ki_ke_sweep_same_int_P300_P300_0.009675852685050798_5.33333_0.csv'


1/0

colors = ['#073b4c','#57ffcd', '#ff479d', '#ffe869','#ff8c00','#04756f']
fs = (5,2)


ntraj=2500
ntimes=24000
Nsamples = 5000
seed = 42
n_model_traj = 4000
witheld=0
test_size=0
multiplexing_df1 = pd.read_csv(same_kdm5b)
multiplexing_df2 = pd.read_csv(same_p300)
int_g1 = multiplexing_df1['green_int_mean'].values.reshape([ntraj,ntimes])    
int_g2 = multiplexing_df2['green_int_mean'].values.reshape([ntraj,ntimes])    

t = np.linspace(0,len(int_g1) - 1,len(int_g1))  #time vector in seconds
labels = np.ones(int_g1.shape[0]*2)
labels[:int_g1.shape[0]] = 0
int_g = np.vstack((int_g1,int_g2))  #merge the files and then let it sort
labels = labels

int_g, labels = mc.even_shuffle_sample(int_g, labels, samples=[int(Nsamples/2), int(Nsamples/2)], seed=seed)
int_g = mc.slice_arr(int_g, 1, 1280)

X_train, X_test, y_train, y_test, X_witheld, y_witheld, Acc_train, Acc_test, Acc_witheld = mc.process_data(int_g, labels, norm='train', seed=seed, witheld = witheld, test_size = test_size, include_acc = True)


plt.figure(dpi=300,  figsize= fs)
x,bins = np.histogram(X_train[labels==1,::100,0].flatten(),bins=30)
#plt.hist(int_g_real_transformed[real_labels==0,::100].flatten(),bins=bins,  density=True, fc=(7/255,59/255,76/255, 0.5));
#plt.hist(int_g_real_transformed[real_labels==1,::100].flatten(),bins=bins, density=True, fc=(239/255,71/255,111/255, 0.5), orientation='horizontal')
#plt.hist(X_train[labels==0,::100,0].flatten(),bins=bins,  ec = colors[0], lw=3, density=True,fc=(0, 0, 1, 0.0), histtype='step');
plt.hist(X_train[labels==0,::100,0].flatten(),bins=bins, lw=3, ec = colors[0], density=True,  histtype='step',)
plt.hist(X_train[labels==1,::100,0].flatten(),bins=bins, lw=3, ec = colors[2], density=True, histtype='step', )
plt.ylim([0,11])
plt.xlim([-0.05,1.05])
plt.ylabel('Density')
plt.xlabel('Normalized Intensity')
plt.savefig('same_int_dist.svg')



multiplexing_df1 = pd.read_csv(similar_kdm5b)
multiplexing_df2 = pd.read_csv(similar_p300)
int_g1 = multiplexing_df1['green_int_mean'].values.reshape([ntraj,ntimes])    
int_g2 = multiplexing_df2['green_int_mean'].values.reshape([ntraj,ntimes])    

t = np.linspace(0,len(int_g1) - 1,len(int_g1))  #time vector in seconds
labels = np.ones(int_g1.shape[0]*2)
labels[:int_g1.shape[0]] = 0
int_g = np.vstack((int_g1,int_g2))  #merge the files and then let it sort
labels = labels

int_g, labels = mc.even_shuffle_sample(int_g, labels, samples=[int(Nsamples/2), int(Nsamples/2)], seed=seed)
int_g = mc.slice_arr(int_g, 1, 1280)

X_train2, X_test, y_train, y_test, X_witheld, y_witheld, Acc_train, Acc_test, Acc_witheld = mc.process_data(int_g, labels, norm='train', seed=seed, witheld = witheld, test_size = test_size, include_acc = True)


plt.figure(dpi=300,  figsize= fs)
x,bins = np.histogram(X_train2[labels==1,::100,0].flatten(),bins=30)
#plt.hist(int_g_real_transformed[real_labels==0,::100].flatten(),bins=bins,  density=True, fc=(7/255,59/255,76/255, 0.5));
#plt.hist(int_g_real_transformed[real_labels==1,::100].flatten(),bins=bins, density=True, fc=(239/255,71/255,111/255, 0.5), orientation='horizontal')
#plt.hist(X_train[labels==0,::100,0].flatten(),bins=bins,  ec = colors[0], lw=3, density=True,fc=(0, 0, 1, 0.0), histtype='step');
plt.hist(X_train2[labels==0,::100,0].flatten(),bins=bins, lw=3, ec = colors[0], density=True,  histtype='step', )
plt.hist(X_train2[labels==1,::100,0].flatten(),bins=bins, lw=3, ec = colors[2], density=True,  histtype='step', )
plt.ylim([0,11])
plt.xlim([-0.05,1.05])
plt.ylabel('Density')
plt.xlabel('Normalized Intensity')
plt.savefig('similar_int_dist.svg')


ntraj=2500
ntimes=12000
Nsamples = 5000
seed = 42
n_model_traj = 4000
witheld=0
test_size=0
different_kdm5b = 'D:/multiplexing_ML/finalized_plots_gaussians/datasets/P300_KDM5B_24000s_different_intensity_gaussian_14scale/ki_ke_sweep_diff_int_KDM5B_KDM5B_0.04241781283138918_5.33333_0.csv'
different_p300 = 'D:/multiplexing_ML/finalized_plots_gaussians/datasets/P300_KDM5B_24000s_different_intensity_gaussian_14scale/ki_ke_sweep_diff_int_P300_P300_0.009675858127721334_5.33333_0.csv'

multiplexing_df1 = pd.read_csv(different_kdm5b)
multiplexing_df2 = pd.read_csv(different_p300)
int_g1 = multiplexing_df1['green_int_mean'].values.reshape([ntraj,ntimes])    
int_g2 = multiplexing_df2['green_int_mean'].values.reshape([ntraj,ntimes])    

t = np.linspace(0,len(int_g1) - 1,len(int_g1))  #time vector in seconds
labels = np.ones(int_g1.shape[0]*2)
labels[:int_g1.shape[0]] = 0
int_g = np.vstack((int_g1,int_g2))  #merge the files and then let it sort
labels = labels

int_g, labels = mc.even_shuffle_sample(int_g, labels, samples=[int(Nsamples/2), int(Nsamples/2)], seed=seed)
int_g = mc.slice_arr(int_g, 1, 1280)

X_train3, X_test, y_train, y_test, X_witheld, y_witheld, Acc_train, Acc_test, Acc_witheld = mc.process_data(int_g, labels, norm='train', seed=seed, witheld = witheld, test_size = test_size, include_acc = True)


plt.figure(dpi=300,  figsize= fs)
x,bins = np.histogram(X_train3[labels==1,::100,0].flatten(),bins=30)
#plt.hist(int_g_real_transformed[real_labels==0,::100].flatten(),bins=bins,  density=True, fc=(7/255,59/255,76/255, 0.5));
#plt.hist(int_g_real_transformed[real_labels==1,::100].flatten(),bins=bins, density=True, fc=(239/255,71/255,111/255, 0.5), orientation='horizontal')
#plt.hist(X_train[labels==0,::100,0].flatten(),bins=bins,  ec = colors[0], lw=3, density=True,fc=(0, 0, 1, 0.0), histtype='step');
plt.hist(X_train3[labels==0,::100,0].flatten(),bins=bins, lw=3, ec = colors[0], density=True,  histtype='step', )
plt.hist(X_train3[labels==1,::100,0].flatten(),bins=bins, lw=3, ec = colors[2], density=True,  histtype='step', )
plt.ylabel('Density')
plt.xlabel('Normalized Intensity')

plt.ylim([0,11])
plt.xlim([-0.05,1.05])
plt.savefig('different_int_dist.svg')




1/0


# similar intensity dataset
Ikey = '../../ML_I_kisdiff/parsweep_img_ML/img_key.csv'
IFkey = '../../ML_IF_kisdiff/parsweep_img_ML/img_key.csv'
Fkey = '../../ML_F_kisdiff/parsweep_img_ML/img_key.csv'



def convert_key_file(f):
    key_file = pd.read_csv(f)
    
    xshape = key_file.shape[0]
    yshape = key_file.shape[1]-1
    
    convert_str = lambda tstr: [x.replace(' ','').replace('(','').replace(')','').replace("'",'') for x in tuple(tstr.split(','))] 
    round_str = lambda fstr: str(np.round(float(fstr),2))
    
    acc_mat = np.zeros([xshape,yshape])
    xlabels = [0,]*xshape
    ylabels = [0,]*yshape
    for i in range(0,xshape):
        for j in range(0,yshape):
            
            ind1,ind2, x,y,acc = convert_str(key_file.iloc[i][j+1])
           
            acc_mat[i,j] = acc
            xlabels[int(ind1)] = x
            ylabels[int(ind2)] = y
        
    return acc_mat, xlabels, ylabels


Imat,x,y = convert_key_file(Ikey)
IFmat,_,_ = convert_key_file(IFkey)
Fmat,_,_ = convert_key_file(Fkey)


i = 9
plt.figure(figsize=(8,4))
plt.plot([int(y) for y in x][:], Imat[:,i]);plt.plot([int(y) for y in x][:], IFmat[:,i]);plt.plot([int(y) for y in x][:], Fmat[:,i]);plt.legend(['I','IF','F']);
plt.gca().set_xticks([int(y) for y in x][:])
plt.gca().set_xticklabels(x[:]); plt.ylabel('Test Accuracy'); plt.title('ML classification over Nframes = %s'%y[i]); plt.xlabel('Frame rate (s)')
plt.savefig('./IIFF_set_framerate.svg')



i = 4
j = -3
plt.figure(figsize=(8,4))
plt.plot([int(x) for x in y][:j], Imat[i,:j]);plt.plot([int(x) for x in y][:j], IFmat[i,:j]);plt.plot([int(x) for x in y][:j], Fmat[i,:j]);plt.legend(['I','IF','F']);
plt.gca().set_xticks([int(x) for x in y][:j])
plt.gca().set_xticklabels(y[:j]); plt.ylabel('Test Accuracy'); plt.title('ML classification at frame rate = %ss'%x[i]); plt.xlabel('Number of Frames (s)')
plt.savefig('./IIFF_set_frames.svg')


'''
i = 9
plt.figure(figsize=(8,4))
plt.plot([int(y) for y in x][:-1], Imat[:-1,i]);plt.plot([int(y) for y in x][:-1], IFmat[:-1,i]);plt.plot([int(y) for y in x][:-1], Fmat[:-1,i]);plt.legend(['I','IF','F']);
plt.gca().set_xticks([int(y) for y in x][:-1])
plt.gca().set_xticklabels(x[:-1]); plt.ylabel('Test Accuracy'); plt.title('ML classification over Nframes = %s'%y[i]); plt.xlabel('Frame rate (s)')
plt.savefig('./IIFF.svg')


plt.figure()
plt.plot([int(y) for y in x], Imat[:,1]);plt.plot([int(y) for y in x], IFmat[:,1]);plt.plot([int(y) for y in x], Fmat[:,1]);plt.legend(['I','IF','F']);
plt.gca().set_xticks([int(y) for y in x])
plt.gca().set_xticklabels(x); plt.ylabel('Test Accuracy'); plt.title('Nframes = %s'%y[1]); plt.xlabel('Frame rate (s)')



plt.figure()
plt.plot([int(y) for y in x], Imat[:,0]);plt.plot([int(y) for y in x], IFmat[:,0]);plt.plot([int(y) for y in x], Fmat[:,0]);plt.legend(['I','IF','F']);
plt.gca().set_xticks([int(y) for y in x])
plt.gca().set_xticklabels(x); plt.ylabel('Test Accuracy'); plt.title('Nframes = %s'%y[0]); plt.xlabel('Frame rate (s)')



plt.figure()
plt.plot([int(y) for y in x][:-2], Imat[:-2,3]);plt.plot([int(y) for y in x][:-2], IFmat[:-2,3]);plt.plot([int(y) for y in x][:-2], Fmat[:-2,3]);plt.legend(['I','IF','F']);
plt.gca().set_xticks([int(y) for y in x][:-2])
plt.gca().set_xticklabels(x[:-2]); plt.ylabel('Test Accuracy'); plt.title('Nframes = %s'%y[3]); plt.xlabel('Frame rate (s)')



'''