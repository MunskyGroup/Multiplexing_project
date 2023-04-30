# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 14:21:24 2022

@author: willi
"""


import pandas   
from IPython.display import Image, display
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import os
import tqdm.notebook as tq
import PIL
import ipywidgets as widgets
from ipywidgets import interact
#import imread
#import cv2
#import skimage.io as io

from matplotlib.colors import LogNorm

import argparse
import os
cwd = os.getcwd()
os.chdir('../../')
import apply_style  as aps#apply custom matplotlib style
import multiplex_core as multiplexing_core
os.chdir(cwd)

aps.apply_style()

import pandas as pd

from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix

from tensorflow.keras import layers, models, optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, AveragePooling1D, Conv1D, LeakyReLU, Lambda
from tensorflow.keras.regularizers import l2, l1, l1_l2
from tensorflow.keras.layers import BatchNormalization
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from matplotlib.colors import ListedColormap, LinearSegmentedColormap 
from sklearn import mixture

save_fig = False

print(os.getcwd())

mc = multiplexing_core.multiplexing_core()
aps.apply_style()
files = ['../../datasets/P300_KDM5B_base_experiment_5_cells/base_experiment_KDM5B_0.06_P300_0.06_5.33333_%i.csv'%i for i in range(2)][:3]

ntraj = 50
ntimes = 3000

green_intensities = []
all_labels = []
for path1 in files:
    multiplexing_df = pd.read_csv(path1)
    green_intensities.append(multiplexing_df['green_int_mean'].values.reshape([ntraj,ntimes]))
    all_labels.append(multiplexing_df['Classification'].values.reshape([ntraj,ntimes])[:,0])
    
    #int_g, labels = mc.even_shuffle_sample(int_g, labels, samples=[int(Nsamples/2), int(Nsamples/2)], seed=seed)
    
int_g = np.vstack(green_intensities)
int_g = mc.slice_arr_reverse(int_g, 10, 128)
labels = np.vstack(all_labels).flatten()

mc = multiplexing_core.multiplexing_core()
X_train_real, _, y_train_real, _, _, _, Acc_train_real, _, _ = mc.process_data(int_g, labels, norm='train', seed=42, witheld = 0, test_size = 0, include_acc = True )




acc, acc_error = mc.get_autocov(np.expand_dims(int_g.T,axis=0), norm='i')
acc, acc_error1 = mc.get_autocorr(acc, norm_type='i', norm='g0' )
acc = acc[0].T
acc = acc.reshape(acc.shape[0], acc.shape[1], 1)

plt.figure(dpi=300)
plt.plot(np.mean(acc[labels==0,:,0], axis=0).T,'r'); plt.plot(np.mean(acc[labels==1,:,0],axis=0).T,'b')

mean_kdm5b_acc = np.mean(acc[labels==0,:,0], axis=0).T
mean_p300_acc = np.mean(acc[labels==1,:,0], axis=0).T

average_dwell_kdm5b = np.mean(np.where(mean_kdm5b_acc < 0.01)[0][:10])*10
average_dwell_p300 = np.mean(np.where(mean_p300_acc < 0.01)[0][:10])*10


colors = ['#073b4c','#57ffcd', '#ff479d', '#ffe869','#ff8c00','#04756f']

plt.figure()
n_pts = len(acc)
mean = np.mean(acc[labels==0,:,0],axis=0)
rezero = np.mean(mean[-50:])
mean = (mean - rezero)/(1 - rezero)
sem = np.std( ((acc[labels==0,:,0]) - rezero)/(1-rezero) ,axis=0)

tau_kdm5b = (np.mean(np.where(mean < 0.01)[0][:10])*10)
plt.plot([-5,100],[0,0],'-',color='gray',alpha=.8,label='_nolegend_')
plt.plot([tau_kdm5b/10,tau_kdm5b/10],[-.2,1.1],'-',color='gray',alpha=.8,label='_nolegend_')
plt.text(tau_kdm5b/10-4, .4, 'Tau = %is'%int(tau_kdm5b), rotation=90, color='gray', alpha=.8)
plt.errorbar(np.linspace(0,127,128), mean, yerr=sem, zorder=0, capsize=1, ls='',marker='o', markersize=1.5, lw=1, color=colors[0])



plt.figure()
n_pts = len(acc)
mean = np.mean(acc[labels==1,:,0],axis=0)
rezero = np.mean(mean[-50:])
mean = (mean - rezero)/(1 - rezero)
sem = np.std( ((acc[labels==1,:,0]) - rezero)/(1-rezero) ,axis=0)

tau_p300 = (np.mean(np.where(mean < 0.01)[0][:10])*10)
plt.plot([-5,100],[0,0],'-',color='gray',alpha=.8,label='_nolegend_')
plt.plot([tau_p300/10,tau_p300/10],[-.2,1.1],'-',color='gray',alpha=.8,label='_nolegend_')
plt.text(tau_p300/10-4, .4, 'Tau = %is'%int(tau_p300), rotation=90, color='gray', alpha=.8)
plt.errorbar(np.linspace(0,127,128), mean, yerr=sem, zorder=0, capsize=1, ls='',marker='o', markersize=1.5, lw=1, color=colors[2])


p300_files = ['parsweep_kes_p300_2.0.csv',
              'parsweep_kes_p300_3.111111111111111.csv',
              'parsweep_kes_p300_4.222222222222222.csv',
              'parsweep_kes_p300_5.333333333333334.csv',
              'parsweep_kes_p300_6.444444444444445.csv',
              'parsweep_kes_p300_7.555555555555555.csv',
              'parsweep_kes_p300_9.777777777777779.csv',
              'parsweep_kes_p300_8.666666666666668.csv',
              'parsweep_kes_p300_10.88888888888889.csv',
              'parsweep_kes_p300_12.0.csv']


kdm5b_files = ['parsweep_kes_kdm5b_2.0.csv',
              'parsweep_kes_kdm5b_3.111111111111111.csv',
              'parsweep_kes_kdm5b_4.222222222222222.csv',
              'parsweep_kes_kdm5b_5.333333333333334.csv',
              'parsweep_kes_kdm5b_6.444444444444445.csv',
              'parsweep_kes_kdm5b_7.555555555555555.csv',
              'parsweep_kes_kdm5b_9.777777777777779.csv',
              'parsweep_kes_kdm5b_8.666666666666668.csv',
              'parsweep_kes_kdm5b_10.88888888888889.csv',
              'parsweep_kes_kdm5b_12.0.csv']

path = '../../datasets/par_sweep_kes/'
real_labels = np.copy(labels)

def get_LL(kdm5b_file, p300_file):
  ntraj=2500
  ntimes=3000
  Nsamples = 5000
  seed = 42
  n_model_traj = 4000
  witheld=0
  test_size=0
  multiplexing_df1 = pd.read_csv(path +'/'+ kdm5b_file)
  multiplexing_df2 = pd.read_csv(path +'/'+  p300_file)
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


  model_mean = np.mean(Acc_train[y_train.flatten() == 0],axis=0)[::10]
  rezero_model = np.mean(model_mean[-50:])
  kdm5b_mean_model = (model_mean - rezero_model)/(1 - rezero_model)
  kdm5b_sem_model = np.std( ( (Acc_train[y_train.flatten().astype(int) == 0])[::10] - rezero_model)/(1-rezero_model) ,axis=0)

  model_mean = np.mean(Acc_train[y_train.flatten() == 1],axis=0)[::10]
  rezero_model = np.mean(model_mean[-50:])
  p300_mean_model = (model_mean - rezero_model)/(1 - rezero_model)
  p300_sem_model= np.std( ( (Acc_train[y_train.flatten().astype(int) == 1])[::10] - rezero_model)/(1-rezero_model) ,axis=0)


  #n_pts = len(acc[labels==0,:,0])
  mean = np.mean(acc[real_labels==0,:,0],axis=0)
  rezero = np.mean(mean[-50:])
  kdm5b_mean_data = (mean - rezero)/(1 - rezero)
  kdm5b_std_data = np.std( ((acc[real_labels==0,:,0]) - rezero)/(1-rezero) ,axis=0)

  #n_pts = len(acc[labels==1,:,0])
  mean = np.mean(acc[real_labels==1,:,0],axis=0)
  rezero = np.mean(mean[-50:])
  p300_mean_data = (mean - rezero)/(1 - rezero)
  p300_std_data = np.std( ((acc[real_labels==1,:,0]) - rezero)/(1-rezero) ,axis=0)

  n = 50
  sigma_model_p300 = p300_sem_model[1:n]/np.sqrt(n_model_traj)
  sigma_model_kdm5b = kdm5b_sem_model[1:n]/np.sqrt(n_model_traj)
  LL_p300 = -.5*np.log(2*np.pi)*n + np.sum(  -np.log(sigma_model_p300) -  ((p300_mean_data[1:n] - p300_mean_model[1:n])**2)/ (2*sigma_model_p300**2))
  LL_kdm5b = -.5*np.log(2*np.pi)*n + np.sum(  -np.log(sigma_model_kdm5b) -  ((kdm5b_mean_data[1:n] - kdm5b_mean_model[1:n])**2)/ (2*sigma_model_kdm5b**2))

  return LL_p300,LL_kdm5b, kdm5b_mean_model, p300_mean_model, kdm5b_mean_data, p300_mean_data,  sigma_model_kdm5b, sigma_model_p300



inds = []
lls = []
lls_p300 = []
lls_kdm5b = []
#for i in range(len(kdm5b_files)):
#  for j in range(len(p300_files)):
 #   lls.append(sum(get_LL(kdm5b_files[i], p300_files[j])))
#    inds.append([i,j])

for i in range(len(kdm5b_files)):
  lls_kdm5b.append(get_LL(kdm5b_files[i], p300_files[0]) )

for i in range(len(p300_files)):
  lls_p300.append(get_LL(kdm5b_files[0], p300_files[i]) )





def bootstrap_data(size, label, n=100):
  bootstrapped_data = []
  #X_train, X_test, y_train, y_test, X_witheld, y_witheld, Acc_train, Acc_test, Acc_witheld = mc.process_data(int_g, labels, norm='train', seed=seed, witheld = witheld, test_size = test_size, include_acc = True )
  for i in range(n):
    m = np.mean(Acc_train[y_train.flatten() == label][np.random.choice(2000, size=size, replace=False)][:,::10,:],axis=0)
    rezero_model = np.mean(m[-50:])
    m = (m - rezero_model)/(1 - rezero_model)
    s = np.std(  ((Acc_train[y_train.flatten() == label]-rezero_model)/(1-rezero_model))[np.random.choice(2000, size=size, replace=False)][:,::10,:],axis=0)
    bootstrapped_data.append( [ m, s, m+s, m-s
                               ])
  return np.array(bootstrapped_data)



ntraj=2500
ntimes=3000
Nsamples=5000
seed=42
witheld=0
test_size=0
#multiplexing_df1 = pd.read_csv('/content/drive/MyDrive/parsweep_kes_kdm5b_12.0.csv')
#multiplexing_df2 = pd.read_csv('/content/drive/MyDrive/parsweep_kes_p300_7.555555555555555.csv' )

multiplexing_df1 = pd.read_csv(path +'/'+kdm5b_files[3])
multiplexing_df2 = pd.read_csv(path +'/'+  p300_files[3])
int_g1 = multiplexing_df1['green_int_mean'].values.reshape([ntraj,ntimes])    
int_g2 = multiplexing_df2['green_int_mean'].values.reshape([ntraj,ntimes])    

t = np.linspace(0,len(int_g1) - 1,len(int_g1))  #time vector in seconds
labels = np.ones(int_g1.shape[0]*2)
labels[:int_g1.shape[0]] = 0
int_g = np.vstack((int_g1,int_g2))  #merge the files and then let it sort
labels = labels

int_g, labels = mc.even_shuffle_sample(int_g, labels, samples=[int(Nsamples/2), int(Nsamples/2)], seed=seed)
int_g = mc.slice_arr(int_g, 1, 1280)

X_train, X_test, y_train, y_test, X_witheld, y_witheld, Acc_train, Acc_test, Acc_witheld = mc.process_data(int_g, labels, norm='train', seed=seed, witheld = witheld, test_size = test_size, include_acc = True )


plt.figure(figsize = (8,3))
#n_pts = len(Acc_train_real[y_train_real.flatten().astype(int) == 0])
mean = np.mean(acc[real_labels==0,:,0],axis=0)
rezero = np.mean(mean[-64:])
mean = (mean - rezero)/(1 - rezero)
sem = np.std( ((acc[real_labels==0,:,0]) - rezero)/(1-rezero) ,axis=0)

tau_kdm5b = (np.mean(np.where(mean < 0.01)[0][:5])*10)
plt.plot([-5,100],[0,0],'-',color='gray',alpha=.8,label='_nolegend_')
#plt.plot([tau_kdm5b/10,tau_kdm5b/10],[-.2,1.1],'-',color='gray',alpha=.8,label='_nolegend_')
#plt.text(tau_kdm5b/10-4, .4, 'Tau = %is'%int(tau_kdm5b), rotation=90, color='gray', alpha=.8)

#sem = np.std(Acc_train_real[y_train_real.flatten().astype(int) == 0],axis=0)

plt.errorbar(np.linspace(0,127,128), mean, yerr=sem, zorder=0, capsize=1, ls='',marker='o', markersize=1.5, lw=1, color=colors[0])

model_mean = np.mean(Acc_train[y_train.flatten() == 0],axis=0)[::10]
rezero_model = np.mean(model_mean[-64:])
model_mean = (model_mean - rezero_model)/(1 - rezero_model)

d = bootstrap_data(27,0,n=100)
#plt.plot(np.mean(d[:,2,:,0],axis=0),color='gray', label='_nolegend_', alpha=.3)
#plt.plot(np.mean(d[:,3,:,0],axis=0), color='gray', label='_nolegend_', alpha=.3)
plt.fill_between(np.linspace(0,127,128), np.mean(d[:,2,:,0],axis=0), np.mean(d[:,3,:,0],axis=0),alpha=.5, color='gray', label='_nolegend_')

tau_kdm5b_model = (np.mean(np.where(model_mean < 0.01)[0][:5])*10)
plt.plot([-5,100],[0,0],'-',color='gray',alpha=.8,label='_nolegend_')
plt.plot([tau_kdm5b_model/10,tau_kdm5b_model/10],[-.2,1.1],'-',color='gray',alpha=.8,label='_nolegend_')
plt.text(tau_kdm5b_model/10-4, .4, 'Tau = %is'%int(tau_kdm5b_model), rotation=90, color='gray', alpha=.8)
plt.plot(model_mean,alpha=1, zorder=2, color='k')






plt.xlim([-5,100])
plt.ylim([-.2,1.1])
plt.title('KDM5B Autocorrelation')
plt.legend(['Model','Data'])
plt.xlabel(r'Delay, $\tau$, (10 s)')
plt.savefig('kdm5b_acc.svg')


plt.figure(figsize = (8,3))

#n_pts = len(Acc_train_real[y_train_real.flatten().astype(int) == 1])
mean = np.mean(acc[real_labels==1,:,0],axis=0)
rezero = np.mean(mean[-64:])
mean = (mean - rezero)/(1 - rezero)
sem = np.std( ((acc[real_labels==1,:,0]) - rezero)/(1-rezero) ,axis=0)
tau_p300 = (np.mean(np.where(mean < 0.01)[0][:5])*10)

plt.plot([-5,100],[0,0],'-',color='gray',alpha=.8,label='_nolegend_')
#plt.plot([tau_p300/10,tau_p300/10],[-.2,1.1],'-',color='gray',alpha=.8,label='_nolegend_')
#plt.text(tau_p300/10-4, .4, 'Tau = %is'%int(tau_p300), rotation=90, color='gray', alpha=.8)

#sem = np.std(Acc_train_real[y_train_real.flatten().astype(int) == 1],axis=0)

plt.errorbar(np.linspace(0,127,128), mean, yerr=sem,  zorder=0, capsize=1, ls='',marker='o', markersize=1.5, lw=1,  color=colors[2])

model_mean = np.mean(Acc_train[y_train.flatten() == 1],axis=0)[::10]
rezero_model = np.mean(model_mean[-64:])
model_mean = (model_mean - rezero_model)/(1 - rezero_model)
d = bootstrap_data(19,1,n=100)
#plt.plot(np.mean(d[:,2,:,0],axis=0),color='gray', label='_nolegend_', alpha=.3)
#plt.plot(np.mean(d[:,3,:,0],axis=0),color='gray', label='_nolegend_', alpha=.3)
plt.fill_between(np.linspace(0,127,128), np.mean(d[:,2,:,0],axis=0), np.mean(d[:,3,:,0],axis=0),alpha=.5, color='gray', label='_nolegend_')

tau_p300_model = (np.mean(np.where(model_mean < 0.01)[0][:5])*10)
plt.plot([-5,100],[0,0],'-',color='gray',alpha=.8,label='_nolegend_')
plt.plot([tau_p300_model/10,tau_p300_model/10],[-.2,1.1],'-',color='gray',alpha=.8,label='_nolegend_')
plt.text(tau_p300_model/10-4, .4, 'Tau = %is'%int(tau_p300_model), rotation=90, color='gray', alpha=.8)
plt.plot(model_mean,alpha=1, zorder=2, color='k')


plt.title('P300 Autocorrelation')
plt.legend(['Model','Data'])
plt.xlabel(r'Delay, $\tau$, (10 s)')
plt.xlim([-5,100])
plt.ylim([-.2,1.1])
if save_fig:
    plt.savefig('p300_acc.svg')


scaler = MinMaxScaler()
scaler.fit(int_g[:,::10])
int_g_real = np.vstack(green_intensities)
int_g_real = mc.slice_arr_reverse(int_g_real, 10, 128)
int_g_real_transformed = scaler.transform(int_g_real)

plt.figure(dpi=300)
x,bins = np.histogram(X_train[labels==0,::100,0].flatten(),bins=30)
plt.hist(int_g_real_transformed[real_labels==0,::100].flatten(),bins=bins,  density=True, fc=(7/255,59/255,76/255, 0.5));
plt.hist(int_g_real_transformed[real_labels==1,::100].flatten(),bins=bins, density=True, fc=(239/255,71/255,111/255, 0.5))
plt.hist(X_train[labels==0,::100,0].flatten(),bins=bins,  ec = colors[0], lw=3, density=True,fc=(0, 0, 1, 0.0), histtype='step');
plt.hist(X_train[labels==1,::100,0].flatten(),bins=bins, lw=3, ec = colors[2], density=True, fc=(0, 0, 1, 0.0), histtype='step')
plt.xlabel('Normalized Intensity (normalized by model data)')
plt.ylabel('Density')

plt.legend(['KDM5B Model', 'P300 Model', 'KDM5B Data', 'P300 Data'])
plt.title('Intensity Distribution')
if save_fig:
    plt.savefig('intensity_dist.svg')


fig5 = plt.figure(dpi=300, constrained_layout=True)
widths = [3, 1]
heights = [1, 1]
spec5 = fig5.add_gridspec(ncols=2, nrows=2, width_ratios=widths,
                          height_ratios=heights)


ax = fig5.add_subplot(spec5[0, 0])

mean = np.mean(acc[real_labels==0,:,0],axis=0)
rezero = np.mean(mean[-64:])
mean = (mean - rezero)/(1 - rezero)
sem = np.std( ((acc[real_labels==0,:,0]) - rezero)/(1-rezero) ,axis=0)

tau_kdm5b = (np.mean(np.where(mean < 0.01)[0][:5])*10)
ax.plot([-5,100],[0,0],'-',color='gray',alpha=.8,label='_nolegend_')
#plt.plot([tau_kdm5b/10,tau_kdm5b/10],[-.2,1.1],'-',color='gray',alpha=.8,label='_nolegend_')
#plt.text(tau_kdm5b/10-4, .4, 'Tau = %is'%int(tau_kdm5b), rotation=90, color='gray', alpha=.8)

#sem = np.std(Acc_train_real[y_train_real.flatten().astype(int) == 0],axis=0)

ax.errorbar(np.linspace(0,127,128), mean, yerr=sem, zorder=0, capsize=1, ls='',marker='o', markersize=1.5, lw=1, color=colors[0])

model_mean = np.mean(Acc_train[y_train.flatten() == 0],axis=0)[::10]
rezero_model = np.mean(model_mean[-64:])
model_mean = (model_mean - rezero_model)/(1 - rezero_model)

d = bootstrap_data(27,0,n=100)
#plt.plot(np.mean(d[:,2,:,0],axis=0),color='gray', label='_nolegend_', alpha=.3)
#plt.plot(np.mean(d[:,3,:,0],axis=0), color='gray', label='_nolegend_', alpha=.3)
ax.fill_between(np.linspace(0,127,128), np.mean(d[:,2,:,0],axis=0), np.mean(d[:,3,:,0],axis=0),alpha=.5, color='gray', label='_nolegend_')

tau_kdm5b_model = (np.mean(np.where(model_mean < 0.01)[0][:5])*10)
ax.plot([-5,100],[0,0],'-',color='gray',alpha=.8,label='_nolegend_')
ax.plot([tau_kdm5b_model/10,tau_kdm5b_model/10],[-.2,1.1],'-',color='gray',alpha=.8,label='_nolegend_')
ax.text(tau_kdm5b_model/10-4, .4, 'Tau = %is'%int(tau_kdm5b_model), rotation=90, color='gray', alpha=.8)
ax.plot(model_mean,alpha=1, zorder=2, color='k')

ax.set_xlim([-5,100])
ax.set_ylim([-.2,1.1])
ax.set_title('KDM5B Autocorrelation')
ax.legend(['Model','Data'])
plt.xlabel(r'Delay, $\tau$, (10 s)')

ax = fig5.add_subplot(spec5[1, 0])

mean = np.mean(acc[real_labels==1,:,0],axis=0)
rezero = np.mean(mean[-64:])
mean = (mean - rezero)/(1 - rezero)
sem = np.std( ((acc[real_labels==1,:,0]) - rezero)/(1-rezero) ,axis=0)

tau_kdm5b = (np.mean(np.where(mean < 0.01)[0][:5])*10)
ax.plot([-5,100],[0,0],'-',color='gray',alpha=.8,label='_nolegend_')
#plt.plot([tau_kdm5b/10,tau_kdm5b/10],[-.2,1.1],'-',color='gray',alpha=.8,label='_nolegend_')
#plt.text(tau_kdm5b/10-4, .4, 'Tau = %is'%int(tau_kdm5b), rotation=90, color='gray', alpha=.8)

#sem = np.std(Acc_train_real[y_train_real.flatten().astype(int) == 0],axis=0)

ax.errorbar(np.linspace(0,127,128), mean, yerr=sem, zorder=0, capsize=1, ls='',marker='o', markersize=1.5, lw=1, color=colors[2])

model_mean = np.mean(Acc_train[y_train.flatten() == 0],axis=0)[::10]
rezero_model = np.mean(model_mean[-64:])
model_mean = (model_mean - rezero_model)/(1 - rezero_model)

d = bootstrap_data(27,0,n=100)
#plt.plot(np.mean(d[:,2,:,0],axis=0),color='gray', label='_nolegend_', alpha=.3)
#plt.plot(np.mean(d[:,3,:,0],axis=0), color='gray', label='_nolegend_', alpha=.3)
ax.fill_between(np.linspace(0,127,128), np.mean(d[:,2,:,0],axis=0), np.mean(d[:,3,:,0],axis=0),alpha=.5, color='gray', label='_nolegend_')

tau_kdm5b_model = (np.mean(np.where(model_mean < 0.01)[0][:5])*10)
ax.plot([-5,100],[0,0],'-',color='gray',alpha=.8,label='_nolegend_')
ax.plot([tau_p300_model/10,tau_p300_model/10],[-.2,1.1],'-',color='gray',alpha=.8,label='_nolegend_')
ax.text(tau_p300_model/10-4, .4, 'Tau = %is'%int(tau_p300_model), rotation=90, color='gray', alpha=.8)
ax.plot(model_mean,alpha=1, zorder=2, color='k')

ax.set_xlim([-5,100])
ax.set_ylim([-.2,1.1])
ax.set_title('P300 Autocorrelation')
ax.legend(['Model','Data'])
plt.xlabel(r'Delay, $\tau$, (10 s)')


ax = fig5.add_subplot(spec5[0, 1])

x,bins = np.histogram(X_train[labels==0,::100,0].flatten(),bins=30)
ax.hist(int_g_real_transformed[real_labels==0,::100].flatten(),bins=bins,  density=True, fc=(7/255,59/255,76/255, 0.5), orientation='horizontal');
#plt.hist(int_g_real_transformed[real_labels==1,::100].flatten(),bins=bins, density=True, fc=(239/255,71/255,111/255, 0.5))
ax.hist(X_train[labels==0,::100,0].flatten(),bins=bins,  ec = colors[0], lw=3, density=True,fc=(0, 0, 1, 0.0), histtype='step', orientation='horizontal');
#plt.hist(X_train[labels==1,::100,0].flatten(),bins=bins, lw=3, ec = colors[2], density=True, fc=(0, 0, 1, 0.0), histtype='step')
#ax.set_ylabel('Normalized Intensity')
ax.set_xlabel('Density')
ax.set_xlim([0,5])
ax.legend(['KDM5B Model', 'KDM5B Data',])
ax.set_title('Intensity Distribution')


ax = fig5.add_subplot(spec5[1, 1])

x,bins = np.histogram(X_train[labels==1,::100,0].flatten(),bins=30)
ax.hist(int_g_real_transformed[real_labels==1,::100].flatten(),bins=bins,  density=True, fc=(239/255,71/255,111/255, 0.5), orientation='horizontal');
#plt.hist(int_g_real_transformed[real_labels==1,::100].flatten(),bins=bins, density=True, fc=(239/255,71/255,111/255, 0.5))
ax.hist(X_train[labels==1,::100,0].flatten(),bins=bins,  ec = colors[2], lw=3, density=True,fc=(0, 0, 1, 0.0), histtype='step', orientation='horizontal');
#plt.hist(X_train[labels==1,::100,0].flatten(),bins=bins, lw=3, ec = colors[2], density=True, fc=(0, 0, 1, 0.0), histtype='step')
#ax.set_ylabel('Normalized Intensity')
ax.set_xlabel('Density')
ax.set_xlim([0,5])
ax.legend(['P300 Model', 'KDM5B Data',])
ax.set_title('Intensity Distribution')
if save_fig:
    plt.savefig('fit.svg')


##############################

plt.figure(dpi=300, figsize= (4,8))
x,bins = np.histogram(X_train[labels==0,::100,0].flatten(),bins=30)
plt.hist(int_g_real_transformed[real_labels==0,::100].flatten(),bins=bins,  density=True, fc=(7/255,59/255,76/255, 0.5), orientation='horizontal');
#plt.hist(int_g_real_transformed[real_labels==1,::100].flatten(),bins=bins, density=True, fc=(239/255,71/255,111/255, 0.5))
plt.hist(X_train[labels==0,::100,0].flatten(),bins=bins,  ec = colors[0], lw=3, density=True,fc=(0, 0, 1, 0.0), histtype='step', orientation='horizontal');
#plt.hist(X_train[labels==1,::100,0].flatten(),bins=bins, lw=3, ec = colors[2], density=True, fc=(0, 0, 1, 0.0), histtype='step')
plt.ylabel('Normalized Intensity')
plt.xlabel('Density')

plt.legend(['KDM5B Model', 'KDM5B Data',])
plt.title('Intensity Distribution')
if save_fig:
    plt.savefig('intensity_dist_kdm5b.svg')


plt.figure(dpi=300,  figsize= (4,8))
x,bins = np.histogram(X_train[labels==1,::100,0].flatten(),bins=30)
#plt.hist(int_g_real_transformed[real_labels==0,::100].flatten(),bins=bins,  density=True, fc=(7/255,59/255,76/255, 0.5));
plt.hist(int_g_real_transformed[real_labels==1,::100].flatten(),bins=bins, density=True, fc=(239/255,71/255,111/255, 0.5), orientation='horizontal')
#plt.hist(X_train[labels==0,::100,0].flatten(),bins=bins,  ec = colors[0], lw=3, density=True,fc=(0, 0, 1, 0.0), histtype='step');
plt.hist(X_train[labels==1,::100,0].flatten(),bins=bins, lw=3, ec = colors[2], density=True, fc=(0, 0, 1, 0.0), histtype='step', orientation='horizontal')
plt.ylabel('Normalized Intensity')
plt.xlabel('Density')

plt.legend([ 'P300 Model', 'P300 Data'])
plt.title('Intensity Distribution')
if save_fig:
    plt.savefig('intensity_dist_p300.svg')


import tifffile as tiff
cell_0 = tiff.imread('../../datasets/P300_KDM5B_base_experiment_5_cells/base_experiment0.tif')[:1280:10]
cell_1 = tiff.imread('../../datasets/P300_KDM5B_base_experiment_5_cells/base_experiment1.tif')[:1280:10]
cell_2 = tiff.imread('../../datasets/P300_KDM5B_base_experiment_5_cells/base_experiment2.tif')[:1280:10]


def quantile_norm(movie, q ):  #quantile norm
    norm_movie = np.zeros(movie.shape)
    for i in range(movie.shape[-1]):
        chn_movie = movie[:,:,:,i]
        max_val = np.quantile(chn_movie, q)
        min_val = np.quantile(chn_movie, .005)
        norm_movie[:,:,:,i] = (chn_movie - min_val)/(max_val - min_val)
        
    norm_movie[norm_movie > 1] = 1
    norm_movie[norm_movie < 0] = 0
    return norm_movie

cell_0_norm = quantile_norm(cell_0,.99)
cell_0_norm[:,:,:,2] = 0
plt.imshow(cell_0_norm[0])
if save_fig:
    plt.savefig('cell0.svg')

plt.figure(dpi=200)
cell_1_norm = quantile_norm(cell_1,.99)
cell_1_norm[:,:,:,2] = 0
plt.imshow(cell_1_norm[0])
if save_fig:
    plt.savefig('cell1.svg')

plt.figure(dpi=200)
cell_2_norm = quantile_norm(cell_2,.99)
cell_2_norm[:,:,:,2] = 0
plt.imshow(cell_2_norm[0])

if save_fig:
    plt.savefig('cell2.svg')

multiplexing_df = pd.read_csv(files[0])
particle_id = 10
particle_df = multiplexing_df[multiplexing_df['particle'] == particle_id]
x,y = particle_df['x'].values[:1280:10], particle_df['y'].values[:1280:10]
particle_int_g = particle_df.background_int_mean_green.values[:1280:10] 
particle_int_r = particle_df.background_int_mean_red.values[:1280:10]
particle_int_b =  particle_df.background_int_mean_blue.values[:1280:10]
particle_int_g_spot = particle_df.green_int_mean.values[:1280:10] + particle_df.background_int_mean_green.values[:1280:10] 
particle_int_g_spot_nobg = particle_df.green_int_mean.values[:1280:10] 


sub = 10
width = 6
points = np.linspace(0,127,sub).astype(int)
fig,ax = plt.subplots(2,sub)
for i in range(sub):
    #ax[0,i].imshow(cell_0_norm[points[i], int(y[i] - width):int(y[i]+width), int(x[i] - width):int(x[i]+width),0 ], cmap='Reds_r')
    ax[0,i].imshow(cell_0_norm[points[i], int(y[points[i]] - width):int(y[points[i]]+width+1), int(x[points[i]] - width):int(x[points[i]]+width+1),: ], cmap='Greens_r')
    ax[0,i].set_axis_off()
    ax[1,i].imshow(cell_0_norm[points[i], int(y[points[i]] - width):int(y[points[i]]+width+1), int(x[points[i]] - width):int(x[points[i]]+width+1),1], cmap='Greens_r')
    ax[1,i].set_axis_off()

if save_fig:
    plt.savefig('spots.svg')

plt.figure(figsize=(10,3))
plt.plot(np.linspace(0,127,128), particle_int_r,'r');
plt.plot(np.linspace(0,127,128), particle_int_g,'k');
plt.plot(np.linspace(0,127,128), particle_int_g_spot,'#096e03');
plt.plot(np.linspace(0,127,128), particle_int_g_spot_nobg,'#47ff0f');
plt.xlabel('Time (10s frame rate)')
plt.ylabel('Intensity (simulated AU)')
plt.legend(['R bg', 'G bg', 'G bg + G spot', 'G spot only'])
plt.ylim([0,13000])
plt.plot(points, particle_int_g_spot[points], 'mo')
if save_fig:
    plt.savefig('int_trace.svg')



real_labels = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])

predicted_labels = np.array([False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False,  True,  True,
        True,  True,  True, False,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False,  True,  True,  True,  True,  True,  True,
        True, False,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False,  True,
       False,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True, False,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True],)


conf = confusion_matrix(real_labels, predicted_labels)

offset = .09
offset2 = -.06
print(conf)



cell_0_norm = quantile_norm(cell_0,.99)
cell_0_norm[:,:,:,2] = 0
plt.imshow(cell_0_norm[0])

multiplexing_df = pd.read_csv(files[0])
x0, y0 = multiplexing_df['x'].values.reshape([50,3000])[:,0],  multiplexing_df['y'].values.reshape([50,3000])[:,0]

colors2 = []
colors3 = []
for i in range(50):
    if i < 25:
        if predicted_labels[i] == real_labels[i]:
            colors2.append('#073b4c') 
            colors3.append('none')
        else:
            colors2.append('none')
            colors3.append('w')
    else:
        if predicted_labels[i] == real_labels[i]:
            colors2.append('#ff479d') 
            colors3.append('none')
        else:
            colors2.append('none')
            colors3.append('w')
            

        
plt.scatter(x0,y0,marker='o', edgecolors=colors2, facecolors='none', lw=2)
plt.scatter(x0,y0,marker='x', c=colors3, lw=2)
if save_fig:
    plt.savefig('cell0_labeled.svg')



plt.figure(dpi=200)
cell_1_norm = quantile_norm(cell_1,.99)
cell_1_norm[:,:,:,2] = 0

plt.imshow(cell_1_norm[0])

multiplexing_df = pd.read_csv(files[1])
x0, y0 = multiplexing_df['x'].values.reshape([50,3000])[:,0],  multiplexing_df['y'].values.reshape([50,3000])[:,0]

colors2 = []
colors3 = []
for i in range(50,100):
    if i < 75:
        if predicted_labels[i] == real_labels[i]:
            colors2.append('#073b4c') 
            colors3.append('none')
        else:
            colors2.append('none')
            colors3.append('w')
    else:
        if predicted_labels[i] == real_labels[i]:
            colors2.append('#ff479d') 
            colors3.append('none')
        else:
            colors2.append('none')
            colors3.append('w')
            

        
plt.scatter(x0,y0,marker='o', edgecolors=colors2, facecolors='none', lw=2)
plt.scatter(x0,y0,marker='x', c=colors3, lw=2)
if save_fig:
    plt.savefig('cell1_labeled.svg')

plt.figure(dpi=200)
cell_2_norm = quantile_norm(cell_2,.99)
cell_2_norm[:,:,:,2] = 0
plt.imshow(cell_2_norm[0])

multiplexing_df = pd.read_csv(files[2])
x0, y0 = multiplexing_df['x'].values.reshape([50,3000])[:,0],  multiplexing_df['y'].values.reshape([50,3000])[:,0]

colors2 = []
colors3 = []
for i in range(100,150):
    if i < 125:
        if predicted_labels[i] == real_labels[i]:
            colors2.append('#073b4c') 
            colors3.append('none')
        else:
            colors2.append('none')
            colors3.append('w')
    else:
        if predicted_labels[i] == real_labels[i]:
            colors2.append('#ff479d') 
            colors3.append('none')
        else:
            colors2.append('none')
            colors3.append('w')
            

        
plt.scatter(x0,y0,marker='o', edgecolors=colors2, facecolors='none', lw=2)
plt.scatter(x0,y0,marker='x', c=colors3, lw=2)
if save_fig:
    plt.savefig('cell2_labeled.svg')


##############################################################################
# Train a classifier on simulated data and apply to the 2 example cells
##############################################################################

Fr = 10
Nframes = 128
ntraj = 2500
ntimes = 3000
Nsamples = 5000
seed = 42
witheld = 1000
test_size = 0

model_file = './dp_example_simulated_data_32_5.h5'
retrain=False

path = '.'

p300_files = ['parsweep_kes_p300_2.0.csv',
              'parsweep_kes_p300_3.111111111111111.csv',
              'parsweep_kes_p300_4.222222222222222.csv',
              'parsweep_kes_p300_5.333333333333334.csv',
              'parsweep_kes_p300_6.444444444444445.csv',
              'parsweep_kes_p300_7.555555555555555.csv',
              'parsweep_kes_p300_9.777777777777779.csv',
              'parsweep_kes_p300_8.666666666666668.csv',
              'parsweep_kes_p300_10.88888888888889.csv',
              'parsweep_kes_p300_12.0.csv']


kdm5b_files = ['parsweep_kes_kdm5b_2.0.csv',
              'parsweep_kes_kdm5b_3.111111111111111.csv',
              'parsweep_kes_kdm5b_4.222222222222222.csv',
              'parsweep_kes_kdm5b_5.333333333333334.csv',
              'parsweep_kes_kdm5b_6.444444444444445.csv',
              'parsweep_kes_kdm5b_7.555555555555555.csv',
              'parsweep_kes_kdm5b_9.777777777777779.csv',
              'parsweep_kes_kdm5b_8.666666666666668.csv',
              'parsweep_kes_kdm5b_10.88888888888889.csv',
              'parsweep_kes_kdm5b_12.0.csv']

multiplexing_df1 = pd.read_csv('../../datasets/par_sweep_kes/' + kdm5b_files[3])
multiplexing_df2 = pd.read_csv('../../datasets/par_sweep_kes/'  + p300_files[3])
int_g1 = multiplexing_df1['green_int_mean'].values.reshape([ntraj,ntimes])    
int_g2 = multiplexing_df2['green_int_mean'].values.reshape([ntraj,ntimes])    

t = np.linspace(0,len(int_g1) - 1,len(int_g1))  #time vector in seconds
labels = np.ones(int_g1.shape[0]*2)
labels[:int_g1.shape[0]] = 0
int_g = np.vstack((int_g1,int_g2))  #merge the files and then let it sort
labels = labels

int_g, labels = mc.even_shuffle_sample(int_g, labels, samples=[int(Nsamples/2), int(Nsamples/2)], seed=seed)
int_g = mc.slice_arr(int_g, Fr, Nframes)

X_train, X_test, y_train, y_test, X_witheld, y_witheld, Acc_train, Acc_test, Acc_witheld = mc.process_data(int_g, labels, norm='train', seed=seed, witheld = witheld, test_size = test_size, include_acc = True )

# full model
def signal_model(input_size_1, kernel_size, filters):
    model = Sequential()
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, kernel_regularizer=l1_l2(l1=1e-5), padding="same", input_shape=(input_size_1, 1)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(AveragePooling1D(pool_size=2))  
    model.add(Flatten())
    return model

def freq_model(input_size_2, kernel_size, filters):
    model = Sequential()
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, kernel_regularizer=l1_l2(l1=1e-5), padding="same", input_shape=(input_size_2, 1)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(AveragePooling1D(pool_size=2))  
    model.add(Flatten())
    return model

def create_model(input_size_1, input_size_2, N_neurons, kernel_size, filters, lr):

    combi_input = keras.layers.Input(shape = (input_size_1+input_size_2,1) ) 
    
    Input_1 = Lambda(lambda x: x[:,:input_size_1,:])(combi_input)
    Input_2 = Lambda(lambda x: x[:,input_size_1:,:])(combi_input)

    signal_output = signal_model(input_size_1, kernel_size, filters)(Input_1)
    freq_output = freq_model(input_size_2, kernel_size, filters)(Input_2)

    cat_output = keras.layers.concatenate([signal_output, freq_output])

    dense_output =  Dense(N_neurons, kernel_regularizer=l1_l2(l1=1e-5), activation = LeakyReLU(alpha=0.3))(cat_output)
    model_out = Dense(1,activation='sigmoid')(dense_output)

    optmizer = keras.optimizers.Adam(lr=lr)
    model = keras.Model(inputs=[combi_input], outputs=model_out)
    model.compile(loss='binary_crossentropy', optimizer=optmizer, metrics=['accuracy'],)

    return model

X_TRAIN = np.concatenate((X_train, Acc_train),axis=1)
X_WITHELD = np.concatenate((X_witheld, Acc_witheld),axis=1)

#_TRAIN = Acc_train
#X_WITHELD = Acc_witheld    

seed = 7
np.random.seed(seed)
  

model_CV = KerasClassifier(build_fn=create_model, verbose=0)

if retrain:
  filters = [16, 32, 64]
  kernel_size = [3, 5, 7]
  batches = [16, 32, 64]
  epochs = [50, 100]
  lrs = [.001]
  neurons = [200]
  inputs_1 = [X_train.shape[1]]
  inputs_2 = [Acc_train.shape[1]]

  distributions = dict(input_size_1 = inputs_1, input_size_2 = inputs_2, kernel_size = kernel_size, filters = filters, epochs= epochs, batch_size= batches, lr=lrs, N_neurons = neurons)
  random = RandomizedSearchCV(model_CV, distributions, n_iter= 2, verbose= 0, n_jobs= 1, cv=3)
  random_result = random.fit(X_TRAIN, y_train)

  best_model = random_result.best_estimator_.model
  best_params = random_result.best_params_
  best_kernel = best_params['kernel_size']
  best_filter = best_params['filters']

  clf = random_result.best_estimator_
  acc = clf.score(X_WITHELD, y_witheld)
  model_path = os.path.join('.', 'drive', 'MyDrive', 'dp_'+ 'real_data' + '_'  + str(best_filter) + '_' + str(best_kernel)  + '.h5')
  best_model.save(model_path)   
else:
  #format: model_i_j_kernelsize_filters
  filters = int(model_file.split('_')[-2])
  kernel_size = int( model_file.split('_')[-1] .split('.')[0] )

  inputs_1 = [X_train.shape[1], Acc_train.shape[1]]
  best_model = create_model(X_train.shape[1], Acc_train.shape[1], 200, kernel_size, filters, .001)
  best_model.load_weights(model_file)

  clf = best_model

  y_pred = best_model.predict(X_WITHELD)
  y_pred = np.argmax(y_pred,axis=1).astype(int)
  y_pred_onehot = np.zeros((y_pred.size, y_pred.max() + 1))
  y_pred_onehot[np.arange(y_pred.size), y_pred] = 1

  acc = 1-np.sum(np.abs(y_pred_onehot- y_witheld))/len(y_pred_onehot)



plt.figure(dpi=300)
cmap = plt.get_cmap('YlGn')
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

new_cmap = truncate_colormap(cmap, 0.1, 1)
my_cmap = new_cmap(np.arange(cmap.N))
my_cmap[:, -1] = .9
my_cmap = ListedColormap(my_cmap)


b_mat = confusion_matrix(y_train,  clf.predict(np.concatenate((X_train, Acc_train),axis=1)).flatten() > .5)
b_mat = b_mat / b_mat.astype(np.float).sum(axis=1)
plt.matshow(b_mat, cmap =my_cmap,)

for i in range(2):
  for j in range(2):
    if i != j:
      plt.text(j,i, str(b_mat[i,j]),
                horizontalalignment='center',
                verticalalignment='center',
                size=10)   
    else:
      plt.text(j,i, str(b_mat[i,j]),
                horizontalalignment='center',
                verticalalignment='center',
                size=10, color='w')    
      
plt.gca().set_xticklabels(['','KDM5B','P300', ], fontsize=8)
plt.gca().set_yticklabels(['','KDM5B','P300',], fontsize=8)
plt.title('Prediction'); plt.ylabel('Actual')
###########################################
if save_fig:
    plt.savefig('conf_training_data.svg')


plt.figure(dpi=300)
cmap = plt.get_cmap('YlGn')
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

new_cmap = truncate_colormap(cmap, 0.1, 1)
my_cmap = new_cmap(np.arange(cmap.N))
my_cmap[:, -1] = .9
my_cmap = ListedColormap(my_cmap)


b_mat = confusion_matrix(y_train_real,  clf.predict(np.concatenate((np.expand_dims(int_g_real_transformed,axis=-1), Acc_train_real),axis=1)).flatten() > .5)
b_mat = b_mat / b_mat.astype(np.float).sum(axis=1)
plt.matshow(b_mat, cmap =my_cmap,)

for i in range(2):
  for j in range(2):
    if i != j:
      plt.text(j,i, str(np.round(b_mat[i,j],3)),
                horizontalalignment='center',
                verticalalignment='center',
                size=10)   
    else:
      plt.text(j,i, str(np.round(b_mat[i,j],3)),
                horizontalalignment='center',
                verticalalignment='center',
                size=10, color='w')    
      
plt.gca().set_xticklabels(['','KDM5B','P300', ], fontsize=8)
plt.gca().set_yticklabels(['','KDM5B','P300',], fontsize=8)
plt.title('Prediction'); plt.ylabel('Actual')
###########################################
if save_fig:
    plt.savefig('conf_sim_data.svg')


