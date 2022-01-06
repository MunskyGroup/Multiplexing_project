# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 13:16:44 2021

@author: willi
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import ListedColormap
import matplotlib
import pandas as pd

import argparse
import yaml


####################################
# plot a ML run directory's acc_mats
#
###################################

parser = argparse.ArgumentParser(description='get paths and save dir and acc_file_name')
parser.add_argument('--target_dir', dest='target_dir', type=str,)
args = parser.parse_args()
target_dir = args.target_dir

#target_dir = 'ML_run1'
acc_mat_name = 'acc_mat'
save_dir = 'plots'
format_type = 'png'
flipped = False

## load the metadata
with open(os.path.join('.',target_dir,"metadata.yaml"), "r") as stream:
    try:
        metadata = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
        
#fr = metadata['global_framerate'] 
#nframes = metadata['global_N_frames'] 
#time = fr*nframes
time = 3

kdm5b_L = 4647
p300_L = 7257
Lt = 1011
n_epitopes = 10

if not os.path.exists(os.path.join('.', target_dir,save_dir)):
    os.makedirs(os.path.join('.', target_dir,save_dir))


acc_mats = []
key_csvs = []
for root, subdirs, files in os.walk(os.path.join('.',target_dir)):
    for f in files:
        if 'key.csv' in f:
            key_csvs.append(os.path.join(root,f))
        if '.npy' in f and acc_mat_name in f:
            acc_mats.append(os.path.join(root,f))

print('found acc_mats:')
print(acc_mats)

print('_______________________')             
            
def metric(L1,L2,Lt1,Lt2,Ne1,Ne2, ki1,ki2,ke1,ke2,time):
    '''
    

    Parameters
    ----------
    L1 : int
        length 1 (nt).
    L2 : int
        length 2 (nt).
    Lt1 : int
        tag length 1 (nt).
    Lt2 : int
        tag length 2 (nt).
    Ne1 : int
        number epitopes 1.
    Ne2 : int
        number epitopes 2.
    ki1 : float
        iniation rate 1.
    ki2 : float
        iniation rate 2.
    ke1 : float
        elongation rate 1.
    ke2 : TYPE
        elongation rate 2.
    time : float
        trajectory length in seconds.

    Returns
    -------
    float
        LEARNABILITY METRIC from 0 (identical processes) to > 1 for learnable.

    '''
    # fold difference in characteristic wait times (frequency) from spot1 to spot2
    # fold difference in expected average intensity from spot1 to spot2
    # scale this metric by amount of information (completed translations)
    
    #effective length of construct (ribosomes past center of tag)
    L1a = (L1 - Lt1/2)/3 
    L2a = (L2 - Lt2/2)/3
    
    #characteristic wait time 
    tau1 = L1a/ke1
    tau2 = L2a/ke2 
    
    #average intensity (ne = number epitopes)
    int_1 = ki1*tau1*Ne1
    int_2 = ki2*tau2*Ne2
    
    intfold = max((int_1/int_2), int_2/int_1) #fold changes
    taufold = max((tau2/tau1), tau1/tau2)

    #scale by amount of information ~ number of complete translation events per trajectory
    expected_complete_translations_1 = (time-tau1)*ki1
    expected_complete_translations_2 = (time-tau2)*ki2
    nrib_threshold = 10
    s1 = min(expected_complete_translations_1, nrib_threshold) / nrib_threshold
    s2 = min(expected_complete_translations_2, nrib_threshold) / nrib_threshold
    
    return abs(2 - (taufold+intfold))*s1*s2 

def make_plot(acc_mat_file, key_file, save_name, format_type):
    
    xlabel = ''
    ylabel = ''
    
    if '_kes' in save_name:
        xlabel = r'$k_{elongation}$ gene 1'
        ylabel = r'$k_{elongation}$ gene 2'

    if '_kis' in save_name:
        xlabel = r'$k_{initation}$ gene 1'
        ylabel = r'$k_{initation}$ gene 2'    

    if '_keki' in save_name:
        xlabel = r'$k_{initation}$'
        ylabel = r'$k_{elongation}$'    

    if '_img' in save_name:
        xlabel = r'$framerate$ (s)'
        ylabel = r'$time$ (s)'   
    
    key_df = pd.read_csv(key_file)
    
    horizontal_labels = list(key_df.columns[1:])
    vertical_labels = [str(x) for x in key_df['Unnamed: 0']]
    
    try:
        float(horizontal_labels[0])
        horizontal_labels = [str(np.round(float(x),2)) for x in horizontal_labels]
        float(vertical_labels[0])
        vertical_labels = [str(np.round(float(x),2)) for x in vertical_labels]
    except:
        pass
    
    acc_mat = np.load(acc_mat_file)

    cmap = plt.cm.viridis_r
    my_cmap = cmap(np.arange(cmap.N))
    my_cmap[:, -1] = .9
    my_cmap = ListedColormap(my_cmap)

    if flipped:
        acc_mat_plot = np.flipud(acc_mat.T)    
    else:
        acc_mat_plot = acc_mat  

    fig,ax = plt.subplots(1,1,dpi=120) #facecolor = '#0F1217')
    
    amat = acc_mat_plot
    
    if '_cl' in save_name:
        A = np.random.randint(0, 1, 100).reshape(10, 10)
        
        mask =  np.tri(A.shape[0], k=-1)
        if flipped:
            mask = mask.T
        acc_mat_plot = np.ma.array(acc_mat_plot,mask=mask)
    
    b = ax.imshow(acc_mat_plot,cmap =my_cmap)
    ax.set_yticks(np.arange(10),)
    ax.set_xticks( np.arange(10))
    
    
    ax.set_yticklabels(vertical_labels, fontdict = {'fontsize':7})
    ax.set_xticklabels(horizontal_labels, fontdict = {'fontsize':7},rotation=45)
    fig.colorbar(b)
    #ax.plot([.05],[7.55555],'r*',markersize=10)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    ax.set_title('Test Acc (train on each combo)')
    
    if '_cl' in save_name:
        for x in range(10):
            for y in range(10):
                if not flipped:
                    if x <= y:
                        ax.text(y,x, '%.2f' % amat[x, y],
                                 horizontalalignment='center',
                                 verticalalignment='center',
                                 size=5)
                else:
                    if x >= y:
                        ax.text(y,x, '%.2f' % amat[x, y],
                                 horizontalalignment='center',
                                 verticalalignment='center',
                                 size=5)                    
                    
    else:
        for x in range(10):
            for y in range(10):
                ax.text(y,x, '%.2f' % amat[x, y],
                         horizontalalignment='center',
                         verticalalignment='center',
                         size=5)
    
    plt.savefig(save_name, format=format_type)



for acc in acc_mats:
    print('making heatmap....')
    print(acc)
    
    
    
    save_name = os.path.split(acc)[1].split('.')[0]
    file_path = os.path.split(acc)[0]
    key_file = save_name.split('_')[-1] + '_key.csv'
    key_file = os.path.join(file_path,key_file)
    make_plot(acc, key_file, os.path.join('.',target_dir,save_dir,save_name) + '.' +format_type, format_type)

