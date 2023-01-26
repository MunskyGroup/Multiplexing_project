# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 16:21:04 2022

@author: willi
"""

import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import pandas as pd
########################################
dark = False
if not dark: ##06d6a0, ef476f
    colors = ['#073b4c', '#8a3619','#06d6a0','#7400b8','#073b4c', '#118ab2',]
else:
    plt.style.use('dark_background')
    plt.rcParams.update({'axes.facecolor'      : '#131313'  , 
'figure.facecolor' : '#131313' ,
'figure.edgecolor' : '#131313' , 
'savefig.facecolor' : '#131313'  , 
'savefig.edgecolor' :'#131313'})


    colors = ['#118ab2','#57ffcd', '#ff479d', '#ffe869','#ff8c00','#04756f']

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 12}

save = False

plt.rcParams.update({'font.size': 12, 'font.weight':'bold','font.family':'normal'  }   )
plt.rcParams.update({'axes.prop_cycle':cycler(color=colors)})

plt.rcParams.update({'axes.prop_cycle':cycler(color=colors)})
plt.rcParams.update({'axes.prop_cycle':cycler(color=colors)})


plt.rcParams.update({'xtick.major.width'   : 2.8 })
plt.rcParams.update({'xtick.labelsize'   : 12 })



plt.rcParams.update({'ytick.major.width'   : 2.8 })
plt.rcParams.update({'ytick.labelsize'   : 12})

plt.rcParams.update({'axes.titleweight'   : 'bold'})
plt.rcParams.update({'axes.titlesize'   : 10})
plt.rcParams.update({'axes.labelweight'   : 'bold'})
plt.rcParams.update({'axes.labelsize'   : 12})

plt.rcParams.update({'axes.linewidth':2.8})
plt.rcParams.update({'axes.labelpad':8})
plt.rcParams.update({'axes.titlepad':10})
plt.rcParams.update({'figure.dpi':300})


f = '../../ML_run_320_5s_wfreq/parsweep_keki_ML/keki_key.csv'
key_file = pd.read_csv(f)
convert_str = lambda tstr: [x.replace(' ','').replace('(','').replace(')','').replace("'",'') for x in tuple(tstr.split(','))] 

acc_mat = np.zeros([10,10])
xlabels = [0,]*10
ylabels = [0,]*10
for i in range(0,10):
    for j in range(0,10):
        ind1,ind2, x,y,acc = convert_str(key_file.iloc[i][j+1])
       
        acc_mat[i,j] = acc
        xlabels[int(ind1)] = x
        ylabels[int(ind2)] = y
            
p = [(0, 2), (1, 4), (2, 5), (3, 6), (4, 8), (5, 8)]
Ls = [1200, 1200, 1734, 2265, 2799, 3333, 3867, 4401, 4932, 5466, 6000]
Ldiffs = [1065, 1599, 1602, 1602, 2133, 1599]
accs = [.904, .938, .85, .872, .904, .915]



total_ribosomes_watched_fold = np.zeros([10,10])
for i in range(0,10):
    for j in range(0,10):
        ind1,ind2, x,y,acc = convert_str(key_file.iloc[i][j+1])
       
        acc_mat[i,j] = acc
        xlabels[int(ind1)] = x
        ylabels[int(ind2)] = y
            

ke_ratio = []
accuracy = []
for i in range(10):
    for j in range(10):
        
        ki = float(xlabels[i])
        ke = float(ylabels[j])
        
        p300 =  (1 - (1011)/(2*8268))*(2756/ke)*ki
        kdm5b =  (1 - (1011)/(2*5658))*(5658/3/ke)*ki
        #int_fold = 2756/ke/320
        ke_ratio.append(np.mean([p300,kdm5b]))
        accuracy.append(acc_mat[i,j])

def movmean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

a = .3
n = 10

plt.figure()
plt.semilogx(ke_ratio,accuracy,'o', alpha = a)

plt.semilogx(movmean(np.array(ke_ratio)[np.argsort(ke_ratio).astype(int)],n), movmean(np.array(accuracy)[np.argsort(ke_ratio).astype(int)],n),label='_nolegend_',color=colors[0] )


#plt.semilogx([1/1.4612937433722164, 1/1.4612937433722164],[0.5,1],'b--', label='_nolegend_')
#plt.semilogx([.08, 10],[.8,.8],'k-', label='_nolegend_')
#plt.plot([0,3.5],[.8,.8],'b--')
#plt.plot([0.5,.5],[.5,1],'b--')




f = '../../ML_run_1280_5s_wfreq/parsweep_keki_ML/keki_key.csv'
key_file = pd.read_csv(f)
convert_str = lambda tstr: [x.replace(' ','').replace('(','').replace(')','').replace("'",'') for x in tuple(tstr.split(','))] 

acc_mat = np.zeros([10,10])
xlabels = [0,]*10
ylabels = [0,]*10
for i in range(0,10):
    for j in range(0,10):
        ind1,ind2, x,y,acc = convert_str(key_file.iloc[i][j+1])
       
        acc_mat[i,j] = acc
        xlabels[int(ind1)] = x
        ylabels[int(ind2)] = y
ke_ratio = []
accuracy = []
for i in range(10):
    for j in range(10):
        
        ki = float(xlabels[i])
        ke = float(ylabels[j])
        
        p300 =  (1 - (1011)/(2*8268))*(2756/ke)*ki
        kdm5b =  (1 - (1011)/(2*5658))*(5658/3/ke)*ki
        #int_fold = 2756/ke/320
        ke_ratio.append(np.mean([p300,kdm5b]) )
        accuracy.append(acc_mat[i,j])


plt.semilogx(ke_ratio,accuracy,'o', alpha=a)
plt.semilogx(movmean(np.array(ke_ratio)[np.argsort(ke_ratio).astype(int)],n), movmean(np.array(accuracy)[np.argsort(ke_ratio).astype(int)],n),label='_nolegend_',color=colors[1] )


f = '../../ML_run_3000_2s_wfreq/parsweep_keki_ML/keki_key.csv'
key_file = pd.read_csv(f)
convert_str = lambda tstr: [x.replace(' ','').replace('(','').replace(')','').replace("'",'') for x in tuple(tstr.split(','))] 

acc_mat = np.zeros([10,10])
xlabels = [0,]*10
ylabels = [0,]*10
for i in range(0,10):
    for j in range(0,10):
        ind1,ind2, x,y,acc = convert_str(key_file.iloc[i][j+1])
       
        acc_mat[i,j] = acc
        xlabels[int(ind1)] = x
        ylabels[int(ind2)] = y
ke_ratio = []
accuracy = []
for i in range(10):
    for j in range(10):
        
        ki = float(xlabels[i])
        ke = float(ylabels[j])
        
        int_fold =  (1 - (1011)/(2*8268))*(2756/ke)*ki
        #int_fold = 2756/ke/320
        
        ke_ratio.append(int_fold)
        accuracy.append(acc_mat[i,j])


plt.semilogx(ke_ratio,accuracy,'o', alpha=a)
plt.semilogx(movmean(np.array(ke_ratio)[np.argsort(ke_ratio).astype(int)],n), movmean(np.array(accuracy)[np.argsort(ke_ratio).astype(int)],n),label='_nolegend_',color=colors[2] )


plt.semilogx([.08, 200],[.8,.8],'k-', label='_nolegend_')
plt.xlim([1,125])
plt.ylim([0.6,1.05])
#plt.xlabel('Times KDM5B dwell time sampled')
plt.xlabel('Average Spot Intensity (UMP)')
#lt.annotate('0.684', xy=(1/1.4612937433722164 - .1, .5), xytext=(.2, .6),
#            arrowprops=dict(facecolor='black', shrink=0.0005,width=.4))
#plt.xlim([0.08,15])
plt.ylabel('Test Accuracy')
plt.legend(['64 frames, 5s FR', '128 frames, 5s FR','1500 frames, 2s FR'])
plt.title(r'Both mRNAs same $k_i$ and $k_e$ dataset')
plt.savefig('./keki_relation.svg')
