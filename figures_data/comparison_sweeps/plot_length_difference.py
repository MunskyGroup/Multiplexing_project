# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 23:03:19 2022

@author: willi
"""

import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
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



cl_acc_mat = np.load('../../ML_run_320_5s_wfreq/parsweep_cl_ML/acc_mat_cl.npy')

p = [(0, 2), (1, 4), (2, 5), (3, 6), (4, 8), (5, 8)]
Ls = [ 1200, 1734, 2265, 2799, 3333, 3867, 4401, 4647, 4932, 5466, 6000, 7257]
Ldiffs = [1065, 1599, 1602, 1602, 2133, 1599]
accs = [.904, .938, .85, .872, .904, .915]


def movmean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

#plt.plot(accs,Ldiffs,'o')

n = 10
a = .3
#cum_L = np.array([ 0, 534, 1065, 1599, 2133, 2667, 3201, 3732, 4266, 4800])

acc_to_plot = []
ldiffs = []
lfold = []
x = cl_acc_mat - .5
for i in range(12):
    current_L = Ls[i]
    #cum_L = np.array([0,] + np.cumsum(    ((np.array(Ls)[1:] - np.array(Ls)[:-1])[i:] )/Ls[i+1]).tolist())
    tmp_Ls = np.array((np.array(Ls)[i:]/ current_L).tolist()   ) 
    
    acc_to_plot.append(cl_acc_mat[i,:][np.where(cl_acc_mat[i,:] > 0)].tolist())
    #ldiffs.append(cum_L[np.where(cl_acc_mat[i,:] > 0)[0]-i].tolist())
    lfold.append(tmp_Ls[np.where(cl_acc_mat[i,:] > 0)[0] - i].tolist())

#ldiffs = [item for sublist in ldiffs for item in sublist]
lfold = [item for sublist in lfold for item in sublist]
acc_to_plot = [item for sublist in acc_to_plot for item in sublist]

lfold_5 = lfold
acc_to_plot_5 = acc_to_plot

#plt.plot(acc_to_plot,ldiffs,'o')
plt.figure()
plt.plot(lfold,acc_to_plot,'o', alpha = a)

plt.plot(movmean(np.array(lfold)[np.argsort(lfold).astype(int)],n), movmean(np.array(acc_to_plot)[np.argsort(lfold).astype(int)],n),label='_nolegend_',color=colors[0] )

plt.plot([.8,3.5],[.8,.8],'k-',label='_nolegend_')
plt.plot([1.40,1.4],[.4,1.1],'b--',label='_nolegend_')



cl_acc_mat = np.load('../../ML_run_1280_5s_wfreq/parsweep_cl_ML/acc_mat_cl.npy')

p = [(0, 2), (1, 4), (2, 5), (3, 6), (4, 8), (5, 8)]
Ls = [ 1200, 1734, 2265, 2799, 3333, 3867, 4401, 4647, 4932, 5466, 6000, 7257]


#plt.plot(accs,Ldiffs,'o')


#cum_L = np.array([ 0, 534, 1065, 1599, 2133, 2667, 3201, 3732, 4266, 4800])

acc_to_plot = []
ldiffs = []
lfold = []
x = cl_acc_mat - .5
for i in range(12):
    current_L = Ls[i]
    #cum_L = np.array([0,] + np.cumsum(    ((np.array(Ls)[1:] - np.array(Ls)[:-1])[i:] )/Ls[i+1]).tolist())
    tmp_Ls = np.array((np.array(Ls)[i:]/ current_L).tolist()   ) 
    
    acc_to_plot.append(cl_acc_mat[i,:][np.where(cl_acc_mat[i,:] > 0)].tolist())
    #ldiffs.append(cum_L[np.where(cl_acc_mat[i,:] > 0)[0]-i].tolist())
    lfold.append(tmp_Ls[np.where(cl_acc_mat[i,:] > 0)[0] - i].tolist())

#ldiffs = [item for sublist in ldiffs for item in sublist]
lfold = [item for sublist in lfold for item in sublist]
acc_to_plot = [item for sublist in acc_to_plot for item in sublist]

lfold_10 = lfold
acc_to_plot_10 = acc_to_plot


plt.plot(lfold,acc_to_plot,'o',alpha=a)
plt.plot(movmean(np.array(lfold)[np.argsort(lfold).astype(int)],n), movmean(np.array(acc_to_plot)[np.argsort(lfold).astype(int)],n),label='_nolegend_',color=colors[1] )








cl_acc_mat = np.load('../../ML_run_3000_2s_wfreq/parsweep_cl_ML/acc_mat_cl.npy')

p = [(0, 2), (1, 4), (2, 5), (3, 6), (4, 8), (5, 8)]

#plt.plot(accs,Ldiffs,'o')


#cum_L = np.array([ 0, 534, 1065, 1599, 2133, 2667, 3201, 3732, 4266, 4800])

acc_to_plot = []
ldiffs = []
lfold = []
x = cl_acc_mat - .5
for i in range(12):
    current_L = Ls[i]
    #cum_L = np.array([0,] + np.cumsum(    ((np.array(Ls)[1:] - np.array(Ls)[:-1])[i:] )/Ls[i+1]).tolist())
    tmp_Ls = np.array((np.array(Ls)[i:]/ current_L).tolist()   ) 
    
    acc_to_plot.append(cl_acc_mat[i,:][np.where(cl_acc_mat[i,:] > 0)].tolist())
    #ldiffs.append(cum_L[np.where(cl_acc_mat[i,:] > 0)[0]-i].tolist())
    lfold.append(tmp_Ls[np.where(cl_acc_mat[i,:] > 0)[0] - i].tolist())

#ldiffs = [item for sublist in ldiffs for item in sublist]
lfold = [item for sublist in lfold for item in sublist]
acc_to_plot = [item for sublist in acc_to_plot for item in sublist]

lfold_50 = lfold
acc_to_plot_50 = acc_to_plot


plt.plot(lfold,acc_to_plot,'o',alpha=a)
plt.plot(movmean(np.array(lfold)[np.argsort(lfold).astype(int)],n), movmean(np.array(acc_to_plot)[np.argsort(lfold).astype(int)],n),label='_nolegend_',color=colors[2] )


#plt.plot([0,3.5],[.8,.8],'r--',label='_nolegend_')
#plt.plot([1.12,1.12],[.4,1.1],'--',color = '#37ad72', label='_nolegend_')
#plt.plot([1.14,1.14],[.4,1.1],'r--',label='_nolegend_')
plt.annotate('1.40', xy=(1.42, .5), xytext=(2, .7),
            arrowprops=dict(facecolor='blue', edgecolor='blue', shrink=0.0005,width=.4))
plt.legend(['64 frames, 5s FR', '128 frames, 5s FR','1500 frames, 2s FR'])
plt.xlabel('Length Fold Difference $(L_{mRNA,2} / L_{mRNA,1})$')
plt.ylabel('Test Accuracy')
plt.title('mRNAs with differing $L_{mRNA}$ values dataset')
plt.xlim([.8,3])
plt.ylim([0.45,1.05])
plt.savefig('./ldiff_relation.svg')


