# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 19:57:22 2022

@author: willi
"""
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
########################################
dark = False
if not dark:
    colors = ['#073b4c', '#ef476f','#06d6a0','#7400b8','#073b4c', '#118ab2',]
else:
    plt.style.use('dark_background')
    plt.rcParams.update({'axes.facecolor'      : '#131313'  , 
'figure.facecolor' : '#131313' ,
'figure.edgecolor' : '#131313' , 
'savefig.facecolor' : '#131313'  , 
'savefig.edgecolor' :'#131313'})


    colors = ['#118ab2','#57ffcd', '#04756f', '#ff479d', '#ffe869','#ff8c00',]

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



acc_base = np.load('../../ML_experiments/ML_run_tag_base/parsweep_img_ML/acc_mat_img.npy')[0][0]
acc_3prime = np.load('../../ML_experiments/ML_run_tag_3prime/parsweep_img_ML/acc_mat_img.npy')[0][0]
acc_split = np.load('../../ML_experiments/ML_run_tag_split/parsweep_img_ML/acc_mat_img.npy')[0][0]
acc_minus5 = np.load('../../ML_experiments/ML_run_tag_minus5/parsweep_img_ML/acc_mat_img.npy')[0][0]
acc_plus5 = np.load('../../ML_experiments/ML_run_tag_plus5/parsweep_img_ML/acc_mat_img.npy')[0][0]

fig, ax = plt.subplots(1,1,dpi=300)
ax.bar([0,1,2,3,4],[acc_base, acc_3prime, acc_split, acc_minus5, acc_plus5])
ax.set_xticks([0,1,2,3,4])
ax.set_xticklabels(['base',"3' tag", 'split tag','minus 5 epitopes','plus 5 epitopes'], rotation=90)





fig, ax = plt.subplots(figsize=(3,8))
ax.barh([2],[acc_base], facecolor=colors[0], height=.4)
ax.barh([1.5],[acc_split], left=[0], facecolor=colors[1],height=.4 )
ax.barh([1],[acc_plus5 ], left=[0], facecolor=colors[2], height=.4 )
ax.barh([.5],[acc_minus5 ], left=[0] , facecolor=colors[3], height=.4)
ax.barh([0],[acc_3prime], left = [0], facecolor=colors[5], height=.4 )
plt.plot([1,1],[-.5,4.5],'g--')
plt.plot([.5,.5],[-.5,4.5],'r--')
plt.ylim([-.45,2.25])
plt.xlim([.45,1.03])

plt.text(.52, 1.95, '50%')
plt.text(.6, 1.45, '82.1%')
plt.text(.65, .95, '83.4%')
plt.text(.65, .45, '93.8%')
plt.text(.65, -.05, '99.8%')

plt.savefig('./tagging_scheme.svg')


