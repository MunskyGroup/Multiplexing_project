# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 11:24:40 2021

@author: willi
"""

import numpy as np
import matplotlib.pyplot as plt


names2 = ['RRAGC','ORC2','LONRF2', 'EDEM3','TRIM33','MAP3K6','COL3A1','KDM6B','PHIP','DOCK8' ]

acc_mat_parsweep5000 = np.load('./acc_mat_cl.npy')
#acc_mat_parsweep5000 = np.flipud(acc_mat_parsweep5000)




from matplotlib.colors import ListedColormap
cmap = plt.cm.viridis_r
my_cmap = cmap(np.arange(cmap.N))
my_cmap[:, -1] = .7
my_cmap = ListedColormap(my_cmap)


def learnable_metric(L1,L2,Lt1,Lt2,Ne1,Ne2, ki1,ki2,ke1,ke2,):
    L1a = (L1 - Lt1/2)/3
    L2a = (L2 - Lt2/2)/3
    
    tau1 = L1a/ke1
    tau2 = L2a/ke2 
    
    int_1 = ki1*tau1*Ne1
    int_2 = ki2*tau2*Ne2
    
    intfold = max((int_1/int_2), int_2/int_1)
    taufold = max((tau2/tau1), tau1/tau2)

    return abs(2 - (taufold+intfold))


diff_mat = np.zeros([10,10])
sizes = [1200, 1734, 2265, 2799, 3333, 3867, 4401, 4932, 5466, 6000]

for i in range(10):
    for j in range(10):
                
        diff_mat[i,j] = learnable_metric(sizes[i],sizes[j],1011,1011,10,10, .06,.06,5.33,5.33) 
            



fig,ax = plt.subplots(1,1,dpi=120)
b = ax.imshow(np.flipud(acc_mat_parsweep5000.T),cmap =my_cmap)
ax.set_yticks(np.arange(10),)
ax.set_xticks( np.arange(10))


ax.set_yticklabels([x for x in names2][::-1], fontdict = {'fontsize':7})
ax.set_xticklabels([x for x in names2], fontdict = {'fontsize':7},rotation=45)
fig.colorbar(b)
#ax.plot([.05],[7.55555],'r*',markersize=10)
ax.set_xlabel('Gene 1')
ax.set_ylabel('Gene 2')
ax.set_title('Test Accuracy over construct lengths')
amat = np.flipud(acc_mat_parsweep5000.T)
for x in range(10):
    for y in range(10):
        
        ax.text(y,x, '%.2f' % amat[x, y],
                 horizontalalignment='center',
                 verticalalignment='center',
                 size=5)

import matplotlib
for x in range(10):
    for y in range(10):
        if np.flipud(diff_mat.T)[x,y] < 1:
            rect1 = matplotlib.patches.Rectangle((y-.5,x-.5), 1,1,
                     edgecolor='r',fill=0)
            ax.add_patch(rect1)
            




fig,ax = plt.subplots(1,1,dpi=120)
b = ax.imshow(np.flipud(diff_mat.T),cmap =my_cmap)
ax.set_yticks(np.arange(10),)
ax.set_xticks( np.arange(10))
for x in range(10):
    for y in range(10):
        
        ax.text(y,x, '%.2f' % np.flipud(diff_mat.T)[x, y],
                 horizontalalignment='center',
                 verticalalignment='center',
                 size=5)

ax.set_yticklabels([x for x in names2][::-1], fontdict = {'fontsize':7})
ax.set_xticklabels([x for x in names2], fontdict = {'fontsize':7},rotation=45)

#ax.plot([.05],[7.55555],'r*',markersize=10)
ax.set_xlabel('Gene 1')
ax.set_ylabel('Gene 2')
ax.set_title('Fold change length (NT)')
import matplotlib
for x in range(10):
    for y in range(10):
        if np.flipud(diff_mat.T)[x,y] < 1:
            rect1 = matplotlib.patches.Rectangle((y-.5,x-.5), 1,1,
                     edgecolor='r',fill=0)
            ax.add_patch(rect1)
            