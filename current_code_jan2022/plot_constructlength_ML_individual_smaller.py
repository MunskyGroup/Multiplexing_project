# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 11:24:40 2021

@author: willi
"""

import numpy as np
import matplotlib.pyplot as plt


names1 = ['GOLT1A','CDC42','MYOZ3','GPN2','CD46','LRRC42','SPATA6','SCP2','BBS5','CAMK2B']


acc_mat_parsweep5000 = np.load('./acc_mat_cl_smaller.npy')
#acc_mat_parsweep5000 = np.flipud(acc_mat_parsweep5000)


x,y = np.tril_indices(10,-1)
xx = x
yy = y
vals = acc_mat_parsweep5000[np.triu_indices(10,1)][::-1]

acc_mat_parsweep5000 = np.flipud(acc_mat_parsweep5000.T)
columns = np.linspace(1,9,9).astype(int)[::-1]

#for i in range()
ksum = 0
for i in range(len(columns)):
    acc_mat_parsweep5000[columns[i], i+1:] = vals[::-1][ksum:ksum+columns[i]] 
    print(columns[i],i+1)
    print(vals[::-1][ksum:ksum+columns[i]] )
    ksum+=columns[i]

from matplotlib.colors import ListedColormap
cmap = plt.cm.viridis_r
my_cmap = cmap(np.arange(cmap.N))
my_cmap[:, -1] = .7
my_cmap = ListedColormap(my_cmap)


fig,ax = plt.subplots(1,1,dpi=120)
b = ax.imshow(acc_mat_parsweep5000,cmap =my_cmap)
ax.set_yticks(np.arange(10),)
ax.set_xticks( np.arange(10))


ax.set_yticklabels([x for x in names1][::-1], fontdict = {'fontsize':7})
ax.set_xticklabels([x for x in names1], fontdict = {'fontsize':7},rotation=45)
fig.colorbar(b)
#ax.plot([.05],[7.55555],'r*',markersize=10)
ax.set_xlabel('Gene 1')
ax.set_ylabel('Gene 2')
ax.set_title('Test Accuracy over construct lengths')
amat = acc_mat_parsweep5000
#amat = np.flipud(acc_mat_parsweep5000.T)
for x in range(10):
    for y in range(10):
        
        ax.text(y,x, '%.2f' % amat[x, y],
                 horizontalalignment='center',
                 verticalalignment='center',
                 size=5)


sizes = [1200, 1734, 2265, 2799, 3333, 3867, 4401, 4932, 5466, 6000]
diff_mat = np.zeros([10,10])

for i in range(10):
    for j in range(10):
        diff = max(sizes[j] /sizes[i], sizes[i]/sizes[j])
        if diff != 0:
            diff_mat[i,j] = diff

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

ax.set_yticklabels([x for x in names1][::-1], fontdict = {'fontsize':7})
ax.set_xticklabels([x for x in names1], fontdict = {'fontsize':7},rotation=45)

#ax.plot([.05],[7.55555],'r*',markersize=10)
ax.set_xlabel('Gene 1')
ax.set_ylabel('Gene 2')
ax.set_title('Fold change length (NT)')