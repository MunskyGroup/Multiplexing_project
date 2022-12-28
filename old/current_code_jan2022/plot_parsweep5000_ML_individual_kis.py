# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 10:27:10 2021

@author: willi
"""
import numpy as np
import matplotlib.pyplot as plt

kis = ['0.1','0.09000000000000001','0.08','0.07','0.06000000000000001','0.05000000000000001',
       '0.04000000000000001','0.030000000000000006','0.020000000000000004','0.01']
kes = ['2.0','3.111111111111111','4.222222222222222','5.333333333333334',
       '6.444444444444445','7.555555555555555','8.666666666666668','9.777777777777779',
       '10.88888888888889','12.0']


inds = np.empty([10,10], dtype=object)
for i in range(10):
    for j in range(10):
        inds[i,j] = (float(kis[i]), float(kes[j]))


acc_mat_parsweep5000 = np.load('./acc_mat_par_ki.npy')
acc_mat_parsweep5000 = np.flipud(acc_mat_parsweep5000) #fix for kis


kdm5b_nt = 4647 + 1011/2
p300_nt = 7257 + 1011/2


def learnable_metric(L1,L2,Lt1,Lt2,Ne1,Ne2, ki1,ki2,ke1,ke2,time):
    L1a = (L1 - Lt1/2)/3
    L2a = (L2 - Lt2/2)/3
    
    tau1 = L1a/ke1
    tau2 = L2a/ke2 
    
    int_1 = ki1*tau1*Ne1
    int_2 = ki2*tau2*Ne2
    
    intfold = max((int_1/int_2), int_2/int_1)
    taufold = max((tau2/tau1), tau1/tau2)

    nribs1 = ki1*time
    nribs2 = ki2*time
    
    nribfold = max(nribs1/nribs2, nribs2/nribs1)

    return abs(2 - (taufold+intfold))* (nribs1+nribs2)/2*.005


ki_float1 = [np.round(float(x),2) for x in kis][::-1]
ki_float2 = [np.round(float(x),2) for x in kis]
ke_float1 = [np.round(float(x),2) for x in kes]
ke_float2 = [np.round(float(x),2) for x in kes]


diff_mat = np.zeros([10,10])

for i in range(10):
    for j in range(10):
        
        tau1 = kdm5b_nt/3/5.33
        tau2 = p300_nt/3/5.33
        int_1 = ki_float1[i]*tau1
        int_2 = ki_float2[j]*tau2
        
        diff = max((int_1/int_2), int_2/int_1)
        
        
        
        if diff != 0:
            diff_mat[i,j] = learnable_metric(4647,7257,1011,1011,10,10, ki_float1[i],ki_float2[j],5.33,5.33,3000) 
            #diff_mat[i,j] = diff


from matplotlib.colors import ListedColormap
cmap = plt.cm.viridis_r
my_cmap = cmap(np.arange(cmap.N))
my_cmap[:, -1] = .7
my_cmap = ListedColormap(my_cmap)


fig,ax = plt.subplots(1,1,dpi=120)
b = ax.imshow(np.flipud(acc_mat_parsweep5000.T),cmap =my_cmap)
ax.set_yticks(np.arange(10),)
ax.set_xticks( np.arange(10))


ax.set_yticklabels([str(np.round(float(x),2)) for x in kis], fontdict = {'fontsize':7})
ax.set_xticklabels([str(np.round(float(x),2)) for x in kis][::-1], fontdict = {'fontsize':7})
fig.colorbar(b)
#ax.plot([.05],[7.55555],'r*',markersize=10)
ax.set_xlabel('ki1')
ax.set_ylabel('ki2')
ax.set_title('Test Accuracy over training Par combo')
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
names1 = [str(np.round(float(x),2)) for x in kis]
ax.set_yticklabels([x for x in names1], fontdict = {'fontsize':7})
ax.set_xticklabels([x for x in names1][::-1], fontdict = {'fontsize':7},rotation=45)

#ax.plot([.05],[7.55555],'r*',markersize=10)
ax.set_xlabel('ki 1')
ax.set_ylabel('ki 2')
ax.set_title(r'fold change Intensity')

import matplotlib
for x in range(10):
    for y in range(10):
        if np.flipud(diff_mat.T)[x,y] < 1:
            rect1 = matplotlib.patches.Rectangle((y-.5,x-.5), 1,1,
                     edgecolor='r',fill=0)
            ax.add_patch(rect1)
