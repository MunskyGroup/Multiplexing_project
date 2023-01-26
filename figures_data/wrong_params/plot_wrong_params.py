# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 20:02:51 2022

@author: willi
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 12:32:29 2022

@author: willi
"""

import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import pandas as pd

import os
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib

import yaml
from cycler import cycler
import matplotlib.patches as patches
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


cmap = plt.get_cmap('Spectral')

#my_cmap = cmap(np.arange(cmap.N))
#my_cmap[:, -1] = .9
#my_cmap = ListedColormap(my_cmap)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

new_cmap = truncate_colormap(cmap, 0.3, 1)
my_cmap = new_cmap(np.arange(cmap.N))
my_cmap[:, -1] = .9
my_cmap = ListedColormap(my_cmap)


acc_mat_self = np.zeros([10,10])
acc_mat_og = np.load('D:/multiplexing_ML/acc_grid_keki.npy')
key_file = 'D:/multiplexing_ML/finalized_plots_gaussians/ML_run_320_5s_wfreq/parsweep_keki_ML/keki_key.csv'

for i in range(10): 
    for j in range(10):
        acc_mat_self[i,j] = acc_mat_og[i,j,i,j]
        
        
key_file = pd.read_csv(key_file)

xshape = key_file.shape[0]
yshape = key_file.shape[1]-1

convert_str = lambda tstr: [x.replace(' ','').replace('(','').replace(')','').replace("'",'') for x in tuple(tstr.split(','))] 
round_str = lambda fstr: str(np.round(float(fstr),2))

#acc_mat = np.zeros([xshape,yshape])
xlabels = [0,]*xshape
ylabels = [0,]*yshape
for i in range(0,xshape):
    for j in range(0,yshape):
        
        ind1,ind2, x,y,acc = convert_str(key_file.iloc[i][j+1])
       
        #acc_mat[i,j] = acc
        xlabels[int(ind1)] = x
        ylabels[int(ind2)] = y


#acc_mat = (np.sum(acc_mat, axis=(2,3))/100).T[0]

topleft = (0,0)
widthx = 10
widthy = 10


acc_mat = (np.sum(  acc_mat_og[:,:,topleft[0]:topleft[0]+widthx,topleft[1]:topleft[1]+widthy] > .7, axis=(2,3) )/100   ).T[0]

#acc_mat = (np.std(acc_mat_og[:,:,topleft[0]:topleft[0]+widthx,topleft[1]:topleft[1]+widthy], axis=(2,3))).T[0]
#acc_mat = acc_mat_self.T

### mi mj di dj
#acc_mat = acc_mat_og[:,:,5,3].T[0]
#acc_mat = acc_mat_og[5,3,:,:].T[0]
rect = patches.Rectangle((topleft[0]-.5, topleft[1]-.5), widthx, widthy, linewidth=1, edgecolor='r', facecolor='none')

xlabel = ''
ylabel = ''
title = ''

xlabel = r'$k_{i}$ (1/s)'
ylabel = r'$k_{e}$ (aa/s)'  
title = r'Test Accuracy over $k_i$ and $k_e$ pairs'
ylabels = [round_str(x) for x in ylabels]
xlabels = [round_str(x) for x in xlabels]



##################################################


fig,ax = plt.subplots(1,1,dpi=120, tight_layout=True) 
b = ax.imshow(acc_mat,cmap ='Greens', vmin = 0, vmax = 1,origin='lower' )
ax.set_yticks(np.arange(yshape),)
ax.set_xticks( np.arange(xshape))
fig.colorbar(b)
ax.set_yticklabels(ylabels, fontdict = {'fontsize':7})
ax.set_xticklabels(xlabels, fontdict = {'fontsize':7},rotation=45)

 
#ax.plot([.05],[7.55555],'r*',markersize=10)

ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)
ax.add_patch(rect)
ax.set_title(title)

for x in range(yshape):
    for y in range(xshape):
        if acc_mat[x, y] != 0:
            
            ax.text(y,x, '%.2f' % acc_mat[x, y],
                     horizontalalignment='center',
                     verticalalignment='center',
                     size=5)
        else:
            ax.text(y,x, '-',
                     horizontalalignment='center',
                     verticalalignment='center',
                     size=5)                    

plt.savefig('keki_wrong_param_n70.svg')











#################

topleft = (8,1)
widthx = 1
widthy = 1


#acc_mat = (np.std(acc_mat_og[:,:,topleft[0]:topleft[0]+widthx,topleft[1]:topleft[1]+widthy], axis=(2,3))).T[0]
#acc_mat = acc_mat_self.T

### mi mj di dj
#acc_mat = acc_mat_og[:,:,5,3].T[0]
acc_mat = acc_mat_og[8,1,:,:].T[0]
rect = patches.Rectangle((topleft[0]-.5, topleft[1]-.5), widthx, widthy, linewidth=1, edgecolor='r', facecolor='none')

xlabel = ''
ylabel = ''
title = ''

xlabel = r'$k_{i}$ (1/s)'
ylabel = r'$k_{e}$ (aa/s)'  
title = r'Test Accuracy over $k_i$ and $k_e$ pairs'
ylabels = [round_str(x) for x in ylabels]
xlabels = [round_str(x) for x in xlabels]



##################################################


fig,ax = plt.subplots(1,1,dpi=120, tight_layout=True) 
b = ax.imshow(acc_mat,cmap =my_cmap, vmin = .5, vmax = 1,origin='lower' )
ax.set_yticks(np.arange(yshape),)
ax.set_xticks( np.arange(xshape))
fig.colorbar(b)
ax.set_yticklabels(ylabels, fontdict = {'fontsize':7})
ax.set_xticklabels(xlabels, fontdict = {'fontsize':7},rotation=45)

 
#ax.plot([.05],[7.55555],'r*',markersize=10)

ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)
ax.add_patch(rect)
ax.set_title(title)

for x in range(yshape):
    for y in range(xshape):
        if acc_mat[x, y] != 0:
            
            ax.text(y,x, '%.2f' % acc_mat[x, y],
                     horizontalalignment='center',
                     verticalalignment='center',
                     size=5)
        else:
            ax.text(y,x, '-',
                     horizontalalignment='center',
                     verticalalignment='center',
                     size=5)                    

plt.savefig('keki_wrong_param_m81.svg')



##################

topleft = (1,8)
widthx = 1
widthy = 1


#acc_mat = (np.std(acc_mat_og[:,:,topleft[0]:topleft[0]+widthx,topleft[1]:topleft[1]+widthy], axis=(2,3))).T[0]
#acc_mat = acc_mat_self.T

### mi mj di dj
#acc_mat = acc_mat_og[:,:,5,3].T[0]
acc_mat = acc_mat_og[1,8,:,:].T[0]
#acc_mat = (np.std(acc_mat_og[:,:,topleft[0]:topleft[0]+widthx,topleft[1]:topleft[1]+widthy], axis=(2,3))).T[0]
#acc_mat = acc_mat_self.T

### mi mj di dj
#acc_mat = acc_mat_og[:,:,5,3].T[0]
#acc_mat = acc_mat_og[5,3,:,:].T[0]
rect = patches.Rectangle((topleft[0]-.5, topleft[1]-.5), widthx, widthy, linewidth=1, edgecolor='r', facecolor='none')

xlabel = ''
ylabel = ''
title = ''

xlabel = r'$k_{i}$ (1/s)'
ylabel = r'$k_{e}$ (aa/s)'   
title = r'Test Accuracy over $k_i$ and $k_e$ pairs'
ylabels = [round_str(x) for x in ylabels]
xlabels = [round_str(x) for x in xlabels]



##################################################


fig,ax = plt.subplots(1,1,dpi=120, tight_layout=True) 
b = ax.imshow(acc_mat,cmap =my_cmap, vmin = .5, vmax = 1,origin='lower' )
ax.set_yticks(np.arange(yshape),)
ax.set_xticks( np.arange(xshape))
fig.colorbar(b)
ax.set_yticklabels(ylabels, fontdict = {'fontsize':7})
ax.set_xticklabels(xlabels, fontdict = {'fontsize':7},rotation=45)

 
#ax.plot([.05],[7.55555],'r*',markersize=10)

ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)
ax.add_patch(rect)
ax.set_title(title)

for x in range(yshape):
    for y in range(xshape):
        if acc_mat[x, y] != 0:
            
            ax.text(y,x, '%.2f' % acc_mat[x, y],
                     horizontalalignment='center',
                     verticalalignment='center',
                     size=5)
        else:
            ax.text(y,x, '-',
                     horizontalalignment='center',
                     verticalalignment='center',
                     size=5)                    

plt.savefig('keki_wrong_param_m18.svg')



###################


topleft = (6,4)
widthx = 1
widthy = 1


#acc_mat = (np.std(acc_mat_og[:,:,topleft[0]:topleft[0]+widthx,topleft[1]:topleft[1]+widthy], axis=(2,3))).T[0]
#acc_mat = acc_mat_self.T

### mi mj di dj
#acc_mat = acc_mat_og[:,:,5,3].T[0]
acc_mat = acc_mat_og[6,4,:,:].T[0]
#acc_mat = (np.std(acc_mat_og[:,:,topleft[0]:topleft[0]+widthx,topleft[1]:topleft[1]+widthy], axis=(2,3))).T[0]
#acc_mat = acc_mat_self.T

### mi mj di dj
#acc_mat = acc_mat_og[:,:,5,3].T[0]
#acc_mat = acc_mat_og[5,3,:,:].T[0]
rect = patches.Rectangle((topleft[0]-.5, topleft[1]-.5), widthx, widthy, linewidth=1, edgecolor='r', facecolor='none')

xlabel = ''
ylabel = ''
title = ''

xlabel = r'$k_{i}$ (1/s)'
ylabel = r'$k_{e}$ (aa/s)'  
title = r'Test Accuracy over $k_i$ and $k_e$ pairs'
ylabels = [round_str(x) for x in ylabels]
xlabels = [round_str(x) for x in xlabels]



##################################################


fig,ax = plt.subplots(1,1,dpi=120, tight_layout=True) 
b = ax.imshow(acc_mat,cmap =my_cmap, vmin = .5, vmax = 1,origin='lower' )
ax.set_yticks(np.arange(yshape),)
ax.set_xticks( np.arange(xshape))
fig.colorbar(b)
ax.set_yticklabels(ylabels, fontdict = {'fontsize':7})
ax.set_xticklabels(xlabels, fontdict = {'fontsize':7},rotation=45)

 
#ax.plot([.05],[7.55555],'r*',markersize=10)

ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)
ax.add_patch(rect)
ax.set_title(title)

for x in range(yshape):
    for y in range(xshape):
        if acc_mat[x, y] != 0:
            
            ax.text(y,x, '%.2f' % acc_mat[x, y],
                     horizontalalignment='center',
                     verticalalignment='center',
                     size=5)
        else:
            ax.text(y,x, '-',
                     horizontalalignment='center',
                     verticalalignment='center',
                     size=5)                    

plt.savefig('keki_wrong_param_m64.svg')




topleft = (9,6)
widthx = 1
widthy = 1


#acc_mat = (np.std(acc_mat_og[:,:,topleft[0]:topleft[0]+widthx,topleft[1]:topleft[1]+widthy], axis=(2,3))).T[0]
#acc_mat = acc_mat_self.T

### mi mj di dj
#acc_mat = acc_mat_og[:,:,5,3].T[0]
acc_mat = acc_mat_og[9,6,:,:].T[0]
#acc_mat = (np.std(acc_mat_og[:,:,topleft[0]:topleft[0]+widthx,topleft[1]:topleft[1]+widthy], axis=(2,3))).T[0]
#acc_mat = acc_mat_self.T

### mi mj di dj
#acc_mat = acc_mat_og[:,:,5,3].T[0]
#acc_mat = acc_mat_og[5,3,:,:].T[0]
rect = patches.Rectangle((topleft[0]-.5, topleft[1]-.5), widthx, widthy, linewidth=1, edgecolor='r', facecolor='none')

xlabel = ''
ylabel = ''
title = ''

xlabel = r'$k_{i}$ (1/s)'
ylabel = r'$k_{e}$ (aa/s)'  
title = r'Test Accuracy over $k_i$ and $k_e$ pairs'
ylabels = [round_str(x) for x in ylabels]
xlabels = [round_str(x) for x in xlabels]



##################################################


fig,ax = plt.subplots(1,1,dpi=120, tight_layout=True) 
b = ax.imshow(acc_mat,cmap =my_cmap, vmin = .5, vmax = 1,origin='lower' )
ax.set_yticks(np.arange(yshape),)
ax.set_xticks( np.arange(xshape))
fig.colorbar(b)
ax.set_yticklabels(ylabels, fontdict = {'fontsize':7})
ax.set_xticklabels(xlabels, fontdict = {'fontsize':7},rotation=45)

 
#ax.plot([.05],[7.55555],'r*',markersize=10)

ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)
ax.add_patch(rect)
ax.set_title(title)
print(np.mean(acc_mat))
for x in range(yshape):
    for y in range(xshape):
        if acc_mat[x, y] != 0:
            
            ax.text(y,x, '%.2f' % acc_mat[x, y],
                     horizontalalignment='center',
                     verticalalignment='center',
                     size=5)
        else:
            ax.text(y,x, '-',
                     horizontalalignment='center',
                     verticalalignment='center',
                     size=5)               





topleft = (9,6)
widthx = 1
widthy = 1


#acc_mat = (np.std(acc_mat_og[:,:,topleft[0]:topleft[0]+widthx,topleft[1]:topleft[1]+widthy], axis=(2,3))).T[0]
#acc_mat = acc_mat_self.T

### mi mj di dj
#acc_mat = acc_mat_og[:,:,5,3].T[0]

acc_mat_avs = np.zeros([10,10])
for i in range(10):
    for j in range(10):
        acc_mat_avs[i,j] = np.mean(acc_mat_og[i,j,:,:].T[0])

acc_mat = acc_mat_avs.T
#acc_mat = acc_mat_og[9,6,:,:].T[0]
#acc_mat = (np.std(acc_mat_og[:,:,topleft[0]:topleft[0]+widthx,topleft[1]:topleft[1]+widthy], axis=(2,3))).T[0]
#acc_mat = acc_mat_self.T

### mi mj di dj
#acc_mat = acc_mat_og[:,:,5,3].T[0]
#acc_mat = acc_mat_og[5,3,:,:].T[0]
rect = patches.Rectangle((topleft[0]-.5, topleft[1]-.5), widthx, widthy, linewidth=1, edgecolor='r', facecolor='none')

xlabel = ''
ylabel = ''
title = ''

xlabel = r'$k_{i}$ (1/s)'
ylabel = r'$k_{e}$ (aa/s)'   
title = r'Test Accuracy over $k_i$ and $k_e$ pairs'
ylabels = [round_str(x) for x in ylabels]
xlabels = [round_str(x) for x in xlabels]



##################################################


fig,ax = plt.subplots(1,1,dpi=120, tight_layout=True) 
b = ax.imshow(acc_mat,cmap =my_cmap, vmin = .5, vmax = 1,origin='lower' )
ax.set_yticks(np.arange(yshape),)
ax.set_xticks( np.arange(xshape))
fig.colorbar(b)
ax.set_yticklabels(ylabels, fontdict = {'fontsize':7})
ax.set_xticklabels(xlabels, fontdict = {'fontsize':7},rotation=45)

 
#ax.plot([.05],[7.55555],'r*',markersize=10)

ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)
ax.add_patch(rect)
ax.set_title(title)
print(np.mean(acc_mat))
for x in range(yshape):
    for y in range(xshape):
        if acc_mat[x, y] != 0:
            
            ax.text(y,x, '%.3f' % acc_mat[x, y],
                     horizontalalignment='center',
                     verticalalignment='center',
                     size=5)
        else:
            ax.text(y,x, '-',
                     horizontalalignment='center',
                     verticalalignment='center',
                     size=5)               

plt.savefig('keki_wrong_param_av.svg')
