# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 18:15:19 2022

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

aps.apply_style()

accs = np.load('D:/multiplexing_ML/finalized_plots_gaussians/figures_data/data_size_sweep/accs_data_size.npy')
accs_2 = np.load('D:/multiplexing_ML/finalized_plots_gaussians/figures_data/data_size_sweep/accs_2_data_size.npy')

a = np.vstack([accs,accs_2[3:]])

plt.figure(dpi=300, figsize=(5,3))
data_size = [50,100,150,200,250,300,350,400,500,600,700,800,900,1000,1250] + np.linspace(100,4000,10).astype(int).tolist()[3:]
plt.errorbar(data_size, np.mean(a,axis=1), yerr= np.std(a,axis=1), capsize=2, linestyle='', marker='o')
plt.xlabel('Training data size')
plt.ylabel('Test Accuracy')

plt.savefig('training_data_size.svg')
#plt.xlim([0,000])



from scipy import optimize as opt

data_sizes =  [50,100,150,200,250,300,350,400,500,600,700,800,900,1000] + [1500,2000]

test_accs_combined = np.zeros([16,4])
test_accs_1 = np.load('.//test_accs_1.npy');
test_accs = np.load('./test_accs (1).npy');
test_accs_2 = np.load('./test_accs_2.npy')

test_accs_combined[:14] = test_accs_1
test_accs_combined[14:,:2] = test_accs
test_accs_combined[14:,2:] = test_accs

ds3  = [10,15,20,25,30,35,40]*10
ys3 =  test_accs_2.flatten(order='F')

ds4  = [1250,1750]*2
test_accs_4 = np.load('./test_accs_3.npy')
ys4 =  test_accs_4.flatten(order='F')

ds5  =  [75,125,165,225,275,325,376,425,550,675,775]*5
test_accs_5 = np.load('./test_accs_4.npy')
ys5 =  test_accs_5.flatten(order='F')

efun = lambda x, k, n, m: (1 / ( 1  + (k / x)**n))*.39 +.5

xs = np.hstack([np.array(data_sizes*4), np.array(ds3), np.array(ds4), np.array(ds5)])
ys = np.hstack([test_accs_combined.flatten(order='F'), ys3, ys4, ys5])


def movmean(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n



bins = [10,100,200,300,400,500,600,700,800,900,1000,1250,1500,1750,2000 ]

ystds = []
yms = []
for i in range(len(bins)-1):
    ystd = np.std(ys[(xs > bins[i])*(xs <= bins[i+1])])
    ym = np.mean(ys[(xs > bins[i])*(xs <= bins[i+1])])
    ystds.append(ystd)
    yms.append(ym)


#efun = lambda x, a,b : np.sqrt(x + a) + b
def fun(x):
    ypred = efun(xs, x[0], x[1], x[2])
    return np.sum(np.abs(ypred-ys)**2)

def fun2(x):
    ypred = efun(bins[:-1], x[0], x[1], x[2])
    return np.sum((np.abs(ypred-yms)/ystds)**2)


par = opt.minimize(fun2, (.5,.5,.87), method='TNC', bounds = ((0,100), (0,10), (0,1),))
plt.figure()
plt.plot(np.linspace(.01,2100,4000), efun(np.linspace(.01,2100,4000), par.x[0],par.x[1], par.x[2]))
plt.scatter(np.hstack( [xs[:13], xs[16:-2]])  ,np.hstack([ ys[:13], ys[16:-2]]) ,marker='.',alpha=.2)
ds3  = [10,15,20,25,30,35,40]*10
plt.scatter(ds3, test_accs_2.flatten(order='F'),marker='.',alpha=.2, c='#073b4c')
ax = plt.gca()


#for i in range(len(bins)-1):
#    ax.add_patch(Rectangle((bins[i], yms[i] - 2*ystds[i]), bins[i+1] - bins[i],  2*ystds[i], alpha=.3))


#plt.plot(bins[:-1],yms)

plt.xlabel('Training data size')
plt.ylabel('Test accuracy')
#plt.legend([r'Trend $y = 0.5 + \frac{0.39}{1 + (%.3f/x)^{%.3f}}$ '%(par.x[0], par.x[1],), 'Single classifier'])
plt.legend(['Trend', 'Single classifier'])

plt.savefig('training_data_size.svg')

