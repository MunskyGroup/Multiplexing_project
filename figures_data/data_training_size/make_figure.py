# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 14:35:54 2023

@author: willi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as opt

data_sizes =  [50,100,150,200,250,300,350,400,500,600,700,800,900,1000] + [1500,2000]

test_accs_combined = np.zeros([16,4])
test_accs_1 = np.load('C:/Users/willi/Downloads/test_accs_1.npy');
test_accs = np.load('C:/Users/willi/Downloads/test_accs (1).npy');

test_accs_combined[:14] = test_accs_1
test_accs_combined[14:,:2] = test_accs
test_accs_combined[14:,2:] = test_accs

efun = lambda x, k: .87*(1-np.exp(-k*x)) 


efun = lambda x, k, n, m: (1 / ( 1  + (k / x)**n))*m 

xs = np.array(data_sizes*4)
ys = test_accs_combined.flatten(order='F')

par,cov = opt.curve_fit(efun, xs, ys, p0=[.008, .875-.5,], bounds =(1e-8, 1), verbose=True)

plt.plot(data_sizes, efun(np.array(data_sizes), par[0], par[1],))
plt.scatter(xs,ys)


#efun = lambda x, a,b : np.sqrt(x + a) + b
def fun(x):
    ypred = efun(xs, x[0], x[1], x[2])
    return np.sum(np.abs(ypred-ys)**2)

par = opt.minimize(fun, (.5,.5,.87), method='TNC', bounds = ((0,10), (0,10), (0,3),))

plt.plot(np.linspace(10,2100,4000), efun(np.linspace(10,2100,4000), par.x[0],par.x[1], par.x[2]+.0075))
plt.scatter(xs,ys)

