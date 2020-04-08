# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 10:29:00 2019

@author: William
"""
import os
os.chdir('../../rSNAPsim')
import rSNAPsim
os.chdir('../translation_ml/codes')

import matplotlib.pyplot as plt


bright_green = '#00ff44'
bright_blue = '#00d0ff'

AlexX_1S12F = rSNAPsim.rSNAPsim()
AlexX_1S12F.open_seq_file('Kenneths_constructs/pUB_1xFLAG_12xSun_AlexX_MS2.txt')
#AlexX_1S12F.open_seq_file('Kenneths_constructs/pUB_1xSun_12xFLAG_AlexX_MS2.txt')

AlexX_1S12F.run_default()
a1S12F = AlexX_1S12F.ssa_solver(k_elong_mean=3, k_initiation=.033,n_traj=100,start_time=1000,tf=3000,tstep=3000)
print(a1S12F.dwelltime)
plt.plot(a1S12F.intensity_vec[0].T,color=bright_blue)
plt.plot(a1S12F.intensity_vec[1].T,color=bright_green)


KDM5B_12F =rSNAPsim.rSNAPsim()
KDM5B_12F.open_seq_file('Kenneths_constructs/pUB_minus1PRF_12xSun_KDM5B_MS2.txt')
#KDM5B_12F.open_seq_file('Kenneths_constructs/12xFLAG_ActB_MS2.txt')
KDM5B_12F.run_default()
'''
k12F = KDM5B_12F.ssa_solver(k_elong_mean=3, k_initiation=.033,n_traj=100,start_time=1000,tf=3000,tstep=3000)
print(k12F.dwelltime)
plt.figure()
plt.plot(k12F.intensity_vec.T,color=bright_blue)
'''

seconds = [1,5,10,20,36]
nframes = 300
nspots = [10,20,50,100]

seconds = [36]
nframes = 100
'''
K_dwelltimes = {}
A_dwelltimes = {}
for spots in nspots:
    for sec in seconds:
        if sec == 1:
            a1S12F = AlexX_1S12F.ssa_solver(k_elong_mean=3, k_initiation=.033, n_traj = spots, start_time=500,tf=800,tstep=800+1)
            k12F = KDM5B_12F.ssa_solver(k_elong_mean=3, k_initiation=.033, n_traj = spots, start_time=500,tf=800,tstep=800+1)
        if sec == 3:
            a1S12F = AlexX_1S12F.ssa_solver(k_elong_mean=3, k_initiation=.033, n_traj = spots, start_time=700,tf=1600,tstep=1600/3+1)
            k12F = KDM5B_12F.ssa_solver(k_elong_mean=3, k_initiation=.033, n_traj = spots, start_time=700,tf=1600,tstep=1600/3+1)        
     
        if sec == 5:
            a1S12F = AlexX_1S12F.ssa_solver(k_elong_mean=3, k_initiation=.033, n_traj = spots, start_time=700,tf=2200,tstep=2200/5+1)
            k12F = KDM5B_12F.ssa_solver(k_elong_mean=3, k_initiation=.033, n_traj = spots, start_time=700,tf=2200,tstep=2200/5+1)   
            
    
        if sec == 10:
            a1S12F = AlexX_1S12F.ssa_solver(k_elong_mean=3, k_initiation=.033, n_traj = spots, start_time=700,tf=3700,tstep=3700/10+1)
            k12F = KDM5B_12F.ssa_solver(k_elong_mean=3, k_initiation=.033, n_traj = spots, start_time=700,tf=3700,tstep=3700/10+1)         
        if sec == 20:
            a1S12F = AlexX_1S12F.ssa_solver(k_elong_mean=3, k_initiation=.033, n_traj = spots, start_time=700,tf=6700,tstep=6700/20+1)
            k12F = KDM5B_12F.ssa_solver(k_elong_mean=3, k_initiation=.033, n_traj = spots, start_time=700,tf=6700,tstep=6700/20+1)         
 
        if sec == 36:
            a1S12F = AlexX_1S12F.ssa_solver(k_elong_mean=3, k_initiation=.033, n_traj = spots, start_time=700,tf=3600+700,tstep=4200/36+1)
            k12F = KDM5B_12F.ssa_solver(k_elong_mean=3, k_initiation=.033, n_traj = spots,  start_time=700,tf=3600+700,tstep=4200/36+1)         
                                               
        K_dwelltimes[(spots,sec)] = [k12F.dwelltime, k12F]
        A_dwelltimes[(spots,sec)] = [a1S12F.dwelltime[0], a1S12F]
    
''' 
    
    
kd = []
ad = []
for i in range(1000):
    a1S12F = AlexX_1S12F.ssa_solver(k_elong_mean=3, k_initiation=.033, n_traj = 1, start_time=700,tf=2100+700,tstep=2800/7+1)
    k12F = KDM5B_12F.ssa_solver(k_elong_mean=3, k_initiation=.033, n_traj = 1,  start_time=700,tf=2100+700,tstep=2800/7+1)         
    ad.append(a1S12F.dwelltime[1])
    kd.append(k12F.dwelltime)
    



