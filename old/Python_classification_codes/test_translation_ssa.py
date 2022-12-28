import numpy as np
import ssa_translation
import matplotlib.pyplot as plt
import time
# load the elongation 
kelong = np.loadtxt('elongationrates.txt')
kbind = kelong[0]
kcompl = kelong[-1]
kelong = kelong[1:-1]

t_array = np.array([0,100,500],dtype=np.float64)
t_array = np.linspace(0,1000,1000,dtype=np.float64)
N_rib = 200
result = np.zeros((len(t_array)*N_rib),dtype=np.int32)
#kelong = np.array([3.1,3.2,3.3,3.4,3.5,3.1,3.2,3.3,3.4,3.5],dtype=np.float64)
n_trajectories = 100
start = time.time()
all_results = np.zeros((n_trajectories,N_rib*len(t_array)),dtype=np.int32)
for i in range(n_trajectories):
    result = np.zeros((len(t_array)*N_rib),dtype=np.int32)
    ssa_translation.run_SSA(result,kelong,t_array,kbind,kcompl)
    all_results[i,:] = result
print('time for {0} trajectories {1}'.format(n_trajectories,time.time()-start))
#plt.hist(result[result>0])
#plt.show()
#traj = result.reshape((N_rib,len(t_array))).T
##print('The result is \n {0}'.format(result.reshape((N_rib,len(t_array))).T))
#plt.plot(traj[-1,:])
#plt.show()

# map to fluorescence.
ntimes = len(t_array)
intensity_vec = np.zeros(ntimes)
pv = np.loadtxt('probe_design.txt')
tstart = 0
I = np.zeros((n_trajectories,ntimes-tstart))
for i in range(n_trajectories):
    traj = all_results[i,:].reshape((N_rib,len(t_array))).T
    for j in range(tstart,ntimes):
        temp_output = traj[j,:]
        I[i,j] = np.sum(pv[temp_output[temp_output>0]-1])

# Plotting
all_traj = np.loadtxt('ivec_1000t')
f,ax = plt.subplots(2,1)
ax[0].plot(I[0:5,-500:].T)
ax[1].plot(all_traj[0:5,-500:].T)
f2,ax2 = plt.subplots(1,2)
ax2[0].hist(I[:,-500:].ravel())
ax2[1].hist(all_traj[:100,-500:].ravel())
plt.show()

