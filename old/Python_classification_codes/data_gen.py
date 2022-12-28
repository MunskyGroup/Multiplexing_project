import numpy as np
#import ssa_translation
import os
os.chdir('../../rSNAPsim')
import rSNAPsim
os.chdir('../translation_ml/codes')

import matplotlib.pyplot as plt

# This is a code to simulate data with a given frame rate, trajectory length, etc.

def simulate_data(frame_rate=1, n_frames=200, n_traj=100,gene='KDM5B',noise=2.2,prin=0.004,k_init=0.033,k_init_prior=True,sample_initiation_rate=False):
    '''
    Generate data for a particular gene. 
    '''
    if gene == 'KDM5B':
        ke_mean = 11
        fname = 'KDM5B_withTags.txt' 
    elif gene == 'H2B':
        ke_mean = 8
        fname = 'Plasmid_H2B.txt' 
    elif gene == 'Bactin':
        ke_mean = 6
        fname = 'Bactin_withTags.txt' 
    elif gene == 'p300':
        ke_mean = 6 
        fname = 'pUB_SM_p300_MS2.gb'

    elif gene == 'AlexX_1S12F':
        ke_mean = 3
        fname = 'Kenneths_constructs/pUB_1xSun_12xFLAG_AlexX_MS2.txt'
        watched_intensity = 1
        
    elif gene == 'AlexX_1F12S':
        ke_mean = 3
        fname = 'Kenneths_constructs/pUB_1xFLAG_12xSun_AlexX_MS2.txt'
        watched_intensity = 1    
        
    elif gene == 'KDM5B_12S':
        ke_mean = 3
        fname = 'Kenneths_constructs/pUB_minus1PRF_12xSun_KDM5B_MS2.txt'


    # load the gene sequence.
    m = rSNAPsim.rSNAPsim()
    m.open_seq_file(fname)
    m.run_default()
    # compute "tf" only going to keep the first half of this. 
    tf = frame_rate*n_frames*2
    if tf<500:
        tf = 500
    t_step = frame_rate*tf
    sims = m.ssa_solver(n_traj=n_traj,k_initiation=k_init,k_elong_mean = ke_mean, start_time=700,tf=n_frames+700,tstep=n_frames+701)   
    print(sims.intensity_vec.shape)
    # only keep steady-state frames. 
    if gene == 'AlexX_1F12S':
        sims.intensity_vec = sims.intensity_vec[0]
        
    #sims.intensity_vec = sims.intensity_vec[:,-n_frames:]
    print(sims.intensity_vec.shape)
    
    
    wn = np.random.randn(*sims.intensity_vec.shape)*noise
    sims.intensity_vec_noise = prin*(sims.intensity_vec) + wn
    sims.intensity_vec_noise[sims.intensity_vec_noise<0]=0

    sims.intensity_vec_norm = sims.intensity_vec/np.expand_dims(np.mean(sims.intensity_vec,axis=1),1)
    sims.intensity_vec_norm_noise = sims.intensity_vec_noise/np.expand_dims(np.mean(sims.intensity_vec_noise,axis=1),1)

    sims.centered_intensity_vec = sims.intensity_vec - np.expand_dims(np.mean(sims.intensity_vec,axis=1),1)
    sims.centered_intensity_vec_noise = sims.intensity_vec_noise - np.expand_dims(np.mean(sims.intensity_vec_noise,axis=1),1)
    return sims 

if __name__== '__main__':
    
    for i in range(10):
        print(i)
        gene_name = 'KDM5B_12S'
        sims = simulate_data(frame_rate=1, n_frames=10000, n_traj=100,gene=gene_name)

        data_norm = sims.intensity_vec_norm_noise
        data_raw = sims.intensity_vec
        if i >= 1:
            f1 = open('../data/sim_data/data_norm_master_{0}_{1}_{2}_frames.txt'.format(gene_name,10000,1000), "a")
            f2 = open('../data/sim_data/data_raw_master_{0}_{1}_{2}_frames.txt'.format(gene_name,10000,1000),"a")
            np.savetxt(f1, data_norm,fmt='%1.3f')
            np.savetxt(f2, data_raw.astype(int),fmt='%i')
        else:
            np.savetxt('../data/sim_data/data_norm_master_{0}_{1}_{2}_frames.txt'.format(gene_name,10000,1000),data_norm,fmt='%1.3f')
            np.savetxt('../data/sim_data/data_raw_master_{0}_{1}_{2}_frames.txt'.format(gene_name,10000,1000),data_raw.astype(int),fmt='%i')
            
    plt.plot(sims.intensity_vec.T)
            
    plt.show()
    
    a = np.loadtxt('../data/sim_data/data_norm_master_{0}_{1}_{2}_frames.txt'.format(gene_name,10000,1000))
    print(a.shape)