# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 16:04:05 2022

@author: willi
"""

import numpy as np
import subprocess
import tqdm
import time
import os
import pandas as pd
import shutil

##############################################################################
# Global setup
# This file makes all the simulated cell data used in the paper:
#    
# Data generation for "Using mechanistic models and machine learning
# to design single-color multiplexed Nascent Chain Tracking experiments
# RAYMOND (2023) - wsraymon@colostate.edu
#
#
# Generates simulated NCT cell datasets used for machine learning experiments
# Using rSNAPed and rSNAPsim to simulate fluorescence microscopy videos for
# a given experimental setup.
#
# !!! WARNING !!! 
# If not run in debug mode, this will generate 120 GB of files and WILL TAKE
# UP TO 20 GB of RAM FOR SOME EXPERIMENTS
##############################################################################

spot_size = 3 #pixels
diffusion_rate = .55 #
debug = True


make_list = [
             1,  #24000s same intensity cells (figure4) 
             1,  #24000s similar intensity cells  (figure 4)
             1,  #24000s different intensity cells (figure 4)
             1,  #construct lengths (mRNA_L vs mRNA_L)  (figure 5, figure 6)
             1,  #ke vs ki  (base experiment is in here, figure 3, figure 5, figure 6, figure 8)
             1,  #kes vs kes (figure 5, figure 6)
             1,  #kis vs kis (figure 5, figure 6)
             1,  #alternate tagging (figure 7)
             1,  #multiplexing (figure 9)
             1,
             1,] #photobleaching (figure s3)

if debug:
    print('RUNNING DEBUG....')
    st = time.time()
else:
    print('WARNING THIS WILL MAKE APPROXIMATELY 120GB OF DATA!! AND WILL TAKE 4 WEEKS+')
    input("Press Enter to continue...")

##############################################################################
# 24000s same intensity cells
#
# Run P300 vs KDM5B at the same intensity by manipulating initation rates for
# 24000 total seconds, used to investigate usage of autocorrelation information
# vs frame rate and video length
##############################################################################

if make_list[0]:
    intensity_scale = 14
    sim_time = 24000
    ncells = 50
    save_dir = './P300_KDM5B_24000s_same_intensity_gaussian_14scale'
    
    if debug:
        sim_time = 100
        ncells = 1
        save_dir = './debug/P300_KDM5B_24000s_same_intensity_gaussian_14scale'
    
    desired_average_ribosome_count = 5 #lower polysome count
    elongations = [5.33333]
    inits_kdm5b = lambda x: desired_average_ribosome_count/( 1886/x)
    inits_p300 = lambda x: desired_average_ribosome_count/( 2756/x)
    
    kis_kdm5b = [inits_kdm5b(elongations[0])]
    kis_p300 = [inits_p300(elongations[0])]
    
    
    
    if not os.path.exists(os.path.join('.', save_dir)):
        os.makedirs(os.path.join('.', save_dir))
    
    print('Running 24000s same intensity 50 cells 50 spots data generation...')
    
    construct_files = ['pUB_SM_KDM5B_PP7_coding_sequence.txt', 'pUB_SM_p300_MS2_coding_sequence.txt']
    
    with tqdm.tqdm(total=len(kis_kdm5b)*2) as pbar :
        for i in range(1):
                kes_str = str(elongations[i]) + "," + str(elongations[i])
                kis_str = str(kis_p300[i]) + "," + str(kis_p300[i])           
                
    
    
                file_name = 'ki_ke_sweep_same_int'
                subprocess.run(["python3",
                                "run_rsnaped.py",
                                "--rsnaped_dir=./rsnaped/rsnaped",
                                "--kis=%s"%kis_str,
                                "--kes=%s"%kes_str,
                                "--save_name=%s"%(file_name + '_P300_P300_' + str(kis_p300[i]) + '_' + str(elongations[i])+ '_' + str(i)+'.csv' ),
                                "--n_cells=%s"%str(ncells),
                               "--sim_time=%f"%sim_time,
                               "--n_spots=25,25",
                               "--diff_rates=%s,%s"%(str(diffusion_rate), str(diffusion_rate)),
                               "--genes=pUB_SM_p300_MS2_coding_sequence.txt,pUB_SM_p300_MS2_coding_sequence.txt",
                               "--gene_names=p300,p300",
                               "--intensity_scale=%s"%str(intensity_scale),
                               "--spot_size=%s"%str(spot_size),
                               "--save_vids=0",
                               "--empty_vid_select=generate_from_guassian",
                               "--save_dir=%s"%save_dir], stdout=subprocess.DEVNULL)
                time.sleep(.1)
                pbar.update(1)
                
    
                kis_str = str(kis_kdm5b[i]) + "," + str(kis_kdm5b[i])           
                
    
                file_name = 'ki_ke_sweep_same_int'
                subprocess.run(["python3",
                                "run_rsnaped.py",
                                "--rsnaped_dir=./rsnaped/rsnaped",
                                "--kis=%s"%kis_str,
                                "--kes=%s"%kes_str,
                                "--save_name=%s"%(file_name + '_KDM5B_KDM5B_' + str(kis_kdm5b[i]) + '_' + str(elongations[i])+ '_' + str(i)+'.csv' ),
                                "--n_cells=%s"%str(ncells),
                               "--sim_time=%f"%sim_time,
                               "--n_spots=25,25",
                               "--diff_rates=%s,%s"%(str(diffusion_rate), str(diffusion_rate)),
                               "--genes=pUB_SM_KDM5B_PP7_coding_sequence.txt,pUB_SM_KDM5B_PP7_coding_sequence.txt",
                               "--gene_names=kdm5b,kdm5b",
                               "--intensity_scale=%s"%str(intensity_scale),
                               "--spot_size=%s"%str(spot_size),
                               "--save_vids=0",
                               "--empty_vid_select=generate_from_guassian",
                               "--save_dir=%s"%save_dir], stdout=subprocess.DEVNULL)
                time.sleep(.1)
                pbar.update(1)
                

        

##############################################################################
# 24000s similar intensity cells
#
# Run P300 vs KDM5B at the same intensity by manipulating initation rates for
# 24000 total seconds, used to investigate usage of autocorrelation / intensity
# information vs frame rate and video length
##############################################################################


if make_list[1]:
    intensity_scale = 14
    sim_time = 24000
    ncells = 50
    save_dir = './P300_KDM5B_24000s_similar_intensity_gaussian_14scale'
    
    if debug:
        sim_time = 100
        ncells = 1
        save_dir = './debug/P300_KDM5B_24000s_similar_intensity_gaussian_14scale'
    
    
    desired_average_ribosome_count = 5 #lower polysome count
    elongations = [5.33333]
    inits_kdm5b = lambda x: desired_average_ribosome_count/( 1886/x)
    inits_p300 = lambda x: desired_average_ribosome_count/( 2756/x)
    
    
    
    ### manual initations to make kdm5b have intensity of 6 and p300 4.7 UMP
    kis_kdm5b = [.0186]
    kis_p300 = [5/(2756/5.333333)]
    
    
    
    if not os.path.exists(os.path.join('.', save_dir)):
        os.makedirs(os.path.join('.', save_dir))
    
    print('Running 24000s similar intensity 50 cells 50 spots data generation...')
    
    construct_files = ['pUB_SM_KDM5B_PP7_coding_sequence.txt', 'pUB_SM_p300_MS2_coding_sequence.txt']
    
    with tqdm.tqdm(total=len(kis_kdm5b)*2) as pbar :
        for i in range(1):
                kes_str = str(elongations[i]) + "," + str(elongations[i])
                kis_str = str(kis_p300[i]) + "," + str(kis_p300[i])           
                
    
                file_name = 'ki_ke_sweep_similar_int'
                subprocess.run(["python3",
                                "run_rsnaped.py",
                                "--rsnaped_dir=./rsnaped/rsnaped",
                                "--kis=%s"%kis_str,
                                "--kes=%s"%kes_str,
                                "--save_name=%s"%(file_name + '_P300_P300_' + str(kis_p300[i]) + '_' + str(elongations[i])+ '_' + str(i)+'.csv' ),
                                "--n_cells=%s"%str(ncells),
                               "--sim_time=%f"%sim_time,
                               "--n_spots=25,25",
                               "--diff_rates=%s,%s"%(str(diffusion_rate), str(diffusion_rate)),
                               "--genes=pUB_SM_p300_MS2_coding_sequence.txt,pUB_SM_p300_MS2_coding_sequence.txt",
                               "--gene_names=p300,p300",
                               "--intensity_scale=%s"%str(intensity_scale),
                               "--spot_size=%s"%str(spot_size),
                               "--save_vids=0",
                               "--empty_vid_select=generate_from_guassian",
                               "--save_dir=%s"%save_dir], stdout=subprocess.DEVNULL)
                time.sleep(.1)
                pbar.update(1)
                
                kis_str = str(kis_kdm5b[i]) + "," + str(kis_kdm5b[i])           
                
    
                file_name = 'ki_ke_sweep_similar_int'
                subprocess.run(["python3",
                                "run_rsnaped.py",
                                "--rsnaped_dir=./rsnaped/rsnaped",
                                "--kis=%s"%kis_str,
                                "--kes=%s"%kes_str,
                                "--save_name=%s"%(file_name + '_KDM5B_KDM5B_' + str(kis_kdm5b[i]) + '_' + str(elongations[i])+ '_' + str(i)+'.csv' ),
                                "--n_cells=%s"%str(ncells),
                               "--sim_time=%f"%sim_time,
                               "--n_spots=25,25",
                               "--diff_rates=%s,%s"%(str(diffusion_rate), str(diffusion_rate)),
                               "--genes=pUB_SM_KDM5B_PP7_coding_sequence.txt,pUB_SM_KDM5B_PP7_coding_sequence.txt",
                               "--gene_names=kdm5b,kdm5b",
                               "--intensity_scale=%s"%str(intensity_scale),
                               "--spot_size=%s"%str(spot_size),
                               "--save_vids=0",
                               "--empty_vid_select=generate_from_guassian",
                               "--save_dir=%s"%save_dir], stdout=subprocess.DEVNULL)
                time.sleep(.1)
                pbar.update(1)
                


##############################################################################
# 24000s different intensity cells
#
# Run P300 vs KDM5B at different intensities by manipulating initation rates for
# 24000 total seconds, used to investigate usage of autocorrelation information
# vs frame rate and video length
##############################################################################


if make_list[2]:
    intensity_scale = 14
    sim_time = 24000
    ncells = 50
    save_dir = './P300_KDM5B_24000s_different_intensity_gaussian_14scale'
    
    if debug:
        sim_time = 100
        ncells = 1
        save_dir = './debug/P300_KDM5B_24000s_different_intensity_gaussian_14scale'
    
    
    desired_average_ribosome_count = 5 #lower polysome count
    elongations = [5.33333]
    inits_kdm5b = lambda x: desired_average_ribosome_count/( 1886/x)
    inits_p300 = lambda x: desired_average_ribosome_count/( 2756/x)
    
    
    
    ### manual initations to make kdm5b have a different intensity
    kis_kdm5b = [15/(1886/5.333333)]
    kis_p300 = [5/(2756/5.333333)]

    
    
    
    if not os.path.exists(os.path.join('.', save_dir)):
        os.makedirs(os.path.join('.', save_dir))
    
    print('Running 24000s different intensity 50 cells 50 spots data generation...')
    
    construct_files = ['pUB_SM_KDM5B_PP7_coding_sequence.txt', 'pUB_SM_p300_MS2_coding_sequence.txt']
    
    with tqdm.tqdm(total=len(kis_kdm5b)*2) as pbar :
        for i in range(1):
                kes_str = str(elongations[i]) + "," + str(elongations[i])
                kis_str = str(kis_p300[i]) + "," + str(kis_p300[i])           
                
    
                file_name = 'ki_ke_sweep_similar_int'
                subprocess.run(["python3",
                                "run_rsnaped.py",
                                "--rsnaped_dir=./rsnaped/rsnaped",
                                "--kis=%s"%kis_str,
                                "--kes=%s"%kes_str,
                                "--save_name=%s"%(file_name + '_P300_P300_' + str(kis_p300[i]) + '_' + str(elongations[i])+ '_' + str(i)+'.csv' ),
                                "--n_cells=%s"%str(ncells),
                               "--sim_time=%f"%sim_time,
                               "--n_spots=25,25",
                               "--diff_rates=%s,%s"%(str(diffusion_rate), str(diffusion_rate)),
                               "--genes=pUB_SM_p300_MS2_coding_sequence.txt,pUB_SM_p300_MS2_coding_sequence.txt",
                               "--gene_names=p300,p300",
                               "--intensity_scale=%s"%str(intensity_scale),
                               "--spot_size=%s"%str(spot_size),
                               "--save_vids=0",
                               "--empty_vid_select=generate_from_guassian",
                               "--save_dir=%s"%save_dir], stdout=subprocess.DEVNULL)
                time.sleep(.1)
                pbar.update(1)
                
                kis_str = str(kis_kdm5b[i]) + "," + str(kis_kdm5b[i])           
                
    
                file_name = 'ki_ke_sweep_similar_int'
                subprocess.run(["python3",
                                "run_rsnaped.py",
                                "--rsnaped_dir=./rsnaped/rsnaped",
                                "--kis=%s"%kis_str,
                                "--kes=%s"%kes_str,
                                "--save_name=%s"%(file_name + '_KDM5B_KDM5B_' + str(kis_kdm5b[i]) + '_' + str(elongations[i])+ '_' + str(i)+'.csv' ),
                                "--n_cells=%s"%str(ncells),
                               "--sim_time=%f"%sim_time,
                               "--n_spots=25,25",
                               "--diff_rates=%s,%s"%(str(diffusion_rate), str(diffusion_rate)),
                               "--genes=pUB_SM_KDM5B_PP7_coding_sequence.txt,pUB_SM_KDM5B_PP7_coding_sequence.txt",
                               "--gene_names=kdm5b,kdm5b",
                               "--intensity_scale=%s"%str(intensity_scale),
                               "--spot_size=%s"%str(spot_size),
                               "--save_vids=0",
                               "--empty_vid_select=generate_from_guassian",
                               "--save_dir=%s"%save_dir], stdout=subprocess.DEVNULL)
                time.sleep(.1)
                pbar.update(1)
                



##############################################################################
# KE vs KI parameter sweep, 5000 spots x 3000s
##############################################################################

if make_list[4]:
    
    intensity_scale = 7
    sim_time = 3000
    ncells = 50
    save_dir = "par_sweep_5000"
    ki = np.linspace(0.01, 0.1, 10)
    ke = np.linspace(2, 12, 10)
    
    if debug:
        sim_time = 100
        ncells = 1        
        save_dir = "./debug/par_sweep_5000"
        ki = np.linspace(0.01, 0.1, 2)
        ke = np.linspace(2, 12, 2)
        
    ### KE vs KI

    
    
    if not os.path.exists(os.path.join('.', save_dir)):
        os.makedirs(os.path.join('.', save_dir))
    
    
    print('Running data generation for Ke vs Ki...')
    
    with tqdm.tqdm(total=len(ki)*len(ke)) as pbar :
        for i in range(len(ki)): 
            for j in range(len(ke)):
            
                kes_str = str(ke[j]) + "," + str(ke[j])
                kis_str = str(ki[i]) + "," + str(ki[i])
                file_name = 'ki_ke_sweep_5000spots'
                subprocess.run(["python3",
                                "run_rsnaped.py",
                                "--rsnaped_dir=./rsnaped/rsnaped",
                                "--kis=%s"%kis_str,
                                "--kes=%s"%kes_str,
                                "--save_name=%s"%(file_name + '_' + str(ki[i]) + '_' + str(ke[j])+'.csv' ),
                                "--n_cells=%i"%2*ncells,
                               "--sim_time=%f"%sim_time,
                               "--n_spots=25,25",
                               "--diff_rates=%s,%s"%(str(diffusion_rate), str(diffusion_rate)),
                               "--genes=pUB_SM_KDM5B_PP7_coding_sequence.txt,pUB_SM_p300_MS2_coding_sequence.txt",
                               "--gene_names=kdm5b,p300",
                               "--save_dir=%s"%os.path.join('.', save_dir),
                               "--intensity_scale=%s"%(str(intensity_scale)),
                               "--spot_size=%s"%(str(spot_size))], stdout=subprocess.DEVNULL)
    
    
                time.sleep(.1)
                pbar.update(1)


##############################################################################
# KI vs KI parameter sweep, 5000 spots x 3000s
##############################################################################


if make_list[6]:
    
    intensity_scale = 7
    sim_time = 3000
    ncells = 50
    save_dir = "par_sweep_kis"
    if debug:
        sim_time = 100
        ncells = 1        
        save_dir = "./debug/par_sweep_kis"
        
    
    ### KI vs KI
    ki = np.linspace(0.01, 0.1, 10)
    
    if not os.path.exists(os.path.join('.', save_dir)):
        os.makedirs(os.path.join('.', save_dir))
    
    
    print('Running data generation KI vs KI...')
    
    
    with tqdm.tqdm(total=10) as pbar:
        for i in range(0,10): 
            
            kis_str = str(ki[i]) + "," + str(ki[i])
            file_name = 'parsweep_kis_kdm5b'
            subprocess.run(["python3",
                                "run_rsnaped.py",
                                "--rsnaped_dir=./rsnaped/rsnaped",
                                "--kis=%s"%kis_str,
                                "--kes=%s"%"5.333,5.333",
                                "--save_name=%s"%(file_name + '_' + str(ki[i]) +'.csv' ),
                                "--n_cells=%i"%ncells,
                               "--sim_time=%f"%sim_time,
                               "--n_spots=25,25",
                               "--diff_rates=%s,%s"%(str(diffusion_rate), str(diffusion_rate)),
                               "--genes=pUB_SM_KDM5B_PP7_coding_sequence.txt,pUB_SM_KDM5B_PP7_coding_sequence.txt",
                               "--gene_names=kdm5b,kdm5b",
                               "--save_dir=%s"%os.path.join('.', save_dir),
                               "--intensity_scale=%s"%(str(intensity_scale)),
                               "--spot_size=%s"%(str(spot_size))], stdout=subprocess.DEVNULL)
    
            time.sleep(.1)
            pbar.update(1)
    
    with tqdm.tqdm(total=10) as pbar:
        for i in range(0,10): 
            
            kis_str = str(ki[i]) + "," + str(ki[i])
            file_name = 'parsweep_kis_p300'
            subprocess.run(["python3",
                                "run_rsnaped.py",
                                "--rsnaped_dir=./rsnaped/rsnaped",
                                "--kis=%s"%kis_str,
                                "--kes=%s"%"5.333,5.333",
                                "--save_name=%s"%(file_name + '_' + str(ki[i]) +'.csv' ),
                                "--n_cells=%i"%ncells,
                               "--sim_time=%f"%sim_time,
                               "--n_spots=25,25",
                               "--diff_rates=%s,%s"%(str(diffusion_rate), str(diffusion_rate)),
                               "--genes=pUB_SM_p300_MS2_coding_sequence.txt,pUB_SM_p300_MS2_coding_sequence.txt",
                               "--gene_names=p300,p300",
                               "--save_dir=%s"%os.path.join('.', save_dir),
                               "--intensity_scale=%s"%(str(intensity_scale)),
                               "--spot_size=%s"%(str(spot_size))], stdout=subprocess.DEVNULL)
            time.sleep(.1)
            pbar.update(1)


##############################################################################
# KE vs KE parameter sweep, 5000 spots x 3000s
##############################################################################

if make_list[5]:
    
    intensity_scale = 7
    sim_time = 3000
    ncells = 50
    save_dir = "par_sweep_kes"
    if debug:
        sim_time = 100
        ncells = 1        
        save_dir = "./debug/par_sweep_kes"
        
    ke = np.linspace(2, 12, 10)
    
    
    print('Running data generation KE vs KE...')
    import os
    
    
    if not os.path.exists(os.path.join('.', save_dir)):
        os.makedirs(os.path.join('.', save_dir))
    
    
    with tqdm.tqdm(total=10) as pbar:
        for i in range(0,10): 
            
            kes_str = str(ke[i]) + "," + str(ke[i])
            file_name = 'parsweep_kes_kdm5b'
            subprocess.run(["python3",
                                "run_rsnaped.py",
                                "--rsnaped_dir=./rsnaped/rsnaped",
                                "--kis=%s"%"0.06,0.06",
                                "--kes=%s"%kes_str,
                                "--save_name=%s"%(file_name + '_' + str(ke[i]) +'.csv' ),
                                "--n_cells=%i"%ncells,
                               "--sim_time=%f"%sim_time,
                               "--n_spots=25,25",
                               "--diff_rates=%s,%s"%(str(diffusion_rate), str(diffusion_rate)),
                               "--genes=pUB_SM_KDM5B_PP7_coding_sequence.txt,pUB_SM_KDM5B_PP7_coding_sequence.txt",
                               "--gene_names=kdm5b,kdm5b",
                               "--save_dir=%s"%os.path.join('.', save_dir),
                               "--intensity_scale=%s"%(str(intensity_scale)),
                               "--spot_size=%s"%(str(spot_size))], stdout=subprocess.DEVNULL)
            time.sleep(.1)
            pbar.update(1)
    
    with tqdm.tqdm(total=10) as pbar:
        for i in range(0,10): 
            
            kes_str = str(ke[i]) + "," + str(ke[i])
            file_name = 'parsweep_kes_p300'
            subprocess.run(["python3",
                                "run_rsnaped.py",
                                "--rsnaped_dir=./rsnaped/rsnaped",
                                "--kis=%s"%"0.06,0.06",
                                "--kes=%s"%kes_str,
                                "--save_name=%s"%(file_name + '_' + str(ke[i]) +'.csv' ),
                                "--n_cells=%i"%ncells,
                               "--sim_time=%f"%sim_time,
                               "--n_spots=25,25",
                               "--diff_rates=%s,%s"%(str(diffusion_rate), str(diffusion_rate)),
                               "--genes=pUB_SM_p300_MS2_coding_sequence.txt,pUB_SM_p300_MS2_coding_sequence.txt",
                               "--gene_names=p300,p300",
                               "--save_dir=%s"%os.path.join('.', save_dir),
                               "--intensity_scale=%s"%(str(intensity_scale)),
                               "--spot_size=%s"%(str(spot_size))], stdout=subprocess.DEVNULL)
            time.sleep(.1)
            pbar.update(1)




##############################################################################
# Varying length gene data set
##############################################################################

if make_list[3]:
    
    intensity_scale = 7
    sim_time = 3000
    ncells = 50
    save_dir = "./construct_length_dataset_larger_range_14scale"
    if debug:
        sim_time = 100
        ncells = 1        
        save_dir = "./debug/construct_length_dataset_larger_range_14scale"
        
    
    
    if not os.path.exists(os.path.join('.', save_dir)):
        os.makedirs(os.path.join('.', save_dir))
    

    
    print('Running data generation for construct lengths...')
    import os
    construct_files = os.listdir('./variable_length_genes_larger_range')
    print(len(construct_files))
    
    
    previous_datasets = os.listdir(save_dir)
    previous_sets = [set([x.split('.')[0].split('_')[-1], x.split('.')[0].split('_')[-2]  ]) for x in previous_datasets]
    print(previous_sets)
    with tqdm.tqdm(total=len(construct_files)) as pbar :
    
        for i in range(len(construct_files)):
            g1 = construct_files[i].replace('.fasta','')
            g2 = construct_files[i].replace('.fasta','')
            print('_________')
            print(g1,' ',g2)
    
            if set([g1,g2 ]) not in previous_sets:
    
                    
                kis_str = '.06,.06'
                kes_str = '5.33,5.33'
                construct_file = construct_files[i]
                file_name = 'construct_lengths'
                subprocess.run(["python3",
                                "run_rsnaped.py",
                                "--rsnaped_dir=./rsnaped/rsnaped",
                                "--kis=%s"%kis_str,
                                "--kes=%s"%kes_str,
                                "--save_name=%s"%(file_name + '_' + construct_files[i][:-6]+'_'+construct_files[i][:-6]+'.csv'),
                                "--n_cells=%i"%ncells,
                               "--sim_time=%f"%sim_time,
                               "--n_spots=25,25",
                               "--diff_rates=%s,%s"%(str(diffusion_rate), str(diffusion_rate)),
                               "--genes=%s,%s"%(construct_files[i],construct_files[i]),
                               "--gene_names=%s,%s"%(construct_files[i][:-6],construct_files[i][:-6]),
                               "--save_dir=%s"%save_dir,
                               "--intensity_scale=%s"%(str(intensity_scale)),
                               "--spot_size=%s"%(str(spot_size))], stdout=subprocess.DEVNULL)
               
                previous_sets.append(set([g1,g2]))
            else:
                print('already made... skipping....')
    
            time.sleep(.1)
            pbar.update(1)
    
    
    construct_files = ['pUB_SM_KDM5B_PP7_coding_sequence.txt', 'pUB_SM_p300_MS2_coding_sequence.txt']
    names = ['KDM5B','P300']
    
    with tqdm.tqdm(total=2) as pbar :
    
        for i in range(len(construct_files)):
    
                    
            kis_str = '.06,.06'
            kes_str = '5.33,5.33'
            construct_file = construct_files[i]
            file_name = 'construct_lengths'
            subprocess.run(["python3",
                            "run_rsnaped.py",
                            "--rsnaped_dir=./rsnaped/rsnaped",
                            "--kis=%s"%kis_str,
                            "--kes=%s"%kes_str,
                            "--save_name=%s"%(file_name + '_' + names[i]+'_'+ names[i] +'.csv'),
                            "--n_cells=%i"%ncells,
                            "--sim_time=%f"%sim_time,
                            "--n_spots=25,25",
                            "--diff_rates=%s,%s"%(str(diffusion_rate), str(diffusion_rate)),
                            "--genes=%s,%s"%(construct_files[i],construct_files[i]),
                            "--gene_names=%s,%s"%(names[i],names[i]),
                            "--save_dir=%s"%save_dir,
                            "--save_vids=%s"%(str(0)),
                            "--intensity_scale=%s"%(str(intensity_scale)),
                            "--spot_size=%s"%(str(spot_size))], stdout=subprocess.DEVNULL)
            
    
            time.sleep(.1)
            pbar.update(1)



##############################################################################
# Alternate Tagging schemes Experiment
##############################################################################

if make_list[7]:


    ke_kdm5b = 7.555555
    ke_p300 = 11.04
    
    save_dir = "par_sweep_different_tags_gaussian"
    intensity_scale = 7
    sim_time = 3000
    ncells = 50
    N_total_videos = 1
    if debug:
        sim_time = 100
        ncells = 1        
        save_dir = "./debug/par_sweep_different_tags_gaussian"
        
        
    print('Running different tagging scheme experiments...')
    import os
    
    
    
    if not os.path.exists(os.path.join('.', save_dir)):
        os.makedirs(os.path.join('.', save_dir))
    
    i = 0
    with tqdm.tqdm(total=5) as pbar:
        if 1 == 1:
            # Plus 5 epitopes
            kes_str = str(ke_kdm5b ) + "," + str(ke_kdm5b )
            file_name = 'parsweep_kdm5b_plus5tag'
            subprocess.run(["python3",
                                "run_rsnaped.py",
                                "--rsnaped_dir=./rsnaped/rsnaped",
                                "--kis=%s"%"0.06,0.06",
                                "--kes=%s"%kes_str,
                                "--save_name=%s"%(file_name + '_' + str(ke_kdm5b) +'.csv' ),
                                "--n_cells=%i"%ncells,
                               "--sim_time=%f"%sim_time,
                               "--n_spots=25,25",
                               "--diff_rates=%s,%s"%(str(diffusion_rate), str(diffusion_rate)),
                               "--genes=plus5_kdm5b.txt,plus5_kdm5b.txt",
                               "--gene_names=kdm5b_plus5,kdm5b_plus5",
                               "--save_dir=%s"%os.path.join('.', save_dir),
                               "--empty_vid_select=generate_from_guassian",
                               "--intensity_scale=%s"%(str(intensity_scale)),
                               "--spot_size=%s"%(str(spot_size))], stdout=subprocess.DEVNULL)
            time.sleep(.1)
            pbar.update(1)
    
    
    
            # minus 5 epitopes
            kes_str = str(ke_kdm5b ) + "," + str(ke_kdm5b )
            file_name = 'parsweep_kdm5b_minus5tag'
            subprocess.run(["python3",
                                "run_rsnaped.py",
                                "--rsnaped_dir=./rsnaped/rsnaped",
                                "--kis=%s"%"0.06,0.06",
                                "--kes=%s"%kes_str,
                                "--save_name=%s"%(file_name + '_' + str(ke_kdm5b) +'.csv' ),
                                "--n_cells=%i"%ncells,
                               "--sim_time=%f"%sim_time,
                               "--n_spots=25,25",
                               "--diff_rates=%s,%s"%(str(diffusion_rate), str(diffusion_rate)),
                               "--genes=minus5_kdm5b.txt,minus5_kdm5b.txt",
                               "--gene_names=kdm5b_minus5,kdm5b_minus5",
                               "--save_dir=%s"%os.path.join('.', save_dir),
                               "--empty_vid_select=generate_from_guassian",
                               "--intensity_scale=%s"%(str(intensity_scale)),
                               "--spot_size=%s"%(str(spot_size))], stdout=subprocess.DEVNULL)
            time.sleep(.1)
            pbar.update(1)
    
            
            # kdm5b no changes
            kes_str = str(ke_kdm5b ) + "," + str(ke_kdm5b )
            file_name = 'parsweep_kdm5b_base'
            subprocess.run(["python3",
                                "run_rsnaped.py",
                                "--rsnaped_dir=./rsnaped/rsnaped",
                                "--kis=%s"%"0.06,0.06",
                                "--kes=%s"%kes_str,
                                "--save_name=%s"%(file_name + '_' + str(ke_kdm5b) +'.csv' ),
                                "--n_cells=%i"%ncells,
                               "--sim_time=%f"%sim_time,
                               "--n_spots=25,25",
                               "--diff_rates=%s,%s"%(str(diffusion_rate), str(diffusion_rate)),
                               "--genes=pUB_SM_KDM5B_PP7_coding_sequence.txt,pUB_SM_KDM5B_PP7_coding_sequence.txt",
                               "--gene_names=kdm5b,kdm5b",
                               "--save_dir=%s"%os.path.join('.', save_dir),
                               "--empty_vid_select=generate_from_guassian",
                               "--intensity_scale=%s"%(str(intensity_scale)),
                               "--spot_size=%s"%(str(spot_size))], stdout=subprocess.DEVNULL)
            time.sleep(.1)
            pbar.update(1)
    
    
            
            # split tag, 3 on the 3' end 7 on the 5' end
            kes_str = str(ke_kdm5b ) + "," + str(ke_kdm5b )
            file_name = 'parsweep_kdm5b_splittag'
            subprocess.run(["python3",
                                "run_rsnaped.py",
                                "--rsnaped_dir=./rsnaped/rsnaped",
                                "--kis=%s"%"0.06,0.06",
                                "--kes=%s"%kes_str,
                                "--save_name=%s"%(file_name + '_' + str(ke_kdm5b) +'.csv' ),
                                "--n_cells=%i"%ncells,
                               "--sim_time=%f"%sim_time,
                               "--n_spots=25,25",
                               "--diff_rates=%s,%s"%(str(diffusion_rate), str(diffusion_rate)),
                               "--genes=splittag_kdm5b.txt,splittag_kdm5b.txt",
                               "--gene_names=kdm5b_splittag,kdm5b_splittag",
                               "--save_dir=%s"%os.path.join('.', save_dir),
                               "--empty_vid_select=generate_from_guassian",
                               "--intensity_scale=%s"%(str(intensity_scale)),
                               "--spot_size=%s"%(str(spot_size))], stdout=subprocess.DEVNULL)
            time.sleep(.1)
            pbar.update(1)
            
            # 3'UTR 10x Flag tag
            kes_str = str(ke_kdm5b ) + "," + str(ke_kdm5b )
            file_name = 'parsweep_kdm5b_threeprimetag'
            subprocess.run(["python3",
                                "run_rsnaped.py",
                                "--rsnaped_dir=./rsnaped/rsnaped",
                                "--kis=%s"%"0.06,0.06",
                                "--kes=%s"%kes_str,
                                "--save_name=%s"%(file_name + '_' + str(ke_kdm5b) +'.csv' ),
                                "--n_cells=%i"%ncells,
                               "--sim_time=%f"%sim_time,
                               "--n_spots=25,25",
                               "--diff_rates=%s,%s"%(str(diffusion_rate), str(diffusion_rate)),
                               "--genes=threeprimetag_kdm5b.txt,threeprimetag_kdm5b.txt",
                               "--gene_names=kdm5b_threeprimetag,kdm5b_threeprimetag",
                               "--save_dir=%s"%os.path.join('.', save_dir),
                               "--empty_vid_select=generate_from_guassian",
                               "--intensity_scale=%s"%(str(intensity_scale)),
                               "--spot_size=%s"%(str(spot_size))], stdout=subprocess.DEVNULL)
            time.sleep(.1)
            pbar.update(1)
            
            # Base p300 construct
            kes_str = str(ke_p300 ) + "," + str(ke_p300 )
            file_name = 'parsweep_p300'
            subprocess.run(["python3",
                                "run_rsnaped.py",
                                "--rsnaped_dir=./rsnaped/rsnaped",
                                "--kis=%s"%"0.06,0.06",
                                "--kes=%s"%kes_str,
                                "--save_name=%s"%(file_name + '_' + str(ke_p300) +'.csv' ),
                                "--n_cells=%i"%ncells,
                               "--sim_time=%f"%sim_time,
                               "--n_spots=25,25",
                               "--diff_rates=%s,%s"%(str(diffusion_rate), str(diffusion_rate)),
                               "--genes=pUB_SM_p300_MS2_coding_sequence.txt,pUB_SM_p300_MS2_coding_sequence.txt",
                               "--gene_names=p300,p300",
                               "--save_dir=%s"%os.path.join('.', save_dir),
                               "--empty_vid_select=generate_from_guassian",
                               "--intensity_scale=%s"%(str(intensity_scale)),
                               "--spot_size=%s"%(str(spot_size))], stdout=subprocess.DEVNULL)
            time.sleep(.1)
            pbar.update(1)
    
        
    



##############################################################################
# Multiplexing cells, 4 genes in the green channel, 3 in the blue
##############################################################################

if make_list[8]:
    
    tagging_regime = {'color 0': ['ORC2', 'TRIM33', 'PHIP'],
     'color 1': ['RRAGC', 'LONRF2', 'MAP3K6', 'DOCK8'],
     'missing': ['COL3A1', 'KDM6B', 'EDEM3']}
    
    
    intensity_scale = 14
    N_total_videos = 50 
    sim_time = 1000
    ki = .06 #1/s
    ke = 5.3333 # aa/s
    nspots_each = 10
    data_dir = 'multiplexing_vids_gaussian_14scale'
    
    if debug:
        intensity_scale = 14
        N_total_videos = 1 
        sim_time = 100
        nspots_each = 1
        data_dir = './debug/multiplexing_vids_gaussian_14scale'        
        
    if not os.path.exists(os.path.join('.', data_dir)):
        os.makedirs(os.path.join('.', data_dir))
    
    
    print('Running data generation for 50 multiplexing cells...')
    
    with tqdm.tqdm(total=N_total_videos) as pbar:
        for i in range(N_total_videos):
    
            video_name = ''
            
            print('________')
            
            gene_names = 'ORC2,TRIM33,PHIP,RRAGC,LONRF2,MAP3K6,DOCK8'
            construct_file1 =  'ORC2' + '.fasta'
            construct_file2 = 'TRIM33' + '.fasta'
            construct_file3 =  'PHIP' + '.fasta'
            construct_file4 =   'RRAGC' + '.fasta'
            construct_file5 = 'LONRF2' + '.fasta'
            construct_file6 =   'MAP3K6' + '.fasta'
            construct_file7 =   'DOCK8' + '.fasta'
            file_str = construct_file1 + ','+ construct_file2 + ',' + construct_file3+ ','+  construct_file4  + ','+  construct_file5 + ','+  construct_file6  + ','+  construct_file7
            file_name = 'multiplexing_7_'
            
            subprocess.run(["python3",
                            "run_rsnaped.py",
                            "--rsnaped_dir=./rsnaped/rsnaped",
                            "--kis=%s"%(((str(ki)+ ',')*7)[:-1]),
                            "--kes=%s"%(((str(ke)+ ',')*7)[:-1]),
                            "--save_name=%s"%(file_name + '_' + 'cell' + str(i)),
                            "--n_cells=%s"%str(N_total_videos),
                           "--sim_time=%f"%sim_time,
                           "--n_spots=%s"%((str(nspots_each)+ ',')*7)[:-1],
                           "--diff_rates=%s"%((str(diffusion_rate)+ ',')*7)[:-1],
                           "--genes=%s"%file_str,
                           "--gene_names=%s"%gene_names,
                           "--protein_channels=%s"%'2,2,2,1,1,1,1',
                           "--save_dir=./%s"%data_dir,
                           "--save_vids=0",
                           "--empty_vid_select=generate_from_guassian",
                           "--mRNA_tag_channels=%s"%((str(0)+ ',')*7)[:-1],
                           "--mRNA_tag_intensity_scale=%s"%((str(0)+ ',')*7)[:-1],
                           "--intensity_scale=%s"%(str(intensity_scale)),
                           "--spot_size=%s"%(str(spot_size))], stdout=subprocess.DEVNULL)
    
        
    
            #os.rename('./cell_0.tif', './' + file_name  + str(i) + '.tif')
            time.sleep(.1)
            pbar.update(1)
        
##############################################################################
# single example multiplexing video for labeling the figure
##############################################################################


if make_list[8]:
    intensity_scale = 14
    sim_time = 1000
    ncells = 1
    N_total_videos = 1
    
    data_dir = './multiplexing_vids_gaussian_14scale'
    if debug:
        intensity_scale = 14
        N_total_videos = 1 
        sim_time = 100
        nspots_each = 10
        data_dir = './debug/multiplexing_vids_gaussian_14scale'  
        
    print('Running data generation for 1 multiplexing cells and saving video...')
    with tqdm.tqdm(total=1) as pbar:
        for i in range(N_total_videos):
    
            video_name = ''
            
            print('________')
            
            gene_names = 'ORC2,TRIM33,PHIP,RRAGC,LONRF2,MAP3K6,DOCK8'
            construct_file1 =  'ORC2' + '.fasta'
            construct_file2 = 'TRIM33' + '.fasta'
            construct_file3 =  'PHIP' + '.fasta'
            construct_file4 =   'RRAGC' + '.fasta'
            construct_file5 = 'LONRF2' + '.fasta'
            construct_file6 =   'MAP3K6' + '.fasta'
            construct_file7 =   'DOCK8' + '.fasta'
            file_str = construct_file1 + ','+ construct_file2 + ',' + construct_file3+ ','+  construct_file4  + ','+  construct_file5 + ','+  construct_file6  + ','+  construct_file7
            file_name = 'multiplexing_7_'
            
            subprocess.run(["python3",
                            "run_rsnaped.py",
                            "--rsnaped_dir=./rsnaped/rsnaped",
                            "--kis=%s"%(((str(ki)+ ',')*7)[:-1]),
                            "--kes=%s"%(((str(ke)+ ',')*7)[:-1]),
                            "--save_name=%s"%(file_name + '_' + 'cell_single' + str(i)),
                            "--n_cells=1",
                           "--sim_time=%f"%sim_time,
                           "--n_spots=%s"%((str(nspots_each)+ ',')*7)[:-1],
                           "--diff_rates=%s"%((str(diffusion_rate)+ ',')*7)[:-1],
                           "--genes=%s"%file_str,
                           "--gene_names=%s"%gene_names,
                           "--protein_channels=%s"%'2,2,2,1,1,1,1',
                           "--save_dir=%s"%data_dir,
                           "--save_vids=1",
                           "--empty_vid_select=generate_from_guassian",
                           "--mRNA_tag_channels=%s"%((str(0)+ ',')*7)[:-1],
                           "--mRNA_tag_intensity_scale=%s"%((str(0)+ ',')*7)[:-1],
                           "--intensity_scale=%s"%(str(intensity_scale)),
                           "--spot_size=%s"%(str(spot_size))], stdout=subprocess.DEVNULL)
    
    
            os.rename('./cell_0.tif', './' + file_name  + str(i) + '.tif')
            shutil.move('./' + file_name  + str(i) + '.tif', data_dir + file_name  + str(i) + '.tif' )
            time.sleep(.1)
            pbar.update(1)
    


##############################################################################
# photobleaching examples generate 75 cells at varying photobleaching rates
##############################################################################


if make_list[9]:
    intensity_scale = 14
    sim_time = 1000
    ncells = 1
    N_total_videos = 1
    N_cells = 75
    data_dir = './P300_KDM5B_350s_base_pb'
    
    if debug:
        intensity_scale = 14
        N_total_videos = 1 
        N_cells = 1
        sim_time = 100
        ki = .06 #1/s
        ke = 5.3333 # aa/s
        nspots_each = 10
        data_dir = './debug/P300_KDM5B_350s_base_pb'  
        
    desired_average_ribosome_count = 5 #lower polysome count
    elongations = [5.33333]
    inits_kdm5b = lambda x: desired_average_ribosome_count/( 1886/x)
    inits_p300 = lambda x: desired_average_ribosome_count/( 2756/x)
    
    
    
    ### manual initations to make kdm5b have intensity of 5.5
    kis_kdm5b = [0.06]
    kis_p300 = [0.06]
    
    #kis_kdm5b = [inits_kdm5b(elongations[0])]
    #kis_p300 = [inits_p300(elongations[0])]
    
    print(kis_kdm5b)
    print(kis_p300)
    
    
    
    if not os.path.exists(os.path.join('.', data_dir)):
        os.makedirs(os.path.join('.', data_dir))
    
    
    sim_time = 350
    frame_rate = 5
    nframes = 128
    
    percentage_drop_per_frame = np.array([1, .999,.998, .996, .994, .99, .98, .965, .96, .955, .95])
    pb_rates = np.abs(-np.log(percentage_drop_per_frame)/5) # convert percentage drop per frame to e^-alpha*t
    pb_rates[0] = 0
    
    
    print('Running photobleaching experiments...')
    
    construct_files = ['pUB_SM_KDM5B_PP7_coding_sequence.txt', 'pUB_SM_p300_MS2_coding_sequence.txt']
    print(len(construct_files))
    
    
    
    with tqdm.tqdm(total=len(pb_rates)*2) as pbar :
    
        for i in range(len(pb_rates)):
    
            
                kes_str = str(elongations[0]) + "," + str(elongations[0])
                kis_str = str(kis_p300[0]) + "," + str(kis_p300[0])           
    
                file_name = 'p300_base_pb'
                subprocess.run(["python3",
                                "run_rsnaped.py",
                                "--rsnaped_dir=./rsnaped/rsnaped",
                                "--kis=%s"%kis_str,
                                "--kes=%s"%kes_str,
                                "--save_name=%s"%(file_name + '_P300_P300_' + str(kis_p300[0]) + '_' + str(elongations[0])+ '_' + str(i)+'.csv' ),
                                "--n_cells=%s"%str(N_cells),
                               "--sim_time=%f"%sim_time,
                               "--n_spots=25,25",
                               "--diff_rates=.55,.55",
                               "--genes=pUB_SM_p300_MS2_coding_sequence.txt,pUB_SM_p300_MS2_coding_sequence.txt",
                               "--gene_names=p300,p300",
                               "--mRNA_tag_channels=0,0",
                               "--mRNA_tag_intensity_scale=500,500",
                               "--mRNA_tag_intensity_type=constant",
                               "--intensity_scale=14",
                               "--spot_size=5",
                               "--save_vids=0",
                               "--empty_vid_select=generate_from_guassian",
                               "--pb=constant",
                               "--pb_vid_mu=%s"% '{:f}'.format(pb_rates[i]) + ',' + '{:f}'.format(pb_rates[i])+ ',' +'{:f}'.format(pb_rates[i]),
                               "--pb_vid_var=%s"% '{:f}'.format(pb_rates[i]/5)  + ',' + '{:f}'.format(pb_rates[i]/5)   + ',' + '{:f}'.format(pb_rates[i]/5) ,
                               "--pb_spot_mu=%s"%'{:f}'.format(pb_rates[i])  + ',' + '{:f}'.format(pb_rates[i])  + ',' +'{:f}'.format(pb_rates[i])  ,
                               "--pb_spot_var=%s"%'{:f}'.format(pb_rates[i]/5)  + ',' + '{:f}'.format(pb_rates[i]/5)  + ',' + '{:f}'.format(pb_rates[i]/5)  ,
                               "--save_dir=%s"%data_dir,
                               "--tracking=1"], stdout=subprocess.DEVNULL)
                
                #os.rename('./cell_0.tif', './' + file_name  + str(i) + '.tif')
                time.sleep(.1)
                pbar.update(1)
    
    
                kis_str = str(kis_kdm5b[0]) + "," + str(kis_kdm5b[0])           
                
    
                file_name = 'kdm5b_base_pb'
                subprocess.run(["python3",
                                "run_rsnaped.py",
                                "--rsnaped_dir=./rsnaped/rsnaped",
                                "--kis=%s"%kis_str,
                                "--kes=%s"%kes_str,
                                "--save_name=%s"%(file_name + '_KDM5B_KDM5B_' + str(kis_kdm5b[0]) + '_' + str(elongations[0])+ '_' + str(i)+'.csv' ),
                                "--n_cells=%s"%str(N_cells),
                               "--sim_time=%f"%sim_time,
                               "--n_spots=25,25",
                               "--diff_rates=.55,.55",
                               "--genes=pUB_SM_KDM5B_PP7_coding_sequence.txt,pUB_SM_KDM5B_PP7_coding_sequence.txt",
                               "--gene_names=kdm5b,kdm5b",
                               "--mRNA_tag_channels=0,0",
                               "--mRNA_tag_intensity_scale=500,500",
                               "--mRNA_tag_intensity_type=constant",
                               "--intensity_scale=14",
                               "--spot_size=5",
                               "--save_vids=0",
                               "--empty_vid_select=generate_from_guassian",
                               "--pb=constant",
                               "--pb_vid_mu=%s"% '{:f}'.format(pb_rates[i]) + ',' + '{:f}'.format(pb_rates[i])+ ',' +'{:f}'.format(pb_rates[i]),
                               "--pb_vid_var=%s"% '{:f}'.format(pb_rates[i]/5)  + ',' + '{:f}'.format(pb_rates[i]/5)   + ',' + '{:f}'.format(pb_rates[i]/5) ,
                               "--pb_spot_mu=%s"%'{:f}'.format(pb_rates[i])  + ',' + '{:f}'.format(pb_rates[i])  + ',' +'{:f}'.format(pb_rates[i])  ,
                               "--pb_spot_var=%s"%'{:f}'.format(pb_rates[i]/5)  + ',' + '{:f}'.format(pb_rates[i]/5)  + ',' + '{:f}'.format(pb_rates[i]/5)  ,
                               "--save_dir=%s"%data_dir,
                               "--tracking=1"], stdout=subprocess.DEVNULL)
                time.sleep(.1)
                pbar.update(1)
                
    
    
    
    with tqdm.tqdm(total=len(pb_rates)) as pbar :
    
        for i in range(len(pb_rates)):
    
            
                kes_str = str(elongations[0]) + "," + str(elongations[0])
                kis_str = str(kis_p300[0]) + "," + str(kis_p300[0])           
                
    
    
    
    
                file_name = 'both_base_pb'
                subprocess.run(["python3",
                                "run_rsnaped.py",
                                "--rsnaped_dir=./rsnaped/rsnaped",
                                "--kis=%s"%kis_str,
                                "--kes=%s"%kes_str,
                                "--save_name=%s"%(file_name + '_KDM5B_P300_' + str(kis_p300[0]) + '_' + str(elongations[0])+ '_video_cell_' + str(i)+'.csv' ),
                                "--n_cells=1",
                               "--sim_time=%f"%300,
                               "--n_spots=25,25",
                               "--diff_rates=.55,.55",
                               "--genes=pUB_SM_KDM5B_PP7_coding_sequence.txt,pUB_SM_p300_MS2_coding_sequence.txt",
                               "--gene_names=kdm5b,p300",
                               "--intensity_scale=14",
                               "--spot_size=5",
                               "--save_vids=1",
                               "--empty_vid_select=generate_from_guassian",
                               "--mRNA_tag_channels=0,0",
                               "--mRNA_tag_intensity_scale=100,100",
                               "--mRNA_tag_intensity_type=constant",
                               "--pb=normal",
                               "--pb_vid_mu=%s"% '{:f}'.format(pb_rates[i]) + ',' + '{:f}'.format(pb_rates[i])+ ',' +'{:f}'.format(pb_rates[i]),
                               "--pb_vid_var=%s"% '{:f}'.format(pb_rates[i]/5)  + ',' + '{:f}'.format(pb_rates[i]/5)   + ',' + '{:f}'.format(pb_rates[i]/5) ,
                               "--pb_spot_mu=%s"%'{:f}'.format(pb_rates[i])  + ',' + '{:f}'.format(pb_rates[i])  + ',' +'{:f}'.format(pb_rates[i])  ,
                               "--pb_spot_var=%s"%'{:f}'.format(pb_rates[i]/5)  + ',' + '{:f}'.format(pb_rates[i]/5)  + ',' + '{:f}'.format(pb_rates[i]/5)  ,
                               "--save_dir=%s"%data_dir,
                               "--tracking=1"], stdout=subprocess.DEVNULL)
                
                os.rename('./cell_0.tif', './' + file_name  + str(i) + '.tif')
                shutil.move('./' + file_name  + str(i) + '.tif', data_dir + file_name  + str(i) + '.tif' )
                time.sleep(.1)
                pbar.update(1)
    


if debug:
   full_time =  time.time() - st
   print('ran all data making commands in debug mode: %s seconds'%(str(full_time)))