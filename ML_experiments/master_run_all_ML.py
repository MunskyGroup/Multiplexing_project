# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 14:30:11 2022

@author: wsraymon
"""

############# main machine learning experiments #############

import os
import subprocess


DEBUG = 1

training_data_size = 1
example_labeling = 1
long_intensity = 1
parsweeps = 1
supplemental_sweeps = 1
tagging = 1
multiplexing = 1
photobleaching = 1


runexp = [
            training_data_size,
            example_labeling,
            long_intensity,
            parsweeps,
            supplemental_sweeps,
            tagging,
            multiplexing,
            photobleaching,
                      ]

def run_heatmaps_TL(base_name,global_fr, global_nf, global_samples, witheld, test_size, subsamples, subsample_size, model_file):
    subprocess.run(['python', './scripts/run_all_heatmaps_cnn_TL.py',
                    '--base_dir=%s'%('./scripts/' + base_command),
                    '--global_fr=%s'%str(global_fr),
                    '--global_nf=%s'%str(global_nf),
                    '--global_samples=%s'%str(global_samples),
                    '--witheld=%s'%(str(witheld)),
                    '--test_size=%s'%(str(test_size)),
                    '--base_model=%s'%model_file, 
                    '--subsample=%s'%(str(subsamples)),
                    '--n_subsamples=%s'%(str(subsamples)),
                    '--debug=%s'%(str(DEBUG)),
                    ])
    
    
    subprocess.run(['python', 'plot_ml_heatmap_from_keyfile.py', '--target_dir=%s'%base_name,'--format=svg'])
    subprocess.run(['python', 'plot_ml_heatmap_from_keyfile.py', '--target_dir=%s'%base_name,'--format=png'])


def run_single_grid(base_name, base_command, d1, d2, global_samples, witheld, test_size):
    subprocess.run(['python', './scripts/run_single_commandline.py',
                    '--base_dir=%s'%base_name,
                    '--base_command=%s'%('./scripts/' + base_command),
                    '--global_samples=%s'%str(global_samples),
                    '--witheld=%s'%(str(witheld)),
                    '--datafile2=%s'%(d2),
                    '--datafile1=%s'%(d1),
                    '--test_size=%s'%(str(test_size)),
                    '--test_type=%s'%'freq',
                    '--debug=%s'%(str(DEBUG)),
                    ])
    
    subprocess.run(['python', 'plot_ml_heatmap_from_keyfile.py', '--target_dir=%s'%base_name,'--format=svg'])
    subprocess.run(['python', 'plot_ml_heatmap_from_keyfile.py', '--target_dir=%s'%base_name,'--format=png'])


def run_training_size(base_name, base_command, d1, d2, sizes):
    subprocess.run(['python', './scripts/run_training_size_exp.py',
                    '--base_dir=%s'%base_name,
                    '--base_command=%s'%('./scripts/' + base_command),
                    '--sizes=%s'%str(sizes),
                    '--debug=%s'%(str(DEBUG)),
                    ])
    
    subprocess.run(['python', 'plot_ml_heatmap_from_keyfile.py', '--target_dir=%s'%base_name,'--format=svg'])
    subprocess.run(['python', 'plot_ml_heatmap_from_keyfile.py', '--target_dir=%s'%base_name,'--format=png'])


def run_heatmaps(base_name,base_command,global_fr, global_nf, global_samples, witheld, test_size, runwho):
    subprocess.run(['python', './scripts/run_heatmaps_commandline.py',
                    '--base_dir=%s'%base_name,
                    '--base_command=%s'% ('./scripts/' + base_command),
                    '--global_fr=%s'%str(global_fr),
                    '--global_nf=%s'%str(global_nf),
                    '--global_samples=%s'%str(global_samples),
                    '--witheld=%s'%(str(witheld)),
                    '--test_size=%s'%(str(test_size)),
                    '--cl=%s'%(str(runwho[0])),
                    '--img=%s'%(str(runwho[1])),
                    '--keki=%s'%(str(runwho[2])),
                    '--kes=%s'%(str(runwho[3])),
                    '--kis=%s'%(str(runwho[4])),
                    '--debug=%s'%(str(DEBUG)),
    
                    ])
    
    
    subprocess.run(['python', 'plot_ml_heatmap_from_keyfile.py', '--target_dir=%s'%base_name,'--format=svg'])
    subprocess.run(['python', 'plot_ml_heatmap_from_keyfile.py', '--target_dir=%s'%base_name,'--format=png'])


def run_heatmaps_14scale(base_name,base_command,global_fr, global_nf, global_samples, witheld, test_size, runwho):
    subprocess.run(['python', './scripts/run_heatmaps_commandline_14scale.py',
                    '--base_dir=%s'%base_name,
                    '--base_command=%s'%('./scripts/' + base_command),
                    '--global_fr=%s'%str(global_fr),
                    '--global_nf=%s'%str(global_nf),
                    '--global_samples=%s'%str(global_samples),
                    '--witheld=%s'%(str(witheld)),
                    '--test_size=%s'%(str(test_size)),
                    '--cl=%s'%(str(runwho[0])),
                    '--img=%s'%(str(runwho[1])),
                    '--keki=%s'%(str(runwho[2])),
                    '--kes=%s'%(str(runwho[3])),
                    '--kis=%s'%(str(runwho[4])),
                    '--debug=%s'%(str(DEBUG)),
    
                    ])
    
    
    subprocess.run(['python', 'plot_ml_heatmap_from_keyfile.py', '--target_dir=%s'%base_name,'--format=svg'])
    subprocess.run(['python', 'plot_ml_heatmap_from_keyfile.py', '--target_dir=%s'%base_name,'--format=png'])


def run_heatmaps_long(base_name,base_command, global_samples, witheld, test_size,):
    subprocess.run(['python', './scripts/run_heatmaps_commandline_long.py',
                    '--base_dir=%s'%base_name,
                    '--base_command=%s'%('./scripts/' + base_command),
                    '--global_samples=%s'%str(global_samples),
                    '--witheld=%s'%(str(witheld)),
                    '--test_size=%s'%(str(test_size)),
                    '--debug=%s'%(str(DEBUG)),
                    ])
    
    subprocess.run(['python', 'plot_ml_heatmap_from_keyfile.py', '--target_dir=%s'%base_name,'--format=svg'])
    subprocess.run(['python', 'plot_ml_heatmap_from_keyfile.py', '--target_dir=%s'%base_name,'--format=png'])



def run_heatmaps_long_freq_comp(base_name,base_command, global_samples, witheld, test_size, test_type):
    subprocess.run(['python', './scripts/run_heatmaps_commandline_long_freq_comp.py',
                    '--base_dir=%s'%base_name,
                    '--base_command=%s'%('./scripts/' + base_command),
                    '--global_samples=%s'%str(global_samples),
                    '--witheld=%s'%(str(witheld)),
                    '--test_size=%s'%(str(test_size)),
                    '--test_type=%s'%test_type,
                    '--debug=%s'%(str(DEBUG)),
                    ])
    
    subprocess.run(['python', 'plot_ml_heatmap_from_keyfile.py', '--target_dir=%s'%base_name,'--format=svg'])
    subprocess.run(['python', 'plot_ml_heatmap_from_keyfile.py', '--target_dir=%s'%base_name,'--format=png'])


def run_heatmaps_long_single(base_name, base_command, global_samples, witheld, test_size):
    subprocess.run(['python', './scripts/run_heatmaps_commandline_long_single.py',
                    '--base_dir=%s'%base_name,
                    '--base_command=%s'%base_command,
                    '--global_samples=%s'%str(global_samples),
                    '--witheld=%s'%(str(witheld)),
                    '--test_size=%s'%(str(test_size)),
                    '--debug=%s'%(str(DEBUG)),
                    ])
    
    subprocess.run(['python', 'plot_ml_heatmap_from_keyfile.py', '--target_dir=%s'%base_name,'--format=svg'])
    subprocess.run(['python', 'plot_ml_heatmap_from_keyfile.py', '--target_dir=%s'%base_name,'--format=png'])


def run_heatmaps_frequency_comparison(base_name, base_command, d1, d2, global_samples, witheld, test_size, test_type):
    subprocess.run(['python', './scripts/run_heatmaps_commandline_frequency.py',
                    '--base_dir=%s'%base_name,
                    '--base_command=%s'%('./scripts/' + base_command),
                    '--global_samples=%s'%str(global_samples),
                    '--witheld=%s'%(str(witheld)),
                    '--test_size=%s'%(str(test_size)),
                    '--datafile2=%s'%(d2),
                    '--datafile1=%s'%(d1),
                    '--test_type=%s'%(test_type),
                    '--debug=%s'%(str(DEBUG)),
                    ])
    
    
    subprocess.run(['python', 'plot_ml_heatmap_from_keyfile.py', '--target_dir=%s'%base_name,'--format=svg'])
    subprocess.run(['python', 'plot_ml_heatmap_from_keyfile.py', '--target_dir=%s'%base_name,'--format=png'])



def run_photobleaching(base_name, base_command, global_samples, witheld, test_size, test_type):
    subprocess.run(['python', './scripts/run_heatmaps_commandline_pb2.py',
                    '--base_dir=%s'%base_name,
                    '--base_command=%s'%('./scripts/' + base_command),
                    '--global_samples=%s'%str(global_samples),
                    '--witheld=%s'%(str(witheld)),
                    '--test_size=%s'%(str(test_size)),
                    '--test_type=%s'%(str(test_type)),
                    '--debug=%s'%(str(DEBUG)),
    
                    ])
    
    
    subprocess.run(['python', 'plot_ml_heatmap_from_keyfile.py', '--target_dir=%s'%base_name,'--format=svg'])
    subprocess.run(['python', 'plot_ml_heatmap_from_keyfile.py', '--target_dir=%s'%base_name,'--format=png'])





'''
base_name = 'ML_run_320_5s_wfreq_3000burnin'
base_command = 'run_cnn_commandline_all_w_freq_burnin.py'

# global data setup
global_fr = 5
global_nf = 64
global_samples = 5000
witheld = 1000
test_size = 0

#          cl img  keki  kes kis
runwho =  [0,  0,   1,   1,   0]


run_heatmaps(base_name,base_command,global_fr, global_nf, global_samples, witheld, test_size, runwho)
'''

##############################################################################
# Training data size
##############################################################################

if runexp[0]:
    data_size = [  10,   15,   20,   25,   30,   35,   40,   50,   75,  100,  125,
        150,  165,  200,  225,  250,  275,  300,  325,  350,  376,  400,
        425,  500,  550,  600,  675,  700,  775,  800,  900, 1000, 1250,
       1500, 1750, 2000]
    
    
    print('running training size exp...')
    dirs = ['ML_training_size']
    
    
    data_file1s = ['par_sweep_kes/parsweep_kes_p300_5.333333333333334.csv',]
    
    data_file2s = ['par_sweep_kes/parsweep_kes_kdm5b_5.333333333333334.csv',
                   ]
    base_name = dirs[0]
    data_file1 = data_file1s[0]
    data_file2 = data_file2s[0]
    base_command = 'run_cnn_commandline_all_training_size.py'    
    run_training_size(base_name, base_command, data_file1, data_file2, data_size)


##############################################################################
# Example proof of concept (Figure 3)
##############################################################################


if runexp[1]:
    x=1
    




##############################################################################
# Similar vs Same vs Different intensity 24000 seconds (Figure 4)
##############################################################################


if runexp[2]:

    dirs = ['ML_IF_kisdiff', 'ML_F_kisdiff','ML_I_kisdiff']
    
    
    commands = ['freq', 'no_freq', 'acc_only']
    
    for i in range(len(commands)):
        print(dirs[i])
        base_name = dirs[i]
        base_command = 'run_cnn_commandline_all_w_freq.py'
        
        # global data setup
        retrain = 1
        global_samples = 5000
        witheld = 1000
        test_size = 0
        test_type = commands[i]
            
        run_heatmaps_long_freq_comp(base_name,base_command, global_samples, witheld, test_size, test_type,)
        

##############################################################################
# parameter sweeps (Figure 5)
##############################################################################

if parsweeps:
    base_name = 'ML_run_320_5s_wfreq'
    base_command = 'run_cnn_commandline_all_w_freq.py'
    
    # global data setup
    global_fr = 5
    global_nf = 64
    global_samples = 5000
    witheld = 1000
    test_size = 0
    
    #          cl img  keki  kes kis
    runwho =  [1,  1,   1,   1,   1]
    
    
    run_heatmaps(base_name,base_command,global_fr, global_nf, global_samples, witheld, test_size, runwho)
    
    
    

##############################################################################
# Tagging experiment (Figure 7)
##############################################################################

if tagging:

    print('running tags...')
    dirs = ['ML_run_tag_base', 'ML_run_tag_3prime', 'ML_run_tag_split', 'ML_run_tag_plus5',  'ML_run_tag_minus5' ]
    
    
    data_file1s = ['par_sweep_different_tags/parsweep_p300_11.04.csv',
                   'par_sweep_different_tags/parsweep_p300_11.04.csv',
                   'par_sweep_different_tags/parsweep_p300_11.04.csv',
                   'par_sweep_different_tags/parsweep_p300_11.04.csv',
                   'par_sweep_different_tags/parsweep_p300_11.04.csv']
    
    data_file2s = ['par_sweep_different_tags/parsweep_kdm5b_base_7.555555.csv',
                   'par_sweep_different_tags/parsweep_kdm5b_threeprimetag_7.555555.csv',
                   'par_sweep_different_tags/parsweep_kdm5b_splittag_7.555555.csv',
                   'par_sweep_different_tags/parsweep_kdm5b_plus5tag_7.555555.csv',
                   'par_sweep_different_tags/parsweep_kdm5b_minus5tag_7.555555.csv'
                   ]
    
    for i in range(len(dirs)):
        base_name = dirs[i]
        base_command = 'run_cnn_commandline_all_w_freq.py'
    
        # global data setup
        retrain = 1
        global_samples = 5000
        witheld = 1000
        test_size = 0
        data_file1 = data_file1s[i]
        data_file2 = data_file2s[i]
    #          cl img  keki  kes kis
        runwho =  [0,  1,   0,   0,   0]
    
        run_single_grid(base_name,base_command, data_file1, data_file2, global_samples, witheld, test_size,)
    
    
##############################################################################
# longer supplemental sweeps (Figure 6 and S4)
##############################################################################

if supplemental_sweeps:
    base_name = 'ML_run_3000_2s_wfreq'
    base_command = 'run_cnn_commandline_all_w_freq.py'
    
    # global data setup
    global_fr = 2
    global_nf = 1500
    retrain = 1
    global_samples = 5000
    witheld = 1000
    test_size = 0
    
    #          cl img  keki  kes kis
    runwho =  [1,  1,   1,   1,   1]
    run_heatmaps(base_name,base_command,global_fr, global_nf, global_samples, witheld, test_size, runwho)
    
    ############# supplemental experiments #############
    
    
    base_name = 'ML_run_1280_10s_wfreq'
    base_command = 'run_cnn_commandline_all_w_freq.py'
    
    # global data setup
    global_fr = 10
    global_nf = 128
    retrain = 1
    global_samples = 5000
    witheld = 1000
    test_size = 0
    #          cl img  keki  kes kis
    runwho =  [1,  1,   1,   1,   1]
    run_heatmaps(base_name,base_command,global_fr, global_nf, global_samples, witheld, test_size, runwho)
    
    ############# supplemental experiments #############
    
    
    base_name = 'ML_run_1280_5s_wfreq'
    base_command = 'run_cnn_commandline_all_w_freq.py'
    
    # global data setup
    global_fr = 5
    global_nf = 256
    retrain = 1
    global_samples = 5000
    witheld = 1000
    test_size = 0
    #          cl img  keki  kes kis
    runwho =  [1,  1,   1,   1,   1]
    run_heatmaps(base_name,base_command,global_fr, global_nf, global_samples, witheld, test_size, runwho)
    
    
##############################################################################
# Multiplexing (Figure 9)
##############################################################################

 
if multiplexing: 
    
    # Retrain the classifiers for 14 intensity scale for construct length
    base_name = 'ML_CL_14scale'
    base_command = 'run_cnn_commandline_all_w_freq.py'
    
    # global data setup
    global_fr = 5
    global_nf = 64
    global_samples = 5000
    witheld = 1000
    test_size = 0
    
    #          cl img  keki  kes kis
    runwho =  [1,  0,0,0,0]
    run_heatmaps_14scale(base_name,base_command,global_fr, global_nf, global_samples, witheld, test_size, runwho)
    
    
    # apply those new classifiers to the multiplexing dataset
    dirs = ['ML_multiplexing_green', 'ML_multiplexing_blue']
    
    files = ['./datasets/construct_length_dataset_larger_range/construct_lengths_RRAGC_RRAGC.csv,./datasets/construct_length_dataset_larger_range/construct_lengths_LONRF2_LONRF2.csv,./datasets/construct_length_dataset_larger_range/construct_lengths_MAP3K6_MAP3K6.csv,./datasets/construct_length_dataset_larger_range/construct_lengths_DOCK8_DOCK8.csv',
             './datasets/construct_length_dataset_larger_range/construct_lengths_ORC2_ORC2.csv,./datasets/construct_length_dataset_larger_range/construct_lengths_TRIM33_TRIM33.csv,./datasets/construct_length_dataset_larger_range/construct_lengths_PHIP_PHIP.csv']
    
    colors = ['green','blue']
    nfiles = [4,3]
    
    retrain = 1
    global_samples = 5000
    witheld = 1000
    test_size = 0
    model_base_name = 'cnn_par'
    for i in range(2):
        if not os.path.exists(os.path.join('.', dirs[i])):
            os.makedirs(os.path.join('.', dirs[i]))
        
        subprocess.run(['python', 'run_cnn_commandline_all_multiplexing.py',
                        '--files=%s'% files[i],
                        '--n_files=%s'% str(nfiles[i]),
                        '--save_model=%s'%str(1),
                        '--save_dir=%s'%os.path.join('.',dirs[i]),
                        '--acc_file=%s'%'acc_mat_img.npy',
                        '--retrain=%s'%str(retrain),
                        '--model_file=%s'%'unused',
                        '--model_name=%s'%(model_base_name),
                        '--model_sub_name=%s'%colors[i],
                        '--color=%s'%colors[i],
                        '--verbose=%s'%str(0),
                        '--Fr=%s'%str(5),
                        '--NFr=%s'%str(64),
                        '--ntimes=%s'%(str(3000)),                                   
                        '--witheld=%s'%(str(witheld)),
                        '--test_size=%s'%(str(test_size)),
                        ])


##############################################################################
# PHOTOBLEACHING ML (figure s3)
##############################################################################
if runexp[-1]:
    
    base_name = 'PB'
    base_command = 'run_cnn_commandline_all_w_freq.py'
    
    # global data setup
    retrain = 1
    global_samples = 1400
    witheld = 500
    test_size = 0
    test_type = 'base'
    
    run_photobleaching(base_name,base_command, global_samples, witheld, test_size,'base')
    run_photobleaching(base_name,base_command, global_samples, witheld, test_size,'wo_correction')
    
        

