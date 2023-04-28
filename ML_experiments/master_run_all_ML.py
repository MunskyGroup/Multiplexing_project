# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 14:30:11 2022

@author: wsraymon
"""

############# main experiment #############

import os
import subprocess



def run_heatmaps_TL(base_name,global_fr, global_nf, global_samples, witheld, test_size, subsamples, subsample_size, model_file):
    subprocess.run(['python', 'run_all_heatmaps_cnn_TL.py',
                    '--base_dir=%s'%base_name,
                    '--global_fr=%s'%str(global_fr),
                    '--global_nf=%s'%str(global_nf),
                    '--global_samples=%s'%str(global_samples),
                    '--witheld=%s'%(str(witheld)),
                    '--test_size=%s'%(str(test_size)),
                    '--base_model=%s'%model_file, 
                    '--subsample=%s'%(str(subsamples)),
                    '--n_subsamples=%s'%(str(subsamples)),
                    ])
    
    
    subprocess.run(['python', 'plot_ml_heatmap_from_keyfile.py', '--target_dir=%s'%base_name,'--format=svg'])
    subprocess.run(['python', 'plot_ml_heatmap_from_keyfile.py', '--target_dir=%s'%base_name,'--format=png'])


def run_single_grid(base_name, base_command, d1, d2, global_samples, witheld, test_size):
    subprocess.run(['python', 'run_single_commandline.py',
                    '--base_dir=%s'%base_name,
                    '--base_command=%s'%base_command,
                    '--global_samples=%s'%str(global_samples),
                    '--witheld=%s'%(str(witheld)),
                    '--datafile2=%s'%(d2),
                    '--datafile1=%s'%(d1),
                    '--test_size=%s'%(str(test_size)),
                    '--test_type=%s'%'freq'
                    ])
    
    subprocess.run(['python', 'plot_ml_heatmap_from_keyfile.py', '--target_dir=%s'%base_name,'--format=svg'])
    subprocess.run(['python', 'plot_ml_heatmap_from_keyfile.py', '--target_dir=%s'%base_name,'--format=png'])




def run_heatmaps(base_name,base_command,global_fr, global_nf, global_samples, witheld, test_size, runwho):
    subprocess.run(['python', 'run_heatmaps_commandline.py',
                    '--base_dir=%s'%base_name,
                    '--base_command=%s'%base_command,
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
    
                    ])
    
    
    subprocess.run(['python', 'plot_ml_heatmap_from_keyfile.py', '--target_dir=%s'%base_name,'--format=svg'])
    subprocess.run(['python', 'plot_ml_heatmap_from_keyfile.py', '--target_dir=%s'%base_name,'--format=png'])


def run_heatmaps_14scale(base_name,base_command,global_fr, global_nf, global_samples, witheld, test_size, runwho):
    subprocess.run(['python', 'run_heatmaps_commandline_14scale.py',
                    '--base_dir=%s'%base_name,
                    '--base_command=%s'%base_command,
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
    
                    ])
    
    
    subprocess.run(['python', 'plot_ml_heatmap_from_keyfile.py', '--target_dir=%s'%base_name,'--format=svg'])
    subprocess.run(['python', 'plot_ml_heatmap_from_keyfile.py', '--target_dir=%s'%base_name,'--format=png'])


def run_heatmaps_long(base_name,base_command, global_samples, witheld, test_size,):
    subprocess.run(['python', 'run_heatmaps_commandline_long.py',
                    '--base_dir=%s'%base_name,
                    '--base_command=%s'%base_command,
                    '--global_samples=%s'%str(global_samples),
                    '--witheld=%s'%(str(witheld)),
                    '--test_size=%s'%(str(test_size)),
                    ])
    
    subprocess.run(['python', 'plot_ml_heatmap_from_keyfile.py', '--target_dir=%s'%base_name,'--format=svg'])
    subprocess.run(['python', 'plot_ml_heatmap_from_keyfile.py', '--target_dir=%s'%base_name,'--format=png'])



def run_heatmaps_long_freq_comp(base_name,base_command, global_samples, witheld, test_size, test_type):
    subprocess.run(['python', 'run_heatmaps_commandline_long_freq_comp.py',
                    '--base_dir=%s'%base_name,
                    '--base_command=%s'%base_command,
                    '--global_samples=%s'%str(global_samples),
                    '--witheld=%s'%(str(witheld)),
                    '--test_size=%s'%(str(test_size)),
                    '--test_type=%s'%test_type,
                    ])
    
    subprocess.run(['python', 'plot_ml_heatmap_from_keyfile.py', '--target_dir=%s'%base_name,'--format=svg'])
    subprocess.run(['python', 'plot_ml_heatmap_from_keyfile.py', '--target_dir=%s'%base_name,'--format=png'])


def run_heatmaps_long_single(base_name, base_command, global_samples, witheld, test_size):
    subprocess.run(['python', 'run_heatmaps_commandline_long_single.py',
                    '--base_dir=%s'%base_name,
                    '--base_command=%s'%base_command,
                    '--global_samples=%s'%str(global_samples),
                    '--witheld=%s'%(str(witheld)),
                    '--test_size=%s'%(str(test_size)),
                    ])
    
    subprocess.run(['python', 'plot_ml_heatmap_from_keyfile.py', '--target_dir=%s'%base_name,'--format=svg'])
    subprocess.run(['python', 'plot_ml_heatmap_from_keyfile.py', '--target_dir=%s'%base_name,'--format=png'])


def run_heatmaps_frequency_comparison(base_name, base_command, d1, d2, global_samples, witheld, test_size, test_type):
    subprocess.run(['python', 'run_heatmaps_commandline_frequency.py',
                    '--base_dir=%s'%base_name,
                    '--base_command=%s'%base_command,
                    '--global_samples=%s'%str(global_samples),
                    '--witheld=%s'%(str(witheld)),
                    '--test_size=%s'%(str(test_size)),
                    '--datafile2=%s'%(d2),
                    '--datafile1=%s'%(d1),
                    '--test_type=%s'%(test_type)
                    ])
    
    
    subprocess.run(['python', 'plot_ml_heatmap_from_keyfile.py', '--target_dir=%s'%base_name,'--format=svg'])
    subprocess.run(['python', 'plot_ml_heatmap_from_keyfile.py', '--target_dir=%s'%base_name,'--format=png'])



def run_photobleaching(base_name, base_command, global_samples, witheld, test_size, test_type):
    subprocess.run(['python', 'run_heatmaps_commandline_pb2.py',
                    '--base_dir=%s'%base_name,
                    '--base_command=%s'%base_command,
                    '--global_samples=%s'%str(global_samples),
                    '--witheld=%s'%(str(witheld)),
                    '--test_size=%s'%(str(test_size)),
                    '--test_type=%s'%(str(test_type)),
    
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


print('1')
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

    

'''


base_name = 'single_image_classification'
base_command = 'run_cnn_commandline_all_w_freq.py'

# global data setup
retrain = 1
global_samples = 5000
witheld = 1000
test_size = 0
data_file1 = 'kdm5b_base_pb_KDM5B_KDM5B_0.06_5.33333_0.csv'
data_file2 = 'p300_base_pb_P300_P300_0.06_5.33333_0.csv'
#          cl img  keki  kes kis
runwho =  [0,  1,   0,   0,   0]

run_single_grid(base_name,base_command, data_file1, data_file2, global_samples, witheld, test_size,)


'''


'''
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



############# main experiment lower numbers #############

base_name = 'ML_run_320_5s_wfreq_500samples'
base_command = 'run_cnn_commandline_all_w_freq.py'

# global data setup
global_fr = 5
global_nf = 64
retrain = 1
global_samples = 500
witheld = 100
test_size = 0

#          cl img  keki  kes kis
runwho =  [1,  1,   1,   1,   1]
run_heatmaps(base_name,base_command,global_fr, global_nf, global_samples, witheld, test_size, runwho)

############# supplemental experiment with long time #############


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



############# Long imaging time same intensity plot ####################

base_name = 'ML_imaging_long_same_int_24000_14scale'
base_command = 'run_cnn_commandline_all_w_freq.py'

# global data setup
global_samples = 5000
witheld = 1000
test_size = 0

run_heatmaps_long(base_name,base_command, global_samples, witheld, test_size)





base_name = 'ML_imaging_long_same_int_24000_14scale'
base_command = 'run_cnn_commandline_all_w_freq.py'

# global data setup
global_samples = 5000
witheld = 1000
test_size = 0

run_heatmaps_long_single(base_name,base_command, global_samples, witheld, test_size)



############# I vs I+F vs F ##########################

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
    



############# Different Tags ###############

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






############# Frequency only experiments #############







dirs = ['ML_run_img_same_int', 'ML_run_img_normal', 'ML_run_img_just_acc', 'ML_run_img_zero_mean',  'ML_run_img_standardize' ]
commands = ['freq', 'freq','acc_only','zero_mean','standardize']

data_file1s = ['P300_KDM5B_12000s_Same_intensity/ki_ke_sweep_same_int_12000_P300_P300_0.00967579825834543_5.3333_0.csv',
               'imaging_conditions_long_time/ki_ke_sweep_same_int_P300_P300_0.06_5.33_0.csv',
               'imaging_conditions_long_time/ki_ke_sweep_same_int_P300_P300_0.06_5.33_0.csv',
               'imaging_conditions_long_time/ki_ke_sweep_same_int_P300_P300_0.06_5.33_0.csv',
               'imaging_conditions_long_time/ki_ke_sweep_same_int_P300_P300_0.06_5.33_0.csv']

data_file2s = ['P300_KDM5B_12000s_Same_intensity/ki_ke_sweep_same_int_12000_KDM5B_KDM5B_0.014139183457051962_5.3333_0.csv',
               'imaging_conditions_long_time/ki_ke_sweep_same_int_KDM5B_KDM5B_0.06_5.33_0.csv',
               'imaging_conditions_long_time/ki_ke_sweep_same_int_KDM5B_KDM5B_0.06_5.33_0.csv',
               'imaging_conditions_long_time/ki_ke_sweep_same_int_KDM5B_KDM5B_0.06_5.33_0.csv',
               'imaging_conditions_long_time/ki_ke_sweep_same_int_KDM5B_KDM5B_0.06_5.33_0.csv'
               ]

#for i in range(,len(dirs)):
for i in range(1):
    base_name = dirs[i]
    base_command = 'run_cnn_commandline_all_w_freq.py'
    
    # global data setup
    retrain = 1
    global_samples = 5000
    witheld = 1000
    test_size = 0
    test_type = commands[i]
    
    data_file1 = data_file1s[i]
    data_file2 = data_file2s[i]
    #          cl img  keki  kes kis
    runwho =  [0,  1,   0,   0,   0]
    
    run_heatmaps_frequency_comparison(base_name,base_command, data_file1, data_file2, global_samples, witheld, test_size, test_type,)
    








############# Transfer Learning experiments #############




base_name = 'ML_run_320_5s_TL'

# global data setup
global_fr = 5
global_nf = 64
global_samples = 5000

subsamples = 10
subsample_size = 500

witheld = 1000
test_size = 0

model_file = './ML_run_320_5s_wfreq/parsweep_keki_ML/cnn_par_keki_16_3__5_3.h5'

run_heatmaps_TL(base_name,global_fr, global_nf, global_samples, witheld, test_size, subsamples, subsample_size, model_file)







############# Large multiplexing Learning experiments #############

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
'''