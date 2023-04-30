# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 11:23:06 2023

@author: willi
"""


##############################################################################
# This code runs the rSNAPed to generate a simulated cell dataset
##############################################################################



############################################################################
# PARSE ARGUMENTS HERE
import argparse
parser = argparse.ArgumentParser(description='Generate an rsnaped datafile')

parser.add_argument('--rsnaped_dir', dest='rsnaped_directory', action='store',
                    default='.',
                    help='directory where rsnaped is stored / installed if needed')

parser.add_argument('--genes', dest='genes', action='store', 
                    default='pUB_SM_KDM5B_PP7.gb,pUB_SM_p300_MS2.gb',
                    help = 'elongation rates of each gene')

parser.add_argument('--gene_names', dest='gene_names', action='store', 
                    default='kdm5b,p300',
                    help = 'elongation rates of each gene')
                    
parser.add_argument('--kis', dest='kis', action='store',
                    default='.033,',
                    help='initation rates of each gene')
                    
parser.add_argument('--kes', dest='kes', action='store', default = '10,',
                    help = 'elonogation rates of each gene')
                    
                                      
parser.add_argument('--sim_time', dest='simulation_time_in_sec', default='600',
                    help='Simulation time in seconds of the datafile')
                  
parser.add_argument('--n_cells', dest='n_cells', default='50',
                    help='How many cells to simulate')
                    
parser.add_argument('--n_spots', dest='n_spots', default='20,',
                    help='How many spots per mrna type to make per cell')
                    
parser.add_argument('--diff_rates', dest='diff_rates', default='.7,.7',
                    help='Diffusion rate per spot type')

parser.add_argument('--empty_vid_select', dest='frame_selection_empty_video', default='generate_from_guassian',
                    help='shuffle type for blank vids')
                    
parser.add_argument('--protein_channels', dest='protein_channels', default='1,1',
                    help='Channel for each protein signal')

parser.add_argument('--mRNA_tag_channels', dest='mrna_tag_channels', default='0,0',
                    help='Channel for each mRNA constitiuative tag')         

parser.add_argument('--save_dir', dest='save_dir', default='.',
                    help='Directory to save in')  

parser.add_argument('--save_name', dest='save_name', default='tmp_df',
                    help='Name of file to save')       

parser.add_argument('--save_vids', dest='save_vids', default='0',
                    help='Name of file to save')                      
                                                       
parser.add_argument('--intensity_scale', dest='intensity_scale', default=1,type=float,
                    help='scale of intensity')           

parser.add_argument('--spot_size', dest='spot_size', default=3,type=int,
                    help='size of the spot matrix nxn')           

parser.add_argument('--data_format', dest='data_format', default='short',type=str,
                    help='include_extra_info')    
parser.add_argument('--tracking', dest='tracking', default='0',type=str,
                    help='include_extra_info')    



parser.add_argument('--pb', dest='simulate_photobleaching', default='none',type=str,
                    help='include_extra_info')    
parser.add_argument('--pb_vid_mu', dest='simulated_pb_video_mu', default='0,0,0',type=str,
                    help='include_extra_info')    
parser.add_argument('--pb_vid_var', dest='simulated_pb_video_var', default='0,0,0',type=str,
                    help='include_extra_info')    
parser.add_argument('--pb_spot_mu', dest='simulated_pb_spot_mu', default='0,0,0',type=str,
                    help='include_extra_info')    
parser.add_argument('--pb_spot_var', dest='simulated_pb_spot_var', default='0,0,0',type=str,
                    help='include_extra_info')    

parser.add_argument('--mRNA_tag_intensity_scale', dest='mRNA_tag_intensity_scale', default='0,0',type=str,
                    help='include_extra_info')    
parser.add_argument('--mRNA_tag_intensity_type', dest='mRNA_tag_intensity_type', default='constant',type=str,
                    help='include_extra_info')    

                      
                      
args = parser.parse_args()         
print(args)                               
############################################################################
#convert list inputs

input_int = lambda strinput: [int(x) for x in strinput.split(',')]
input_str = lambda strinput: [x for x in strinput.split(',')]
input_float = lambda strinput: [float(x) for x in strinput.split(',')]

simulation_time_in_sec = float(args.simulation_time_in_sec)
number_cells = int(args.n_cells)
frame_selection_empty_video = args.frame_selection_empty_video
list_gene_sequences = input_str(args.genes)
list_number_spots = input_int(args.n_spots)
dataframe_format = args.data_format
list_target_channels_proteins = input_int(args.protein_channels)
list_target_channels_mRNA = input_int(args.mrna_tag_channels)
list_diffusion_coefficients =input_float(args.diff_rates)
list_label_names = input_str(args.gene_names) # list of strings used to generate a classification field in the output data frame
list_elongation_rates = input_float(args.kes) # elongation rates aa/sec
list_initation_rates = input_float(args.kis) # initiation rates 1/sec


simulate_photobleaching = args.simulate_photobleaching
simulated_pb_video_mu = input_float(args.simulated_pb_video_mu)
simulated_pb_spot_mu = input_float(args.simulated_pb_spot_mu)
simulated_pb_video_var = input_float(args.simulated_pb_video_var)
simulated_pb_spot_var = input_float(args.simulated_pb_spot_var)

mRNA_tag_intensity_scale = input_int(args.mRNA_tag_intensity_scale)
mRNA_tag_intensity_type = args.mRNA_tag_intensity_type

save_dir = args.save_dir
save_name = args.save_name
save_vids = int(args.save_vids)
intensity_scale = args.intensity_scale
tracking = int(args.tracking)

spot_size = args.spot_size

############################################################################
### Imports
import rsnapsim as rsim  #import the rsnapsim
import numpy as np
import rsnapsim as rss
import sys
from sys import platform
from skimage import io 
from skimage.io import imread
from skimage.measure import find_contours
from random import randrange
import pandas as pd
import os; from os import listdir; from os.path import isfile, join
import re
import shutil
import pathlib
from random import randrange
import time

st = time.time()




############################################################################

# Defining directories

#sequences_dir = current_dir.joinpath('rsnaped','DataBases','gene_files')
#video_dir = current_dir.joinpath('rsnaped','DataBases','videos_for_sim_cell')
#rsnaped_dir = current_dir.joinpath('rsnaped','rsnaped')
current_dir = pathlib.Path().absolute()
rsnaped_dir = pathlib.Path(args.rsnaped_directory)
rsnap_top_level = rsnaped_dir.parents[0]
video_dir = rsnap_top_level.joinpath('DataBases','videos_for_sim_cell')
sequences_dir = rsnap_top_level.joinpath('DataBases','gene_files')
masks_dir = rsnap_top_level.joinpath('DataBases','masks_for_sim_cell')



# Importing rsnaped
sys.path.append(str(rsnaped_dir))
current_dir = os.getcwd()
os.chdir(str(rsnaped_dir))
#import rsnaped as rsp
import rsnaped as rsp
os.chdir(current_dir)
current_dir = pathlib.Path().absolute()

############################################################################
#setup gene files
gene_seqs = []
for i in range(len(list_gene_sequences)):
    gene_seqs.append(str(sequences_dir.joinpath(list_gene_sequences[i])))

list_gene_sequences = gene_seqs # path to gene sequences

############################################################################


list_files_names = sorted([f for f in listdir(video_dir) if isfile(join(video_dir, f)) and ('.tif') in f], key=str.lower)  # reading all tif files in the folder
list_files_names.sort(key=lambda f: int(re.sub('\D', '', f)))  # sorting the index in numerical order
path_files = [ str(video_dir.joinpath(f).resolve()) for f in list_files_names ] # creating the complete path for each file
num_cell_shapes = len(path_files)

############################################################################
# RUN THE CELL SIMULATIONS

def run_simulations(list_gene_sequences, list_number_spots, list_target_channels_proteins, list_target_channels_mRNA,list_diffusion_coefficients,list_label_names, list_elongation_rates, list_initation_rates, frame_selection_empty_video = 'linear_interpolation',simulation_time_in_sec = 100,step_size_in_sec = 1,save_as_tif = 0,save_dataframe = 0,create_temp_folder = 0, spot_size = spot_size, number_cells=1,intensity_scale_ch0=intensity_scale,intensity_scale_ch1=intensity_scale,intensity_scale_ch2=intensity_scale, tracking=tracking, mRNA_tag_intensity_type=mRNA_tag_intensity_type, mRNA_tag_intensity_scale=mRNA_tag_intensity_scale):
  list_dataframe_simulated_cell =[]
  list_ssa_all_cells_and_genes =[]
  #list_videos =[]
  for i in range (0,number_cells ):
    saved_file_name = 'cell_' + str(i)  # if the video or dataframe are save, this variable assigns the name to the files
    sel_shape = randrange(num_cell_shapes)
    video_path = path_files[sel_shape]
    inial_video = io.imread(video_path) # video with empty cell
    mask_image = imread(masks_dir.joinpath('mask_cell_shape_'+str(sel_shape)+'.tif'))
    
    # CALL THE GENE MULTIPLEXING CODE IN RSNAPED
    tensor_vid, single_dataframe_simulated_cell,list_ssa = rsp.SimulatedCellMultiplexing(inial_video,list_gene_sequences,list_number_spots,list_target_channels_proteins,list_target_channels_mRNA, list_diffusion_coefficients,list_label_names,list_elongation_rates,list_initation_rates,simulation_time_in_sec,step_size_in_sec,save_as_tif, save_dataframe, saved_file_name,create_temp_folder, mask_image=mask_image, cell_number =i,frame_selection_empty_video=frame_selection_empty_video,spot_size =spot_size, intensity_scale_ch0=intensity_scale_ch0, intensity_scale_ch1=intensity_scale_ch1, intensity_scale_ch2=intensity_scale_ch2, dataframe_format=dataframe_format,
                                                                                        simulate_photobleaching = simulate_photobleaching,
                                                                                        simulated_pb_video_mu = simulated_pb_video_mu,
                                                                                        simulated_pb_spot_mu = simulated_pb_spot_mu,
                                                                                        simulated_pb_video_var = simulated_pb_video_var,
                                                                                        simulated_pb_spot_var = simulated_pb_spot_var,                                                                                         
                                                                                         list_target_channels_mRNA_intensity = mRNA_tag_intensity_scale,
                                                                                         simulated_RNA_intensities_method = mRNA_tag_intensity_type, ).make_simulation()
    
    
    master_trackpy_df = ''
    master_trackpy_df_wo_correction = ''
    found_particles = sum(list_number_spots)
    print(tensor_vid.shape)
    
    
    # PERFORM TRACKING IF NEEDED
    if tracking:

                #[T, Y, X, C]
      print(tensor_vid.shape)  #slice to correct frame rate here
      #tensor_vid = tensor_vid[:640:5,:,:,:]
      print(tensor_vid.shape) 
      target_dir = current_dir.joinpath('P300_KDM5B_3000s_base_pb')
      selected_channel_tracking = 0
      selected_channel_segmentation = 1
      intensity_calculation_method = 'disk_donut'  # options are : 'total_intensity' and 'disk_donut' 'gaussian_fit'
      mask_selection_method = 'max_area'           # options are : 'max_spots' and 'max_area' 
      use_optimization_for_tracking = 1            # 0 not using, 1 is using optimization
      min_percentage_time_tracking = 0.2            # (normalized) minimum time to consider a trajectory.
      particle_detection_size = 7                  # spot size for the simulation and tracking.
      selected_channel = 0                         # Selected channel for tracking
      average_cell_diameter = 200                    # cell diameter
      intensity_threshold_tracking = None          # intensity threshold. If None, the code uses automatic detection # 
      real_positions_dataframe = None #pd.read_csv(video_dir.joinpath('both_base_pb_KDM5B_P300_0.06_5.33333_video_cell_0.csv'))      
          
          
      
      list_DataFrame_tracking, _, _, _ = rsp.image_processing(tensor_vid, mask_image, files_dir_path_processing=target_dir,
                                                            particle_size=particle_detection_size,
                                                            selected_channel_tracking = selected_channel_tracking,
                                                            selected_channel_segmentation = selected_channel_segmentation,
                                                            intensity_calculation_method =intensity_calculation_method, 
                                                            mask_selection_method = mask_selection_method,
                                                            show_plot=False,
                                                            use_optimization_for_tracking=use_optimization_for_tracking,
                                                            real_positions_dataframe = real_positions_dataframe,
                                                            average_cell_diameter=average_cell_diameter,
                                                            print_process_times=False,
                                                            min_percentage_time_tracking=min_percentage_time_tracking,
                                                            dataframe_format='short')
    
      trackpy_df_wo_correction = list_DataFrame_tracking[0]
      
      # APPLY PHOTOBLEACH CORRECTION IF NEEDED
      if simulate_photobleaching: 
         # apply photobleach correction to the tensor video coming out before tracking
        tensor_vid, exponentials, opt, cov = rsp.PhotobleachingCorrectionVideo(tensor_vid, mask_image,).apply_photobleaching_correction()

        list_DataFrame_tracking, _, _, _ = rsp.image_processing( tensor_vid, mask_image, files_dir_path_processing=target_dir,
                                                              particle_size=particle_detection_size,
                                                              selected_channel_tracking = selected_channel_tracking,
                                                              selected_channel_segmentation = selected_channel_segmentation,
                                                              intensity_calculation_method =intensity_calculation_method, 
                                                              mask_selection_method = mask_selection_method,
                                                              show_plot=False,
                                                              use_optimization_for_tracking=use_optimization_for_tracking,
                                                              real_positions_dataframe = real_positions_dataframe,
                                                              average_cell_diameter=average_cell_diameter,
                                                              print_process_times=False,
                                                              min_percentage_time_tracking=min_percentage_time_tracking,
                                                              dataframe_format='short')
      
      trackpy_df = list_DataFrame_tracking[0]
      trackpy_df['cell_number'] = [i,]*trackpy_df.shape[0]   
      trackpy_df_wo_correction['cell_number'] = [i,]*trackpy_df_wo_correction.shape[0]     

      if i == 0:
        master_trackpy_df = trackpy_df.copy()
        master_trackpy_df_wo_correction = trackpy_df_wo_correction.copy()
        #master_trackpy_df = master_trackpy_df.append(trackpy_df, ignore_index=True)
        #master_trackpy_df_wo_correction = master_trackpy_df_wo_correction.append(trackpy_df_wo_correction, ignore_index=True)
      else:
        master_trackpy_df = master_trackpy_df.append(trackpy_df, ignore_index=True)
        master_trackpy_df_wo_correction = master_trackpy_df_wo_correction.append(trackpy_df_wo_correction, ignore_index=True)


    list_dataframe_simulated_cell.append(single_dataframe_simulated_cell)
    list_ssa_all_cells_and_genes.append(list_ssa)
    #list_videos.append(tensor_video)

  dataframe_simulated_cells = pd.concat(list_dataframe_simulated_cell)
  return dataframe_simulated_cells , list_ssa_all_cells_and_genes, master_trackpy_df, master_trackpy_df_wo_correction #, list_videos

############################################################################


#Make the dataframe from rSNAPed
dataframe_simulated_cells_condition_0 , list_ssa_all_cells_and_genes_condition_0, trackpy_df, trackpy_df_wo_correction = run_simulations(list_gene_sequences,
 list_number_spots,
 list_target_channels_proteins,
 list_target_channels_mRNA,
 list_diffusion_coefficients,
 list_label_names,
 list_elongation_rates,
 list_initation_rates,
 frame_selection_empty_video = frame_selection_empty_video,
 simulation_time_in_sec = simulation_time_in_sec,
 step_size_in_sec = 1,
 save_as_tif = save_vids,
 save_dataframe = 0,
 create_temp_folder = 0,
 spot_size = spot_size,  
 number_cells=number_cells,
 intensity_scale_ch0=intensity_scale,
 intensity_scale_ch1=intensity_scale,
 intensity_scale_ch2=intensity_scale)
 

##Patch the csv file so its a bit smaller to save 
int_labels = [int(x==list_label_names[1]) for x in dataframe_simulated_cells_condition_0['Classification']]
dataframe_simulated_cells_condition_0['Classification'] = int_labels

if dataframe_format == 'short':
    df_to_save = dataframe_simulated_cells_condition_0.drop(columns=['red_int_mean', 'red_int_std', 'SNR_red',
                                                'background_int_mean_red', 'background_int_std_red','blue_int_mean',
                                                'blue_int_std', 'SNR_blue', 'background_int_mean_blue', 'background_int_std_blue' ])
else:
    df_to_save = dataframe_simulated_cells_condition_0


# Save the final data frame
save_path = pathlib.Path(save_dir).joinpath(save_name)
df_to_save.to_csv(str(save_path))
if tracking:
    
    trackpy_df.to_csv(str(save_path) + '_tracking_w_correction')
    trackpy_df_wo_correction.to_csv(str(save_path) + '_tracking_wo_correction')
    np.save((str(save_path) + '_ssas'), np.array(list_ssa_all_cells_and_genes_condition_0), )
