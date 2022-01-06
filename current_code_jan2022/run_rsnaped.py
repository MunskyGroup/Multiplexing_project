
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

parser.add_argument('--empty_vid_select', dest='frame_selection_empty_video', default='shuffle',
                    help='shuffle type for blank vids')
                    
parser.add_argument('--protein_channels', dest='protein_channels', default='1,1',
                    help='Channel for each protein signal')

parser.add_argument('--mRNA_tag_channels', dest='mrna_tag_channels', default='0,2',
                    help='Channel for each mRNA constitiuative tag')         

parser.add_argument('--save_dir', dest='save_dir', default='.',
                    help='Directory to save in')  

parser.add_argument('--save_name', dest='save_name', default='tmp_df',
                    help='Name of file to save')       

parser.add_argument('--save_vids', dest='save_vids', default='0',
                    help='Name of file to save')                      
                                                       
                                        
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
list_target_channels_proteins = input_int(args.protein_channels)
list_target_channels_mRNA = input_int(args.mrna_tag_channels)
list_diffusion_coefficients =input_float(args.diff_rates)
list_label_names = input_str(args.gene_names) # list of strings used to generate a classification field in the output data frame
list_elongation_rates = input_float(args.kes) # elongation rates aa/sec
list_initation_rates = input_float(args.kis) # initiation rates 1/sec
save_dir = args.save_dir
save_name = args.save_name
save_vids = int(args.save_vids)
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

#python3 run_rsnaped.py --rsnaped_dir=~/Desktop/rsnaped/ --kis=.033,.033 --kes=10,10 --save_name='test_df' --

############################################################################

# Defining directories
current_dir = pathlib.Path().absolute()

#sequences_dir = current_dir.joinpath('rsnaped','DataBases','gene_files')
#video_dir = current_dir.joinpath('rsnaped','DataBases','videos_for_sim_cell')
#rsnaped_dir = current_dir.joinpath('rsnaped','rsnaped')

rsnaped_dir = pathlib.Path(args.rsnaped_directory)
rsnap_top_level = rsnaped_dir.parents[0]
video_dir = rsnap_top_level.joinpath('DataBases','videos_for_sim_cell')
sequences_dir = rsnap_top_level.joinpath('DataBases','gene_files')

# Importing rsnaped
sys.path.append(str(rsnaped_dir))
current_dir = os.getcwd()
os.chdir(str(rsnaped_dir))
#import rsnaped as rsp
import rsnaped as rsp
os.chdir(current_dir)


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

def run_simulations(list_gene_sequences, list_number_spots, list_target_channels_proteins, list_target_channels_mRNA,list_diffusion_coefficients,list_label_names, list_elongation_rates, list_initation_rates, frame_selection_empty_video = 'shuffle',simulation_time_in_sec = 100,step_size_in_sec = 1,save_as_tif = 0,save_dataframe = 0,create_temp_folder = 0, spot_size = 3, number_cells=1):
  list_dataframe_simulated_cell =[]
  list_ssa_all_cells_and_genes =[]
  #list_videos =[]
  for i in range (0,number_cells ):
    saved_file_name = 'cell_' + str(i)  # if the video or dataframe are save, this variable assigns the name to the files
    sel_shape = randrange(num_cell_shapes)
    video_path = path_files[sel_shape]
    inial_video = io.imread(video_path) # video with empty cell
    _, single_dataframe_simulated_cell,list_ssa = rsp.SimulatedCellMultiplexing(inial_video,list_gene_sequences,list_number_spots,list_target_channels_proteins,list_target_channels_mRNA, list_diffusion_coefficients,list_label_names,list_elongation_rates,list_initation_rates,simulation_time_in_sec,step_size_in_sec,save_as_tif, save_dataframe, saved_file_name,create_temp_folder,cell_number =i,frame_selection_empty_video=frame_selection_empty_video,spot_size =spot_size).make_simulation()
    list_dataframe_simulated_cell.append(single_dataframe_simulated_cell)
    list_ssa_all_cells_and_genes.append(list_ssa)
    #list_videos.append(tensor_video)
  dataframe_simulated_cells = pd.concat(list_dataframe_simulated_cell)
  return dataframe_simulated_cells , list_ssa_all_cells_and_genes #, list_videos

############################################################################

print(rsp.__dict__)
#Make the dataframe from rSNAPed
dataframe_simulated_cells_condition_0 , list_ssa_all_cells_and_genes_condition_0 = run_simulations(list_gene_sequences,
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
 spot_size = 3,
 number_cells=number_cells)
 
 
 


##Patch the csv file so its a bit smaller to save 
int_labels = [int(x==list_label_names[1]) for x in dataframe_simulated_cells_condition_0['Classification']]
dataframe_simulated_cells_condition_0['Classification'] = int_labels

df_to_save = dataframe_simulated_cells_condition_0.drop(columns=['red_int_mean', 'red_int_std', 'SNR_red',
                                            'background_int_mean_red', 'background_int_std_red','blue_int_mean',
                                            'blue_int_std', 'SNR_blue', 'background_int_mean_blue', 'background_int_std_blue' ])

save_path = pathlib.Path(save_dir).joinpath(save_name)
df_to_save.to_csv(str(save_path))

