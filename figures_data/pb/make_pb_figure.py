# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 19:22:21 2023

@author: willi
"""
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import pandas as pd
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from skimage.io import imread
########################################
dark = False
if not dark: ##06d6a0, ef476f
    colors = ['#073b4c', '#8a3619','#06d6a0','#7400b8','#073b4c', '#118ab2',]
else:
    plt.style.use('dark_background')
    plt.rcParams.update({'axes.facecolor'      : '#131313'  , 
'figure.facecolor' : '#131313' ,
'figure.edgecolor' : '#131313' , 
'savefig.facecolor' : '#131313'  , 
'savefig.edgecolor' :'#131313'})


    colors = ['#118ab2','#57ffcd', '#ff479d', '#ffe869','#ff8c00','#04756f']

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 12}

save = False

plt.rcParams.update({'font.size': 12, 'font.weight':'bold','font.family':'normal'  }   )
plt.rcParams.update({'axes.prop_cycle':cycler(color=colors)})

plt.rcParams.update({'axes.prop_cycle':cycler(color=colors)})
plt.rcParams.update({'axes.prop_cycle':cycler(color=colors)})


plt.rcParams.update({'xtick.major.width'   : 2.8 })
plt.rcParams.update({'xtick.labelsize'   : 12 })



plt.rcParams.update({'ytick.major.width'   : 2.8 })
plt.rcParams.update({'ytick.labelsize'   : 12})

plt.rcParams.update({'axes.titleweight'   : 'bold'})
plt.rcParams.update({'axes.titlesize'   : 10})
plt.rcParams.update({'axes.labelweight'   : 'bold'})
plt.rcParams.update({'axes.labelsize'   : 12})

plt.rcParams.update({'axes.linewidth':2.8})
plt.rcParams.update({'axes.labelpad':8})
plt.rcParams.update({'axes.titlepad':10})
plt.rcParams.update({'figure.dpi':300})




target_dir_data = '../../datasets/P300_KDM5B_350s_base_pb'
target_dir_data2 = '../../datasets/P300_KDM5B_350s_base_pb_combined'
target_dir_ml = '../../ML_PB/parsweep_pb_ML'

##############################################################################
# Plotting true spot percentages kept
##############################################################################
percentage_drop_per_frame = np.array([1, .999,.998, .996, .994, .99, .98, .965, .96, .955, .95])
pb_rates = np.abs(-np.log(percentage_drop_per_frame)/5) # convert percentage drop per frame to e^-alpha*t
pb_rates[0] = 0



n0 = [np.load('%s%i/p300_base_pb_P300_P300_0.06_5.33333_%i_tracking_wo_correction_intensities.npy'%(target_dir_data,1,i)).shape[0] for i in range(11)]
n1 = [np.load('%s%i/p300_base_pb_P300_P300_0.06_5.33333_%i_tracking_wo_correction_intensities.npy'%(target_dir_data,2,i)).shape[0] for i in range(11)]
n2 = [np.load('%s%i/p300_base_pb_P300_P300_0.06_5.33333_%i_tracking_wo_correction_intensities.npy'%(target_dir_data,3,i)).shape[0] for i in range(11)]

n_p300 = np.array([n0,n1,n2])

n0 = [np.load('%s%i/kdm5b_base_pb_KDM5B_KDM5B_0.06_5.33333_%i_tracking_wo_correction_intensities.npy'%(target_dir_data,1,i)).shape[0] for i in range(11)]
n1 = [np.load('%s%i/kdm5b_base_pb_KDM5B_KDM5B_0.06_5.33333_%i_tracking_wo_correction_intensities.npy'%(target_dir_data,2,i)).shape[0] for i in range(11)]
n2 = [np.load('%s%i/kdm5b_base_pb_KDM5B_KDM5B_0.06_5.33333_%i_tracking_wo_correction_intensities.npy'%(target_dir_data,3,i)).shape[0] for i in range(11)]


n_kdm5b = np.array([n0,n1,n2])

n = (n_kdm5b + n_p300 )/(25*50*2)

plt.figure()
plt.errorbar(1-percentage_drop_per_frame,np.mean(n,axis=0), yerr = np.std(n,axis=0), capsize=2)
plt.xlabel('% loss per 5 seconds')
plt.ylabel('% spots kept')


plt.figure()
plt.plot(1-percentage_drop_per_frame,n[0])
plt.plot(1-percentage_drop_per_frame,n[1])
plt.plot(1-percentage_drop_per_frame,n[2])
plt.xlabel('% loss per 5 seconds')
plt.ylabel('% spots kept')

plt.figure()
plt.plot(1-percentage_drop_per_frame, np.sum(n_kdm5b + n_p300,axis=0)/(2*25*50*3),'o-')
plt.xlabel('% loss per 5 seconds')
plt.ylabel('% spots kept')
plt.savefig('pb_tracking.svg')

##############################################################################
# Plotting all (Fake/True/Fragmented) spot percentages kept
##############################################################################

particle_matching_array = np.zeros([2,3,11,4])

for j in range(1,4):
    target_dir = 'P300_KDM5B_3000s_base_pb%i'%j
    # KDM5B
    for i in range(11):
        a = np.load('%s%i/kdm5b_base_pb_KDM5B_KDM5B_0.06_5.33333_%i_tracking_wo_correction_matching.npy'%(target_dir_data,j,i))
        real_particles = 1250
        real_particles_found = np.sum(a[:,-2])
        fake_particles_found = np.sum(a[:,-2] == 0)
        kept_particles_real = np.sum(a[:,-3])
        kept_particles_fake = np.sum((a[:,3][a[:,-2] == 0] >= 300))
        
        particle_matching_array[0,j-1,i,:] = [real_particles_found, fake_particles_found, kept_particles_real, kept_particles_fake]
    
    #p300
    for i in range(11):
        a = np.load('%s%i/p300_base_pb_P300_P300_0.06_5.33333_%i_tracking_wo_correction_matching.npy'%(target_dir_data,j,i))
        real_particles = 1250
        real_particles_found = np.sum(a[:,-2])
        fake_particles_found = np.sum(a[:,-2] == 0)
        kept_particles_real = np.sum(a[:,-3])
        kept_particles_fake = np.sum((a[:,3][a[:,-2] == 0] >= 300))
        
        particle_matching_array[1,j-1,i,:] = [real_particles_found, fake_particles_found, kept_particles_real, kept_particles_fake]

# array of gene, dataset, pb rate, type of spot
master_matching_array = np.sum(np.sum(particle_matching_array,axis=0),axis=0)



plt.figure()
plt.plot(1-percentage_drop_per_frame, (master_matching_array[:,0]+master_matching_array[:,1])/(2500*3),'o-')
plt.plot(1-percentage_drop_per_frame, master_matching_array[:,0]/(2500*3),'o-')
plt.plot(1-percentage_drop_per_frame, (master_matching_array[:,2]+master_matching_array[:,3])/(2500*3),'o-')
plt.plot(1-percentage_drop_per_frame, master_matching_array[:,2]/(2500*3),'o-')

plt.plot(1-percentage_drop_per_frame, [1,]*11, 'r--')
plt.xlabel('% loss per 5 seconds')
plt.ylabel('N spots / total real spots')
plt.legend(['total spots found', 'real spots found', 'spots kept after 300s threshold', 'real spots kept after 300s threshold'], bbox_to_anchor=(1.1, 1.05))
plt.savefig('pb_tracking.svg')


##############################################################################
# ML accuracy
##############################################################################

wos = ['%s/acc_mat_pb_wo.npy'%target_dir_ml]
reals = ['%s/acc_mat_pb.npy'%target_dir_ml]

wo_acc = np.array([(np.load(x))[0,:11] for x in wos])
real_acc = np.array([np.load(x)[0,:11] for x in reals])

plt.figure()
plt.errorbar(1-percentage_drop_per_frame, np.mean(real_acc,axis=0), yerr= np.std(real_acc,axis=0), capsize=2)
plt.errorbar(1-percentage_drop_per_frame, np.mean(wo_acc,axis=0), yerr= np.std(wo_acc,axis=0), capsize=2)
plt.xlabel('% loss per 5 seconds')
plt.ylabel('Classifier Accuracy')
plt.legend(['perfect information','tracking-wo-correction'])

plt.figure()
plt.plot(1-percentage_drop_per_frame, real_acc[0],'o-')
plt.plot(1-percentage_drop_per_frame,wo_acc[0],'o-' )
plt.xlabel('% loss per 5 seconds')
plt.ylabel('Classifier Accuracy')
plt.legend(['perfect tracking','realistic tracking'])

nns = np.sum(n_kdm5b + n_p300,axis=0)
plt.text((1-percentage_drop_per_frame)[0]-.002,wo_acc[0][0]+.005, nns[0], fontsize=7, color=colors[1])
#plt.text((1-percentage_drop_per_frame)[0]-.002,wo_acc[0][0]+.015, 'training size', fontsize=7, color=colors[1])
plt.text((1-percentage_drop_per_frame)[1],wo_acc[0][1]-.01, nns[1], fontsize=7, color=colors[1])
plt.text((1-percentage_drop_per_frame)[2]-.0003,wo_acc[0][2]+.005, nns[2], fontsize=7, color=colors[1])
plt.text((1-percentage_drop_per_frame)[3]+.001,wo_acc[0][3]-.005, nns[3], fontsize=7, color=colors[1])
plt.text((1-percentage_drop_per_frame)[4],wo_acc[0][4]+.005, nns[4], fontsize=7, color=colors[1])
plt.text((1-percentage_drop_per_frame)[5]+.001,wo_acc[0][5]-.005, nns[5], fontsize=7, color=colors[1])
plt.text((1-percentage_drop_per_frame)[6],wo_acc[0][6]+.005, nns[6], fontsize=7, color=colors[1])
plt.text((1-percentage_drop_per_frame)[7],wo_acc[0][7]-.01, nns[7], fontsize=7, color=colors[1])
plt.text((1-percentage_drop_per_frame)[8],wo_acc[0][8]+.005, nns[8], fontsize=7, color=colors[1])
plt.text((1-percentage_drop_per_frame)[9],wo_acc[0][9]+.005, nns[9], fontsize=7, color=colors[1])
plt.text((1-percentage_drop_per_frame)[10]-.001,wo_acc[0][10]+.008, nns[10], fontsize=7, color=colors[1])


plt.text((1-percentage_drop_per_frame)[0],real_acc[0][0]+.033, 'data set size', fontsize=7, color=colors[0])
plt.text((1-percentage_drop_per_frame)[0],real_acc[0][0]+.04, 2500, fontsize=7, color=colors[0])
plt.ylim([.72,.93])
plt.savefig('pb_accuracy.svg')


##############################################################################
# Combine and save spots (unused)
##############################################################################
'''
n0 = [np.load('D:/multiplexing_ML/finalized_plots_gaussians/P300_KDM5B_3000s_base_pb1/kdm5b_base_pb_KDM5B_KDM5B_0.06_5.33333_%i_tracking_wo_correction_intensities.npy'%i) for i in range(11)]
n1 = [np.load('D:/multiplexing_ML/finalized_plots_gaussians/P300_KDM5B_3000s_base_pb2/kdm5b_base_pb_KDM5B_KDM5B_0.06_5.33333_%i_tracking_wo_correction_intensities.npy'%i)for i in range(11)]
n2 = [np.load('D:/multiplexing_ML/finalized_plots_gaussians/P300_KDM5B_3000s_base_pb3/kdm5b_base_pb_KDM5B_KDM5B_0.06_5.33333_%i_tracking_wo_correction_intensities.npy'%i)for i in range(11)]

combined = [np.vstack([n0[i], n1[i], n2[i]]) for i in range(11)]
[np.save('D:/multiplexing_ML/finalized_plots_gaussians/P300_KDM5B_3000s_base_pb_combined/kdm5b_base_pb_KDM5B_KDM5B_0.06_5.33333_%i_tracking_wo_correction_intensities.npy'%i, combined[i]) for i in range(11)]


n0 = [np.load('D:/multiplexing_ML/finalized_plots_gaussians/P300_KDM5B_3000s_base_pb1/p300_base_pb_P300_P300_0.06_5.33333_%i_tracking_wo_correction_intensities.npy'%i) for i in range(11)]
n1 = [np.load('D:/multiplexing_ML/finalized_plots_gaussians/P300_KDM5B_3000s_base_pb2/p300_base_pb_P300_P300_0.06_5.33333_%i_tracking_wo_correction_intensities.npy'%i) for i in range(11)]
n2 = [np.load('D:/multiplexing_ML/finalized_plots_gaussians/P300_KDM5B_3000s_base_pb3/p300_base_pb_P300_P300_0.06_5.33333_%i_tracking_wo_correction_intensities.npy'%i) for i in range(11)]


combined = [np.vstack([n0[i], n1[i], n2[i]]) for i in range(11)]
[np.save('D:/multiplexing_ML/finalized_plots_gaussians/P300_KDM5B_3000s_base_pb_combined/p300_base_pb_P300_P300_0.06_5.33333_%i_tracking_wo_correction_intensities.npy'%i, combined[i]) for i in range(11)]
'''

##############################################################################
# Plot the PB rates
##############################################################################

cmap = plt.get_cmap('Spectral_r')

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

new_cmap = truncate_colormap(cmap, 0.1, 1)
my_cmap = new_cmap(np.arange(cmap.N))
my_cmap[:, -1] = 1
my_cmap = ListedColormap(my_cmap)



plt.figure(dpi=300)
pb_rates2 = pb_rates
[plt.plot(np.linspace(0,500,500), np.exp(-pb_rates2[i]*np.linspace(0,500, 500)), color = my_cmap(i/11)) for i in range(0,11)];

plt.xlabel('Time (s)')
plt.ylabel('Intensity scale')

plt.text(470, 1.007, str(np.round((1-percentage_drop_per_frame)[0]*100, 2)) + '%', color=my_cmap(0/11), fontsize=8)
plt.text(470, .92,  str(np.round((1-percentage_drop_per_frame)[1]*100, 2))+ '%', color=my_cmap(1/11), fontsize=8)
plt.text(470, .85,  str(np.round((1-percentage_drop_per_frame)[2]*100, 2))+ '%', color=my_cmap(2/11), fontsize=8)
plt.text(470, .7,  str(np.round((1-percentage_drop_per_frame)[3]*100, 2))+ '%', color=my_cmap(3/11), fontsize=8)
plt.text(470, .6,  str(np.round((1-percentage_drop_per_frame)[4]*100, 2))+ '%', color=my_cmap(4/11), fontsize=8)
plt.text(470, .42,  str(np.round((1-percentage_drop_per_frame)[5]*100, 2))+ '%', color=my_cmap(5/11), fontsize=8)
plt.text(470, .18,  str(np.round((1-percentage_drop_per_frame)[6]*100, 2))+ '%', color=my_cmap(6/11), fontsize=8)
plt.text(470, .05,  str(np.round((1-percentage_drop_per_frame)[7]*100, 2))+ '%', color=my_cmap(7/11), fontsize=8)
plt.text(470, -.03,  str(np.round((1-percentage_drop_per_frame)[-1]*100, 2))+ '%', color=my_cmap(10/11), fontsize=8)
plt.savefig('pb_rates.svg')


##############################################################################
# PLOTS OF EXAMPLE CELLS
##############################################################################

fig,ax = plt.subplots(nrows=11,ncols=6, dpi=300, tight_layout=True)
for i in range(11):
    video = imread('%s/both_base_pb%i.tif'%(target_dir_data+'_vids',i))[[1,50,150,200,250,300],:,:,1]
    for j in range(6):
        ax[i,j].imshow(video[j], cmap='Greens_r')
        ax[i,j].axis('off')

plt.savefig('pb_vids_g.svg')



fig,ax = plt.subplots(nrows=11,ncols=6, dpi=300, tight_layout=True)
for i in range(11):
    video = imread('%s/both_base_pb%i.tif'%(target_dir_data+'_vids',i))[[1,50,150,200,250,300],:,:,0]
    for j in range(6):
        ax[i,j].imshow(video[j], cmap='Reds_r')
        ax[i,j].axis('off')

plt.savefig('pb_vids_r.svg')


##############################################################################
# JOY PLOTS OF INTENSITY DISTRIBUTIONS (UNUSED)
##############################################################################


1/0
import pandas as pd
import joypy

int_gs_kdm5b = []
int_gs_labels_kdm5b = []
int_gs_p300 = []
int_gs_labels_p300= []

for i in range(11):
    plt.figure()
    int_g1 = np.load('D:/multiplexing_ML/finalized_plots_gaussians/P300_KDM5B_3000s_base_pb1/p300_base_pb_P300_P300_0.06_5.33333_%i_tracking_wo_correction_intensities.npy'%i)
    int_g2 = np.load('D:/multiplexing_ML/finalized_plots_gaussians/P300_KDM5B_3000s_base_pb1/kdm5b_base_pb_KDM5B_KDM5B_0.06_5.33333_%i_tracking_wo_correction_intensities.npy'%i)
    int_gs_kdm5b.append(int_g2.flatten().tolist())
    int_gs_labels_kdm5b.append([i]*len(int_g2.flatten()))
    int_gs_p300.append(int_g1.flatten().tolist())
    int_gs_labels_p300.append([i]*len(int_g1.flatten()))
    
int_df_kdm5b = pd.DataFrame({'pb':[item for sublist in int_gs_labels_kdm5b for item in sublist], 'I':[item for sublist in int_gs_kdm5b for item in sublist]})
int_df_p300 = pd.DataFrame({'pb':[item for sublist in int_gs_labels_p300 for item in sublist], 'I':[item for sublist in int_gs_p300 for item in sublist]})

joypy.joyplot(int_df_kdm5b, by='pb')
plt.xlim([0,3000])
plt.xlabel('Intensity')
plt.title('KDM5B Tracked (G)')

joypy.joyplot(int_df_p300, by='pb')
plt.xlim([0,3000])    

plt.xlabel('Intensity')
plt.title('P300 Tracked (G)')
int_gs_kdm5b = []
int_gs_labels_kdm5b = []
int_gs_p300 = []
int_gs_labels_p300= []


for i in range(11):
    plt.figure()
    
    df_real1 ='D:/multiplexing_ML/finalized_plots_gaussians/P300_KDM5B_3000s_base_pb1/kdm5b_base_pb_KDM5B_KDM5B_0.06_5.33333_%i.csv'%i
    df_real2 ='D:/multiplexing_ML/finalized_plots_gaussians/P300_KDM5B_3000s_base_pb1/p300_base_pb_P300_P300_0.06_5.33333_%i.csv'%i
    multiplexing_df1 = pd.read_csv(df_real1)
    multiplexing_df2 = pd.read_csv(df_real2)

    ntraj = 1250
    ntimes = 350
    int_g2 = multiplexing_df1['green_int_mean'].values.reshape([ntraj,ntimes])    
    int_g1 = multiplexing_df2['green_int_mean'].values.reshape([ntraj,ntimes])   
 
    int_gs_kdm5b.append(int_g2.flatten().tolist())
    int_gs_labels_kdm5b.append([i]*len(int_g2.flatten()))
    int_gs_p300.append(int_g1.flatten().tolist())
    int_gs_labels_p300.append([i]*len(int_g1.flatten()))
int_df_kdm5b = pd.DataFrame({'pb':[item for sublist in int_gs_labels_kdm5b for item in sublist], 'I':[item for sublist in int_gs_kdm5b for item in sublist]})
int_df_p300 = pd.DataFrame({'pb':[item for sublist in int_gs_labels_p300 for item in sublist], 'I':[item for sublist in int_gs_p300 for item in sublist]})

joypy.joyplot(int_df_kdm5b, by='pb')
plt.xlim([0,8000])
plt.xlabel('Intensity')
plt.title('KDM5B All (G)')
joypy.joyplot(int_df_p300, by='pb')
plt.xlim([0,8000])
plt.xlabel('Intensity')
plt.title('P300 All (G)')



for i in range(11):
    plt.figure()
    
    df_real1 ='D:/multiplexing_ML/finalized_plots_gaussians/P300_KDM5B_3000s_base_pb1/kdm5b_base_pb_KDM5B_KDM5B_0.06_5.33333_%i.csv'%i
    df_real2 ='D:/multiplexing_ML/finalized_plots_gaussians/P300_KDM5B_3000s_base_pb1/p300_base_pb_P300_P300_0.06_5.33333_%i.csv'%i
    multiplexing_df1 = pd.read_csv(df_real1)
    multiplexing_df2 = pd.read_csv(df_real2)

    ntraj = 1250
    ntimes = 350
    int_g2 = multiplexing_df1['red_int_mean'].values.reshape([ntraj,ntimes])    
    int_g1 = multiplexing_df2['red_int_mean'].values.reshape([ntraj,ntimes])   
 
    int_gs_kdm5b.append(int_g2.flatten().tolist())
    int_gs_labels_kdm5b.append([i]*len(int_g2.flatten()))
    int_gs_p300.append(int_g1.flatten().tolist())
    int_gs_labels_p300.append([i]*len(int_g1.flatten()))
int_df_kdm5b = pd.DataFrame({'pb':[item for sublist in int_gs_labels_kdm5b for item in sublist], 'I':[item for sublist in int_gs_kdm5b for item in sublist]})
int_df_p300 = pd.DataFrame({'pb':[item for sublist in int_gs_labels_p300 for item in sublist], 'I':[item for sublist in int_gs_p300 for item in sublist]})

joypy.joyplot(int_df_kdm5b, by='pb')
plt.xlim([0,8000])
plt.xlabel('Intensity')
plt.title('KDM5B All (R)')
joypy.joyplot(int_df_p300, by='pb')
plt.xlim([0,8000])
plt.xlabel('Intensity')
plt.title('P300 All (R)')

