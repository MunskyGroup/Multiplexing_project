# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 14:15:44 2022

@author: willi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from cycler import cycler

###############################################################################
#  Style file for paper, applies a consistent style to each figure / plot
#  Usage: from apply_style import apply_style; apply_style()
###############################################################################

def make_heatmaps_from_keyfile(target_dir, format_type='svg',):
    '''
    Generates the ML parameter sweep heatmaps

    Parameters
    ----------
    target_dir : str
        target directory to make an ML heat map for
    format_type : str, optional
        string file extension to save the figure as, ex: svg or png. The default is 'svg'.

    Returns
    -------
    None
    '''
    dark = False
    flipped_x = False
    flipped_y = False
    metric_on = False
    c_map = 'Spectral'
    
    sep_default_cl = 0
    
    vscale_min = .5
    vscale_max = 1
    
    cmap = plt.get_cmap(c_map)
    
    def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
        new_cmap = LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
        return new_cmap
    
    new_cmap = truncate_colormap(cmap, 0.3, 1)
    my_cmap = new_cmap(np.arange(cmap.N))
    my_cmap[:, -1] = .9
    my_cmap = ListedColormap(my_cmap)
    
    
    ##############################################################################
    
    colors = ['#ef476f', '#073b4c','#06d6a0','#7400b8','#073b4c', '#118ab2',]

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
    plt.rcParams.update({'figure.dpi':120})
    
    ##############################################################################
    
    #target_dir = 'ML_run1'
    acc_mat_name = 'acc_mat'
    save_dir = 'plots'
    
    
    if not os.path.exists(os.path.join('.', target_dir,save_dir)):
        os.makedirs(os.path.join('.', target_dir,save_dir))
    
    
    acc_mats = []
    key_csvs = []
    for root, subdirs, files in os.walk(os.path.join('.',target_dir)):
        for f in files:
            if 'key.csv' in f:
                key_csvs.append(os.path.join(root,f))
            if '.npy' in f and acc_mat_name in f:
                acc_mats.append(os.path.join(root,f))
    

    def make_plot(f, save_name, format_str):
        
        
        
        key_file = pd.read_csv(f)
        
        xshape = key_file.shape[0]
        yshape = key_file.shape[1]-1
        
        convert_str = lambda tstr: [x.replace(' ','').replace('(','').replace(')','').replace("'",'') for x in tuple(tstr.split(','))] 
        round_str = lambda fstr: str(np.round(float(fstr),2))
        
        acc_mat = np.zeros([xshape,yshape])
        xlabels = [0,]*xshape
        ylabels = [0,]*yshape
        for i in range(0,xshape):
            for j in range(0,yshape):
                
                ind1,ind2, x,y,acc = convert_str(key_file.iloc[i][j+1])
               
                acc_mat[i,j] = acc
                xlabels[int(ind1)] = x
                ylabels[int(ind2)] = y
            
        acc_mat = acc_mat.T
        xlabel = ''
        ylabel = ''
        title = ''
        if 'kes' in save_name:
            xlabel = r'$k_{e}$ mRNA 1 (aa/s)'
            ylabel = r'$k_{e}$ mRNA 2 (aa/s)'
            title = 'Test Accuracy over mRNAs with different $k_e$'
            ylabels = [round_str(x) for x in ylabels]
            xlabels = [round_str(x) for x in xlabels]
        if 'kis' in save_name:
            xlabel = r'$k_{i}$ mRNA 1 (1/s)'
            ylabel = r'$k_{i}$ mRNA 2 (1/s)'
            title = 'Test Accuracy over mRNAs with different $k_i$'
            ylabels = [round_str(x) for x in ylabels]
            xlabels = [round_str(x) for x in xlabels]    
        
        if 'keki' in save_name:
            xlabel = r'$k_{i}$ (1/s)'
            ylabel = r'$k_{e}$ (aa/s)'  
            title = r'Test Accuracy over $k_i$ and $k_e$ pairs'
            ylabels = [round_str(x) for x in ylabels]
            xlabels = [round_str(x) for x in xlabels]
        if 'img' in save_name:
            xlabel = r'Frame Rate, $FR$ (s)'
            ylabel = r'Number of Frames, $n_F$'
            title = 'Test Accuracy over Imaging Conditions'
            ylabels = [str(int(float(round_str(x)))) for x in ylabels]
            xlabels = [str(int(float(round_str(x)))) for x in xlabels]
            
        if 'cl' in save_name:
            title = 'Test Accuracy vs Construct Length'
            xlabel = 'mRNA length (NT)'
            if not sep_default_cl:
                xlabels = ['1200','1734','2265','2799','3333','3867','4401','4647','4932','5466','6000','7257' ]
            else: 
                xlabels = ['1200','1734','2265','2799','3333','3867','4401','4932','5466','6000','4647','7257' ]
        if 'cl' in save_name:
            A = np.random.randint(0, 1, xshape*yshape).reshape(xshape, yshape)
            
            mask =  np.tri(A.shape[0], k=-1)
            acc_mat = np.ma.array(acc_mat.T,mask=mask)   
        
        ##################################################
        
    
        fig,ax = plt.subplots(1,1,dpi=120, tight_layout=True) 
        
        
        b = ax.imshow(acc_mat,cmap =my_cmap, vmin = vscale_min, vmax = vscale_max,origin='lower' )
        ax.set_yticks(np.arange(yshape),)
        ax.set_xticks( np.arange(xshape))
        fig.colorbar(b)
        ax.set_yticklabels(ylabels, fontdict = {'fontsize':7})
        ax.set_xticklabels(xlabels, fontdict = {'fontsize':7},rotation=45)
        
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        ax.set_title(title)
        if 'cl' in save_name:
            for x in range(yshape):
                for y in range(xshape):
                    if x <= y:
                        ax.text(y,x, '%.2f' % acc_mat[x, y],
                                 horizontalalignment='center',
                                 verticalalignment='center',
                                 size=5)
        else:
            for x in range(yshape):
                for y in range(xshape):
                    if acc_mat[x, y] != 0:
                        
                        ax.text(y,x, '%.2f' % acc_mat[x, y],
                                 horizontalalignment='center',
                                 verticalalignment='center',
                                 size=5)
                    else:
                        ax.text(y,x, '-',
                                 horizontalalignment='center',
                                 verticalalignment='center',
                                 size=5)                    
                        
        
        plt.savefig(save_name, format=format_type, transparent=True)
        
    i = 0
    
    for key in key_csvs:
        print(key)
        print('making heatmap....')
        
        save_name = os.path.split(key)[1].split('.')[0] + '_%i'%i
        file_path = os.path.split(key)[0]
        #key_file = save_name.split('_')[-1] + '_key.csv'
        #key_file = os.path.join(file_path,key_file)
        i+=1
        print(os.getcwd())
        make_plot( key, os.path.join(os.getcwd(),save_name) + '.' +format_type, format_type)
    




def make_heatmaps_from_keyfile_big(target_dir, format_type='svg',):
    '''
    Generates the large ML parameter sweep heatmaps

    Parameters
    ----------
    target_dir : str
        target directory to make an ML heat map for
    format_type : str, optional
        string file extension to save the figure as, ex: svg or png. The default is 'svg'.

    Returns
    -------
    None
    '''
    dark = False
    flipped_x = False
    flipped_y = False
    metric_on = False
    c_map = 'Spectral'
    
    sep_default_cl = 0
    
    vscale_min = .5
    vscale_max = 1
    
    cmap = plt.get_cmap(c_map)

    
    def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
        new_cmap = LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
        return new_cmap
    
    new_cmap = truncate_colormap(cmap, 0.3, 1)
    my_cmap = new_cmap(np.arange(cmap.N))
    my_cmap[:, -1] = .9
    my_cmap = ListedColormap(my_cmap)
    
    
    ##############################################################################
    
    colors = ['#ef476f', '#073b4c','#06d6a0','#7400b8','#073b4c', '#118ab2',]
    save = False
    
    plt.rcParams.update({'font.size': 8, 'font.weight':'bold','font.family':'normal'  }   )
    plt.rcParams.update({'axes.prop_cycle':cycler(color=colors)})
    
    plt.rcParams.update({'axes.prop_cycle':cycler(color=colors)})
    plt.rcParams.update({'axes.prop_cycle':cycler(color=colors)})
    
    
    plt.rcParams.update({'xtick.major.width'   : 2.8 })
    plt.rcParams.update({'xtick.labelsize'   : 8 })
    
    
    
    plt.rcParams.update({'ytick.major.width'   : 2.8 })
    plt.rcParams.update({'ytick.labelsize'   : 8})
    
    plt.rcParams.update({'axes.titleweight'   : 'bold'})
    plt.rcParams.update({'axes.titlesize'   : 10})
    plt.rcParams.update({'axes.labelweight'   : 'bold'})
    plt.rcParams.update({'axes.labelsize'   : 8})
    
    plt.rcParams.update({'axes.linewidth':2.8})
    plt.rcParams.update({'axes.labelpad':8})
    plt.rcParams.update({'axes.titlepad':10})
    plt.rcParams.update({'figure.dpi':120})
    
    ##############################################################################
    
    #target_dir = 'ML_run1'
    acc_mat_name = 'acc_mat'
    save_dir = 'plots'
    
    
    if not os.path.exists(os.path.join('.', target_dir,save_dir)):
        os.makedirs(os.path.join('.', target_dir,save_dir))
    
    
    acc_mats = []
    key_csvs = []
    for root, subdirs, files in os.walk(os.path.join('.',target_dir)):
        for f in files:
            if 'key.csv' in f:
                key_csvs.append(os.path.join(root,f))
            if '.npy' in f and acc_mat_name in f:
                acc_mats.append(os.path.join(root,f))
    

    def make_plot(f, save_name, format_str):
        key_file = pd.read_csv(f)
        
        xshape = key_file.shape[0]
        yshape = key_file.shape[1]-1
        
        convert_str = lambda tstr: [x.replace(' ','').replace('(','').replace(')','').replace("'",'') for x in tuple(tstr.split(','))] 
        round_str = lambda fstr: str(np.round(float(fstr),2))
        
        acc_mat = np.zeros([xshape,yshape])
        xlabels = [0,]*xshape
        ylabels = [0,]*yshape
        for i in range(0,xshape):
            for j in range(0,yshape):
                
                ind1,ind2, x,y,acc = convert_str(key_file.iloc[i][j+1])
               
                acc_mat[i,j] = acc
                xlabels[int(ind1)] = x
                ylabels[int(ind2)] = y
            
        acc_mat = acc_mat.T
        xlabel = ''
        ylabel = ''
        title = ''
        if 'kes' in save_name:
            xlabel = r'$k_{elongation}$ mRNA 1'
            ylabel = r'$k_{elongation}$ mRNA 2'
            title = 'Test Accuracy over mRNAs with different $k_e$'
            ylabels = [round_str(x) for x in ylabels]
            xlabels = [round_str(x) for x in xlabels]
        if 'kis' in save_name:
            xlabel = r'$k_{initation}$ mRNA 1'
            ylabel = r'$k_{initation}$ mRNA 2'
            title = 'Test Accuracy over mRNAs with different $k_i$'
            ylabels = [round_str(x) for x in ylabels]
            xlabels = [round_str(x) for x in xlabels]    
        
        if 'keki' in save_name:
            xlabel = r'$k_{initation}$'
            ylabel = r'$k_{elongation}$'  
            title = r'Test Accuracy over $k_i$ and $k_e$ pairs'
            ylabels = [round_str(x) for x in ylabels]
            xlabels = [round_str(x) for x in xlabels]
        if 'img' in save_name:
            xlabel = r'Frame Rate, $FR$ (s)'
            ylabel = r'Number of Frames, $n_F$'
            title = 'Test Accuracy over Imaging Conditions'
            ylabels = [str(int(float(round_str(x)))) for x in ylabels]
            xlabels = [str(int(float(round_str(x)))) for x in xlabels]
            
        if 'cl' in save_name:
            title = 'Test Accuracy vs Construct Length'
            xlabel = 'mRNA length (NT)'
            if not sep_default_cl:
                xlabels = ['1200','1734','2265','2799','3333','3867','4401','4647','4932','5466','6000','7257' ]
            else: 
                xlabels = ['1200','1734','2265','2799','3333','3867','4401','4932','5466','6000','4647','7257' ]
        if 'cl' in save_name:
            A = np.random.randint(0, 1, xshape*yshape).reshape(xshape, yshape)
            
            mask =  np.tri(A.shape[0], k=-1)
            acc_mat = np.ma.array(acc_mat.T,mask=mask)   
        
        ##################################################
        
    
        fig,ax = plt.subplots(1,1,dpi=300, tight_layout=True) 
        
        
        b = ax.imshow(acc_mat,cmap =my_cmap, vmin = vscale_min, vmax = vscale_max,origin='lower' )
        ax.set_yticks(np.arange(yshape),)
        ax.set_xticks( np.arange(xshape))
        fig.colorbar(b)
        ax.set_yticklabels(ylabels, fontdict = {'fontsize':7})
        ax.set_xticklabels(xlabels, fontdict = {'fontsize':7},rotation=45)
        
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        ax.set_title(title)
        if 'cl' in save_name:
            for x in range(yshape):
                for y in range(xshape):
                    if x <= y:
                        ax.text(y,x, '%.2f' % acc_mat[x, y],
                                 horizontalalignment='center',
                                 verticalalignment='center',
                                 size=4)
        else:
            for x in range(yshape):
                for y in range(xshape):
                    if acc_mat[x, y] != 0:
                        
                        ax.text(y,x, '%.2f' % acc_mat[x, y],
                                 horizontalalignment='center',
                                 verticalalignment='center',
                                 size=4)
                    else:
                        ax.text(y,x, '-',
                                 horizontalalignment='center',
                                 verticalalignment='center',
                                 size=4)                    
                        
        
        plt.savefig(save_name, format=format_type, transparent=True)
        
    i = 0
    
    for key in key_csvs:
        print(key)
        print('making heatmap....')
        
        save_name = os.path.split(key)[1].split('.')[0] + '_%i'%i
        file_path = os.path.split(key)[0]
        #key_file = save_name.split('_')[-1] + '_key.csv'
        #key_file = os.path.join(file_path,key_file)
        i+=1
        print(os.getcwd())
        make_plot( key, os.path.join(os.getcwd(),save_name) + '.' +format_type, format_type)
    



def apply_style(dark=False):
    '''
    Applies a consistent matplotlib style to any figure for the paper

    Parameters
    ----------
    dark : bool, optional
        Use a dark theme instead of bright. The default is False.

    Returns
    -------
    None.

    '''
    if not dark:
        colors = ['#073b4c', '#ef476f','#06d6a0','#7400b8','#073b4c', '#118ab2',]
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