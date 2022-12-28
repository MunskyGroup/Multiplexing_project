# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 11:53:27 2022

@author: willi
"""

### Supplemental figure showing gaussian generation of new video backgrounds

# imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

import os
cwd = os.getcwd()
os.chdir('../../')
import apply_style #apply custom matplotlib style
os.chdir(cwd)
apply_style.apply_style()

n = 19  # frames -1 for change in frames


## Gaussian generated video Z scores

plt.figure()
plt.gca().set_aspect('equal')

generated_video_gaussian = np.load('./generated_video_gaussian.npy')
original_video = np.load('./original_video.npy')


def quantile_norm(movie, q, min_val=None ):
   max_val = np.quantile(movie, q)
   if min_val == None:
    min_val = np.quantile(movie, .005)
   norm_movie = (movie - min_val)/(max_val - min_val)
   norm_movie[norm_movie > 1] = 1
   norm_movie[norm_movie < 0] = 0
   return norm_movie

min_val = np.quantile(original_video, .005)
norm_original = quantile_norm(original_video,.95)
norm_gaussian = quantile_norm(generated_video_gaussian, .95, min_val = min_val)


for i in range(n):
  z0 = (generated_video_gaussian[i] - np.mean(generated_video_gaussian, axis=0)) / np.std(generated_video_gaussian,axis=0)
  z1 = (generated_video_gaussian[i+1] - np.mean(generated_video_gaussian, axis=0)) / np.std(generated_video_gaussian,axis=0)
  plt.scatter(z0.flatten(),z1.flatten(),  s=1, alpha=.05, facecolor=cm.jet(i/n))

plt.plot([0,0],[-4,4], 'k--')
plt.plot([-4,4],[0,0], 'k--')
plt.ylabel(r'Z-score $tau=1$')
plt.xlabel(r'Z-score $tau=0$')
plt.savefig('./gzscore_gaussian_generated.png',  transparent=True )



## Original video Z scores

plt.figure()
plt.gca().set_aspect('equal')

for i in range(0,n):
  z0 = (original_video[i] - np.mean(original_video, axis=0)) / np.std(original_video,axis=0)
  z1 = (original_video[i+1] - np.mean(original_video, axis=0)) / np.std(original_video,axis=0)
  plt.scatter(z0.flatten(),z1.flatten(),  s=1, alpha=.05, facecolor=cm.jet(i/n))

plt.plot([0,0],[-4,4], 'k--')
plt.plot([-4,4],[0,0], 'k--')
plt.xlim([-4.2,4.2])
plt.ylim([-4.2,4.2])
plt.ylabel(r'Z-score $tau=1$')
plt.xlabel(r'Z-score $tau=0$')
plt.savefig('./gzscore_orig.png', transparent=True )

# CDF comparison

plt.figure(figsize=(5,4))
x, bins = np.histogram(norm_original.flatten(), bins=100)
x2, _ = np.histogram(norm_gaussian.flatten(), bins = bins)

x = np.array(x.tolist() + [0])
x2 = np.array(x2.tolist() + [0])

plt.plot(bins+(1/200),np.cumsum(x)/np.cumsum(x)[-1] )
plt.plot(bins+(1/200),np.cumsum(x2)/np.cumsum(x)[-1] ,'--')

plt.ylim([-.1,1])
plt.xlabel('Normalized pixel value')
plt.ylabel('CDF')
plt.legend(['Original Video (20 Frames)', 'Generated Video (20 Frames)'])
plt.savefig('video_density.svg')



# Frame 0 of original and gaussian generated video

fig,ax = plt.subplots(1,1)
ax.imshow(norm_original[0], cmap='Greens_r', vmax=1, vmin=0)
plt.gca().xaxis.set_ticks([]);plt.gca().yaxis.set_ticks([]) 


a = [Rectangle([200,60],50,50, edgecolor='red'), Rectangle([170,265],20,20, edgecolor='white')]

pc = PatchCollection(a, facecolor='None', alpha=1,
                      match_original=True)

ax.add_collection(pc)



fig,ax = plt.subplots(1,1)
ax.imshow(norm_gaussian[0], cmap='Greens_r', vmax=1, vmin=0)
plt.gca().xaxis.set_ticks([]);plt.gca().yaxis.set_ticks([]) 


a = [Rectangle([200,60],50,50, edgecolor='red'), Rectangle([170,265],20,20, edgecolor='white')]

pc = PatchCollection(a, facecolor='None', alpha=1,
                      match_original=True)

ax.add_collection(pc)

###########

# Zoomed in segments on areas of interest

fig,ax = plt.subplots(1,1, )
ax.imshow(norm_original[0, 60:110, 200:250], cmap='Greens_r', vmax=1, vmin=0)
plt.gca().xaxis.set_ticks([]);plt.gca().yaxis.set_ticks([]) 

fig,ax = plt.subplots(1,1, )
ax.imshow(norm_original[9, 60:110, 200:250], cmap='Greens_r', vmax=1, vmin=0)
plt.gca().xaxis.set_ticks([]);plt.gca().yaxis.set_ticks([]) 


fig,ax = plt.subplots(1,1, )
ax.imshow(norm_original[19, 60:110, 200:250], cmap='Greens_r', vmax=1, vmin=0)
plt.gca().xaxis.set_ticks([]);plt.gca().yaxis.set_ticks([]) 


fig,ax = plt.subplots(1,1, )
ax.imshow(norm_gaussian[0, 60:110, 200:250], cmap='Greens_r', vmax=1, vmin=0)
plt.gca().xaxis.set_ticks([]);plt.gca().yaxis.set_ticks([]) 

fig,ax = plt.subplots(1,1, )
ax.imshow(norm_gaussian[9, 60:110, 200:250], cmap='Greens_r', vmax=1, vmin=0)
plt.gca().xaxis.set_ticks([]);plt.gca().yaxis.set_ticks([]) 


fig,ax = plt.subplots(1,1, )
ax.imshow(norm_gaussian[19, 60:110, 200:250], cmap='Greens_r', vmax=1, vmin=0)
plt.gca().xaxis.set_ticks([]);plt.gca().yaxis.set_ticks([]) 




############



# Zoomed in segments on areas of normal area

fig,ax = plt.subplots(1,1, )
ax.imshow(norm_original[0, 265:285, 170:190], cmap='Greens_r', vmax=1, vmin=0)
plt.gca().xaxis.set_ticks([]);plt.gca().yaxis.set_ticks([]) 


fig,ax = plt.subplots(1,1, )
ax.imshow(norm_original[9, 265:285, 170:190], cmap='Greens_r', vmax=1, vmin=0)
plt.gca().xaxis.set_ticks([]);plt.gca().yaxis.set_ticks([]) 



fig,ax = plt.subplots(1,1, )
ax.imshow(norm_original[19, 265:285, 170:190], cmap='Greens_r', vmax=1, vmin=0)
plt.gca().xaxis.set_ticks([]);plt.gca().yaxis.set_ticks([]) 



fig,ax = plt.subplots(1,1, )
ax.imshow(norm_gaussian[0, 265:285, 170:190], cmap='Greens_r', vmax=1, vmin=0)
plt.gca().xaxis.set_ticks([]);plt.gca().yaxis.set_ticks([]) 


fig,ax = plt.subplots(1,1, )
ax.imshow(norm_gaussian[9, 265:285, 170:190], cmap='Greens_r', vmax=1, vmin=0)
plt.gca().xaxis.set_ticks([]);plt.gca().yaxis.set_ticks([]) 



fig,ax = plt.subplots(1,1, )
ax.imshow(norm_gaussian[19, 265:285, 170:190], cmap='Greens_r', vmax=1, vmin=0)
plt.gca().xaxis.set_ticks([]);plt.gca().yaxis.set_ticks([]) 




