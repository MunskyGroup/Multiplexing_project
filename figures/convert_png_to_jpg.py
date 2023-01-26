# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 11:23:16 2023

@author: willi
"""


import numpy as np
import os 
from PIL import Image

files = os.listdir('.')

for f in files:
    if f[-6:] == '_w.png':
        im = Image.open(f)
        rgba = np.array(im)
        #rgba[rgba[...,-1]==0] = [255,255,255,0]
        
        Image.fromarray(rgba).convert('RGB').save(f[:-6] + '.jpg', subsampling=0, quality=100)
        
        