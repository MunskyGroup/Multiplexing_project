# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 21:58:25 2022

@author: willi
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

import os
cwd = os.getcwd()
os.chdir('../../')
import apply_style  as aps#apply custom matplotlib style
os.chdir(cwd)

target_dir = '../../ML_run_3000_2s_wfreq'
aps.make_heatmaps_from_keyfile(target_dir)