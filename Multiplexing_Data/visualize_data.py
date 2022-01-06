# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 11:06:05 2020

@author: willi
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file = '20191025_1xFLAG-12xSun-AlexX-MS2_12xSun-KDM5B-MS2_12xFLAG-H2B-MS2_MCP-CAAX.xls'

data = pd.read_excel(file)
data.head()