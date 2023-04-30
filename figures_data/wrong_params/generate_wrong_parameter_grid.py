# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 13:26:14 2023

@author: willi
"""


##############################################################################
#
# This code generates a matrix using the ke vs ki models / dataset
# Returns a 10,10,10,10,1 matrix of  model ki, model ke, data ki, data ke, acc 
# examples:
# Model 8,1 | applied to all data would be acc_keki_grid[8,1,:,:]
# All models applied to dataset 3,4 would be acc_keki_grid[:,:,3,4]
##############################################################################

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
sess = tf.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)
import tqdm

import logging
logging.getLogger('tensorflow').setLevel(logging.FATAL)

'''
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("no connection to gpu")
'''
import numpy as np
import pandas as pd
from tensorflow import keras

from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, AveragePooling1D, Conv1D, LeakyReLU, Lambda
from tensorflow.keras.regularizers import l1_l2

from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import os 

cwd = os.getcwd()
os.chdir('../../')
from multiplex_core import multiplexing_core 
os.chdir(cwd)

mc = multiplexing_core()

model_dir = '../../ML_experiments/ML_run_320_5s_wfreq/parsweep_keki_ML/'

input_size = 64
def load_model(model_file):

    def signal_model(input_size_1, kernel_size, filters):
        model = Sequential()
        model.add(Conv1D(filters=filters, kernel_size=kernel_size, kernel_regularizer=l1_l2(l1=1e-5), padding="same", input_shape=(input_size_1, 1)))
        model.add(LeakyReLU(alpha=0.3))
        model.add(AveragePooling1D(pool_size=2))  
        model.add(Flatten())
        return model
    
    def freq_model(input_size_2, kernel_size, filters):
        model = Sequential()
        model.add(Conv1D(filters=filters, kernel_size=kernel_size, kernel_regularizer=l1_l2(l1=1e-5), padding="same", input_shape=(input_size_2, 1)))
        model.add(LeakyReLU(alpha=0.3))
        model.add(AveragePooling1D(pool_size=2))  
        model.add(Flatten())
        return model
    
    def create_model(input_size_1, input_size_2, N_neurons, kernel_size, filters, lr):
    
        combi_input = keras.layers.Input(shape = (input_size_1+input_size_2,1) ) 
        
        Input_1 = Lambda(lambda x: x[:,:input_size_1,:])(combi_input)
        Input_2 = Lambda(lambda x: x[:,input_size_1:,:])(combi_input)
    
        signal_output = signal_model(input_size_1, kernel_size, filters)(Input_1)
        freq_output = freq_model(input_size_2, kernel_size, filters)(Input_2)
    
        cat_output = keras.layers.concatenate([signal_output, freq_output])
    
        dense_output =  Dense(N_neurons, kernel_regularizer=l1_l2(l1=1e-5), activation = LeakyReLU(alpha=0.3))(cat_output)
        model_out = Dense(1,activation='sigmoid')(dense_output)
    
        optmizer = keras.optimizers.Adam(lr=lr)
        model = keras.Model(inputs=[combi_input], outputs=model_out)
        model.compile(loss='binary_crossentropy', optimizer=optmizer, metrics=['accuracy'],)
    
        return model
    
    kernel_size = int(model_file.split('_')[-4])
    filters = int( model_file.split('_')[-5] .split('.')[0] )
    model = create_model(input_size,input_size, 200, kernel_size, filters, .001)
    model.load_weights(model_file)
    return model


data_path = '../../datasets/par_sweep_5000/'
n_traj = 1000

kis = ['0.01','0.020000000000000004', '0.030000000000000006','0.04000000000000001','0.05000000000000001',
       '0.06000000000000001','0.07','0.08','0.09000000000000001','0.1']
kes = ['2.0','3.111111111111111','4.222222222222222','5.333333333333334','6.444444444444445','7.555555555555555','8.666666666666668',
       '9.777777777777779','10.88888888888889','12.0']


print('making model grid....')
model_grid = []
model_keys = []
with tqdm.tqdm(range(100)) as pbar:
  for i in range(10):
    model_sub_grid = []
    model_sub_keys = []
    for j in range(10):
      model_file = [x for x in os.listdir(model_dir) if x[-7:] == '_%i_%i.h5'%(i,j)][0]
      model_file = model_dir + '%s'%model_file
      model_sub_grid.append(load_model(model_file))
      model_sub_keys.append((i,j))
      pbar.update(1)
    model_grid.append(model_sub_grid)
    model_keys.append(model_sub_keys)



print('making_data_grid....')
data_grid = []
label_grid = []
with tqdm.tqdm(range(100)) as pbar:
    for i in range(10):
        sub_data_grid = []
        sub_label_grid = []
        for j in range(10):
            data_file = data_path + 'ki_ke_sweep_5000spots_%s_%s.csv'%(kis[i],kes[j])
            multiplexing_df1 = pd.read_csv(data_file)
            int_g1 = multiplexing_df1['green_int_mean'].values.reshape([5000,3000])    
            labels = multiplexing_df1['Classification'].values.reshape([5000,3000])[:,0]
            int_g, labels = mc.even_shuffle_sample(int_g1, labels, samples=[2500, 2500], seed=42)
            int_g = mc.slice_arr(int_g, 5, 64)
            _, _, _, _, X_witheld, y_witheld, _, _, Acc_witheld = mc.process_data(int_g, labels, norm='train', seed=42, witheld = 1000, test_size = 0, include_acc = True )
            
            x = np.concatenate((X_witheld, Acc_witheld),axis=1)
            y = y_witheld
            sub_data_grid.append(x)
            sub_label_grid.append(y)
            pbar.update(1)
            
        data_grid.append(sub_data_grid)
        label_grid.append(sub_label_grid)


print('running_all_predictions...')
y_pred = np.zeros([10,10,10,10,1000])
acc_grid = np.zeros([10,10,10,10,1])
with tqdm.tqdm(range(100*100)) as pbar:
    for m in range(10):
        for k in range(10):
            model = model_grid[m][k]
            for i in range(10):
                for j in range(10):
                    y_hat = model.predict(data_grid[i][j])
                    y_pred[m,k,i,j,:] = y_hat.flatten()
                    print(y_hat.dtype)
                    print(label_grid[i][j].dtype)
                    print(y_hat.shape)
                    print(label_grid[i][j].shape)
                    acc_grid[m,k,i,j,0] = accuracy_score(label_grid[i][j].astype(int), (y_hat.flatten() > .5).astype(int))
                    

np.save('./y_predicted_keki.npy', y_pred)
np.save('./acc_grid_keki.npy', acc_grid)
np.save('./y_true_keki.npy', np.array(label_grid))