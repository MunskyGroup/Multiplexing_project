# -*- coding: utf-8 -*-
"""cnn.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1D0wYRLSFlS7ExoHUBASQSXxTaXsTHg-t
"""


import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
sess = tf.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

#if tf.test.gpu_device_name():
#    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
#else:
#    print("no connection to gpu")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix

from tensorflow.keras import layers, models, optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, AveragePooling1D, Conv1D, LeakyReLU
from tensorflow.keras.regularizers import l2, l1, l1_l2
from tensorflow.keras.layers import BatchNormalization
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from sklearn import mixture

# path to data
# parse arguments 

##############
parser = argparse.ArgumentParser(description='get paths and save dir and acc_file_name')
parser.add_argument('--i', dest='ind1', type=int,)
parser.add_argument('--j', dest='ind2', type=int,)
parser.add_argument('--data_file1', dest='path1', type=str,)
parser.add_argument('--data_file2', dest='path2', type=str,)
parser.add_argument('--save_model', dest='save_model', type=bool,)
parser.add_argument('--save_dir', dest='save_dir', type=str,)
parser.add_argument('--acc_file', dest='acc_file',type=str)
parser.add_argument('--retrain', dest='retrain', type=bool)
parser.add_argument('--model_file', dest='model_file', type=str)
parser.add_argument('--model_name', dest='model_name', type=str)
parser.add_argument('--verbose', dest='verbose', type=str)
parser.add_argument('--Fr', dest='framerate', type=int)
parser.add_argument('--NFr', dest='Nframes', type=int)
###############

args = parser.parse_args()


verbose = args.verbose
ind1 = args.ind1
ind2 = args.ind2
path1 = args.path1
path2 = args.path2
save_dir = args.save_dir
save_model = args.save_model
model_file = args.model_file
acc_file = args.acc_file
retrain = args.retrain
model_name = args.model_name
Nframes = args.Nframes
Fr = args.framerate

def slice_arr(array, FR, Nframes,axis=1):
    total_time = FR*Nframes
    if total_time > 3000:
        print('WARNING: desired slicing regime is not possible, making as many frames as possible')
        return array[:,::FR]
    return array[:,::FR][:,:Nframes]

if verbose:
    print('reading csvs....')
multiplexing_df1 = pd.read_csv(path1)
multiplexing_df2 = pd.read_csv(path2)

def convert_labels_to_onehot(labels):
    '''
    converts labels in the format 1xN, [0,0,1,2,3,...] to onehot encoding,
    ie: N_classes x N,  [[1,0,0,0],[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]     
    '''
    onehotlabels = np.zeros((labels.shape[0],len(np.unique(labels))))
    for i in range(len(onehotlabels)):
        onehotlabels[i,labels[i]] = 1
    return onehotlabels

def df_to_array(dataframe_simulated_cell):
    total_particles = 0
    for cell in set(dataframe_simulated_cell['cell_number']):
        total_particles += len(set(dataframe_simulated_cell[dataframe_simulated_cell['cell_number'] == 0]['particle'] ))

  #preallocate numpy array sof n_particles by nframes
    I_g = np.zeros([total_particles, np.max(dataframe_simulated_cell['frame'])+1] )  #intensity green
    I_g_std = np.zeros([total_particles, np.max(dataframe_simulated_cell['frame'])+1] ) #intensity green std
    x_loc = np.zeros([total_particles, np.max(dataframe_simulated_cell['frame'])+1] ) #x loc
    y_loc = np.zeros([total_particles, np.max(dataframe_simulated_cell['frame'])+1] ) #y_loc
    I_r_std   = np.zeros([total_particles, (np.max(dataframe_simulated_cell['frame'])+1)] ) #intensity red
    I_r = np.zeros([total_particles, (np.max(dataframe_simulated_cell['frame'])+1) ] ) #intensity red std
    labels = np.zeros([total_particles])
    label_list = list(set(np.unique(dataframe_simulated_cell['Classification'])))
    k = 0
    for cell in set(dataframe_simulated_cell['cell_number']):  #for every cell 
        for particle in set(dataframe_simulated_cell[dataframe_simulated_cell['cell_number'] == 0]['particle'] ): #for every particle
            tmpdf = dataframe_simulated_cell[(dataframe_simulated_cell['cell_number'] == cell) & (dataframe_simulated_cell['particle'] == particle)]  #slice the dataframe
            maxframe = np.max(tmpdf['frame'])
            minframe = np.min(tmpdf['frame'])
            I_g[k, 0:(maxframe+1-minframe)] = tmpdf['green_int_mean']  #fill the arrays to return out
            x_loc[k, 0:(maxframe+1-minframe)] = tmpdf['x']
            y_loc[k, 0:(maxframe+1-minframe)] = tmpdf['y']
            I_g_std[k, 0:(maxframe+1-minframe)] = tmpdf['green_int_std']
            #I_r[k, 0:(maxframe+1-minframe)] = tmpdf['red_int_mean']
            #I_r_std[k, 0:(maxframe+1-minframe)] = tmpdf['red_int_std']
            labels[k] = label_list.index(list(set(np.unique(tmpdf['Classification'])))[0])

            k+=1 #iterate over k (total particles)
    return I_g, I_g_std, I_r, I_r_std, labels, x_loc,y_loc   #return everything backout

if verbose:
    print('converting csv....')
    
int_g1 = multiplexing_df1['green_int_mean'].values.reshape([2500,3000])    
int_g2 = multiplexing_df2['green_int_mean'].values.reshape([2500,3000])        
#int_g1, _, _, _, _, _, _ = df_to_array(multiplexing_df1)
#int_g2, _, _, _, _, _, _ = df_to_array(multiplexing_df2)
t = np.linspace(0,len(int_g1) - 1,len(int_g1))  #time vector in seconds



labels = np.ones(int_g1.shape[0]*2)
labels[:int_g1.shape[0]] = 0
labels_onehot = convert_labels_to_onehot(labels.astype(int))

int_g = np.vstack((int_g1,int_g2)) #merge the files and then let it sort
scaler = MinMaxScaler()
int_g = scaler.fit_transform(int_g)

#check shape
int_g = slice_arr(int_g, Fr, Nframes)
Nframes = int_g.shape[1]


X_train, X_test, y_train, y_test = train_test_split(int_g, labels, test_size=.2, random_state= 42)
X_test.shape, y_test.shape, X_train.shape

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

y_train = np.asarray(y_train).astype('float32').reshape((-1,1))
y_test = np.asarray(y_test).astype('float32').reshape((-1,1))


input_list = lambda strinput: [int(x) for x in strinput.split(',')]

def create_model(kernel_sizes, filter_sizes, pool_sizes):
    
    
    #encoding layers
    maxpooling=True
    strides = "1,1,1"
    total_encoding_layers = 2
    activation_function = 'tanh'
    w,h = Nframes,1
    center_layer_size = 2
    model = Sequential()
    model.add(keras.Input(shape=(w, 1))) # input layer

    for i in range(total_encoding_layers):
      pool_size = input_list(pool_sizes)[i]
      stride_size =  input_list(strides)[i]
      model.add(keras.layers.Conv1D(input_list(filter_sizes)[i], input_list(kernel_sizes)[i], stride_size, activation=activation_function,padding='same'))
      if maxpooling:
        model.add(keras.layers.MaxPooling1D(pool_size= pool_size))

    center_shape_in = (int(w/np.prod(input_list(pool_sizes ))) ,int(input_list(filter_sizes)[-1]))
    center_shape_flat = center_shape_in[0]*center_shape_in[1]
    
    ## center layers
    model.add(keras.layers.Flatten())
    model.add( keras.layers.Dense(center_layer_size, activation='tanh',))
    model.add( keras.layers.Dense(center_shape_flat, activation='tanh',))
    model.add(keras.layers.Reshape(center_shape_in ))
        
    
    ## decoding layers layers layers
    
    for i in range(total_encoding_layers-1,-1,-1):
      
      pool_size = input_list(pool_sizes)[i]
      stride_size =  input_list(strides)[i]
      model.add(keras.layers.Conv1D(input_list(filter_sizes)[i], input_list(kernel_sizes)[i], stride_size, activation=activation_function,padding='same'))
      if maxpooling:
        model.add(keras.layers.UpSampling1D(   pool_size ))
    
    model.add(keras.layers.Conv1D(1, input_list(kernel_sizes)[-1],  stride_size, activation=activation_function,padding='same'))
       

    model.compile(loss='MSE',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model




if retrain:
    if verbose:
        print('training....')
    seed = 7
    np.random.seed(seed)
     
    
    model_CV = KerasClassifier(build_fn=create_model, verbose=0)
    
    filters= ["64,16",]
    kernel_size = ["4,2"]
    pools = ["4,2"]
    batches = [16, 32, 64]
    epochs = [50, 100]
    
    
    #distributions = dict(kernel_size = kernel_size, filters = filters, pools=pools, epochs= epochs, batch_size= batches)
    #random = RandomizedSearchCV(model_CV, distributions, n_iter= 2, verbose= 0, n_jobs= 1, cv=3)
    #random_result = random.fit(X_train, y_train)
    
    DAE = create_model(kernel_size[0], filters[0], pools[0])

    DAE.fit(x = X_train, y=X_train, validation_data = (X_test,X_test) ,epochs=40,verbose=0 )

    def encode(x,center_layer=5):
      y = DAE.layers[0](x)
      for i in range(1,center_layer+1):
        y = DAE.layers[i](y)
      return y
  
    
    latents = sess.run(encode(X_train.astype(np.float32)))
    latents_test = sess.run(encode(X_test.astype(np.float32)))
        
    gmm = mixture.GaussianMixture(n_components=2,
                                  covariance_type='full').fit(latents)
    
    gmm_labels = gmm.fit_predict(latents_test)  
    
    acc = accuracy_score(y_test,gmm_labels)
    if acc < .5:
        acc = 1-acc

    '''
    if verbose:
        print(random_result.best_score_)
        print(random_result.best_params_)
    best_model = random_result.best_estimator_.model
    best_params = random_result.best_params_
    best_kernel = best_params['kernel_size']
    best_filter = best_params['filters']
    best_pool = best_params['pools']
    '''
        
    #clf = random_result.best_estimator_
    #if verbose:
        #print('Test accuracy: %.3f' % clf.score(X_test, y_test))
   # acc = clf.score(X_test, y_test)
    
    if save_model:
        model_path = os.path.join('.', save_dir,model_name + '_'+ str(ind1) + '_' + str(ind2) + '.h5')
        DAE.save(model_path)
        
        gmm_name = os.path.join('.', save_dir,model_name + '_'+ str(ind1) + '_' + str(ind2))
        np.save(gmm_name + '_weights', gmm.weights_, allow_pickle=False)
        np.save(gmm_name + '_means', gmm.means_, allow_pickle=False)
        np.save(gmm_name + '_covariances', gmm.covariances_, allow_pickle=False)

    '''
    load in
    means = np.load(gmm_name + '_means.npy')
    covar = np.load(gmm_name + '_covariances.npy')
    loaded_gmm = mixture.GaussianMixture(n_components = len(means), covariance_type='full')
    loaded_gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covar))
    loaded_gmm.weights_ = np.load(gmm_name + '_weights.npy')
    loaded_gmm.means_ = means
    loaded_gmm.covariances_ = covar
    '''

else:
    if verbose:
        print('testing saved model....')
    #format: model_i_j_kernelsize_filters
    kernel_size = int(model_file.split('_')[-2])
    filters = int( model_file.split('_')[-1] .split('.')[0] )
    model = create_model(kernel_size,filters)
    model.load_weights(model_file)
    
    y_pred = model.predict(int_g)

    acc = accuracy_score(np.argmax(y_pred,axis=1), labels)
    if verbose:
        print('acc: %f '%acc)

if verbose:
    print('saving acc_mat...')
acc_path = os.path.join('.', save_dir, acc_file)
if not os.path.exists(acc_path):
    acc_mat = np.zeros([10,10])
    
else:    
    acc_mat = np.load(acc_path)
    
acc_mat[ind1,ind2] = acc
np.save(acc_path, acc_mat)
if verbose:
    print('Done.')