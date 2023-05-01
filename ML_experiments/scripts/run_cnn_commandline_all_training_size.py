# -*- coding: utf-8 -*-

###############################################################################
#
#  This code runs the classification given an intensity dataset and saves / 
#  returns the classifier and accuracy for a given condition. This is specific
#  to the training data size experiment, it will use a consistent data set
#  and will slice larger and larger sections to train on.
#
#  Arguments:
#    
#  --i    index one to save to the accuracy matrix, int
#  --j    index two to save to the accuracy matrix, int
#  --data_file1   path to data file 1, str
#  --data_file2   path to data file 2, str
#  --save_model   save the model, bool
#  --save_dir     path to save the model to, str
#  --acc_file     name of the accuracy file to save to, str
#  --retrain      retrain the model or use the model fpath, bool
#  --model_file   path to model to use, str
#  --model_name   name of the model to use, str
#  --verbose      verbose output, bool
#  --Fr           frame rate / frame interval to slice too, int
#  --NFr          number of frames to slice too, int
#  --ntraj        number of total trajectories per gene in each data file, int
#  --ntimes       total time in the original data, int
#  --two_files    data is in two files or one, bool
#  --Nsamples     number of total samples inside the data files, int
#  --witheld      how many samples to withold for test accuracy, int
#  --test_size    percentage to withold for test size, optional, float
#  --seed         rng seed, int
#
#  --test_type    test type? freq, no_freq, acc_only, standardize, zero mean  
#                 freq is used in the paper, meaning it includes the frequency 
#                 half of the architecture. no_freq is intensity only
#                 acc_only would be frequency only classification
#                 standardization would standardize both intensities
#                 zero mean would be zeroing the data means before classifying
#  --data_size    size of the training data to use
#  --subfold      which subfold to save too, ie original data is 5000 spots
#                 so we can only break this into 4 subfolds of 1000 spots + 
#                 1 1000 spot witheld dataset. subfold denotes which subfold
#                 to use.
###############################################################################


import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
sess = tf.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)


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

from multiplex_core import multiplexing_core 

mc = multiplexing_core()


# parse arguments 

##############
parser = argparse.ArgumentParser(description='get paths and save dir and acc_file_name')
parser.add_argument('--i', dest='ind1', type=int,)
parser.add_argument('--j', dest='ind2', type=int,)
parser.add_argument('--data_file1', dest='path1', type=str,)
parser.add_argument('--data_file2', dest='path2', type=str,default='')
parser.add_argument('--save_model', dest='save_model', type=bool,)
parser.add_argument('--save_dir', dest='save_dir', type=str,)
parser.add_argument('--acc_file', dest='acc_file',type=str)
parser.add_argument('--retrain', dest='retrain', type=bool)
parser.add_argument('--model_file', dest='model_file', type=str)
parser.add_argument('--model_name', dest='model_name', type=str)
parser.add_argument('--verbose', dest='verbose', type=str)
parser.add_argument('--Fr', dest='framerate', type=int)
parser.add_argument('--NFr', dest='Nframes', type=int)
parser.add_argument('--ntraj', dest='ntraj',type=int, default=2500)
parser.add_argument('--ntimes',dest='ntimes',type=int, default=3000)
parser.add_argument('--two_files', dest='two_files', type=int)
parser.add_argument('--Nsamples', dest='Nsamples',type=int, default=5000)
parser.add_argument('--witheld',dest='witheld',type=int)
parser.add_argument('--test_size',dest='test_size',type=float)
parser.add_argument('--seed',dest='seed',type=int)
parser.add_argument('--subfold',dest='subfold',type=int)
parser.add_argument('--data_size',dest='data_size',type=int)
parser.add_argument('--test_type',dest='test_type',type=str, default='freq')
parser.add_argument('--debug', dest='debug', type=int, default=0)


###############

args = parser.parse_args()

print(args)
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
ntraj= args.ntraj
ntimes= args.ntimes
two_files = args.two_files
Nsamples = args.Nsamples
test_size = args.test_size
witheld = args.witheld
seed = args.seed
subfold = args.subfold
test_type = args.test_type
data_size = args.data_size
debug = args.debug



if verbose:
    print('reading csvs....')

print(path1)
print(path2)
if not debug:
    if two_files:
        if path1[-3:] == 'csv':
            multiplexing_df1 = pd.read_csv(path1)
            multiplexing_df2 = pd.read_csv(path2)
            int_g1 = multiplexing_df1['green_int_mean'].values.reshape([ntraj,ntimes])    
            int_g2 = multiplexing_df2['green_int_mean'].values.reshape([ntraj,ntimes])    
        
            t = np.linspace(0,len(int_g1) - 1,len(int_g1))  #time vector in seconds
            labels = np.ones(int_g1.shape[0]*2)
            labels[:int_g1.shape[0]] = 0
            int_g = np.vstack((int_g1,int_g2))  #merge the files and then let it sort
            labels = labels
            
            int_g, labels = mc.even_shuffle_sample(int_g, labels, samples=[int(Nsamples/2), int(Nsamples/2)], seed=seed)
        
        if path1[-3:] == 'npy':
            int_g1 = np.load(path1)   
            int_g2 = np.load(path2)   
            t = np.linspace(0,len(int_g1) - 1,len(int_g1))  #time vector in seconds
            t = np.linspace(0,len(int_g1) - 1,len(int_g1))  #time vector in seconds
            labels = np.ones(int_g1.shape[0] + int_g2.shape[0])
            labels[:int_g1.shape[0]] = 0
            int_g = np.vstack((int_g1,int_g2))  #merge the files and then let it sort
            labels = labels
            
            int_g, labels = mc.even_shuffle_sample(int_g, labels, samples=[len(int_g1), len(int_g2)], seed=seed)
        
    
    else:    
        multiplexing_df = pd.read_csv(path1)
        int_g = multiplexing_df['green_int_mean'].values.reshape([ntraj,ntimes])
        labels = multiplexing_df['Classification'].values.reshape([ntraj,ntimes])[:,0]
        
        int_g, labels = mc.even_shuffle_sample(int_g, labels, samples=[int(Nsamples/2), int(Nsamples/2)], seed=seed)

else:
    int_g = np.random.randint(0,3000,size=(ntraj+ntraj,ntimes))
    labels = np.ones((int_g.shape[0]))
    labels[:int(int_g.shape[0]/2)] = 0
    t = np.linspace(0,len(int_g) - 1,len(int_g))  #time vector in seconds    



# Slice the data
print(Fr)
print(Nframes)
print(int_g.shape)
int_g = mc.slice_arr(int_g, Fr, Nframes)

print(int_g.shape)
print(labels.shape)
if verbose:
    print('processing data....')
print('***************************')
print('Data size:')
print(data_size)
print('***************************')

print(labels[:25])
#X_train, X_test, y_train, y_test, X_witheld, y_witheld, Acc_train, Acc_test, Acc_witheld = mc.process_data(int_g, labels, norm='train', seed=seed, witheld = witheld, test_size = test_size, include_acc = True )
X_train, X_test, y_train, y_test, X_witheld, y_witheld, Acc_train, Acc_test, Acc_witheld = mc.process_data(np.vstack([int_g[subfold*data_size:(subfold+1)*data_size] , int_g[-1000:]])  , np.hstack([labels[subfold*data_size:(subfold+1)*data_size], labels[-1000:]] ), norm='train', seed=seed, witheld = 1000, test_size = test_size, include_acc = True, shuffle=False )
        

if verbose:
    print('----Data Shapes-----')
    print('training: ')
    print(X_train.shape)
    print('test: ')
    if X_test != None:
        print(X_test.shape)
    else:
        print('None')
    print('witheld: ')
    print(X_witheld.shape)
    print(y_witheld.shape)
    print(np.sum(y_witheld))
    print('--------------------')

########## Define model here ##############


if test_type == 'freq':

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
    
    X_TRAIN = np.concatenate((X_train, Acc_train),axis=1)
    X_WITHELD = np.concatenate((X_witheld, Acc_witheld),axis=1)



if test_type == 'no_freq':
    def create_model(input_size_1, input_size_2, N_neurons, kernel_size, filters, lr):
    
        model = Sequential()
        model.add(Conv1D(filters=filters, kernel_size=kernel_size, kernel_regularizer=l1_l2(l1=1e-5), padding="same", input_shape=(X_train.shape[1], 1)))
        model.add(LeakyReLU(alpha=0.3))
        model.add(AveragePooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(N_neurons, kernel_regularizer=l1_l2(l1=1e-5)))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dense(1, activation='sigmoid'))
        optmizer = keras.optimizers.Adam(lr=lr)
        model.compile(loss='binary_crossentropy', optimizer=optmizer, metrics=['accuracy'])
    
        return model

    X_TRAIN = X_train
    X_WITHELD = X_witheld
    

if test_type == 'acc_only':
    
    def create_model(input_size_1, input_size_2, N_neurons, kernel_size, filters, lr):
    
        model = Sequential()
        model.add(Conv1D(filters=filters, kernel_size=kernel_size, kernel_regularizer=l1_l2(l1=1e-5), padding="same", input_shape=(X_train.shape[1], 1)))
        model.add(LeakyReLU(alpha=0.3))
        model.add(AveragePooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(N_neurons, kernel_regularizer=l1_l2(l1=1e-5)))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dense(1, activation='sigmoid'))
        optimizer = keras.optimizers.Adam(lr=lr)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
        return model

    X_TRAIN = Acc_train
    X_WITHELD = Acc_witheld    
    
if test_type == 'zero_mean':
    

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
    
    
    X_train, X_test, y_train, y_test, X_witheld, y_witheld, Acc_train, Acc_test, Acc_witheld = mc.process_data(int_g, labels, use_norm=False, norm='train', seed=seed, witheld = witheld, test_size = test_size, include_acc = True )

    X_train = X_train[:,:,0] - np.mean(X_train[:,:,0], axis=1)[:,np.newaxis]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    X_witheld = X_witheld[:,:,0] - np.mean(X_witheld[:,:,0], axis=1)[:,np.newaxis]
    X_witheld = X_witheld.reshape(X_witheld.shape[0], X_witheld.shape[1], 1)
    
    X_TRAIN = np.concatenate((X_train, Acc_train),axis=1)
    X_WITHELD = np.concatenate((X_witheld, Acc_witheld),axis=1)



if test_type == 'standardize':
    
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
    
    X_train, X_test, y_train, y_test, X_witheld, y_witheld, Acc_train, Acc_test, Acc_witheld = mc.process_data(int_g, labels, use_norm=False, norm='train', seed=seed, witheld = witheld, test_size = test_size, include_acc = True )

    X_train = (X_train[:,:,0] - np.mean(X_train[:,:,0], axis=1)[:,np.newaxis]) / np.std(X_train[:,:,0],axis=1)[:,np.newaxis]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    X_witheld = ( X_witheld[:,:,0] - np.mean(X_witheld[:,:,0], axis=1)[:,np.newaxis]) / np.std(X_witheld[:,:,0],axis=1)[:,np.newaxis]
    X_witheld = X_witheld.reshape(X_witheld.shape[0], X_witheld.shape[1], 1)
    
    
    X_TRAIN = np.concatenate((X_train, Acc_train),axis=1)
    X_WITHELD = np.concatenate((X_witheld, Acc_witheld),axis=1)
    


print(X_TRAIN.shape)
print(X_WITHELD.shape)
print(y_train.shape)
print(y_witheld.shape)

########## Train model here ##############

if retrain:
    
    if verbose:
        print('training....')
    if not debug:
        seed = 7
        np.random.seed(seed)
         
        
        model_CV = KerasClassifier(build_fn=create_model, verbose=0)
        
        filters = [16, 32, 64]
        kernel_size = [3, 5, 7]
        batches = [16, 32, 64]
        epochs = [50, 100]
        lrs = [.001]
        neurons = [200]
        inputs_1 = [X_train.shape[1]]
        inputs_2 = [Acc_train.shape[1]]
        
        distributions = dict(input_size_1 = inputs_1, input_size_2 = inputs_2, kernel_size = kernel_size, filters = filters, epochs= epochs, batch_size= batches, lr=lrs, N_neurons = neurons)
        random = RandomizedSearchCV(model_CV, distributions, n_iter= 2, verbose= 0, n_jobs= 1, cv=3)
        random_result = random.fit(X_TRAIN, y_train)
        if verbose:
            print(random_result.best_score_)
            print(random_result.best_params_)
        best_model = random_result.best_estimator_.model
        best_params = random_result.best_params_
        best_kernel = best_params['kernel_size']
        best_filter = best_params['filters']
        
            
        clf = random_result.best_estimator_
        if verbose:
            print('Test accuracy: %.3f' % clf.score(X_WITHELD, y_witheld))
        #print(y_witheld.shape)
        #print(np.sum(y_witheld))
        acc = np.sum(np.abs((best_model.predict(X_WITHELD) < .5) - y_witheld))/len(y_witheld)
        #print(y_witheld[:15])
        #acc = clf.score(X_WITHELD, y_witheld)
        
        if save_model:
            model_path = os.path.join('.', save_dir,model_name + '_'  + str(best_filter) + '_' + str(best_kernel) + '_' + '_'+ str(ind1) + '_' + str(ind2) + '.h5')
            best_model.save(model_path)
    else:
        acc = .5
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
    acc_mat = np.zeros([36,5])
    
else:    
    acc_mat = np.load(acc_path)
    
acc_mat[ind1,ind2] = acc
np.save(acc_path, acc_mat)
if verbose:
    print('Done.')