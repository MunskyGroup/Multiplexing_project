# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 11:09:14 2021

@author: willi
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
################

#Machine learning imports for CNNs:
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split

if tf.__version__[:1] == '1.':
  Sequential = keras.layers.Sequential
else:
  Sequential = keras.models.Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D, InputLayer
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.regularizers import l2
#import keras.layers.experimental.preprocessing as preprocessing

from sklearn.metrics import confusion_matrix

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

from keras.models import load_model

class CNN:
    def __init__(self, kernel_size, filters, input_shape, sess):
        self.kernel_size = kernel_size
        self.filters = filters
        self.sess = sess
        self.input_shape = input_shape
        self.model = self.create_model(kernel_size, filters)
        pass

    def create_model(self,kernel_size, filters):
    
        model = Sequential()
        model.add(Conv1D(filters=filters, kernel_size=kernel_size, kernel_regularizer=l1_l2(l1=1e-5), padding="same", input_shape=(self.input_shape, 1)))
        model.add(LeakyReLU(alpha=0.3))
        model.add(AveragePooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(200, kernel_regularizer=l1_l2(l1=1e-5)))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model


class convAE:
    '''
    Convolutional Autoencoder to GMM clustering (unsupervised)
    
    for these to work you need frame counts that are a multiple of 4
    '''
    def __init__(self, DAE, gmm, sess):
        self.DAE = DAE
        self.gmm = gmm
        self.sess = sess

    def encode(self,x,center_layer=5):
        if len(x.shape) < 4:
            x = np.expand_dims(x,axis=-1)
        if x.dtype != np.float32:
            x = x.astype(np.float32)
        y = self.DAE.layers[0](x)
        for i in range(1,center_layer+1):
            y = self.DAE.layers[i](y)
        return y

    def predict(self,x):
        if len(x.shape) < 4:
            x = np.expand_dims(x,axis=-1)
        if x.dtype != np.float32:
            x = x.astype(np.float32)
        latents = self.sess.run(self.encode(x))
        return self.gmm.predict(latents)
    
    def visualize_latent_boundary(self, extent = [-2,-2,2,2]):
        xy,label = self.gmm.sample(1000)
        plt.scatter(*xy.T, c=cm.viridis(label*.5),alpha=1,s=2)
        
        x = np.linspace(np.min(xy[:,0])-.2, np.max(xy[:,0])+.2,25)
        y = np.linspace(np.min(xy[:,1])-.2, np.max(xy[:,1])+.2,25)
        X, Y = np.meshgrid(x, y)
        XX = np.array([X.ravel(), Y.ravel()]).T
        Z = -self.gmm.score_samples(XX)
        Z = Z.reshape(X.shape)
        
        CS = plt.contour(
            X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 3, 10)
        )
        CB = plt.colorbar(CS, shrink=0.8, extend="both")
        #plt.scatter(X_train[:, 0], X_train[:, 1], 0.8)
        
        plt.scatter(*self.gmm.means_.T,marker='x',color='r')
        
        
        plt.title('GMM Latent clustering')
        plt.xlabel('Latent 1')
        plt.ylabel('Latent 2')
        plt.axis("tight")
        plt.show()
        
        
    def cluster_purity(self, N=10000):
        xy,label = self.gmm.sample(N)
        c1 = self.gmm.predict(xy[label == 0])
        c2 = self.gmm.predict(xy[label == 1])
        return (np.sum(c1 == 0) + np.sum(c2 == 1) )/ N
        
    def self_centroid_distance(self, N =10000):
        xy2,label = self.gmm.sample(N)
        centers = self.gmm.means_
        c1_dists = np.sqrt(np.sum(np.square(xy2 - centers[0,:]), axis=1))
        c2_dists = np.sqrt(np.sum(np.square(xy2 - centers[1,:]), axis=1))
        center_dist = np.sum(np.min(np.array([c1_dists,c2_dists]),axis=0))/ len(xy2)       
        return center_dist
        
    def check_centroid_distance(self, samples):
        centers = self.gmm.means_
        
        c1_dists = np.sqrt(np.sum(np.square(samples - centers[0,:]), axis=1))
        c2_dists = np.sqrt(np.sum(np.square(samples - centers[1,:]), axis=1))
        center_dist = np.sum(np.min(np.array([c1_dists,c2_dists]),axis=0))/ len(samples)
        return center_dist
    
    def centroid_metric(self,samples):
        self_center_dist = self.self_centroid_distance()
        sample_dists = self.check_centroid_distance(samples)
        return sample_dists/self_center_dist




    




class multiplexing_data_processing:
    def __init__(self):
        pass

    def slice_arr(array, FR, Nframes,axis=1):
        total_time = FR*Nframes
        if total_time > 3000:
            print('WARNING: desired slicing regime is not possible, making as many frames as possible')
            return array[:,::FR]
        return array[:,::FR][:,:Nframes]
    
    def load_data_fast(self, dataframe_simulated_cell, total_traj=5000, spot_nums=50, time=3000):
        int_g = dataframe_simulated_cell['green_int_mean'].values.reshape([total_traj,time])
        labels = dataframe_simulated_cell['Classification'].values.reshape([total_traj,time])[:,0]
        return int_g, labels
    
    def df_to_array(self, dataframe_simulated_cell, total_traj=5000, spot_nums=50):
  
        labels2 = np.zeros([total_traj,1])
        label = 1
        for i in range(total_traj):
            if i%spot_nums == 0:
                label = (label+1)%2
            labels2[i] = label
          
        # get the total number of particles in all cells
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
        return I_g, I_g_std, labels, x_loc,y_loc, labels2 #I_r, #I_r_std,    #return everything backout
  


    #individual zero mean unit variance #insensitive to uneven numbers of labels
    @staticmethod
    def minmaxscale_signal(signal, axis = 0):
        scaler = MinMaxScaler()
        signal = scaler.fit_transform(signal)
        return signal
    
        
    #individual zero mean unit variance #insensitive to uneven numbers of labels
    @staticmethod
    def standardize_signal(signal, axis = 0):
        return (signal - np.mean(signal, axis=axis)) / np.std(signal,axis=axis)
    
    #sets individual 0-1
    @staticmethod
    def normalize_signal(signal, axis = 0):
        return (signal -  np.min(signal,axis=axis) ) / ( np.max(signal,axis=axis)  - np.min(signal,axis=axis) )
    
    #global mean = 0, unit variance
    @staticmethod
    def standardize_signal_global(signal, axis = 0):
        return (signal - np.mean(signal)) / np.std(signal)
    
    #sets global 0-1
    @staticmethod
    def normalize_signal_global(signal, axis = 0):
        return (signal -  np.min(signal) ) / ( np.max(signal)  - np.min(signal) )
    
    @staticmethod
    def displacement_signal(signal,axis=0):
        new_signal = np.zeros(signal.shape)
        new_signal[:,1:] = signal[:,:-1] - signal[:,1:]
        return new_signal
    
    
    #sets global to 0 to 95th percentile  #sensitive to uneven numbers of labels
    @staticmethod
    def minmax95_signal_global(signal, axis = 0):
        max_95 = np.quantile(signal, .95)
        sig = (signal -  np.min(signal) ) / ( max_95  - np.min(signal) )
        return np.minimum(sig, 1.5)
    
    @staticmethod
    def convert_labels_to_onehot(labels):
        '''
        converts labels in the format 1xN, [0,0,1,2,3,...] to onehot encoding,
        ie: N_classes x N,  [[1,0,0,0],[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]     
        '''
        onehotlabels = np.zeros((labels.shape[0],len(np.unique(labels))))
        for i in range(len(onehotlabels)):
            onehotlabels[i,labels[i]] = 1
        return onehotlabels
    
    @staticmethod
    def build_cnn_model_1D(data_shape, maxpooling,
                    dropout, 
                    activation_function,
                    loss_function,
                    filter_sizes,
                    kernel_sizes,
                    strides,
                    pool_sizes,
                    dense_neurons,
                    dropout_percent,
                    learning_rate,
                    l2_weight):



        names = ['Intensity','Xsig','Ysig']
        w,h = data_shape[1],1
        print(w,h)
        total_classes = 2
        input_list = lambda strinput: [int(x) for x in strinput.split(',')]
        total_dense_layers = len(input_list(dense_neurons))
        
        signal_inputs = keras.Input(shape = (w,h), name=  'input_X')


        cnn_outs = []
        all_inputs = []
        #build sub_cnn's
        for signal_num in range(h):
            sub_cnn_input = keras.layers.Lambda(lambda x: tf.expand_dims(x[:,:,signal_num],axis=-1) )(signal_inputs)
            
              #signal_inputs = keras.Input(shape = (w,1), name=  names[signal_num])
              #all_inputs.append(signal_inputs)
            
            input_list = lambda strinput: [int(x) for x in strinput.split(',')]
            total_convolutional_layers = len(input_list(pool_sizes))
              
            cnn_sub_layers = [ ]
    ## convolutional layers
            for i in range(total_convolutional_layers):
                pool_size = input_list(pool_sizes)[i]
                stride_size =  input_list(strides)[i]
                if i == 0:
                    x = keras.layers.Conv1D(input_list(filter_sizes)[i], (input_list(kernel_sizes)[i]  ), (stride_size), activation=activation_function, kernel_regularizer=l2(l2_weight))(sub_cnn_input)
                else:
                    x = keras.layers.Conv1D(input_list(filter_sizes)[i], (input_list(kernel_sizes)[i]  ), (stride_size), activation=activation_function, kernel_regularizer=l2(l2_weight) )(x)
                cnn_sub_layers = cnn_sub_layers + [x,]
                if maxpooling:
                    x = keras.layers.MaxPooling1D(pool_size=(pool_size))(x)
                    cnn_sub_layers = cnn_sub_layers + [x,]
                
            
            cnn_outs.append(keras.layers.Flatten()(x))
        print(cnn_outs)
        if h > 1:
            dense_inputs = keras.layers.concatenate(cnn_outs)
        else:
            dense_inputs = cnn_outs

        for i in range(total_dense_layers):
            if i == 0:
                x = keras.layers.Dense(input_list(dense_neurons)[i], activation = activation_function)(dense_inputs)
            else:
                x = keras.layers.Dense(input_list(dense_neurons)[i], activation = activation_function)(x)
            if dropout:
                x = keras.layers.Dropout(0.5)(x)

      
          #final classification layer
        if total_classes > 2:
            final_activation = 'softmax'
        else:
            final_activation = 'softmax'

        final_layer = keras.layers.Dense(total_classes, activation = final_activation)(x)
        
        cnn_model2 = keras.Model(inputs = signal_inputs, outputs= final_layer)
        
        cnn_model2.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
              metrics=['accuracy'])

        return cnn_model2



'''
example_dataset = 'D:/multiplexing_ML/construct_length_dataset/construct_length_dataset/construct_lengths_BBS5_CAMK2B.csv'

mp = multiplexing_data_processing()
example_df = pd.read_csv(example_dataset)
I_g, _, labels, _, _,labels2 = mp.df_to_array(example_df)

print(labels.shape)
print(labels2.shape)
int_g_S = mp.standardize_signal(I_g.T).T

print(np.array([labels]))
labels_onehot = mp.convert_labels_to_onehot(np.array([labels]).T.astype(int))

cnn_model2 = mp.build_cnn_model_1D(data_shape=int_g_S.shape, maxpooling= True,
                              dropout = False,
                              dropout_percent = .03,
                              activation_function = 'relu',
                              loss_function = 'categorical_crossentropy',
                              filter_sizes = "25,15,5",
                              kernel_sizes = "4,3,2",
                              strides = "1,1,1",
                              dense_neurons = "200,100,10",
                              pool_sizes = "3,2,1",
                              learning_rate=.0003,
                              l2_weight=.02 )

x_train, x_val, label_train, label_val = train_test_split(np.expand_dims(int_g_S, axis=-1),
                                                            labels_onehot,
                                                            test_size=.2,
                                                            random_state=42)

history = cnn_model2.fit(x = x_train, y= label_train,  validation_data=(x_val,label_val),epochs=40, verbose=0 )

plt.plot(history.history['val_accuracy'])
plt.plot(history.history['accuracy'])
plt.legend(['val_accuracy','accuracy'])

'''