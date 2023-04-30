# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 23:20:02 2023

@author: willi
"""



##############################################################################
#     GENERATE THE PLOTS FOR THE MULTIPLEXING FIGURE
##############################################################################

## IMPORTS
import matplotlib as mpl
from matplotlib.gridspec import GridSpec


from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import pandas   

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import os

import skimage.io as io

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
sess = tf.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)



import pandas as pd

from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix

from tensorflow.keras import layers, models, optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, AveragePooling1D, Conv1D, LeakyReLU, Lambda
from tensorflow.keras.regularizers import l2, l1, l1_l2
from tensorflow.keras.layers import BatchNormalization
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


import os
cwd = os.getcwd()
os.chdir('../../')
import apply_style  as aps#apply custom matplotlib style
import mc_core as multiplexing_core
os.chdir(cwd)

aps.apply_style()

def train_model(files, color, name, model_file=None, retrain=True, include_noise = True):

  mc = multiplexing_core()
  verbose = 1
  ntraj = 2500
  ntimes = 3000
  Nsamples = 5000
  Fr = 5
  Nframes = 64
  seed = 42
  test_type = 'freq'

  witheld = 1500
  test_size = 0 
  file_list = files.replace(' ','').split(',')
  n_files = len(file_list)


  if verbose:
      print('reading csvs....')


  if color == 'green':
      color = 'green_int_mean'
      inverse = 'blue_int_mean'
  if color == 'blue':
      color = 'blue_int_mean'
      inverse = 'green_int_mean'
  if color == 'red':
      color = 'red_int_mean'
  
  
  if include_noise:
    int_g = np.zeros([ntraj*(n_files+1), ntimes])
    labels = np.zeros(ntraj*(n_files+1))
  else:
    int_g = np.zeros([ntraj*(n_files), ntimes])
    labels = np.zeros(ntraj*(n_files))
  i = 0
  for f in file_list:
      multiplexing_df1 = pd.read_csv(f)
      int_g1 = multiplexing_df1[color].values.reshape([ntraj,ntimes])  
      int_g[i*ntraj: (i+1)*ntraj ,:] = int_g1
      labels[i*ntraj:(i+1)*ntraj] = i
      i+=1
  if include_noise:
      multiplexing_df1 = pd.read_csv(file_list[0])
      int_g1 = multiplexing_df1[inverse].values.reshape([ntraj,ntimes])  
      int_g[i*ntraj: (i+1)*ntraj ,:] = int_g1
      labels[i*ntraj:(i+1)*ntraj] = i
      i+=1      
      n_files = n_files+1
      
  int_g, labels = mc.even_shuffle_sample_n(int_g, labels, samples=[int(Nsamples/2)]*n_files, seed=seed)

  t = np.linspace(0,len(int_g1) - 1,len(int_g1))  #time vector in seconds


  # Slice the data
  print(Fr)
  print(Nframes)
  int_g = mc.slice_arr(int_g, Fr, Nframes)

  print(int_g.shape)
  print(labels.shape)
  if verbose:
      print('processing data....')

  X_train, X_test, y_train, y_test, X_witheld, y_witheld, Acc_train, Acc_test, Acc_witheld = mc.process_data_n(int_g, labels, norm='train', seed=seed, witheld = witheld, test_size = test_size, include_acc = True )


  y_train = sess.run(tf.one_hot(y_train, n_files))
  y_train = np.swapaxes(y_train, 1,2)[:,:,0]

  if witheld > 0:
      y_witheld = sess.run(tf.one_hot(y_witheld, n_files))
      y_witheld = np.swapaxes(y_witheld, 1,2)[:,:,0]
  if test_size > 0:
      y_test = tf.one_hot(y_test, n_files).cpu()

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
          #temp_out = Lambda(lambda x: x / temp)(dense_output)
          model_out = Dense(n_files,activation='softmax')(dense_output)
      
          optmizer = keras.optimizers.Adam(lr=lr)
          model = keras.Model(inputs=[combi_input], outputs=model_out)
          model.compile(loss='categorical_crossentropy', optimizer=optmizer, metrics=['accuracy'],)

          return model
      
      X_TRAIN = np.concatenate((X_train, Acc_train),axis=1)
      X_WITHELD = np.concatenate((X_witheld, Acc_witheld),axis=1)



  print(np.unique(labels))
  print(X_TRAIN.shape)
  print(X_WITHELD.shape)
  print(y_train.shape)
  print(y_witheld.shape)

  ########## Train model here ##############
  
  if retrain:
      if verbose:
          print('training....')
      seed = 7
      np.random.seed(seed)
      
      
      model_CV = KerasClassifier(build_fn=create_model, verbose=0)
      
      filters = [16, 32, 64]
      kernel_size = [3, 5, 7]
      batches = [16, 32, 64]
      epochs = [50, 100]
      lrs = [.001]
      neurons = [200]
      #temp = [0.7,1,1.3]
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
      acc = clf.score(X_WITHELD, y_witheld)
      model_path = os.path.join('.', '.', 'mp_'+ name + '_'  + str(best_filter) + '_' + str(best_kernel)  + '.h5')
      best_model.save(model_path)    

      
      #if save_model:
          #model_path = os.path.join('.', save_dir,model_name + '_'  + str(best_filter) + '_' + str(best_kernel) + '_' + '_'+ model_sub_name + '.h5')
          #best_model.save(model_path)

  else:
    #format: model_i_j_kernelsize_filters
    filters = int(model_file.split('_')[-2])
    kernel_size = int( model_file.split('_')[-1] .split('.')[0] )

    inputs_1 = [X_train.shape[1]]
    inputs_2 = [Acc_train.shape[1]]
    best_model = create_model(X_train.shape[1], Acc_train.shape[1], 200, kernel_size, filters, .001)
    best_model.load_weights(model_file)

    y_pred = best_model.predict(X_WITHELD)
    y_pred = np.argmax(y_pred,axis=1).astype(int)
    y_pred_onehot = np.zeros((y_pred.size, y_pred.max() + 1))
    y_pred_onehot[np.arange(y_pred.size), y_pred] = 1


    print( y_witheld.shape)
    print(y_pred_onehot.shape)
    acc = 1-np.sum(np.abs(y_pred_onehot- y_witheld))/len(y_pred_onehot)
    if verbose:
        print('acc: %f '%acc)


  return best_model, acc, int_g


retrain = False
f = '../../datasets/construct_lengths_RRAGC_RRAGC.csv,../../datasets/construct_lengths_LONRF2_LONRF2.csv,../../datasets/construct_lengths_MAP3K6_MAP3K6.csv,../../datasets/construct_lengths_DOCK8_DOCK8.csv'
color='green'

if retrain:
    green_model, green_acc, int_g_g = train_model(f,'green','green_w_noise_and_temp', retrain=1, model_file=None)
else:
    green_model, green_acc, int_g_g = train_model(f,'green','green', retrain=0, model_file='./mp_green_w_noise_16_3.h5')


f = '../../datasets/construct_lengths_ORC2_ORC2.csv,../../datasets/construct_lengths_TRIM33_TRIM33.csv,../../datasets/construct_lengths_PHIP_PHIP.csv'

if retrain:
    blue_model, blue_acc, int_b_g = train_model(f,'green','blue_w_noise_and_temp', retrain=1, model_file=None)
else:
    blue_model, blue_acc, int_b_g = train_model(f,'green','blue', retrain=0, model_file='./mp_blue_w_noise_16_3.h5')


######## apply the classifier and get confidences

data_file_50 = './multiplexing_7__cell0_50'
simulated_cell_dataframe_50 = pandas.read_csv(data_file_50)
shape = (70*50,1000)

int_g_cell_all = simulated_cell_dataframe_50['green_int_mean'].values.reshape(shape)
int_b_cell_all = simulated_cell_dataframe_50['blue_int_mean'].values.reshape(shape)
cells = simulated_cell_dataframe_50['cell_number'].values.reshape(shape)

g_labels = [0,]*10 + [1,]*10 + [2,]*10 + [3,]*10 
b_labels = [0,]*10 + [1,]*10 + [2,]*10 

g_labels = [item for sublist in [g_labels]*50 for item in sublist]
b_labels = [item for sublist in [b_labels]*50 for item in sublist]

g_noise_labels = [4,]*500
b_noise_labels = [3,]*500

g_labels = g_labels + g_noise_labels
b_labels = b_labels + b_noise_labels

for i in range(50):
  if i ==0:
    b_50 = int_b_cell_all[70*i:70*i+70][:30]
    g_50 = int_g_cell_all[70*i:70*i+70][30:]
  else:
    b_50 = np.vstack((b_50, int_b_cell_all[70*i:70*i+70][:30] ))
    g_50 = np.vstack((g_50, int_g_cell_all[70*i:70*i+70][30:] ))

for i in range(50):
  b_50 = np.vstack((b_50, int_g_cell_all[70*i:70*i+70][:10] ))
  g_50 = np.vstack((g_50, int_b_cell_all[70*i:70*i+70][30:40] ))

def slice_arr(array, FR, Nframes,axis=1):
    total_time = FR*Nframes
    if total_time > 3000:
        print('WARNING: desired slicing regime is not possible, making as many frames as possible')
        return array[:,::FR]
    return array[:,::FR][:,:Nframes]


g_50 = slice_arr(g_50,5,64)
b_50 = slice_arr(b_50,5,64)
mc = multiplexing_core()
labels = [0,]*10 + [1,]*10 + [2,]*10 + [3,]*10 
data, data_acc_b_50 = mc.process_test_data(b_50, include_acc=True )

b_50_64 = np.concatenate((data, data_acc_b_50),axis=1)
b_50_64_labels = labels

#int_g_64_5_array = np.array(int_g_64_5_labels).astype(int)
#int_g_64_5_labels = np.zeros((int_g_64_5_array.size, int_g_64_5_array.max()+1))
#int_g_64_5_labels[np.arange(int_g_64_5_array.size), int_g_64_5_array] = 1


labels = [0,]*10 + [1,]*10 + [2,]*10
data, data_acc_g_50 = mc.process_test_data(g_50, include_acc=True )

g_50_64 = np.concatenate((data, data_acc_g_50),axis=1)
g_50_64_labels = labels

#int_b_64_5_array = np.array(int_b_64_5_labels).astype(int)
#int_b_64_5_labels = np.zeros((int_b_64_5_array.size, int_b_64_5_array.max()+1))
#int_b_64_5_labels[np.arange(int_b_64_5_array.size), int_b_64_5_array] = 1


def np_onehot(array):
  b = np.zeros((array.size, array.max() + 1))
  b[np.arange(array.size), array] = 1
  return b

scaler = MinMaxScaler()
scaler.fit(int_g_g)
int_g_50_64_transform = scaler.transform(g_50)  
int_g_64_50 = np.concatenate((np.expand_dims(int_g_50_64_transform,-1), data_acc_g_50),axis=1)


scaler = MinMaxScaler()
scaler.fit(int_b_g)
int_b_50_64_transform = scaler.transform(b_50)  
int_b_64_50 = np.concatenate((np.expand_dims(int_b_50_64_transform,-1), data_acc_b_50),axis=1)


g_labels_predicted_50 = np.argmax((green_model.predict(int_g_64_50) > .5).astype(int),axis=1)
b_labels_predicted_50 = np.argmax((blue_model.predict(int_b_64_50) > .5).astype(int),axis=1)

g_onehot_predicted_50 = green_model.predict(int_g_64_50)
b_onehot_predicted_50 = blue_model.predict(int_b_64_50)

g_onehot_true = np_onehot(np.array(g_labels))
b_onehot_true = np_onehot(np.array(b_labels))


g_predict_50 = green_model.predict(int_g_64_50)
b_predict_50 = blue_model.predict(int_b_64_50)


g_confidence_50 = []
b_confidence_50 = []
g_right_50 = []
b_right_50 = []
i = 0
for label in g_labels_predicted_50:
  g_confidence_50.append(g_predict_50[i,label])
  g_right_50.append(g_labels_predicted_50[i] ==   g_labels[i])
  i+=1
i = 0
for label in b_labels_predicted_50:
  b_confidence_50.append(b_predict_50[i,label])
  b_right_50.append(b_labels_predicted_50[i] ==   b_labels[i])
  i+=1

g_true_labels_50 = np.array(g_labels)
b_true_labels_50 = np.array(b_labels)
print('Green Channel Accuracy:')
print(accuracy_score(g_true_labels_50,g_labels_predicted_50))
print('G incorrect conf:')
print(np.mean(np.array(g_confidence_50)[np.where(g_true_labels_50 != g_labels_predicted_50)]))
print(np.std(np.array(g_confidence_50)[np.where(g_true_labels_50 != g_labels_predicted_50)]))
print('G correct conf:')
print(np.mean(np.array(g_confidence_50)[np.where(g_true_labels_50 == g_labels_predicted_50)]))
print(np.std(np.array(g_confidence_50)[np.where(g_true_labels_50 == g_labels_predicted_50)]))

print('______________________')
print('Blue Channel Accuracy:')
print(accuracy_score(b_true_labels_50,b_labels_predicted_50))
print('B incorrect conf:')
print(np.mean(np.array(b_confidence_50)[np.where(b_true_labels_50 != b_labels_predicted_50)]))
print(np.std(np.array(b_confidence_50)[np.where(b_true_labels_50 != b_labels_predicted_50)]))
print('B correct conf:')
print(np.mean(np.array(b_confidence_50)[np.where(b_true_labels_50 == b_labels_predicted_50)]))
print(np.std(np.array(b_confidence_50)[np.where(b_true_labels_50 == b_labels_predicted_50)]))



g_50_acc = accuracy_score(g_true_labels_50,g_labels_predicted_50)

g_50_iconf = np.mean(np.array(g_confidence_50)[np.where(g_true_labels_50 != g_labels_predicted_50)])
g_50_iconf_std = np.std(np.array(g_confidence_50)[np.where(g_true_labels_50 != g_labels_predicted_50)])

g_50_conf = np.mean(np.array(g_confidence_50)[np.where(g_true_labels_50 == g_labels_predicted_50)])
g_50_conf_std =np.std(np.array(g_confidence_50)[np.where(g_true_labels_50 == g_labels_predicted_50)])


b_50_acc = accuracy_score(b_true_labels_50,b_labels_predicted_50)

b_50_iconf = np.mean(np.array(b_confidence_50)[np.where(b_true_labels_50 != b_labels_predicted_50)])
b_50_iconf_std = np.std(np.array(b_confidence_50)[np.where(b_true_labels_50 != b_labels_predicted_50)])

b_50_conf = np.mean(np.array(b_confidence_50)[np.where(b_true_labels_50 == b_labels_predicted_50)])
b_50_conf_std =np.std(np.array(b_confidence_50)[np.where(b_true_labels_50 == b_labels_predicted_50)])



fig = plt.figure(dpi=300)
ax1 = fig.add_subplot(111)


g_true_nonoise_labels_50 = g_true_labels_50[:2000]
b_true_nonoise_labels_50 = b_true_labels_50[:1500]
g_labels_predicted_50_nonoise = g_labels_predicted_50[:2000]
b_labels_predicted_50_nonoise = b_labels_predicted_50[:1500]
g_confidence_50_nonoise = g_confidence_50[:2000]
b_confidence_50_nonoise = b_confidence_50[:1500]

ys = [np.array(g_confidence_50_nonoise)[np.where(g_true_nonoise_labels_50 == g_labels_predicted_50_nonoise)],
      np.array(g_confidence_50_nonoise)[np.where(g_true_nonoise_labels_50 != g_labels_predicted_50_nonoise)],
      np.array(b_confidence_50_nonoise)[np.where(b_true_nonoise_labels_50 == b_labels_predicted_50_nonoise)],
      np.array(b_confidence_50_nonoise)[np.where(b_true_nonoise_labels_50 != b_labels_predicted_50_nonoise)],]

def add_subplot_axes(ax,rect,axisbg='w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)    
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x,y,width,height],facecolor=axisbg)  # matplotlib 2.0+

    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax

colors = ['#ef476f', '#073b4c','#06d6a0','#7400b8','#073b4c', '#118ab2',]
for i in [0,1,2,3]:
  x = np.random.normal(i, 0.04, size=len(ys[i]))+1
  ax1.plot(x, ys[i], '.', alpha=0.5, markersize=.5, color=[colors[2],colors[2], colors[1], colors[1] ][i])

  xx,yy = np.histogram(ys[i], np.linspace(.5,1,100), density=True)
  
  axsub = add_subplot_axes(ax1,[[.11,.115,.07,.847], [.33,.115,.07,.847], [.555,.115,.07,.847], [.775,.115,.07,.847]][i])
  axsub.plot(xx/xx.sum(),yy[:-1],'b-',lw=1)
  axsub.plot([0,0],[.5,1],'k-',lw=.5)
  axsub.plot([0,.1],[1,1],'k-',lw=.5)
  axsub.set_xlim([0,.1])
  #axsub.invert_yaxis()
  axsub.invert_xaxis()
  axsub.axis('off')


ax1.boxplot(np.array(g_confidence_50_nonoise)[np.where(g_true_nonoise_labels_50 == g_labels_predicted_50_nonoise)])
ax1.boxplot(np.array(g_confidence_50_nonoise)[np.where(g_true_nonoise_labels_50 != g_labels_predicted_50_nonoise)], positions=[2])

ax1.boxplot(np.array(b_confidence_50_nonoise)[np.where(b_true_nonoise_labels_50 == b_labels_predicted_50_nonoise)], positions=[3], showfliers=False)
ax1.boxplot(np.array(b_confidence_50_nonoise)[np.where(b_true_nonoise_labels_50 != b_labels_predicted_50_nonoise)], positions=[4])
ax1.plot([0,4.5],[.5,.5],color='gray',alpha=.2)
ax1.plot([0,4.5],[1,1],color='gray',alpha=.2)

ax1.set_ylim([.4,1.05])
ax1.set_xlim([0,4.5])

ax1.set_xticklabels([r'$G_{conf}$',r'$G_{conf, incorrect}$', r'$B_{conf}$',r'$B_{conf, incorrect}$'], fontsize=8)
ax1.set_ylabel('Accuracy')
ax1.text(.7,.45,'n=%s'%(int(len(ys[0])) ), fontsize=8)
ax1.text(1.8,.45,'n=%s'%(int(len(ys[1])) ), fontsize=8)
ax1.text(2.7,.45,'n=%s'%(int(len(ys[2])) ), fontsize=8)
ax1.text(3.8,.45,'n=%s'%(int(len(ys[3])) ), fontsize=8)

plt.savefig('acc.svg')


fig = plt.figure(dpi=300, figsize=(5,3))
ax1 = fig.add_subplot(111)

g_true_nonoise_labels_50 = g_true_labels_50[:2000]
b_true_nonoise_labels_50 = b_true_labels_50[:1500]
g_labels_predicted_50_nonoise = g_labels_predicted_50[:2000]
b_labels_predicted_50_nonoise = b_labels_predicted_50[:1500]
g_confidence_50_nonoise = g_confidence_50[:2000]
b_confidence_50_nonoise = b_confidence_50[:1500]


threshes = np.linspace(0,1,1000)

ys = np.array([(len(g_true_nonoise_labels_50[np.array(g_confidence_50_nonoise) > thresh]), accuracy_score(g_true_nonoise_labels_50[np.array(g_confidence_50_nonoise) > thresh],g_labels_predicted_50_nonoise[np.array(g_confidence_50_nonoise) > thresh])) for thresh in threshes])
plt.plot((2000-ys[:,0])/2000,ys[:,1], color=colors[2])

ys = np.array([(len(b_true_nonoise_labels_50[np.array(b_confidence_50_nonoise) > thresh]), accuracy_score(b_true_nonoise_labels_50[np.array(b_confidence_50_nonoise) > thresh],b_labels_predicted_50_nonoise[np.array(b_confidence_50_nonoise) > thresh])) for thresh in threshes])
plt.plot((1500-ys[:,0])/1500,ys[:,1], color=colors[1])

plt.xlabel('Fraction of spots discarded (sorted by confidence)')
plt.ylabel('Accuracy')
plt.legend(['Green','Blue'])
plt.savefig('accvconf.svg')


##############################################################################
# GREEN CHANNEL/ BLUE CHANNEL CONFUSION PLOT
##############################################################################
plt.figure(dpi=300)
cmap = plt.get_cmap('YlGn')
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

new_cmap = truncate_colormap(cmap, 0.1, 1)
my_cmap = new_cmap(np.arange(cmap.N))
my_cmap[:, -1] = .9
my_cmap = ListedColormap(my_cmap)


gtl = g_true_labels_50+1
gpl = g_labels_predicted_50+1
gtl[gtl == 5] = 0
gpl[gpl == 5] = 0

btl = b_true_labels_50+1
bpl = b_labels_predicted_50+1
btl[btl == 4] = 0
bpl[bpl == 4] = 0

g_mat = confusion_matrix(gtl, gpl)
g_mat = g_mat / g_mat.astype(np.float).sum(axis=1)

b_mat = confusion_matrix(btl, bpl)
b_mat = b_mat / b_mat.astype(np.float).sum(axis=1)
plt.matshow(g_mat, cmap =my_cmap,)
for i in range(5):
  for j in range(5):
    if i != j:
      plt.text(j,i, str(g_mat[i,j]),
                horizontalalignment='center',
                verticalalignment='center',
                size=10)   
    else:
      plt.text(j,i, str(g_mat[i,j]),
                horizontalalignment='center',
                verticalalignment='center',
                size=10, color='w')     

plt.gca().set_xticklabels(['','Noise','RRAGC', 'LONRF2','MAP3K6','DOCK8'], fontsize=8)
plt.gca().set_yticklabels(['','Noise','RRAGC', 'LONRF2','MAP3K6','DOCK8',], fontsize=8)
plt.title('Prediction'); plt.ylabel('Actual')
###########################################
plt.savefig('conf_G.svg')

cmap = plt.get_cmap('Blues')
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

new_cmap = truncate_colormap(cmap, 0.1, 1)
my_cmap = new_cmap(np.arange(cmap.N))
my_cmap[:, -1] = .9
my_cmap = ListedColormap(my_cmap)


plt.figure(dpi=300)
plt.matshow(b_mat, cmap =my_cmap,)

for i in range(4):
  for j in range(4):
    if i != j:
      plt.text(j,i, str(b_mat[i,j]),
                horizontalalignment='center',
                verticalalignment='center',
                size=10)   
    else:
      plt.text(j,i, str(b_mat[i,j]),
                horizontalalignment='center',
                verticalalignment='center',
                size=10, color='w')  
      
plt.gca().set_xticklabels(['','Noise','ORC2','TRIM33','PHIP',], fontsize=8)
plt.gca().set_yticklabels(['','Noise','ORC2','TRIM33','PHIP',], fontsize=8)
plt.title('Prediction'); plt.ylabel('Actual')
plt.savefig('conf_B.svg')



##############################################################################
# Generate the single frame classification of an example multiplexed video
##############################################################################

data_file = './multiplexing_7_cell0.csv.csv'
simulated_cell_dataframe = pandas.read_csv(data_file)

shape = (70,1000)

int_g_cell1 = simulated_cell_dataframe['green_int_mean'].values.reshape(shape)[3*10:,:]
int_b_cell1 = simulated_cell_dataframe['blue_int_mean'].values.reshape(shape)[:30,:]

int_g_bg_cell1 = simulated_cell_dataframe['blue_int_mean'].values.reshape(shape)[3*10:,:]
int_b_bg_cell1 = simulated_cell_dataframe['green_int_mean'].values.reshape(shape)[:30,:]

labels_cell1 = simulated_cell_dataframe['Classification'].values.reshape(shape)

x_cell1 = simulated_cell_dataframe['x'].values.reshape(shape) 
y_cell1 = simulated_cell_dataframe['y'].values.reshape(shape)

def slice_arr(array, FR, Nframes,axis=1):
    total_time = FR*Nframes
    if total_time > 3000:
        print('WARNING: desired slicing regime is not possible, making as many frames as possible')
        return array[:,::FR]
    return array[:,::FR][:,:Nframes]

int_g_cell1_64 = slice_arr(int_g_cell1,5,64)
scaler = MinMaxScaler()
scaler.fit(int_g_g)
int_g_cell1_64_minmax = scaler.transform(int_g_cell1_64)


int_b_cell1_64 = slice_arr(int_b_cell1,5,64)
scaler = MinMaxScaler()
scaler.fit(int_b_g)
int_b_cell1_64_minmax = scaler.fit_transform(int_b_cell1_64)


mc = multiplexing_core()
labels = [0,]*10 + [1,]*10 + [2,]*10 + [3,]*10 
data, data_acc_g = mc.process_test_data(int_g_cell1_64, include_acc=True )

int_g_64_5 = np.concatenate((data, data_acc_g),axis=1)
int_g_64_5_labels = labels

int_g_64_5_array = np.array(int_g_64_5_labels).astype(int)
int_g_64_5_labels = np.zeros((int_g_64_5_array.size, int_g_64_5_array.max()+1))
int_g_64_5_labels[np.arange(int_g_64_5_array.size), int_g_64_5_array] = 1


labels = [0,]*10 + [1,]*10 + [2,]*10
data, data_acc_b = mc.process_test_data(int_b_cell1_64, include_acc=True )

int_b_64_5 = np.concatenate((data, data_acc_b),axis=1)
int_b_64_5_labels = labels

int_b_64_5_array = np.array(int_b_64_5_labels).astype(int)
int_b_64_5_labels = np.zeros((int_b_64_5_array.size, int_b_64_5_array.max()+1))
int_b_64_5_labels[np.arange(int_b_64_5_array.size), int_b_64_5_array] = 1


scaler = MinMaxScaler()
scaler.fit(int_g_g)
int_g_cell1_64_transform = scaler.transform(int_g_cell1_64)  
int_g_64_5 = np.concatenate((np.expand_dims(int_g_cell1_64_transform,-1), data_acc_g),axis=1)


scaler = MinMaxScaler()
scaler.fit(int_b_g)
int_b_cell1_64_transform = scaler.transform(int_b_cell1_64)  
int_b_64_5 = np.concatenate((np.expand_dims(int_b_cell1_64_transform,-1), data_acc_b),axis=1)


g_labels_predicted = np.argmax((green_model.predict(int_g_64_5) > .5).astype(int),axis=1)
b_labels_predicted = np.argmax((blue_model.predict(int_b_64_5) > .5).astype(int),axis=1)

g_predict = green_model.predict(int_g_64_5)
b_predict = blue_model.predict(int_b_64_5)

g_labels_predicted = np.argmax((green_model.predict(int_g_64_5) > .5).astype(int),axis=1)
b_labels_predicted = np.argmax((blue_model.predict(int_b_64_5) > .5).astype(int),axis=1)

g_predict = green_model.predict(int_g_64_5)
b_predict = blue_model.predict(int_b_64_5)

g_confidence = []
b_confidence = []
i = 0
for label in g_labels_predicted:
  g_confidence.append(g_predict[i,label])
  i+=1
i = 0
for label in b_labels_predicted:
  b_confidence.append(b_predict[i,label])
  i+=1
  
predicted_labels_blue = b_labels_predicted
predicted_labels_green = g_labels_predicted


frame = 0 #@param {type:"integer"}


def inside_subplot(ax,rect,axisbg='w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)    
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  
    subax = fig.add_axes([x,y,width,height],facecolor='white') 
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax


frame = 30
fig = plt.figure( dpi=300, constrained_layout=True)
gs = GridSpec(8,4,figure=fig )

gp_ax =  fig.add_subplot(gs[ 0:4, 1])
gl_ax =  fig.add_subplot(gs[ 0:4, 2])
gc_ax =  fig.add_subplot(gs[0:4, 3])

bp_ax =  fig.add_subplot(gs[ 4: , 1])
bl_ax =  fig.add_subplot(gs[ 4: , 2])
bc_ax =  fig.add_subplot(gs[ 4: , 3])

rragc_ax = fig.add_subplot(gs[0,0])
lonrf2_ax = fig.add_subplot(gs[1,0])
map3k6_ax = fig.add_subplot(gs[2,0])
dock8_ax = fig.add_subplot(gs[3,0])

orc2_ax = fig.add_subplot(gs[4,0])
trim33_ax = fig.add_subplot(gs[5,0])
phip_ax = fig.add_subplot(gs[6,0])
wrong_ax = fig.add_subplot(gs[7,0])

val_video = io.imread('./multiplexing_7_1.tif')
val_video.shape

def quantile_norm(movie, q ):
   max_val = np.quantile(movie, q)
   min_val = np.quantile(movie, .005)
   norm_movie = (movie - min_val)/(max_val - min_val)
   norm_movie[norm_movie > 1] = 1
   norm_movie[norm_movie < 0] = 0
   return norm_movie

maxquant = .99
green_vid = quantile_norm(val_video[:,:,:,1], maxquant)
blue_vid = quantile_norm(val_video[:,:,:,2], maxquant)


conf_cmap  = (mpl.colors.ListedColormap(['r','r','r','r','r','#E7B04B', '#F3E377', '#79E888', '#31DAD4','#1789FC']))
predicted_labels_cell1 = predicted_labels_green.tolist()
vid_frame = green_vid[frame,:,:]
gp_ax.imshow(vid_frame,cmap=cm.Greens_r)
c = ['#12153D', '#237167', '#E2D6C9', '#BD571F', '#35b779', '#90d743', '#fde725']

labels_cell1 = [[0]*10,[1]*10,[2]*10,[3]*10,]
labels_cell1 = [item for sublist in labels_cell1 for item in sublist]
colors = [c[int(predicted_labels_cell1[i])] for i in range(0,len(predicted_labels_cell1))]
blank_colors = [[0,0,0,0],]*len(colors)
for i in range(len(colors)):
  if int(predicted_labels_cell1[i]) != int(labels_cell1[i]):
    colors[i] = [0,0,0,0]

for i in range(len(colors)):
  if int(predicted_labels_cell1[i]) != int(labels_cell1[i]):
    blank_colors[i] = 'r'

  #if np.max(predicted_labels_cell1_proba[i,:]) < .75:
    #colors[i] = 'm'


gp_ax.scatter(x_cell1[30:,frame],y_cell1[30:,frame],facecolors='none', edgecolors=colors, s=15)
gp_ax.scatter(x_cell1[30:,frame],y_cell1[30:,frame],facecolors=blank_colors, edgecolors=blank_colors, s=15,  marker='x')
gp_ax.set_title('Predicted labels')

gl_ax.imshow(vid_frame,cmap=cm.Greens_r)

colors = [c[int(labels_cell1[i])] for i in range(0,len(labels_cell1))]


gl_ax.scatter(x_cell1[30:,frame],y_cell1[30:,frame],facecolors='none', edgecolors=colors, s=15)
gl_ax.set_title('True labels')
percent_correct = np.sum(labels_cell1 == np.array(predicted_labels_cell1))/len(np.array(predicted_labels_cell1))
gp_ax.text(10,40,'Acc %1.4f '%percent_correct, color='white', fontsize=8)
gp_ax.text(10,500,'G', color='white', fontsize=8)

gc_ax.imshow(vid_frame,cmap=cm.Greys_r,  alpha=.6)
gc_ax.scatter(x_cell1[30:,frame],y_cell1[30:,frame],facecolors='none', edgecolors=conf_cmap(g_confidence), s=15)
gc_ax.set_title('Confidence')

gp_ax.axes.xaxis.set_visible(False)
gp_ax.axes.yaxis.set_visible(False)
gl_ax.axes.xaxis.set_visible(False)
gl_ax.axes.yaxis.set_visible(False)
gc_ax.axes.xaxis.set_visible(False)
gc_ax.axes.yaxis.set_visible(False)
gp_ax.set_ylabel('Green Channel')
labels_cell1 = [ [0,]*10, [1]*10, [2]*10, ]
labels_cell1 = [item for sublist in labels_cell1 for item in sublist]
predicted_labels_cell1 = predicted_labels_blue.tolist()
vid_frame = blue_vid[frame,:,:]
bp_ax.imshow(vid_frame,cmap=cm.Blues_r)


cax = inside_subplot(gc_ax, [1.2,.15,.1,.4] )

cmap = (mpl.colors.ListedColormap(['r','r','r','r','r','#E7B04B', '#F3E377', '#79E888', '#31DAD4','#1789FC']))

bounds = [0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
cb = fig.colorbar(
    mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
    cax=cax,
    ticks=bounds[4:],
    spacing='uniform',
    orientation='vertical',
)
cax.set_ylim([.4,1])
cb.outline.set_linewidth(0)
cb.ax.tick_params(labelsize=5, width=.5, length=1)



##############################

c2 = ['#3D3337', '#7D5EB8', '#6AC1B8', '#BD571F', '#35b779', '#90d743', '#fde725']

colors = [c2[int(predicted_labels_cell1[i])] for i in range(0,len(predicted_labels_cell1))]
blank_colors = [[0,0,0,0],]*len(colors)
for i in range(len(colors)):
  if int(predicted_labels_cell1[i]) != int(labels_cell1[i]):
    colors[i] = [0,0,0,0]

for i in range(len(colors)):
  if int(predicted_labels_cell1[i]) != int(labels_cell1[i]):
    blank_colors[i] = 'r'

  #if np.max(predicted_labels_cell1_proba[i,:]) < .75:
    #colors[i] = 'm'


bp_ax.scatter(x_cell1[:30,frame],y_cell1[:30,frame],facecolors='none', edgecolors=colors, s=15)
bp_ax.scatter(x_cell1[:30,frame],y_cell1[:30,frame],facecolors=blank_colors, edgecolors=blank_colors, s=15, marker='x')
bp_ax.set_ylabel('Blue Channel')

bl_ax.imshow(vid_frame,cmap=cm.Blues_r)

colors = [c2[int(labels_cell1[i])] for i in range(0,len(labels_cell1))]


bl_ax.scatter(x_cell1[:30,frame],y_cell1[:30,frame],facecolors='none', edgecolors=colors, s=15)
#bl_ax.set_title('True labels')
percent_correct = np.sum(labels_cell1 == np.array(predicted_labels_cell1))/len(np.array(predicted_labels_cell1))
bp_ax.text(10,40,'Acc %1.4f '%percent_correct, color='white', fontsize=8)

bp_ax.text(10,500,'B', color='white', fontsize=8)

bc_ax.imshow(vid_frame,cmap=cm.Greys_r, alpha=.6)
a = bc_ax.scatter(x_cell1[:30,frame],y_cell1[:30,frame],facecolors='none', edgecolors=conf_cmap(g_confidence), s=15)
#bc_ax.set_title('Confidence')

bp_ax.axes.xaxis.set_visible(False)
bp_ax.axes.yaxis.set_visible(False)
bl_ax.axes.xaxis.set_visible(False)
bl_ax.axes.yaxis.set_visible(False)
bc_ax.axes.xaxis.set_visible(False)
bc_ax.axes.yaxis.set_visible(False)


cax = inside_subplot(bc_ax, [1.2,-.3,.1,.4] )

cmap = (mpl.colors.ListedColormap(['r','r','r','r','r','#E7B04B', '#F3E377', '#79E888', '#31DAD4','#1789FC']))

bounds = [0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
cb = fig.colorbar(
    mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
    cax=cax,
    ticks=bounds[4:],
    spacing='uniform',
    orientation='vertical',
)
cax.set_ylim([.4,1])
cb.outline.set_linewidth(0)
cb.ax.tick_params(labelsize=5, width=.5, length=1)


spots = [rragc_ax, lonrf2_ax, map3k6_ax, dock8_ax, orc2_ax, trim33_ax, phip_ax]
spot_inds = [30, 30+12, 30+22, 25+32, 3, 1+12, 1+22]
cc = ['#12153D', '#237167', '#E2D6C9', '#BD571F', '#3D3337', '#7D5EB8', '#6AC1B8']
labels = ['RRAGC', 'LONRF2','MAP3K6','DOCK8','ORC2','TRIM33','PHIP']
channels = [1,1,1,1,2,2,2]
def subplot_spot(tmpax, index, color, channel, label):
  if channel == 1:
    vid_frame = green_vid[frame,:,:]
  if channel == 2:
    vid_frame = blue_vid[frame,:,:]
  x,y = x_cell1[index,frame],y_cell1[index,frame]
  cmap = ['Reds_r','Greens_r','Blues_r'][channel]
  tmpax.imshow(vid_frame[y-5:y+5,x-5:x+5], cmap=cmap)
  tmpax.axes.xaxis.set_visible(False)
  tmpax.axes.set_yticks([])
  tmpax.set_ylabel(label, fontsize=4)
  for spine in tmpax.spines.values():
      spine.set_edgecolor(color)
      spine.set_linewidth(5)

[subplot_spot(spots[i], spot_inds[i], cc[i], channels[i], labels[i]) for i in range(7)]

subplot_spot(wrong_ax, 8,'r' ,2,'Wrong')
wrong_ax.scatter(5,5,color='r', marker='x')

#plt.subplots_adjust(wspace=0,hspace=0)
plt.savefig('multiplexing_7.svg')
