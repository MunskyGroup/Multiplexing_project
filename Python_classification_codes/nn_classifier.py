import numpy as np
import tensorflow as tf
from tensorflow import keras
import data_fit as df
import matplotlib.pyplot as plt

class nn_classifier():
    def __init__(self):
        pass

    def load_data(self,fname,photobleaching=True):
        '''
        load a dataset, return data object
        '''
        # Get some training data. 
        data = df.Data()
        data.load_data_v2(fname)
        if photobleaching:
            data.correct_for_photobleaching(data.all_spots,normalized=False)
        self.data = data
        return data

    def create_nn_model(self,n_nodes=50):
        '''
        Create the NN model, compile it. 
        '''
        self.model = keras.Sequential([
        keras.layers.Dense(n_nodes, activation=tf.nn.relu),
        keras.layers.Dense(2, activation=tf.nn.softmax)])
        self.model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    def create_cnn_model(self,data_len):
        '''
        Create a convolutional neural network model. 
        '''
        self.model = keras.Sequential([
            keras.layers.Conv1D(input_shape=(data_len,1), filters = 1, kernel_size=5, strides=1,padding='valid',activation=tf.nn.relu),
            #keras.layers.MaxPool1D(pool_size=2),
            keras.layers.Conv1D(filters=4, kernel_size=2, strides=1, padding='same', activation = tf.nn.relu),
            keras.layers.MaxPool1D(pool_size=2),
            keras.layers.Flatten(),
            keras.layers.Dense(2,activation = tf.nn.softmax)]) 
        self.model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    def visualize_model_layers(self,test_data):
        '''
        For each layer in the CNN, visualize a given input from each class to see how it works. 
        '''
        layer_outputs = [layer.output for layer in self.model.layers]
        activation_model = keras.models.Model(inputs=self.model.input,outputs=layer_outputs)
        f,ax = plt.subplots(len(self.model.layers)-1,2)
        f2,ax2 = plt.subplots(1,2)
        f3,ax3 = plt.subplots(len(self.model.layers)-1,2)
        for k in range(2):
            ax[0,k].set_title('Convolutional layers for sample trajectory {0}'.format(k))
            ax[0,k].plot(test_data[k,:])
            activations = activation_model.predict(np.expand_dims(np.reshape(test_data[k,:],(1,len(test_data[k,:]))),axis=2))
            for j in range(len(activations)-2):
                try:
                    z = np.zeros(activations[j].shape[1])
                    w = self.model.layers[j].get_weights()[0].ravel()
                    z[10:10+len(w)] = w
                    ax3[j,k].step(np.arange(activations[j].shape[1]),z)
                except:
                    continue
                for l in range(activations[j].shape[2]):
                    ax[j+1,k].plot(activations[j][0,:,l])
            ax2[k].set_title('Flattened conv {0}'.format(k))
            m1 = ax2[k].imshow(np.hstack((activations[-2].T,self.model.layers[-1].get_weights()[0])),aspect='auto',cmap='RdBu')
            f2.colorbar(m1,ax=ax2[k])
        f2.tight_layout()
        f.tight_layout()
        # this is the tensor-flowy way of doing the same as above. 
#        graph = tf.Graph()
#        with graph.as_default():
#            # set placeholders
#            inputs_ = tf.placeholder(tf.float32,[None,data_len,n_channels],name = 'inputs')
#            labels_ = tf.placeholder(tf.float32,[None,n_classes],name='labels')
#            keep_prob_ = tf.placeholder(tf.float32,name='learning_rate')
#
#            # construct the network
#            conv1 = tf.layers.conv1d(inputs=inputs_, filters = 2, kernel_size=2, strides=1,padding='same',activation=tf.nn.relu)
#            max_pool_1 = tf.layers.max_pooling1d
#
#            # (batch, 64, 18) -> (batch, 32, 36)
#            conv2 = tf.layers.conv1d
#            max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2, padding='same')
#        
#            # (batch, 32, 36) -> (batch, 16, 72)
#            conv3 = tf.layers.conv1d(inputs=max_pool_2, filters=8, kernel_size=2, strides=1, padding='same', activation = tf.nn.relu)
#            max_pool_3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2, padding='same')
#            # (batch, 16, 72) -> (batch, 8, 144)
#            conv4 =tf.layers.conv1d(inputs=max_pool_3, filters=16, kernel_size=2, strides=1, padding='same', activation = tf.nn.relu)
#            max_pool_4 = tf.layers.max_pooling1d(inputs=conv4, pool_size=2, strides=2, padding='same')
#
#            # flatten max_pool_4. 
#            flat = tf.reshape(max_pool_4, (-1,16))
#            flat = tf.nn.dropout(flat,keep_prob=keep_prob_)
#
#            # map this to fully connected layer
#            logits = tf.layers.dense(flat,n_classes)
#            # Cost function and optimizer
#            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
#                labels=labels_))
#            optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost)
#        
#            # Accuracy
#            correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_, 1))
#            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
