import numpy as np
import matplotlib.pyplot as plt
import nn_classifier as ml

# load data, trim, combine:
M = ml.nn_classifier()
M.load_data('../data/myTrainingDataset_1.xls')
# try again except with FT of data. 
data1 = np.copy(M.data.all_spots_trunc)
#data1 = np.fft.fft(M.data.all_spots_trunc)
data1 = data1[:,:100]
np.random.shuffle(data1)
train_data1 = data1[:-4,:]
test_data1 = data1[-4:,:]

M.load_data('../data/myTrainingDataset_2.xls')
data2 = np.copy(M.data.all_spots_trunc)
np.random.shuffle(data2)
#data2 = np.fft.fft(M.data.all_spots_trunc)
labels = np.zeros(data1.shape[0]+data2.shape[0])
labels[data1.shape[0]:] += 1
train_data2 = data2[:-4,:]
test_data2 = data2[-4:,:]
data = np.vstack((data1,data2))
train_data = np.vstack((train_data1,train_data2))
test_data = np.vstack((test_data1,test_data2))
train_labels = np.zeros(train_data1.shape[0]+train_data2.shape[0])
train_labels[train_data1.shape[0]:] += 1
test_labels = np.zeros(test_data1.shape[0]+test_data2.shape[0])
test_labels[test_data1.shape[0]:] += 1

#f,ax = plt.subplots(2,1)
#ax[0].plot(data1[:10,:].real.T,'k')
#ax[0].plot(data2[:10,:].real.T,'r')
#ax[1].plot(data1[:10,:].imag.T,'k')
#ax[1].plot(data2[:10,:].imag.T,'r')
#f2,ax2 = plt.subplots()
#ax2.hist(data1.real.ravel(),30,color='k')
#ax2.hist(data2.real.ravel(),30,color='r')
#plt.show()

# get a model
M.create_cnn_model()
# expand train data because
train_data = np.expand_dims(train_data,axis=2)
M.model.fit(train_data,train_labels,epochs=2000)

# test the model
predictions = M.model.predict(np.expand_dims(test_data,axis=2))
print(test_labels)
print(predictions[:,0]<predictions[:,1])
print(predictions)
