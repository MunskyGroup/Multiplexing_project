import numpy as np
import matplotlib.pyplot as plt
import nn_classifier as ml

M = ml.nn_classifier()
M.load_data('../data/H2B.xls')
kdm5b_data = np.copy(M.data.all_spots_trunc)
ncol = kdm5b_data.shape[1]
# load data, trim, combine:
split = 7500
noise = 2.2 
prin = .004
#prin = 1

data1 = np.loadtxt('../data/kmdb5_10000.txt',delimiter=',')
#ncol = data1.shape[1]-1
data1 = data1[:,250:250+ncol]
#data1 = (data1.T/np.max(data1,axis=1)).T
data1 = prin*((data1.T-np.mean(data1,axis=1)).T+np.random.randn(*data1.shape)*noise)
np.random.shuffle(data1)
train_data1 = data1[:split,:]
test_data1 = data1[split:,:]

data2 = np.loadtxt('../data/h2b_10000.txt',delimiter=',')
data2 = data2[:,250:250+ncol]
data2 = prin*((data2.T-np.mean(data2,axis=1)).T+np.random.randn(*data1.shape)*noise)
#data2 = (data2.T/np.max(data2,axis=1)).T
np.random.shuffle(data2)
train_data2 = data2[:split,:]
test_data2 = data2[split:,:]

labels = np.zeros(data1.shape[0]+data2.shape[0])
labels[data1.shape[0]:] += 1
data = np.vstack((data1,data2))
train_data = np.vstack((train_data1,train_data2))
test_data = np.vstack((test_data1,test_data2))
train_labels = np.zeros(train_data1.shape[0]+train_data2.shape[0])
train_labels[train_data1.shape[0]:] += 1
test_labels = np.zeros(test_data1.shape[0]+test_data2.shape[0])
test_labels[test_data1.shape[0]:] += 1

# get a model
M.create_cnn_model(data_len=train_data1.shape[1])
# expand train data because
train_data = np.expand_dims(train_data,axis=2)
M.model.fit(train_data,train_labels,epochs=500)

# test the model
predictions = M.model.predict(np.expand_dims(test_data,axis=2))
# make some plots
f,ax = plt.subplots(1,2,figsize=(8,4))
ax[0].hist(predictions[:2500,0],color='orange')
ax[1].hist(predictions[2500:,0],color='mediumorchid') 
f.savefig('sim_predictions.eps')
f2,ax2 = plt.subplots(figsize=(8,2))
for i in range(3):
    ax2.plot(np.arange(data1.shape[1]),train_data1[i,:].T,'orange')
    ax2.plot(np.arange(data1.shape[1]),train_data2[i,:].T,'mediumorchid')

f2.savefig('trajectories.eps'.format(i))
print(test_labels)
print(predictions[:,0]<predictions[:,1])
print(predictions)

# try prediciting real data
real_predictions = M.model.predict(np.expand_dims(kdm5b_data,axis=2))
f,ax = plt.subplots(2,1)
ax[0].plot(kdm5b_data[:3].T,color='orange')
ax[1].hist(real_predictions[:,0],color='orange')
f.savefig('real_data.eps')

plt.show()

