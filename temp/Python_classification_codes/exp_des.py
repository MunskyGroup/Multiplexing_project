import numpy as np
import matplotlib.pyplot as plt
import nn_classifier as ml


def load_data(noise,data_len):
    '''
    load the data
    '''
    prin =1
    split = 7500
    data1 = np.loadtxt('../data/kmdb5_10000.txt',delimiter=',')
    ncol = data_len
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
    return train_data,test_data,train_labels,test_labels

def train_model(train_data,train_labels,epochs=50):
    '''
    Train the model.
    '''
    M = ml.nn_classifier()
    M.create_cnn_model(data_len=train_data.shape[1])
    train_data = np.expand_dims(train_data,axis=2)
    M.model.fit(train_data,train_labels,epochs=50)
    return M

def test_model(M,test_data,test_labels):
    '''
    Test the ability to identify.
    '''
    predictions = M.model.predict(np.expand_dims(test_data,axis=2))
    predictions_1 = predictions[:2500,0]
    predictions_2 = predictions[2500:,0]
    p1_correct = np.sum(predictions_1>.5)
    p2_correct = np.sum(predictions_2<.5)
    return (p1_correct+p2_correct)/5000.0

def run_noise_length():
    noises = np.linspace(0,10,10)
    lengths = np.arange(20,250,30)
    ID_matrix = np.zeros((len(noises),len(lengths)))
    for i in range(len(noises)):
        for j in range(len(lengths)):
            print(i)
            print(j)
            train_data,test_data,train_labels,test_labels=load_data(noises[i],lengths[j])
            M = train_model(train_data,train_labels)
            ID_matrix[i,j] = test_model(M,test_data,test_labels)
            print(ID_matrix[i,j])

    np.savetxt('id_matrix.txt',ID_matrix)
    # plt.contourf(ID_matrix)
    # plt.savefig('id_matrix_fig.eps')

def plot_noise_length():
    '''

    '''
    ids = np.loadtxt('id_matrix.txt')
    noises = np.linspace(0,10,10)
    lengths = np.arange(20,250,30)
    L,N = np.meshgrid(lengths,noises)
    f,ax = plt.subplots()
    cp = ax.contourf(L,N,ids)
    cb = f.colorbar(cp)
    f.savefig('id_matrix_fig.eps')
    f2,ax2 = plt.subplots(2,1,figsize=(8,4))
    ax2[0].plot(lengths,ids[2,:],'k',linewidth=2)
    ax2[1].plot(noises,ids[:,3],'k',linewidth=2)
    f2.savefig('exp_deses.eps')
    print(noises[2])
    print(lengths[3])

if __name__=='__main__':
    run_noise_length()
    plot_noise_length()
    #plt.show()
