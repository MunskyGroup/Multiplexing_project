import numpy as np
import matplotlib.pyplot as plt
import data_fit as df
import nn_classifier as ml
import data_gen
from sklearn.utils import shuffle
import os
import tensorflow as tf

def check_data():
    '''
    This is a function to load the real data trajectories to compare to
    the real data
    '''
    # load in some data to check out.
    dKDM5B = df.Data()
    dKDM5B.load_data('../data/myKDM5B_1secInterval.xls',26)
    print('KDM5B has {0} trajectories'.format(dKDM5B.all_spots_norm.shape[0]))
    dKDM5B.all_spots_norm = np.array(dKDM5B.correct_for_photobleaching(dKDM5B.all_spots_trunc,normalized=True)[0])
    print('KDM5B has {0} trajectories'.format(dKDM5B.all_spots_norm.shape[0]))

    dp300 = df.Data()
    #dp300.load_data('../data/myp300_1secInterval.xls',36)
    dp300.load_data_v2('../data/H2B.xls')
    dp300.all_spots_norm = np.array(dKDM5B.correct_for_photobleaching(dp300.all_spots_trunc,normalized=True)[0])
    print('H2B has {0} trajectories'.format(len(dp300.all_spots)))
    # plot some trajectories.
    n_plot = 5
    f,ax = plt.subplots(2,1)
    for i in range(n_plot):
        ax[0].plot(dKDM5B.all_spots_norm[i,:],'k')
        ax[1].plot(dp300.all_spots_norm[i,:],'salmon')


    ax[0].set_title('KDM5B (experimental)')
    ax[1].set_title('H2B (experimental)')

    f3,ax3 = plt.subplots(2,2)
    ax3[0,0].hist(np.array(dKDM5B.all_spots_norm).ravel(),bins=30)
    ax3[0,0].set_title('Exp. KDM5B')
    ax3[1,0].hist(np.array(dp300.all_spots_norm).ravel(),bins=30)
    ax3[1,0].set_title('Exp. H2B')

    # generate some trajectories for comparison
    sim_KDM5B = data_gen.simulate_data(n_frames =100,n_traj=10,gene='KDM5B',noise=.015,sample_initiation_rate=True)
    sim_p300 = data_gen.simulate_data(n_frames = 100,n_traj=10,gene='H2B',noise=.015,k_init=.15,sample_initiation_rate=True)

    ax3[0,1].hist(np.array(sim_KDM5B.intensity_vec_norm_noise.T).ravel(),bins=30)
    ax3[0,1].set_title('Sim. KDM5B')
    ax3[0,1].set_xlim(ax3[0,0].get_xlim())
    ax3[1,1].hist(np.array(sim_p300.intensity_vec_norm_noise.T).ravel(),bins=30)
    ax3[1,1].set_title('Sim. H2B')
    ax3[1,1].set_xlim(ax3[1,0].get_xlim())

    f2,ax2 = plt.subplots(2,1)
    #ax2[0].plot(sim_KDM5B.intensity_vec_norm_noise.T[-100:,:],'orange')
    ax2[0].plot(sim_KDM5B.intensity_vec_norm_noise.T,'orange')
    #ax2[1].plot(sim_p300.intensity_vec_norm_noise.T[-100:,:],'magenta')
    ax2[1].plot(sim_p300.intensity_vec_norm_noise.T,'magenta')
    ax2[0].set_title('KDM5B (simulated)')
    ax2[1].set_title('H2B (simulated)')

    plt.show()

def train_model_multiple_sizes(lengths=[50,100,150,200,250,300],gene1='KDM5B',gene2='H2B',new_data=True,save_data=False,n_traj=1000,n_frames = 500):
    '''
    train multiple NN on trajectories of different lengths, run consensus.
    '''
    noise=1
    if new_data:
        print('...Simulating {0} trajectories for gene {1}'.format(n_traj,gene1))
        sim1=data_gen.simulate_data(gene=gene1,n_traj=n_traj,n_frames =n_frames ,noise=noise)
        data1 = sim1.centered_intensity_vec_noise_norm
        #data1 = sim1.intensity_vec
        print('Done. \n ...Simulating {0} trajectories for gene {1}'.format(n_traj,gene2))
        sim2=data_gen.simulate_data(gene=gene2,n_traj=n_traj, n_frames = n_frames,noise=noise)
        data2 = sim2.centered_intensity_vec_noise_norm
        #data2 = sim2.intensity_vec
        print('Done.')
        framerate = 7
        if save_data:
            np.savetxt('../data/sim_data/{0}_{1}_{2}_{3}_frames.txt'.format(gene1,n_traj,n_frames,framerate),data1)
            np.savetxt('../data/sim_data/{0}_{1}_{2}_{3}_frames.txt'.format(gene2,n_traj,n_frames,framerate),data2)

    else:
        try:
            data1 = np.loadtxt('../data/sim_data/{0}_{1}_{2}_frames.txt'.format(gene1,n_traj,n_frames))
            data2 = np.loadtxt('../data/sim_data/{0}_{1}_{2}_frames.txt'.format(gene2,n_traj,n_frames))
        except:
            print('Unable to load the data, simulating data...')
            print('...Simulating {0} trajectories for gene {1}'.format(n_traj,gene1))
            sim1=data_gen.simulate_data(gene=gene1,n_traj=n_traj)
            data1 = sim1.centered_intensity_vec_noise
            print('Done. \n ...Simulating {0} trajectories for gene {1}'.format(n_traj,gene2))
            sim2=data_gen.simulate_data(gene=gene2,n_traj=n_traj)
            data2 = sim2.centered_intensity_vec_noise
            print('Done.')
            if save_data:
                np.savetxt('../data/sim_data/{0}_{1}_{2}_frames.txt'.format(gene1,n_traj,n_frames),data1)
                np.savetxt('../data/sim_data/{0}_{1}_{2}_frames.txt'.format(gene2,n_traj,n_frames),data2)
    all_models = []
    all_classifications = []
    for i in range(len(lengths)):
        print('Training model for trajectories of length {0}'.format(lengths[i]))
        # split into training and testing sets.
        split = 750
        train_data1 = data1[:split,:lengths[i]]
        test_data1 = data1[split:,:lengths[i]]
        train_data2 = data2[:split,:lengths[i]]
        test_data2 = data2[split:,:lengths[i]]
        # make class labels
        labels = np.zeros(data1.shape[0]+data2.shape[0])
        labels[data1.shape[0]:] += 1
        data = np.vstack((data1,data2))
        train_data = np.vstack((train_data1,train_data2))
        test_data = np.vstack((test_data1,test_data2))
        train_labels = np.zeros(train_data1.shape[0]+train_data2.shape[0])
        train_labels[train_data1.shape[0]:] += 1
        test_labels = np.zeros(test_data1.shape[0]+test_data2.shape[0])
        test_labels[test_data1.shape[0]:] += 1
        # shuffle training data
        #train_data,train_labels = shuffle(train_data,train_labels)
        M = ml.nn_classifier()
        M.create_cnn_model(data_len=train_data1.shape[1])
        # expand train data because tensors
        train_data = np.expand_dims(train_data,axis=2)
        M.model.fit(train_data,train_labels,epochs=300)
        all_models.append(M)
        # for each test trajectory, record the classification
        predictions = M.model.predict(np.expand_dims(test_data,axis=2))
        print(predictions)
        all_classifications.append(predictions)
    # get the consensus
    weights = np.array(lengths)/sum(lengths)
    consensus = all_classifications[0]*weights[0]
    for i in range(1,len(lengths)):
        consensus += weights[i]*all_classifications[i]
    print(consensus)
    for i in range(6):
        print(np.sum(all_classifications[i][:250,0]>0.5))
        print(np.sum(all_classifications[i][250:,0]<0.5))
        print('****')

    print(np.sum(consensus[:250,0]>0.5))
    print(np.sum(consensus[250:,0]<0.5))

    return (M,all_classifications,consensus)


def down_sample_data(data1,data2,desired_frame_rate,ntraj,noise=2.2,prin=.014):
    '''
    This function downsamples large training data files to a desired framerate and adds noise
    
    **noise** - normal variance
    
    **prin** - total scaling
    '''
    
    os.chdir('../data/sim_data/')
    
    data1 = data1[0:ntraj,0:-1:desired_frame_rate]
    data2 = data2[0:ntraj,0:-1:desired_frame_rate]
    
    wn = np.random.randn(*data1.shape)*noise
    data1 = prin*(data1) + wn
    wn = np.random.randn(*data2.shape)*noise
    data2 = prin*(data2) + wn
    
    data1[data1<0] = 0
    data2[data2<0] = 0
    os.chdir('../../codes/')
    return data1,data2


def train_model(gene1='AlexX_1S12F',gene2='KDM5B_12F',norm='norm',new_data=True,save_data=True,n_traj=10000,n_frames = 500, frame_rate=7):
    '''
    train NN classifier
    '''
    if new_data:
        print('...Simulating {0} trajectories for gene {1}'.format(n_traj,gene1))
        sim1=data_gen.simulate_data(gene=gene1,n_traj=n_traj,n_frames =n_frames ,noise=.015,frame_rate=7)
        
        if norm == 'norm':
            data1 = sim1.intensity_vec_norm_noise
        else:
            data1 = sim1.intensity_vec
        #data1 = sim1.intensity_vec
        print('Done. \n ...Simulating {0} trajectories for gene {1}'.format(n_traj,gene2))
        sim2=data_gen.simulate_data(gene=gene2,n_traj=n_traj, n_frames = n_frames,noise=.015,frame_rate=7)
        
        if norm == 'norm':
            data2 = sim2.intensity_vec_norm_noise
        else:
            data2 = sim2.intensity_vec
        #data2 = sim2.intensity_vec
        print('Done.')
        if save_data:
            np.savetxt('../data/sim_data/{0}_{1}_{2}_frames_{3}.txt'.format(gene1,n_traj,n_frames,norm),data1)
            np.savetxt('../data/sim_data/{0}_{1}_{2}_frames_{3}.txt'.format(gene2,n_traj,n_frames,norm),data2)

    else:
        #try:
        data1 = np.loadtxt('../data/sim_data/data_norm_master_{0}_{1}_{2}_frames.txt'.format(gene1,10000,1000,norm))
        data2 = np.loadtxt('../data/sim_data/data_norm_master_{0}_{1}_{2}_frames.txt'.format(gene2,10000,1000,norm))
        data1,data2 = down_sample_data(data1,data2,n_traj,frame_rate)
        #except:
            #print('Unable to load the data, simulating data...')
            #print('...Simulating {0} trajectories for gene {1}'.format(n_traj,gene1))
            #sim1=data_gen.simulate_data(gene=gene1,n_traj=n_traj)
            #data1 = sim1.centered_intensity_vec_noise
            #print('Done. \n ...Simulating {0} trajectories for gene {1}'.format(n_traj,gene2))
            #sim2=data_gen.simulate_data(gene=gene2,n_traj=n_traj)
            #data2 = sim2.centered_intensity_vec_noise
            #print('Done.')
            #if save_data:
                #np.savetxt('../data/sim_data/{0}_{1}_{2}_frames.txt'.format(gene1,n_traj,n_frames),data1)
                #np.savetxt('../data/sim_data/{0}_{1}_{2}_frames.txt'.format(gene2,n_traj,n_frames),data2)

    # split into training and testing sets.
    split = int(np.floor(.75*n_traj))
    train_data1 = data1[:split,:]
    test_data1 = data1[split:,:]
    train_data2 = data2[:split,:]
    test_data2 = data2[split:,:]
    # make class labels
    labels = np.zeros(data1.shape[0]+data2.shape[0])
    labels[data1.shape[0]:] += 1
    data = np.vstack((data1,data2))
    train_data = np.vstack((train_data1,train_data2))
    test_data = np.vstack((test_data1,test_data2))
    train_labels = np.zeros(train_data1.shape[0]+train_data2.shape[0])
    train_labels[train_data1.shape[0]:] += 1
    test_labels = np.zeros(test_data1.shape[0]+test_data2.shape[0])
    test_labels[test_data1.shape[0]:] += 1
    # make the CNN model
    M = ml.nn_classifier()
    opt = tf.keras.optimizers.Adam(lr=0.001)
    M.create_cnn_model(data_len=train_data1.shape[1] )
    # expand train data because tensors
    train_data = np.expand_dims(train_data,axis=2)
    M.model.fit(train_data,train_labels,epochs=20)
    # test the model on simulated data
    predictions = M.model.predict(np.expand_dims(test_data,axis=2))
    # make some plots
    print(predictions)
#    f,ax = plt.subplots(1,2,figsize=(6,2.5))
#    print(test_data1.shape)
#    ax[0].hist(predictions[:test_data1.shape[0],0],color='orange',bins=np.linspace(0,1,18))
#    ax[0].set_xlabel('Probability of {0}'.format(gene1))
#    ax[0].set_ylabel('Number of trajectories')
#    ax[1].hist(predictions[test_data1.shape[0]:,1],color='mediumorchid',bins=np.linspace(0,1,18))
#    ax[1].set_xlabel('Probability of {0}'.format(gene2))
#    f.tight_layout()
#    f.savefig('../figures/sim_data_id.eps')
    # M.visualize_model_layers(np.vstack((train_data1[0,:],train_data1[10,:])))
    return M,predictions



def test_real_data(M):
    '''
    test a trained CNN on real data.
    '''
    dKDM5B = df.Data()
    dKDM5B.load_data('../data/myKDM5B_1secInterval.xls',26)
    print(dKDM5B.all_spots_norm.shape)
    dKDM5B.all_spots_norm = np.array(dKDM5B.correct_for_photobleaching(dKDM5B.all_spots_trunc,normalized=True)[0])
    dp300 = df.Data()
    dp300.load_data_v2('../data/H2B.xls')
    dp300.all_spots_norm = np.array(dKDM5B.correct_for_photobleaching(dp300.all_spots_trunc,normalized=True)[0])

    predictions_KDM5B = M.model.predict(np.expand_dims(dKDM5B.all_spots_norm.T,axis=2))
    predictions_p300 = M.model.predict(np.expand_dims(dp300.all_spots_norm.T,axis=2))
    f,ax = plt.subplots(1,2,figsize=(6,2.5))
    ax[0].hist(predictions_KDM5B[:,0],color='k',bins=np.linspace(0,1,18))
    ax[0].set_xlim([0,1])
    ax[0].set_xlabel('Probability of KDM5B')
    ax[0].set_ylabel('Number of trajectories')
    ax[1].hist(predictions_p300[:,1],color='salmon',bins=np.linspace(0,1,18))
    ax[1].set_xlabel('Probability of H2B')
    ax[1].set_xlim([0,1])
    f.tight_layout()
    f.savefig('../figures/exp_data_id.eps')
    return (predictions_KDM5B,predictions_p300)





if __name__=='__main__':

#    check_data()
    M,p = train_model(gene1='AlexX_1F12S',gene2='KDM5B_12S',new_data=False,save_data=False,n_traj=500,n_frames=300,frame_rate=7)
    (predictions_KDM5B,predictions_H2B) = test_real_data(M)
#    M,all_classification,consensus = train_model_multiple_sizes()
