# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 13:51:22 2022

@author: wsraymon
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.utils import resample

class multiplexing_core:
    '''
    over head class containing some consistent functions to use in the multiplexing paper 
    '''
    def __init__(self):
        pass
    
    
    def get_acc2(self, data, trunc=False):
        '''
        return autocorrelation from a data vector via Fourier transform
    
        Parameters
        ----------
        data : ndarray
            numpy array of data to get the correlation from.
        trunc : bool, optional
            Remove non zero entries and only return decreasing entries.
            The default is False.
    
        Returns
        -------
        autocorrelation : ndarray
            autocorrelation function.
    
        '''
    
        N = len(data)
        fvi = np.fft.fft(data, n=2*N)
        acf = fvi*np.conjugate(fvi)
        acf = np.fft.ifft(acf)
        acf = np.real(acf[:N])/float(N)
        if trunc:
            acf[acf < 0] = 0
            for i in range(1, len(acf)):
                if acf[i] > acf[i-1]:
                    acf[i] = acf[i-1]
        return acf
    
    
    def get_autocov(self,intensity_vec, norm='global'):
        '''
        Return the autocovariance of an intensity vector as defined by:
        
        .. math:: 
            
            ACOV(t, \tau) = cov(X_{t}, X_{\tau}) = E{X_{t}, X_{\tau}} 
        
        There are also several normalization options:
            
        Raw - perform no normalization to each intensity trajectory
        
        Global - subtract the global intensity mean and divide by global
        variance
        
        .. math:: 
            
            \mu_I = E(Intensity) \\
            \sigma_I^2 = V(Intensity) \\
            X_{normalized} = (X(t) - \mu_I) / \sigma_I^2            
    
        Individual - subtact each trajectory by its mean and divide by
        its variance
    
        .. math:: 
            
            X_{normalized} = (X_{i}(t) - E(X_{i}(t))) / V(X_{i}(t))   
    
        Parameters
        ----------
        intensity_vec : ndarray
            intensity tensor of shape (ncolor, ntimes, ntraj).
        norm : str, optional
            normalization to use, 'raw' for no normalization, 'global' to 
            normalize by the gobal intensity moments, 
            'individual' to normalize each trajectory by
            its individual moments. The default is 'global'.
    
        Returns
        -------
        autocorr_vec : ndarray
            returns autorcovariance array of size (ncolor, ntime-1, ntraj).
        autocorr_err : ndarray
            returns autorcovariance SEM array of size (ncolor, ntime-1, ntraj).
    
        '''
        autocorr_vec = np.zeros((intensity_vec.shape))
        autocorr_err = np.zeros((intensity_vec.shape))
        colors = intensity_vec.shape[0]
        n_traj = intensity_vec.shape[2]
    
        for n in range(colors):
            if norm in ['Individual', 'I', 'individual', 'ind','i']:
                for i in range(intensity_vec.shape[2]):
                    ivec = intensity_vec[n, :, i]
                    autocorr_vec[n, :, i] = self.get_acc2(
                        (ivec - np.mean(ivec))/np.var(ivec))
    
            elif norm in ['global', 'Global', 'g', 'G']:
                global_mean = np.mean(intensity_vec[n])
                global_var = np.var(intensity_vec[n])
                for i in range(intensity_vec.shape[2]):
                    autocorr_vec[n, :, i] = self.get_acc2(
                        (intensity_vec[n, :, i]-global_mean)/global_var )
            elif norm in ['raw', 'Raw']:
                for i in range(intensity_vec.shape[2]):
                    autocorr_vec[n, :, i] = self.get_acc2(
                        intensity_vec[n, :, i])
            else:
                print('unrecognized normalization,'/
                      ' please use individual, global, or none')
                return
    
        autocorr_err = 1.0/np.sqrt(n_traj)*np.std(
            autocorr_vec, ddof=1, axis=2)
    
        return autocorr_vec, autocorr_err
    
    def get_g0(self,covariance, mode='interp'):
        '''
        return the normalization point for autocorrelations, g0 delay
    
        Parameters
        ----------
        correlation : ndarray
            numpy array of a cross or autocovariance.
        mode : string, optional
            the type of G0 shot noise to return,
    
            * 'Interp' - will interpolate the g0 position from the G1, G2, and G3 points.
            
            * 'g1' - (second point) g1 will be returned
            
            * 'g0' - g0 will be returned (first point)
            
            * 'max' -maximum of the correlation will be returned
            
            The default is 'interp'.
    
        Returns
        -------
        G0 : float
            point to normalize correlation over.
    
        '''
        if mode.lower() in ['interp', 'inter', 'extrapolate', 'interpolate']:
            X = [1, 2, 3, 4]
            V = covariance[:, X, :]
            G0 = np.interp(0, X, V)
    
        if mode.lower() in ['g1', '1']:
            G0 = covariance[:, 1, :]
    
        if mode.lower() in ['g0', '0']:
            G0 = covariance[:, 0, :]
    
        if mode.lower() in ['max', 'maximum']:
            G0 = np.max(covariance, axis=1)
        return G0
    
    def get_autocorr(self,autocov, norm_type='interp', norm = 'individual'):
        '''
        Given an autocovariance tensor, normalize to the autocorrelation
        
        .. math:: 
            
            ACORR(X(t)) = ACOV(X(t)) / Normalization Constant
            
        where Normalization constant is defined as the delay to divide all 
        autocrrelations by:
            
            * G0 - the first delay without shot noise correction
            
            * G1 - the second delay of the autocorrelation
            
            * interp - interpolated G0, take G1-4 and calculate the G0 without shot noise
            
            
        norm = global will normalize the autocovariance by the global average normalization_constant
        norm = individual will normalize each trajectory by its own normalization_constant
            
        Parameters
        ----------
        autocov : ndarray
            autocovariance tensor of shape (Ncolor, Ntimes, Ntrajectories).
        norm_type : str, optional
            Delay to normalize by, G0, G1 or interp for interpolated G0. The default is 'interp'.
        norm : str, optional
            globally normalize autocovariance or individually normalize.
            Normalize each trajectory by its own g0 or by the global g0. 
            The default is 'individual'
    
        Returns
        -------
        autocorr : ndarray
            autocorrelation tensor of shape (Ncolor, Ntimes, Ntrajectories).
        err_autocorr : ndarray
            SEM autocorrelation tensor of shape (Ncolor, Ntimes, Ntrajectories).
    
        '''
        autocorr = np.copy(autocov)
        n_traj = autocorr.shape[-1]
    
        if norm_type.lower() in ['individual','indiv','i']:
            g0 = self.get_g0(autocov, norm)
            for n in range(autocov.shape[0]):
                autocorr[n] = autocorr[n]/g0[n]
        elif norm_type.lower() in ['global','g']:
            g0 = self.get_g0(autocov, norm)
            g0_mean = np.mean(g0)
            for n in range(autocov.shape[0]):
                autocorr[n] = autocorr[n]/g0_mean     
                
        else: 
        
            msg = 'unrecognized normalization, please use '\
                  'individual, or global for norm arguement'
            print(msg)
            return                 
        err_autocorr =  1.0/np.sqrt(n_traj)*np.std(autocorr, ddof=1, axis=2)
        return autocorr, err_autocorr

    def slice_arr_reverse(self,array, FR, Nframes,axis=1):
        total_time = FR*Nframes
        if total_time > array.shape[1]:
            print('WARNING: desired slicing regime is not possible, making as many frames as possible')
            return array[:,::FR][:,:Nframes]
        return array[:,::FR][:,-Nframes:]
    
    def slice_arr(self,array, FR, Nframes,axis=1):
        total_time = FR*Nframes
        if total_time > array.shape[1]:
            print('WARNING: desired slicing regime is not possible, making as many frames as possible')
            return array[:,::FR][:,:Nframes]
        return array[:,::FR][:,:Nframes]
        
    
    def convert_labels_to_onehot(self,labels):
        '''
        converts labels in the format 1xN, [0,0,1,2,3,...] to onehot encoding,
        ie: N_classes x N,  [[1,0,0,0],[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]     
        '''
        onehotlabels = np.zeros((labels.shape[0],len(np.unique(labels))))
        for i in range(len(onehotlabels)):
            onehotlabels[i,labels[i]] = 1
        return onehotlabels
    

    def process_data_n(self, data, labels, use_norm=True, norm='train_and_test', seed=42, witheld = 1000, test_size = .2, include_acc = False, shuffle=True):
        '''
        

        Parameters
        ----------
        data : ndarray
            ntraj x ntimes intensity data array.
        labels : ndarray
            label array.
        norm : str, optional
            which normalization to use, "train_and_test" will not consider the witheld data
            when performing all data normalization, "train" uses just the training set to normalize all data,
            "all" will use all data to calculate normalization. The default is 'train_and_test'.
        seed : int, optional
            Seed for all randomness. The default is 42.
        witheld : int, optional
            how many trajectories to withold. The default is 1000.
        test_size : float, optional
            percentage to consider the test set. The default is .2.

        Returns
        -------
        X_train : ndarray
            training data.
        X_test : ndarray
            training labels.
        y_train : ndarray
            test data.
        y_test : ndarray
            test labels.
        X_witheld : ndarray
            witheld data.
        y_witheld : ndarray
            witheld labels.

        '''
        unique_labels = np.sort(np.unique(labels).astype(int))
        n_labels = len(unique_labels)
        print(unique_labels)        
        s = []
        for i in range(len(unique_labels)):
            s.append(len(labels == i))
        
        # Shuffle the data so its not from the same cells when we index by labels
        if shuffle:
            data, labels = self.even_shuffle_sample_n(data, labels, samples=s, seed=seed) 
        
        
        if witheld > 0:
            # Witheld data ###################################
            np.random.seed(seed)
            
            int_witheld = []
            labels_witheld = []
            for i in range(len(unique_labels)):
                int_1 = data[labels==i][:int(witheld/n_labels), :] #get n witheld samples
                labels_1 = labels[labels==i][:int(witheld/n_labels)]
                int_witheld.append(int_1)
                labels_witheld.append(labels_1)
                
                
            X_witheld = np.vstack(int_witheld)  #combine and then shuffle in place
            y_witheld = np.hstack(labels_witheld)
    
            int_1_inds = np.random.permutation(len(X_witheld))
            X_witheld = X_witheld[int_1_inds]
            y_witheld = y_witheld[int_1_inds]        
            
            

        # Training data ###################################
            int_training= []
            labels_training = []
            for i in range(len(unique_labels)):     
                int_1 = data[labels==i][int(witheld/n_labels):, :] #get n usable samples                
                labels_1 = labels[labels==i][int(witheld/n_labels):]
                int_training.append(int_1)
                labels_training.append(labels_1)
        
            data = np.vstack(int_training)
            labels = np.hstack(labels_training)
            
            int_1_inds = np.random.permutation(len(data))
 
            data = data[int_1_inds]
            labels = labels[int_1_inds]              
 
        else:
            X_witheld = None
            y_witheld = None
            

        if test_size == 0:
            X_train = data
            y_train = labels
            X_test = None
            y_test = None
        else:
            X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state= seed, stratify=labels)
        
        if use_norm:
            if norm=='train':
                scaler = MinMaxScaler()
                scaler.fit(X_train)
            if norm=='all':
                scaler = MinMaxScaler()
                scaler.fit(np.vstack([X_train, X_test, X_witheld] ))
            if norm=='train_and_test':
                scaler = MinMaxScaler()
                scaler.fit(np.vstack([X_train, X_test] ))         
        
        
            X_train = scaler.transform(X_train)
            if witheld > 0:
                X_witheld = scaler.transform(X_witheld)
            if test_size > 0:
                X_test = scaler.transform(X_test)
        
        
        ### reshape data to correct shape
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        if test_size > 0:
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        if witheld > 0:
            X_witheld = X_witheld.reshape(X_witheld.shape[0], X_witheld.shape[1], 1)
        
        y_train = np.asarray(y_train).astype('float32').reshape((-1,1))
        if test_size > 0:
            y_test = np.asarray(y_test).astype('float32').reshape((-1,1))
        if witheld > 0:
            y_witheld = np.asarray(y_witheld).astype('float32').reshape((-1,1))
        
        if include_acc:
            
            def get_acc(int_g):
                acc, acc_error = self.get_autocov(np.expand_dims(int_g.T,axis=0), norm='ind')
                acc, acc_error1 = self.get_autocorr(acc, norm_type='individual', norm='g0' )
                acc = acc[0].T
                acc = acc.reshape(acc.shape[0], acc.shape[1], 1)
                return acc
            
            Acc_train = get_acc(X_train[:,:,0])
            if test_size > 0:
                Acc_test = get_acc(X_test[:,:,0])
            else:
                Acc_test = None
            if witheld > 0:
                Acc_witheld = get_acc(X_witheld[:,:,0])
            else:
                Acc_witheld = None
            
            return X_train, X_test, y_train, y_test, X_witheld, y_witheld, Acc_train, Acc_test, Acc_witheld
        else:
            
            return X_train, X_test, y_train, y_test, X_witheld, y_witheld
            

    def bootstrap_data(self, X_train, y_train, n_sample, seed):
        return resample(X_train, y_train, n_samples=n_sample, random_state=seed, replace=False)
        


    def process_data(self, data, labels, use_norm=True, norm='train_and_test', seed=42, witheld = 1000, test_size = .2, include_acc = False, shuffle=True):
        '''
        This is the main function to process data before machine learning, takes the intensity numpy array
        and returns X_train, X_test, X_witheld, y_train, y_test, y_train based on desired training sizes.

        Parameters
        ----------
        data : ndarray
            ntraj x ntimes intensity data array.
        labels : ndarray
            label array.
        norm : str, optional
            which normalization to use, "train_and_test" will not consider the witheld data
            when performing all data normalization, "train" uses just the training set to normalize all data,
            "all" will use all data to calculate normalization. The default is 'train_and_test'.
        seed : int, optional
            Seed for all randomness. The default is 42.
        witheld : int, optional
            how many trajectories to withold. The default is 1000.
        test_size : float, optional
            percentage to consider the test set. The default is .2.

        Returns
        -------
        X_train : ndarray
            training data.
        X_test : ndarray
            training labels.
        y_train : ndarray
            test data.
        y_test : ndarray
            test labels.
        X_witheld : ndarray
            witheld data.
        y_witheld : ndarray
            witheld labels.

        '''
        
        
        s1 = len(labels == 0)
        s2 = len(labels == 1)
        
        if shuffle:
            # Shuffle the data so its not from the same cells when we index by labels
            data, labels = self.even_shuffle_sample(data, labels, samples=[s1,s2], seed=seed) 
        
        if witheld > 0:
            # Witheld data ###################################
            np.random.seed(seed)
    
            int_1 = data[labels==0][:int(witheld/2), :] #get n witheld samples
            int_2 = data[labels==1][:int(witheld/2), :] #get n witheld samples
            
            labels_1 = labels[labels==0][:int(witheld/2)]
            labels_2 = labels[labels==1][:int(witheld/2)]
            
            X_witheld = np.vstack([int_1,int_2])  #combine and then shuffle in place
            y_witheld = np.hstack([labels_1,labels_2])
    
            int_1_inds = np.random.permutation(len(X_witheld))
            X_witheld = X_witheld[int_1_inds]
            y_witheld = y_witheld[int_1_inds]        
            
            

        # Training data ###################################
     
            int_1 = data[labels==0][int(witheld/2):, :] #get n usable samples
            int_2 = data[labels==1][int(witheld/2):, :]
            
            labels_1 = labels[labels==0][int(witheld/2):]
            labels_2 = labels[labels==1][int(witheld/2):]
    
            data = np.vstack([int_1,int_2])
            labels = np.hstack([labels_1,labels_2])
            
            int_1_inds = np.random.permutation(len(data))
 
            data = data[int_1_inds]
            labels = labels[int_1_inds]              
 
        else:
            X_witheld = None
            y_witheld = None

        if test_size == 0:
            X_train = data
            y_train = labels
            X_test = None
            y_test = None
        else:
            X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state= seed, stratify=labels)
        
        if use_norm:
            if norm=='train':
                scaler = MinMaxScaler()
                scaler.fit(X_train)
            if norm=='all':
                scaler = MinMaxScaler()
                scaler.fit(np.vstack([X_train, X_test, X_witheld] ))
            if norm=='train_and_test':
                scaler = MinMaxScaler()
                scaler.fit(np.vstack([X_train, X_test] ))         
        
        
            X_train = scaler.transform(X_train)
            if witheld > 0:
                X_witheld = scaler.transform(X_witheld)
            if test_size > 0:
                X_test = scaler.transform(X_test)
        
        
        ### reshape data to correct shape
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        if test_size > 0:
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        if witheld > 0:
            X_witheld = X_witheld.reshape(X_witheld.shape[0], X_witheld.shape[1], 1)
        
        y_train = np.asarray(y_train).astype('float32').reshape((-1,1))
        if test_size > 0:
            y_test = np.asarray(y_test).astype('float32').reshape((-1,1))
        if witheld > 0:
            y_witheld = np.asarray(y_witheld).astype('float32').reshape((-1,1))
        
        if include_acc:
            
            def get_acc(int_g):
                acc, acc_error = self.get_autocov(np.expand_dims(int_g.T,axis=0), norm='ind')
                acc, acc_error1 = self.get_autocorr(acc, norm_type='individual', norm='g0' )
                acc = acc[0].T
                acc = acc.reshape(acc.shape[0], acc.shape[1], 1)
                return acc
            
            Acc_train = get_acc(X_train[:,:,0])
            if test_size > 0:
                Acc_test = get_acc(X_test[:,:,0])
            else:
                Acc_test = None
            if witheld > 0:
                Acc_witheld = get_acc(X_witheld[:,:,0])
            else:
                Acc_witheld = None
            
            return X_train, X_test, y_train, y_test, X_witheld, y_witheld, Acc_train, Acc_test, Acc_witheld
        else:
            
            return X_train, X_test, y_train, y_test, X_witheld, y_witheld
        
    
    
    
    def df_to_arrays(self,dataframe_simulated_cell):
        '''
        convert an rSNAPed data frame into a numpy intensity array
        '''
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
                k+=1
                
        return I_g, I_g_std, I_r, I_r_std, labels, x_loc,y_loc
    
    
    def get_intensity_from_df(self,multiplexing_df, n_traj, n_timepoints, channel='green_int_mean'):
        # convert a data frame into intensity and label arrays
        int_g = multiplexing_df['green_int_mean'].values.reshape([n_traj,n_timepoints])    
        labels = multiplexing_df['Classification'].values.reshape([n_traj,n_timepoints])[:,0]
        
        return int_g, labels
    
    
    def even_shuffle_sample(self, intensity_array, labels, samples = [2500,2500], seed = 42):
        '''
        Evenly shuffle each intensity by label and then sample each intensity.
        
        Return the concatenated shuffled arrays back out 
        '''
        int_1 = intensity_array[labels==0]
        int_2 = intensity_array[labels==1]
        
        labels_1 = labels[labels==0]
        labels_2 = labels[labels==1]
        
        np.random.seed(seed)
        int_1_inds = np.random.permutation(len(int_1))
        int_2_inds = np.random.permutation(len(int_2))
        
        int_1= int_1[int_1_inds][:samples[0], :]
        int_2 = int_2[int_2_inds][:samples[1],:]
        
        labels_1 = labels_1[int_1_inds][:samples[0]]
        labels_2 = labels_2[int_2_inds][:samples[1]]
        
        int_out = np.vstack([int_1,int_2])
        labels_out = np.hstack([labels_1,labels_2])
        
        int_1_inds = np.random.permutation(len(int_out))
        int_out = int_out[int_1_inds]
        labels_out = labels_out[int_1_inds]
        
        return int_out, labels_out

    def process_test_data(self, data, use_norm=True, seed=42, include_acc = True ):

        
        scaler = MinMaxScaler()
        scaler.fit(data)     
        data = scaler.transform(data)    


        
        ### reshape data to correct shape
        data = data.reshape(data.shape[0], data.shape[1], 1)


        if include_acc:
            
            def get_acc(int_g):
                acc, acc_error = self.get_autocov(np.expand_dims(int_g.T,axis=0), norm='ind')
                acc, acc_error1 = self.get_autocorr(acc, norm_type='individual', norm='g0' )
                acc = acc[0].T
                acc = acc.reshape(acc.shape[0], acc.shape[1], 1)
                return acc
            
            acc_data = get_acc(data[:,:,0])
        if include_acc:
          return data, acc_data
        else:
          return data        

    def even_shuffle_sample_n(self, intensity_array, labels, samples = [2500,2500], seed = 42):
        '''
        Evenly shuffle each intensity by label and then sample each intensity.
        
        Return the concatenated shuffled arrays back out 
        '''
        
        np.random.seed(seed)
        unique_labels = np.sort(np.unique(labels).astype(int))
        print(unique_labels)
        ints_out = []
        labels_out = []
        for i in range(len(unique_labels)):
            int_1 = intensity_array[labels==unique_labels[i]]
            labels_1 = labels[labels==unique_labels[i]]
            
            int_1_inds = np.random.permutation(len(int_1))
            int_1= int_1[int_1_inds][:samples[i], :] #shuffle these 
            labels_1 = labels_1[int_1_inds][:samples[i]]
            ints_out.append(int_1)
            labels_out.append(labels_1)
        
        int_out = np.vstack(ints_out)
        labels_out = np.hstack(labels_out)
        
        int_1_inds = np.random.permutation(len(int_out))
        int_out = int_out[int_1_inds]
        labels_out = labels_out[int_1_inds]
        
        return int_out, labels_out        
    
    def minmax_quantile_signal(signal, axis=0, norm='global', quantile=.95, max_outlier=1.5):
        '''
        normalize a singal by its min and max from a quantile,
        leaving trajectories from 0 to 1 at that quantile. Set all outliers
        over a maximum to max_outlier.
        This can be applied globally or individually to each trajectory
        with the flag norm ='global' or norm='indiv'
        
        
        .. code-block::
            
            #global normalization
            S_95= np.quantile(0.95)
            S_normalized = (S - np.min(S)) / (np.quantile(0.95) - np.min(S))
            
            #individual normalization
            S_95= np.quantile(0.95)
            S_normalized = (S - np.min(S,axis=axis)) / (np.quantile(0.95) - np.min(S,axis=axis))
        
    
        Parameters
        ----------
        signal : ndarray
            intensity or signal array.
        axis : int, optional
            axis to apply the standardization over. The default is 0.
        norm : str, optional
            apply this using global mean ('global') and var or individual trajectory
            mean and var ('indiv'). The default is 'global'.
        quantile : float (0-1)
            the % quantile to set to the new 1
        max_outlier : float
            the maximum value to use for any outlier after normalization
    
        Raises
        ------
        UnrecognizedNormalizationError
            Throws an error if the norm is not global or individual.
    
        Returns
        -------
        ndarray
            minmaxed per quantile array over axis desired.
    
        '''
        if norm in ['global', 'Global', 'g', 'G']:
            
            max_95 = np.quantile(signal, quantile)
            sig = (signal -  np.min(signal) ) / ( max_95  - np.min(signal) )
            return np.minimum(sig,max_outlier)
        
        if norm in ['Individual', 'I', 'individual', 'ind','indiv','i']:
            
            max_95 = np.quantile(signal, quantile,axis=axis)
            print(max_95.shape)
            sig = (signal -  np.min(signal,axis=axis) ) / ( max_95 - np.min(signal,axis=axis) )
            return np.minimum(sig,max_outlier)
        
        else:
            msg = 'unrecognized normalization, please use '\
                  'individual, global, or none for norm arguement'
            print(msg)
            return 
    
    
    
    def standardize_signal(signal, axis=0, norm='global'):
        '''
        Perform standardization to set signal mean to 0 with unit variance of 
        1. This can be applied globally or individually to each trajectory
        with the flag norm ='global' or norm='indiv'
    
        Parameters
        ----------
        signal : ndarray
            intensity or signal array.
        axis : int, optional
            axis to apply the standardization over. The default is 0.
        norm : str, optional
            apply this using global mean ('global') and var or individual trajectory
            mean and var ('indiv'). The default is 'global'.
    
        Raises
        ------
        UnrecognizedNormalizationError
            Throws an error if the norm is not global or individual.
    
        Returns
        -------
        ndarray
            standardized array over axis desired.
    
        '''
        if norm in ['global', 'Global', 'g', 'G']:
            return (signal - np.mean(signal)) / np.std(signal)
        if norm in ['Individual', 'I', 'individual', 'ind','indiv','i']:
            return (signal - np.mean(signal, axis=axis)) / np.std(signal,axis=axis)
        else:
            msg = 'unrecognized normalization, please use '\
                  'individual, global, or none for norm arguement'
            print(msg)
            return     
        
            
            
            
        
    
    