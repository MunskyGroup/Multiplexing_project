import sys 
sys.path.append('../')
import matplotlib
matplotlib.use('TkAgg')
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['mathtext.rm'] = 'serif'
import numpy as np
import argparse
#from mechanistic_models import translation_models as models
from pandas import read_excel,ExcelFile
import matplotlib.pyplot as plt
#from tools import generic_solvers
#stat = generic_solvers.GenericStats()
from scipy.optimize import  minimize,curve_fit


class Data:
    def __init__(self):
        pass


    def load_kenneth_data(self,fname):
        f = ExcelFile(fname)
        self.data_file = f
        self.all_spots = [] 
        self.all_spots_cent = []
        df = f.parse()
        
        
        
        R_array = np.array(df['R_GaussianInt'])
        G_array = np.array(df['G_GaussianInt'])
        B_array = np.array(df['B_GaussianInt'])

        R_err_array = np.array(df['R_95Conf'])
        G_err_array = np.array(df['G_95Conf'])
        B_err_array = np.array(df['B_95Conf'])
               
        time_array = np.array(df['Time_(sec)'])
        
        spot_array = np.array(df['Spot_#'])
                                 
        x=1
        R_traj = []
        G_traj = []
        B_traj = []
        time_traj = []
        for i in range(max(spot_array)):
            inds = np.where(spot_array == i)
            R_traj.append(R_array[inds])
            G_traj.append(G_array[inds])
            B_traj.append(B_array[inds])
            time_traj.append(time_array[inds])
            
        
        self.R_traj = R_traj
        self.G_traj = G_traj
        self.B_traj = B_traj
        self.time_traj = time_traj
                                 
        lengths = []
        for i in range(1,78):
            lengths.append(self.time_traj[i][-1])
            
            
        cap_index = 20
        time_traj_trunc = []
        kept_traj = []
        for i in range(1,len(self.time_traj)):
            try:
                time_traj_trunc.append(self.time_traj[i][:cap_index])    
                kept_traj.append(i)
            except:
                pass
            
        R_trunc = []
        B_trunc = []
        G_trunc = []
        for traj in kept_traj:
            R_trunc.append(self.R_traj[traj][:cap_index])    
            G_trunc.append(self.G_traj[traj][:cap_index])   
            B_trunc.append(self.B_traj[traj][:cap_index])   
            
        self.R_trunc = np.array(R_trunc)
        self.G_trunc = np.array(G_trunc)
        self.B_trunc = np.array(B_trunc)
        self.time_trunc = np.array(time_traj_trunc)
        
        
        self.G_trunc[np.where(self.G_trunc > 500)] = 0
        self.B_trunc[np.where(self.B_trunc > 500)] = 0
        self.B_trunc[np.where(self.B_trunc < -500)] = 0
                
                                         
    def load_data(self,fname,nspots):
        '''
        load the experimental data
        '''
        f = ExcelFile(fname)
        self.data_file = f
        self.all_spots = [] 
        self.all_spots_cent = []
        for sheet in range(nspots):
            df = f.parse(sheet)
            if len( np.nonzero(np.array(df['Peak intensity'])<0 )[0])==0 and len( np.nonzero(np.array(df['Peak intensity'])>.02 )[0])==0:
                self.all_spots.append(np.array(df['Peak intensity']))
                self.all_spots_cent.append(self.all_spots[-1]-np.mean(self.all_spots[-1]))
        self.nspots = len(self.all_spots_cent)
        # find the shortest trajectory
        self.min_len = 100000
        self.all_spot_lengths=[]
        for i in range(self.nspots):
            new_len = len(self.all_spots_cent[i])
            self.all_spot_lengths.append(new_len)
            if new_len<self.min_len:
                self.min_len = new_len 
        # make all trajectories into a matrix of equal length   
        self.all_spots_trunc = np.zeros((self.nspots,self.min_len))
        self.all_spots_norm = np.zeros((self.nspots,self.min_len))
        for i in range(self.nspots):
            self.all_spots_trunc[i,:] = self.all_spots[i][:self.min_len]
            self.all_spots_norm[i,:] = self.all_spots[i][:self.min_len]/np.mean(self.all_spots[i][:self.min_len])
        # get the variance of the trajectories
        self.traj_variance = np.var(self.all_spots_trunc,axis=1) 
        self.traj_mean = np.mean(self.all_spots_trunc,axis=1)
        self.all_spots_trunc_cent = self.all_spots_trunc-np.expand_dims(np.mean(self.all_spots_trunc,axis=1),axis=2)

        return self.all_spots 

    def load_data_v2(self,fname):
        '''
        load the experimental data
        '''
        f = ExcelFile(fname)
        self.data_file = f
        self.all_spots = [] 
        self.all_spots_cent = []
        # parse the sheet
        df = f.parse(0)
        # get the trajectory ids
        trajectory_ids = np.array(df['TrackN'])
        print(trajectory_ids)
        self.nspots = np.max(trajectory_ids)
        self.all_lens = []
        for i in range(1,self.nspots+1):
            start,stop = np.where(trajectory_ids == i)[0][[0,-1]]
            tmp = np.array(df['Peak intensity'])[start:stop+1]
            tmp[tmp>.04] = .02
            tmp[tmp<0] = 1e-6
            self.all_spots.append(tmp)
            self.all_spots_cent.append(self.all_spots[-1]-np.mean(self.all_spots[-1]))
            self.all_lens.append(len(tmp))
        # make all trajectories into a matrix of equal length   
        self.min_len = np.min(self.all_lens)
        self.min_len = 100 #ghettoblasthardcode
        self.all_spots_trunc = np.zeros((self.nspots,self.min_len))
        self.all_spots_norm = np.zeros((self.nspots,self.min_len))
        for i in range(self.nspots):
            self.all_spots_trunc[i,:] = self.all_spots[i][:self.min_len]
            self.all_spots_norm[i,:] = self.all_spots[i][:self.min_len]/np.mean(self.all_spots[i][:self.min_len])
        # get the variance of the trajectories
        self.traj_variance = np.var(self.all_spots_trunc,axis=1) 
        self.traj_mean = np.mean(self.all_spots_trunc,axis=1)
        self.all_spots_trunc_cent = self.all_spots_trunc-np.expand_dims(np.mean(self.all_spots_trunc,axis=1),axis=2)
        return self.all_spots 
        
    def get_data_acc(self,nspots):
        ''' 
        get the variance for all the autocorrelations.
        '''
        # get a matrix of autocorrelations
        self.acc_data = np.zeros((nspots,self.min_len))
        self.acc_data_normalized = np.zeros((nspots,self.min_len))
        for i in range(nspots):
            self.acc_data[i,:] = stat.get_acc2(self.all_spots_norm[i,:]-1.0,trunc=True) 
            self.acc_data_normalized[i,:] = self.acc_data[i,:]/self.traj_mean[i]**2
        # get the variance of the autocorrelations at each time      
        self.acc_variance = np.var(self.acc_data,axis=0)
        self.acc_mean = np.mean(self.acc_data,axis=0)
        # get the normalized acc
         
    def get_elongation_time(self,nspots):
        '''
        get the elongation time distributions for each trajectory.
        '''
        self.waiting_times = np.zeros(nspots)
        small = 1e-5
        self.tvec=np.arange(self.acc_data.shape[1])
        for i in range(nspots):
            self.waiting_times[i] = self.tvec[np.where(self.acc_data[i,:]<small)[0][0]]
        return self.waiting_times
    
    def expo(self,x,a,c,d):        
        '''
        exponential function.
        '''
        c = np.abs(c)
        return a*np.exp(-c*x)+d

    def correct_for_photobleaching(self,trajectories,normalized=False):
        '''
        Fit each trajectory to an exponential;
        correct! 
        '''  
        #ntraj,nt = trajectories.shape
        corrected = []  
        exponentials = []
        for i in range(trajectories.shape[0]):
            guess = (.5,.3,0)
            #guess = (.001,.34)
            tvec = range(len(trajectories[i,:]))
            opt,cov = curve_fit(self.expo,tvec,trajectories[i,:],p0=guess,maxfev=10000)
            if normalized:
                exponentials.append(self.expo(tvec,opt[0],opt[1],opt[2]))
            else:
                exponentials.append(self.expo(tvec,1.0,opt[1],1.0))
            #exponentials.append(self.expo(self.tvec,opt[0],opt[1]))
            corrected.append(trajectories[i,:]/exponentials[-1]) 
        self.all_spots_cent = corrected
        return corrected,exponentials
            
class Fit(Data):
    def __init__(self,N,gene):
        '''
        define free parameters, construct a baseline model. 
        ''' 
        self.model = models.translateCorrs() 
        self.model.N = N
        # important timing stuff - defined according to data
        self.frame_rate = 10
        self.model.tf = self.frame_rate*self.min_len
        self.model.ptimes = self.min_len 
        self.model.fi = .1 
        if gene =='KDM5B':
            self.model.N = 19
            probe_design_old = np.loadtxt('out/KDM5B_Probe.txt')
            n_real = len(probe_design_old)
            binary = np.zeros(self.model.N)
            for i in range(self.model.N):
                binary[i] = np.sum(probe_design_old[i*binsize:(i+1)*binsize])
        elif gene == 'p300':
            self.model.N = 28
            probe_design_old = np.loadtxt('out/p300_Probe.txt')
            n_real = len(probe_design_old)
            binary = np.zeros(self.model.N)
            for i in range(self.model.N):
                binary[i] = np.sum(probe_design_old[i*binsize:(i+1)*binsize])
        else:
            print('Gene not recognized')
        self.model.binary = binary 
    
    def loss(self,parameters):
        '''
        get autocorrelations for a given parameter set.   
        compute difference between this and a given trajectorie(s)
        '''
        ke,kb = parameters
        self.model.params['ke']=ke
        self.model.params['kb']=kb
        self.model.get_autocorrelation() 
        for i in range(nspots):
            error +=  np.sum((self.acc_data-cc.acc_model)**2 / self.acc_variance)
        return error 
    
    def fit(self):
        '''
        fit a spot/spots. 
        '''
        res = minimize(self.loss,x=[.1,.5])
        return res.x


#data1 = Data()             
#data2 = Data()

#data1.load_data('../../data/Data_05_30_17/myTrainingDataset_1.xls',20)
#data2.load_data('../../data/Data_05_30_17/myTrainingDataset_2.xls',20)
#data1.get_data_acc(14)
#data2.get_data_acc(10)
#data1.get_elongation_time(14)
#data2.get_elongation_time(10)


#corrected,exponentials = data1.correct_for_photobleaching(data1.all_spots_trunc)
#f,ax = plt.subplots(2,1)
#for i in range(5):
#    ax[0].plot(data1.tvec,data1.all_spots_trunc[i,:])
#    ax[0].plot(data1.tvec,exponentials[i])
#    ax[1].plot(data1.tvec,corrected[i])
#f.show()
    

#f= plt.figure(figsize=(12,6))
#ax1 = f.add_axes([.1,.1,.5,.8])
#ax3 = f.add_axes([.7,.52,.25,.38])
#ax4 = f.add_axes([.7,.1,.25,.38])
#
#ax1.plot(data1.acc_data.T,linewidth=2,color='C0',alpha=.3)
#ax1.plot(data1.acc_mean,linewidth=4,label='p300')
#ax1.plot(data2.acc_data.T,linewidth=2,color='C1',alpha=.3)
#ax1.plot(data2.acc_mean,linewidth=4,label='KDM5B')
#ax1.legend()
#ax1.fill_between(range(len(data1.acc_mean)),data1.acc_mean-np.sqrt(data1.acc_variance),data1.acc_mean+np.sqrt(data1.acc_variance),color='C0',alpha=.25)
#ax1.fill_between(range(len(data2.acc_mean)),data2.acc_mean-np.sqrt(data2.acc_variance),data2.acc_mean+np.sqrt(data2.acc_variance),color='C1',alpha=.25)
#
#ax3.hist(data2.waiting_times,bins=8,color='C1')
#ax4.hist(data1.waiting_times,bins=8,color='C0')
#
##limits
#ax3.set_xlim([0,40])
#ax4.set_xlim([0,40])
#ax3.set_ylim([0,6])
#ax4.set_ylim([0,6])
#ax1.set_ylim([-.1,.5])
#ax1.set_xlim([0,30])
#
## labeling
#ax1.set_xlabel(r'$\tau$',size=25)
#ax1.set_ylabel(r'$R(\tau)$',size=25)
#ax4.set_xlabel(r'$t_{elong}$',size=25)
#ax3.set_ylabel(r'$frequency$',size=25)
#ax4.set_ylabel(r'$frequency$',size=25)
#f.show()



