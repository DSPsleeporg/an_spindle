# -*- coding: utf-8 -*-

"""
This is a parameter search module. With this module, a certain firing pattern 
can be searched randomly with AN model, SAN model and X model. In order to 
lessen dependence of models on parameters, it's important to make various 
parameter sets (models) and then extract common features among them.\
In this script, parameter sets that recapitulate slow wave sleep (SWS) firing
pattern can be searched with algorithms as described in Tatsuki et al., 2016 
and Yoshida et al., 2018.
"""

__author__ = 'Fumiya Tatsuki, Kensuke Yoshida, Tetsuya Yamada, Shoi Shi, Hiroki R. Ueda'
__status__ = 'in prep'
__version__ = '1.0.0'
__date__ = '11 May 2020'


import os
import sys
"""
LIMIT THE NUMBER OF THREADS!
change local env variables BEFORE importing numpy
"""
os.environ["OMP_NUM_THREADS"] = "1"  # 2nd likely
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"  # most likely

from collections import Counter
from datetime import datetime
from multiprocessing import Pool
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from scipy.signal import periodogram
from scipy import signal
from time import time

from anmodel import models


## Run anmodel_rand_param_search.py on the linux terminal (random parameter search by X model) ##
## ex. (on terminal) python anmodel_rand_param_search.py
## make information file in the Info directory
## information file : 
##      ---- before running ----
##      file name : date (ex. 1997_12_22)
##      model : X (or AN / SAN)
##      model_name : (ex. SAN + g_na_v)
##      channel_name and channel_bool_list
##          : determine which channels are included (1：included, 0：not included)
##               channel：determine which channels are included (1：included, 0：not included)
##               channel[0]：g_leak
##               channel[1]：g_na_v
##               channel[2]：g_k_v_hh
##               channel[3]：g_k_v_a
##               channel[4]：g_k_v_si
##               channel[5]：g_ca_v
##               channel[6]：g_k_ca
##               channel[7]：g_na_p
##               channel[8]：g_k_ir
##               channel[9]：g_AMPAR
##               channel[10]：g_NMDAR
##               channel[11]：g_GABAR
##               channel[12]：t_Ca (Ca-pump)
##            (ex. SAN model：[1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1],
##                 YAN model：[1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1])
##      cores : number of cores using for calculation

if __name__ == '__main__':
    import anmodel_rand_param_search
    now = datetime.now()
    date = str(now.year) + "_" + str(now.month) + "_" + str(now.day) 
    home_dir = os.path.expanduser("~")
    spn_dir = "AN_spindle"
    info_dir = "info"
    info_dir = os.path.join(spn_dir, info_dir)

    """
    #### FULL model random parameter search ####
    param_search = anmodel_rand_param_search.RANDOM_PARAM_SEARCH(model="FULL", set_name="FULL_model")
    param_search.multi_singleprocess(state="sws", cores=16)
    """

    #### FULL model random parameter search (change ion concentrations) ####
    info_file = date + "_ion.csv"
    info_path = os.path.join(home_dir, info_dir, info_file)
    info_df = pd.read_csv(info_path, header=None, index_col=0)
    model = info_df.loc["model"][1]
    model_name = info_df.loc["model_name"][1]
    cores = np.int(info_df.loc["cores"][1])
    concentrations = info_df.iloc[3:, ].astype(np.float).to_dict()[1]
    param_search = anmodel_rand_param_search.RANDOM_PARAM_SEARCH(model=model, set_name=model_name, 
                                                                 equil=True, concentrations=concentrations)
    param_search.multi_singleprocess(state="sws", cores=cores)


    """ 
    #### X_model ramdom paramter search ####
    info_file = date + ".csv
    info_path = os.path.join(home_dir, info_dir, info_file)
    info = pd.read_csv(info_path, header=None, index_col=0)
    model = info.loc["model"][1]
    model_name = info.loc["model_name"][1]
    channel = info.loc["channel_bool_list"].astype(np.int64).values
    cores = int(info.loc["cores"][1])
    param_search = anmodel_rand_param_search.RANDOM_PARAM_SEARCH(model=model, channel_bool=channel, set_name=model_name)
    param_search.multi_singleprocess(state="spn", cores=cores)
    """


class RandomParamSearch():
    """ Random parameter search.
    
    Generate parameter sets randomly, and pick up those which recapitulate a cirtain
    firing pattern. 

    Parameters
    ----------
    model : str
        model in which parameter search is conducted
    pattern : str
        searched firing pattern
    time : int or str
        how long to run parameter search (hour), default 48 (2 days)
    channel_bool : list (bool) or None
        Channel lists that X model contains. True means channels incorporated 
        in the model and False means not. The order of the list is the same 
        as other lists or dictionaries that contain channel information in 
        AN model. Example: \
        channel_bool = [
            1,  # leak channel
            0,  # voltage-gated sodium channel
            1,  # HH-type delayed rectifier potassium channel
            0,  # fast A-type potassium channel
            0,  # slowly inactivating potassium channel
            1,  # voltage-gated calcium channel
            1,  # calcium-dependent potassium channel
            1,  # persistent sodium channel
            0,  # inwardly rectifier potassium channel
            0,  # AMPA receptor
            0,  # NMDA receptor
            0,  # GABA receptor
            1,  # calcium pump
        ]\
        This is SAN model, default None
    model_name : str or None
        name of the X model, default None
    ion : bool
        whether you make equiribrium potential variable or not, 
        default False
    concentration : dictionary or str or None
        dictionary of ion concentration, or 'sleep'/'awake' that
        designate typical ion concentrations, default None

    Attributes
    ----------
    tools : object
        Keep attributes and helper functions needed for parameter search.
    pattern : str
        searched firing pattern
    time : int
        how long to run parameter search (hour)
    model_name : str
        model name
    model : object
        Simulation model object. See anmodel.models.py
    """
    def __init__(self, model, pattern='sws', time=48,
                 channel_bool=None, model_name=None, 
                 ion=False, concentration=None):
        self.tools = SearchTools()
        self.pattern = pattern
        self.time = int(time)
        if model == 'AN':
            self.model_name = 'AN'
            self.model = models.ANmodel(ion, concentration)
        if model == 'SAN':
            self.model_name = 'SAN'
            self.model = models.SANmodel(ion, concentration)
        if model == "X":
            self.model_name = model_name
            self.model = models.Xmodel(channel_bool, ion, concentrations)

    ## arguments (see also multi_singleprocess_SPN) :
    ##   core : 0~23 (if cores = 24) and written for distinguish the results calculate in the different cores.
    ##   cores : number of cores that are used for calculating 
    ##   now : datetime.datetime.now()
    ##   rand_seed : random seed for generating random parameters
    def search_singleprocess(self, args):
        """ Random SWS parameter search using single core.

        Parameters
        ----------
        args : list
            core : int
                n th core of designated number of cores
            now : 
                datetime.datetime.now()
        """
        state, core, now, rand_seed = args
        date = f'{now.year}_{now.month}_{now.year}'
        p = Path.cwd().parent
        res_p = p / 'result' / f'{self.pattern}_params' / f'{date}_{self.model_name}'
        res_p.mkdir(exist_ok=True)

        param_df = pd.DataFrame([])
        niter = 0  # number of iteration
        nhit = 0  # number of hits
        nosci = 0  # number of oscillation
        st = time()  # start time : updated every 1 hour
        st_abs = time()  # absolute start time : not updated (fixed)
        np.random.seed(rand_seed)

        while True:
            niter += 1
            new_params = pd.Series(self.model.set_rand_params())
            new_params = pd.DataFrame(new_params).T
            S, info  = self.model.run_odeint()
            if 'Excess work done on this call (perhaps wrong Dfun type).' == info['message']:
                pass
            else:  # spindle ---> return 1 , else ---> return 0
                if np.any(S[4999:9999, 0] == float("inf")) or np.any(S[4999:9999, 0] != S[4999:9999, 0]):
                    judge = "EXCLUDED"
                elif state == "spn":
                    judge = self.tools.check_spindle(v=S[4999:9999, 0]) 
                elif state == "sws":
                    judge = self.tools.check_sws(v=S[4999:9999, 0])

            if judge == "SPN" or judge == "SWS":  # 1: spindle
                print("hit!")
                nhit += 1
                param_df = pd.concat([param_df, new_params])
            else:
                nosci += 1
            
            ## save parameters every 1 hour 
            md = time()
            if (md - st) > 60 * 60:  # 1 hour
                st = time()  # update start time
                with open(res_path+"/"+state+"_"+date_core+"full.pickle", "wb") as f:
                    pickle.dump(niter, f)
                    pickle.dump(param_df, f)
                print(datetime.now(), "Core"+str(core)+": ", len(param_df), " "+state+" kamosirenai pickled.")
            
            ## finish random parameter search after "self.time" hours
            if (md - st_abs) > 60 * 60 * self.time:
                break

    ## arguments : 
    ##   cores : number of cores that are used for calculating 
    def multi_singleprocess(self, state, cores):
        args = []
        now = datetime.now()
        for core in range(cores):
            args.append((state, core, cores, now, np.random.randint(0, 2 ** 32 - 1, 1)))

        print('Random search: using ' + str(cores) +' cores to explore '+state+".")
        with Pool(processes=cores) as pool:
            result = pool.map(self.singleprocess, args)


class SearchTools():
    ## arguments: 
    ##   v: array of membrane potential (in most case: S[:, 0])
    ##
    ## return: 
    ##   peaktime: array of time of the peak membrane potential
    def cal_peaktime(self, s, start=4999, end=10000):
        maxid = signal.argrelmax(s[start:end, 0], order=1)
        peaktime = maxid[0]
        index = np.where(s[peaktime + start, 0] > -20)[0]
        peaktime = peaktime[index]
        return peaktime

    ## arguments: 
    ##   v: array of membrane potential (in most case: S[:, 0])
    ##   isi_thres_for_burst: if inter-spike interval (ms) is longer than this value, those spikes are regarded to be in the same burst
    ##   event_thres_for_burst: if spikes are less than this value, those spikes in the same events are not regarded as the burst
    ##
    ## return:
    ##   burst_index: array of time (or index) when membrane potential showing burst
    ##   ave_fire_in_burst: average number of fires in the single burst (type: scalar)
    ##   num_bursts: number of bursts
    def make_burst_index(self, v, isi_thres_for_burst=50, event_thres_for_burst=4): # extract indexes in burst phase
        peaktime = self.cal_peaktime(v)
        isi = np.diff(peaktime) # interspike interval
        grouped_events = np.split(peaktime, np.where(isi > isi_thres_for_burst)[0]+1)
        burst_events = [x for x in grouped_events if len(x) >= event_thres_for_burst]
        num_bursts = len(burst_events)
        if num_bursts == 0:
            ave_fire_in_burst = 0
            burst_index = []
            return burst_index, ave_fire_in_burst, num_bursts
        else:
            ave_fire_in_burst = len(np.concatenate(burst_events)) / num_bursts # sum_fire / num_burst_events
            intraburst_interval = [np.diff(x).mean() for x in burst_events] # average length of silent phase
            burst_index = []
            for j in range(len(burst_events)):
                index = [i for i in range(5001) if burst_events[j][0]-intraburst_interval[j] < i < burst_events[j][-1]+intraburst_interval[j]]
                burst_index = burst_index + index
            return burst_index, ave_fire_in_burst, num_bursts

    ## arguments:
    ##   v: array of membrane potential (in most case: S[:, 0])
    ##
    ## return:
    ##   v_square: binary array of membrane potential (burst phase: 1, silent phase: 0)
    def square_wave(self, v): # fitting to square wave
        burst_index = self.make_burst_index(v)[0]
        v_square = np.array([1 if i in burst_index else 0 for i in range(5001)]) 
        return v_square

    ## arguments:
    ##   v: array of membrane potential (in most case: S[:, 0])
    ##
    ## return: 
    ##   1: the firing pattern is regarded as spindle
    ##   0: the firing pattern is not regarded as spindle
    def check_spindle(self, v):
        detv = signal.detrend(v)
        f, Spw = periodogram(detv, fs=1000)
        maxamp = max(Spw)
        nummax = Spw.tolist().index(maxamp)
        maxfre = f[nummax]
        ave_fire_in_burst = self.make_burst_index(v)[1]
        numbursts = self.make_burst_index(v)[2]
        numfire = ave_fire_in_burst * numbursts

        v_square = self.square_wave(v)
        v_labeled = pd.DataFrame([v, v_square]).T
        v_groupby = v_labeled.groupby(1)
        
        if np.any(v_square==0) and np.any(v_square==1):
            silent_min = v_groupby.min().iloc[0] # min membrane potential in silent phase
            burst_min = v_groupby.min().iloc[1] # min membrane potential in burst phase
        else:
            burst_min = silent_min = v_groupby.min().iloc[0] # if no burst phase or silent phase

        '''
        ## calculate standard deviation of silent_phase length
        zero_diff = np.diff(np.where(v_square==0))
        silent_index = np.split(np.arange(5001), np.where(zero_diff>1)[1]+1)
        len_silent_phase = [len(silent_index[i]) for i in range(1, len(silent_index)-1)]
        normalized_len_silent_phase = np.array(len_silent_phase) / np.average(len_silent_phase)
        stdev = np.std(normalized_len_silent_phase)
        '''

        if (0.2 < maxfre < 10.2) & (numfire > 5*maxfre-1):
            if (ave_fire_in_burst >= 4) & (silent_min > burst_min).iloc[0]:
                return "SPN"
            else:
                return "noSPN"
        else:
            return "noSPN"

    #### state discrimination based on Tatsuki et al., 2016
    #### -> see also an_bifurcation.py, yan_bifurcation.py
    def calfire(self, v):
        burst = np.zeros(1)
        for i in range(4999):
            if (v[i]+20)*(v[i+1]+20) < 0:
                burst += 1
        return burst//2
    
    def check_sws(self, v):
        detv = signal.detrend(v)
        max_potential = max(detv)
        f, spw = periodogram(detv, fs=1000)
        maxamp = max(spw)  # highest amp freq
        nummax = spw.tolist().index(maxamp)  # get index of maxamp
        maxfre = f[nummax]
        numfire = self.calfire(v)

        if (200 < max_potential):
            return "EXCLUDED"
        elif (maxfre < 0.2) | (numfire < 5*2):
            return "RESTING"
        elif (0.2 < maxfre < 10.2) & (numfire > 5*maxfre-1):
            return "SWS"
        elif (0.2 < maxfre < 10.2) & (numfire <= 5*maxfre-1):
            return "SWS_FEW_SPIKES"
        elif (10.2 < maxfre):
            return "AWAKE"
        else:
            return "EXCLUDED"
