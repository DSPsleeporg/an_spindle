# -*- coding: utf-8 -*-

""" 
This is the module for normalizing the frequency of membrane potential. 
You normalize the frequency of burst firings (1st~6th burst firing) and 
plot normalized membrane potential, Ca, and so on. 
"""

__author__ = 'Tetsuya Yamada'
__status__ = 'Prepared'
__version__ = '1.0.0'
__date__ = '24 Aug 2020'


import os
import sys
"""
LIMIT THE NUMBER OF THREADS!
change local env variables BEFORE importing numpy
"""
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

sys.path.append('../')
sys.path.append('../anmodel')

from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import scipy.stats
import seaborn as sns
from tqdm import tqdm
from typing import Dict, List, Iterator, Optional

import anmodel
import analysistools


class SFA:
    def __init__(self, model: str='AN', wavepattern: str='SPN', 
                 channel_bool: Optional[Dict]=None, 
                 model_name: Optional[str]=None, 
                 ion: bool=False, concentration: Dict=None)-> None:
        """ Normalize the frequency of membrane potential.

        Parameters
        ----------
        model : str
            the type of model a simulation is conducted (ex. AN, SAN, X)
        wavepattern : str
            the collected wavepattrn (ex. SWS, SPN)
        channel_bool[Optional] : Dict
            when X model is selected, you need to choose channels by this
        model_name[Optional] : str
            when X model is selected, you need to designate the model name (ex. RAN)
        ion[Optional] : bool
            whther you take extracellular ion concentration into account
        concentration[Optional] : Dict
            when ion=True, you need to designate initial ion concentrations
        
        """
        self.model = model
        self.wavepattern = wavepattern
        if self.model == 'AN':
            self.model_name = 'AN'
            self.model = anmodel.models.ANmodel(ion, concentration)
        if self.model == 'SAN':
            self.model_name = 'SAN'
            self.model = anmodel.models.SANmodel(ion, concentration)
        if self.model == "X":
            if channel_bool is None:
                raise TypeError('Designate channel in argument of X model.')
            self.model_name = model_name
            self.model = anmodel.models.Xmodel(channel_bool, ion, concentration)

        self.fs = anmodel.analysis.FreqSpike(samp_freq=1000)

    def getinfo(self, v: np.ndarray):
        if self.wavepattern == 'SWS':
            burst_events, _, _, _ = self.fs.get_burstinfo(v, spike='peak')
        elif self.wavepattern == 'SPN':
            burst_events, _, _, _ = self.fs.get_burstinfo(v, spike='bottom')
        
        ratio_lst = []
        for _, burst in enumerate(burst_events):
            isi = np.diff(burst)
            isi_ratio = isi[-1] / isi[0]
            ratio_lst.append(isi_ratio)
        ratio_mean = np.mean(ratio_lst)
        return ratio_mean

    def main(self, filename: str):
        now: datetime = datetime.now()
        date: str = f'{now.year}_{now.month}_{now.day}'
        p: Path = Path.cwd().parents[0]
        res_p: Path = p / 'results' / 'spike_freq_adap' 
        res_p.mkdir(parents=True, exist_ok=True)
        save_p: Path = res_p / f'{date}_{self.model_name}_{self.wavepattern}.pickle'

        data_p: Path = p / 'results' / f'{self.wavepattern}_params' / self.model_name
        time_p: Path = p / 'results' / 'normalization_mp_ca' 
        with open(data_p/filename, 'rb') as f:
            param_df = pickle.load(f)
            param_df.index = range(len(param_df))
        with open(time_p/f'{self.wavepattern}_{self.model_name}_time.pickle', 'rb') as f:
            time_df = pickle.load(f).dropna(how='all')
            time_df.index = range(len(time_df))

        df: pd.DataFrame = pd.DataFrame(columns=['first_last_ratio'])
        for i in tqdm(range(len(time_df))):
            param = param_df.iloc[i, :]
            e = time_df.iloc[i, :]
            if e[0] == None:
                pass
            else:
                samp_len = 10 + ((5000+e[6])//10000) * 10
                self.model.set_params(param)
                s, _ = self.model.run_odeint(samp_freq=1000, samp_len=samp_len)
                v: np.ndarray = s[e[0]:e[6], 0]
                info = self.getinfo(v)
                df.loc[i] = info

        with open(save_p, 'wb') as f:
            pickle.dump(df, f)


if __name__ == '__main__':
    arg: List = sys.argv
    model = arg[1]
    wavepattern = arg[2]
    filename = arg[3]
    if model == 'RAN':
        channel_bool = [1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1]
        model_name = 'RAN'
        sfa = analysistools.spike_freq_adap.SFA(
            model='X', 
            wavepattern=wavepattern, 
            channel_bool=channel_bool, 
            model_name=model_name, 
        )
    else:
        sfa = analysistools.spike_freq_adap.SFA(
            model=model, 
            wavepattern=wavepattern, 
        )
    sfa.main(filename=filename)
    
