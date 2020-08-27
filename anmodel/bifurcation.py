# -*- coding: utf-8 -*-

"""
This is a module for bifurcation analysis. 
"""

__author__ = 'Tetsuya Yamada'
__status__ = 'Editing'
__version__ = '1.1.0'
__date__ = '26 August 2020'


import os
import sys
sys.path.append('../')
sys.path.append('../anmodel')

from copy import copy
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from scipy.stats import gaussian_kde
from tqdm import tqdm
from typing import Dict, List, Iterator, Optional

import anmodel


class Bifurcation:
    def __init__(self, channel: str, magnif: float, 
                 model: str='AN', wavepattern: str='SWS', 
                 channel_bool: Optional[Dict]=None, 
                 model_name: Optional[str]=None, 
                 ion: bool=False, concentration: Dict=None) -> None:
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

        self.samp_freq = 1000
        self.channel = channel
        self.magnif = magnif
        self.wc = anmodel.analysis.WaveCheck()
        self.fs = anmodel.analysis.FreqSpike(samp_freq=self.samp_freq)

    def getinfo(self, v: np.ndarray) -> List[float]:
        if self.wavepattern == 'SWS':
            nspike, _ = self.fs.get_spikeinfo(v)
            sq: np.ndarray = self.fs.square_wave(v, spike='peak')
        elif self.wavepattern == 'SPN':
            nspike, _ = self.fs.get_ahpinfo(v)
            sq: np.ndarray = self.fs.square_wave(v, spike='bottom')
        lenburst: int = len(np.where(sq==1)[0])
        lensilent: int = len(np.where(sq==0)[0])
        reslst: List[float] = [nspike, 
                               nspike / 6, 
                               lenburst, 
                               lenburst / 6, 
                               lensilent, 
                               lensilent / 6
                              ]
        return reslst

    def main(self, filename: str):
        now: datetime = datetime.now()
        date: str = f'{now.year}_{now.month}_{now.day}'
        p: Path = Path.cwd().parents[0]
        res_p: Path = p / 'results' / 'bifurcation' / f'{self.model_name}_{self.wavepattern}'
        res_p.mkdir(parents=True, exist_ok=True)
        save_p: Path = res_p / f'{date}_{self.channel}_{self.magnif}.pickle'

        data_p: Path = p / 'results' / f'{self.wavepattern}_params' / self.model_name
        time_p: Path = p / 'results' / 'normalization_mp_ca' 
        with open(data_p/filename, 'rb') as f:
            param_df = pickle.load(f)
        with open(time_p/f'{self.wavepattern}_{self.model_name}_time.pickle', 'rb') as f:
            time_df = pickle.load(f)  

        data: List = ['nspike', 
                      'average_spike_per_burst', 
                      'lenburst', 
                      'average_length_of_burst', 
                      'lensilent', 
                      'average_length_of_silent'
                      ]
        df: pd.DataFrame = pd.DataFrame(columns=data)
        for i in tqdm(range(len(time_df))):
            param: pd.Series = copy(param_df.iloc[i, :])
            param[self.channel] = param[self.channel] * self.magnif
            e = time_df.iloc[i, :]
            if e[0] == None:
                pass
            else:
                samp_len = 10 + ((5000+e[6])//10000) * 10
                self.model.set_params(param)
                s, _ = self.model.run_odeint(samp_freq=1000, samp_len=samp_len)
                v: np.ndarray = s[e[0]:e[6], 0]
                infolst = self.getinfo(v)
                df.loc[i] = infolst
        with open(save_p, 'wb') as f:
            pickle.dump(df, f)


if __name__ == '__main__':
    arg: List = sys.argv
    channel = arg[1]
    magnif = float(arg[2])
    model = arg[3]
    wavepattern = arg[4]
    filename = arg[5]
    if model == 'X':
        channel_bool = [1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1]
        model_name = 'RAN'
        bfc = anmodel.bifurcation.Bifurcation(
            channel=channel, 
            magnif=magnif, 
            model=model, 
            wavepattern=wavepattern, 
            channel_bool=channel_bool, 
            model_name=model_name, 
        )
    else:
        bfc = anmodel.bifurcation.Bifurcation(
            channel=channel,
            magnif=magnif, 
            model=model, 
            wavepattern=wavepattern, 
        )
    bfc.main(filename=filename)
