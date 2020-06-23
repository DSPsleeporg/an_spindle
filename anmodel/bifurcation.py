# -*- coding: utf-8 -*-

"""
This is a module for bifurcation analysis. 
"""

__author__ = 'Tetsuya Yamada'
__status__ = 'Editing'
__version__ = '1.0.0'
__date__ = '31 May 2020'


import os
import sys
sys.path.append('../')

from copy import copy
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from scipy.stats import gaussian_kde
from typing import Dict, List, Iterator, Optional

import anmodel


class DetailAnalysis:
    def __init__(self, channel: str, magnif: float, 
                 model: str='AN', 
                 channel_bool: Optional[Dict]=None, 
                 model_name: Optional[str]=None, 
                 ion: bool=False, concentration: Dict=None) -> None:
        self.model = model
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

        self.samp_freq = 10000
        self.channel = channel
        self.magnif = magnif
        self.wc = anmodel.analysis.WaveCheck()
        self.fs = anmodel.analysis.FreqSpike(samp_freq=self.samp_freq)

    def main(self, param_df):
        now: datetime = datetime.now()
        date: str = f'{now.year}_{now.month}_{now.day}'
        p: Path = Path.cwd().parents[0]
        res_p: Path = p / 'results' / 'bifurcation' / f'{self.model_name}'
        res_p.mkdir(parents=True, exist_ok=True)
        save_p: Path = res_p / f'{date}_{self.channel}_{self.magnif}.pickle'

        data: List = ['nspike', 'aspb', 'numburst', 'lenburst', 'lensilent']
        df: pd.DataFrame = pd.DataFrame(columns=data)
        for i in range(len(param_df)):
            param: pd.Series = copy(param_df.iloc[i, :])
            param[self.channel] = param[self.channel] * self.magnif
            self.model.set_params(param)
            s, _ = self.model.run_odeint(samp_freq=self.samp_freq)
            v: np.ndarray = s[int(5000*self.samp_freq/1000):, 0]
            nspike, _ = self.fs.get_spikeinfo(v)
            _, aspb, num_burst = self.fs.get_burstinfo(v)
            sq: np.ndarray = self.fs.square_wave(v)
            lenburst: int = len(np.where(sq==1)[0])
            lensilent: int = len(np.where(sq==0)[0])
            df.loc[i] = [nspike, aspb, num_burst, lenburst, lensilent]
        with open(save_p, 'wb') as f:
            pickle.dump(df, f)


if __name__ == '__main__':
    arg: List = sys.argv
    magnif = float(arg[1])
    als = DetailAnalysis(
        channel='g_kvsi',
        magnif=magnif,
        model='AN',
    )
    with open('../results/previous_params/SPN_param_matome_processed.pickle', 'rb') as f:
        df = pickle.load(f)
    als.main(param_df=df)
