# -*- coding: utf-8 -*-

""" 

"""

__author__ = 'Fumiya Tatsuki, Kensuke Yoshida, Tetsuya Yamada, \
              Takahiro Katsumata, Shoi Shi, Hiroki R. Ueda'
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

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from typing import Dict, List, Iterator, Optional

import anmodel

class Normalization:
    def __init__(self, model: str='AN', channel_bool: Optional[Dict]=None, 
                 model_name: Optional[str]=None, 
                 ion: bool=False, concentration: Dict=None)-> None:
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

    def norm(self, param: pd.Series) -> List[int]:
        self.model.set_params(param)
        s, _ = self.model.run_odeint(samp_freq=1000)
        v: np.ndarray = s[5000:, 0]
        vmax: np.float64 = v.max()
        vmin: np.float64 = v.min()
        vrange: np.float64 = vmax - vmin

        ss: List[List] = []
        for i in range(int(vrange)):
            si: List[int] = []
            for j in range(len(v)-1):
                if (v[j]-(vmax-i))*(v[j+1]-(vmax-i))<0:
                    si.append(j)
            ss.append(si)

        d: List[int] = []
        for i, si in enumerate(ss):
            dd = []
            for j in range(len(si)-1):
                dd.append(si[j+1]-si[j])
            if len(dd) == 0:
                d.append(None)
            else:
                d.append(max(dd))

        k: List[int] = []
        maxlen = 0
        for i, si in enumerate(ss):
            if len(si) > maxlen:
                k = [i]
                maxlen = len(si)
            elif len(si) == maxlen:
                k.append(i)
            else:
                pass

        dia: List[int] = []
        for i in k:
            dia.append(d[i])
        h: List[int] = []
        for k in k:
            if d[k] is None:
                pass
            if d[k] == min(dia):
                h.append(k)

        dh = d[h[0]]
        sh = ss[h[0]]
        e = []
        for j in range(len(sh)-1):
            if sh[j+1]-sh[j] >= 0.5 * dh:
                e.append(j)
        st = sh[e[0]]
        en = sh[e[6]]
        return [st, en]

    def main(self, filename: str, wavepattern: str='SPN') -> pd.DataFrame:
        p: Path = Path.cwd().parents[0]
        data_p: Path = p / 'results' / f'{wavepattern}_params' / self.model_name
        with open(data_p/filename, 'rb') as f:
            df = pickle.load(f)
            
        res_df = pd.DataFrame([], columns=['start', 'end'], index=range(len(df)))
        for i in range(len(df)):
            param = df.iloc[i, :]
            res_df.iloc[i, :] = self.norm(param)
        return res_df