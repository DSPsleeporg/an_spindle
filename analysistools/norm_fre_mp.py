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

sys.path.append('../')
sys.path.append('../anmodel')

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from tqdm import tqdm
from typing import Dict, List, Iterator, Optional

import anmodel
import analysistools

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

    def norm_sws(self, param: pd.Series, samp_len: int=10) -> List[int]:
        self.model.set_params(param)
        s, _ = self.model.run_odeint(samp_freq=1000, samp_len=samp_len)
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
        try:
            st = sh[e[0]]
            en = sh[e[6]]
            return [st, en]
        except IndexError:
            self.norm_sws(param=param, samp_len=samp_len*2)

    def norm_spn(self, param: pd.Series, samp_len: int=10) -> List[int]:
        self.model.set_params(param)
        s, _ = self.model.run_odeint(samp_freq=1000, samp_len=samp_len)
        v: np.ndarray = s[5000:, 0]

        als = anmodel.analysis.FreqSpike()
        burstidx, _, _ = als.get_burstinfo(v, spike='bottom')
        
        e = []
        for lst in burstidx:
            e.append(lst[-1])
        try:
            st = e[0]
            en = e[6]
            return [st, en]
        except IndexError:
            self.norm_spn(param=param, samp_len=samp_len*2)


    def main(self, filename: str, wavepattern: str='SPN') -> pd.DataFrame:
        p: Path = Path.cwd().parents[0]
        data_p: Path = p / 'results' / f'{wavepattern}_params' / self.model_name
        res_p: Path = p / 'results' / 'normalization_mp_ca' / f'{wavepattern}_{self.model_name}.pickle'
        with open(data_p/filename, 'rb') as f:
            df = pickle.load(f)

        res_df = pd.DataFrame([], columns=['start', 'end'], index=range(len(df)))
        for i in tqdm(range(len(df))):
            param = df.iloc[i, :]
            if wavepattern == 'SWS':
                res_df.iloc[i, :] = self.norm_sws(param)
            elif wavepattern == 'SPN':
                res_df.iloc[i, :] = self.norm_spn(param)
            else:
                raise NameError(f'Wavepattern {wavepattern} is unvalid.')

            if i%10 == 0:
                with open(res_p, 'wb') as f:
                    pickle.dump(res_df, f)
                print(f'Now i={i}, and pickled')
        # return res_df

    
if __name__ == '__main__':
    arg: List = sys.argv
    model = arg[1]
    filename = arg[2]
    if model == 'X':
        channel_bool = [1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1]
        model_name = 'RAN'
        norm = analysistools.norm_fre_mp.Normalization(
            model, channel_bool, model_name
            )
    else:
        norm = analysistools.norm_fre_mp.Normalization(model)
    norm.main(filename)
