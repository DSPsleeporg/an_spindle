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
    def __init__(self, model: str='AN', wavepattern: str='SPN', 
                 channel_bool: Optional[Dict]=None, 
                 model_name: Optional[str]=None, 
                 ion: bool=False, concentration: Dict=None)-> None:
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
        if len(e) >= 7:
            return sh[e[:7]]
        else:
            self.norm_sws(param=param, samp_len=samp_len*2)

        del(ss)
        del(d)
        del(k)
        del(dia)

    def norm_spn(self, param: pd.Series, samp_len: int=10) -> List[int]:
        self.model.set_params(param)
        s, _ = self.model.run_odeint(samp_freq=1000, samp_len=samp_len)
        v: np.ndarray = s[5000:, 0]

        als = anmodel.analysis.FreqSpike()
        burstidx, _, _ = als.get_burstinfo(v, spike='bottom')
        
        e = []
        for lst in burstidx:
            e.append(lst[-1])
        if len(e) >= 7:
            return e[:7]
        else:
            self.norm_spn(param=param, samp_len=samp_len*2)


    def time(self, filename: str) -> None:
        p: Path = Path.cwd().parents[0]
        data_p: Path = p / 'results' / f'{self.wavepattern}_params' / self.model_name
        res_p: Path = p / 'results' / 'normalization_mp_ca' / f'{self.wavepattern}_{self.model_name}_time.pickle'
        with open(data_p/filename, 'rb') as f:
            df = pickle.load(f)

        res_df = pd.DataFrame([], columns=range(7), index=range(len(df)))
        for i in tqdm(range(len(df))):
            param = df.iloc[i, :]
            if self.wavepattern == 'SWS':
                res_df.iloc[i, :] = self.norm_sws(param)
            elif self.wavepattern == 'SPN':
                res_df.iloc[i, :] = self.norm_spn(param)
            else:
                raise NameError(f'Wavepattern {self.wavepattern} is unvalid.')

            if i%10 == 0:
                with open(res_p, 'wb') as f:
                    pickle.dump(res_df, f)
                print(f'Now i={i}, and pickled')
        with open(res_p, 'wb') as f:
            pickle.dump(res_df, f)

    def mp(self, filename: str) -> None:
        p: Path = Path.cwd().parents[0]
        data_p: Path = p / 'results' / f'{self.wavepattern}_params' / self.model_name
        time_p: Path = p / 'results' / 'normalization_mp_ca' / f'{self.wavepattern}_{self.model_name}_time.pickle'
        with open(data_p/filename, 'rb') as f:
            param_df = pickle.load(f)
        with open(time_p, 'rb') as f:
            time_df = pickle.load(f)            
        
        hm_df = pd.DataFrame([], columns=range(48), index=range(len(time_p)))
        for i in range(len(time_p)):
            param = param_df.iloc[i, :]
            self.model.set_params(param)
            s, _ = self.model.run_odeint(samp_freq=1000, samp_len=samp_len)
            v: np.ndarray = s[5000:, 0]

            e = time_df.iloc[i, :]
            v_norm = []
            for j in range(len(e)-1):
                tlst = np.linspace(e[j], e[j+1], 9, dtype=int)
                for k in range(len(tlst)-1):
                    v_norm.append(v[tlst[k]:tlst[k+1]].var(ddof=0))
            hm_df.iloc[i, :] = v_norm / max(v_norm)
    

if __name__ == '__main__':
    arg: List = sys.argv
    model = arg[1]
    wavepattern = arg[2]
    filename = arg[3]
    if model == 'X':
        channel_bool = [1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1]
        model_name = 'RAN'
        norm = analysistools.norm_fre_mp.Normalization(
            model, wavepattern, channel_bool, model_name
            )
    else:
        norm = analysistools.norm_fre_mp.Normalization(model, wavepattern)
    norm.time(filename)
