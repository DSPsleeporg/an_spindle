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

from copy import copy
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


class Normalization:
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

    def norm_yoshida(self, param: pd.Series, samp_len: int=10) -> List[int]:
        """ Normalize frequency of burst firing in SWS firing pattern.

        Parameters
        ----------
        param : pd.Series or Dict
            single parameter set
        samp_len : int
            sampling time length (sec) (usually 10)
        
        Returns
        ----------
        List[int]
            the index (time (ms)) of the 1st~6th ends of burst firing
        
        Notes
        ----------
            this algorithm is same as Yoshida et al., 2018
        """
        self.model.set_params(param)
        s, _ = self.model.run_odeint(samp_freq=1000, samp_len=samp_len)
        v: np.ndarray = s[5000:, 0]
        del(s)
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
            return [sh[i] for i in range(7)]
        else:
            if samp_len <=20:
                self.norm_sws(param=param, samp_len=samp_len+10)
            else:
                return [None] * 7

    def norm_sws(self, param: pd.Series, 
                 gl: float=None, gl_name: str = None, 
                 samp_len: int=10) -> List[int]:
        """ Normalize frequency of burst firing in SWS firing pattern.

        Parameters
        ----------
        param : pd.Series or Dict
            single parameter set
        gl : float [Optional]
            leak channel (na/k) conductance for bifurcation analysis
        gl_name : str [Optional]
            na / k
        samp_len : int [Optional]
            sampling time length (sec) (usually 10)
        
        Returns
        ----------
        List[int]
            the index (time (ms)) of the 1st~6th ends of burst firing
        """
        self.model.set_params(param)
        if gl_name == 'k':
            self.model.leak.set_gk(gl)
        elif gl_name == 'na':
            self.model.leak.set_gna(gl)

        s, _ = self.model.run_odeint(samp_freq=1000, samp_len=samp_len)
        v: np.ndarray = s[5000:, 0]
        del(s)

        als = anmodel.analysis.FreqSpike()
        _, burstidx, _, _ = als.get_burstinfo(v, spike='peak')

        e = []
        for lst in burstidx:
            e.append(lst[-1])
        if len(e) >= 7:
            return e[:7]
        else:
            if samp_len <= 20:
                self.norm_spn(param=param, samp_len=samp_len+10)
            else:
                return [None] * 7

    def norm_spn(self, param: pd.Series, 
                 gl: float=None, gl_name: str = None, 
                 samp_len: int=10) -> List[int]:
        """ Normalize frequency of burst firing in SPN firing pattern.

        Parameters
        ----------
        param : pd.Series or Dict
            single parameter set
        gl : float [Optional]
            leak channel (na/k) conductance for bifurcation analysis
        gl_name : str [Optional]
            na / k
        samp_len : int [Optional]
            sampling time length (sec) (usually 10)
        
        Returns
        ----------
        List[int]
            the index (time (ms)) of the 1st~6th ends of burst firing
        """
        self.model.set_params(param)
        if gl_name == 'k':
            self.model.leak.set_gk(gl)
        elif gl_name == 'na':
            self.model.leak.set_gna(gl)

        s, _ = self.model.run_odeint(samp_freq=1000, samp_len=samp_len)
        v: np.ndarray = s[5000:, 0]
        del(s)

        als = anmodel.analysis.FreqSpike()
        _, burstidx, _, _ = als.get_burstinfo(v, spike='bottom')
        
        e = []
        for lst in burstidx:
            e.append(lst[-1])
        if len(e) >= 7:
            return e[:7]
        else:
            if samp_len <= 20:
                self.norm_spn(param=param, samp_len=samp_len+10)
            else:
                return [None] * 7

    def time(self, filename: str) -> None:
        """ Calculate time points for 1st~6th burst firing for all parameter sets.

        Parameters
        ----------
        filename : str
            the name of file in which parameter sets are contained
        """
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

    def mp_ca(self, filename: str) -> None:
        """ Calculate normalized mp and ca for plotting heatmap.

        Parameters
        ----------
        filename : str
            the name of file in which parameter sets are contained
        """
        p: Path = Path.cwd().parents[0]
        data_p: Path = p / 'results' / f'{self.wavepattern}_params' / self.model_name
        res_p: Path = p / 'results' / 'normalization_mp_ca'
        with open(data_p/filename, 'rb') as f:
            param_df = pickle.load(f)
        with open(res_p/f'{self.wavepattern}_{self.model_name}_time.pickle', 'rb') as f:
            time_df = pickle.load(f)
        
        hm_df = pd.DataFrame([], columns=range(48), index=range(len(time_df)))
        hm_ca_df = pd.DataFrame([], columns=range(48), index=range(len(time_df)))
        for i in tqdm(range(len(time_df))):
            param = param_df.iloc[i, :]
            e = time_df.iloc[i, :]
            if e[0] == None:
                pass
            else:
                samp_len = 10 + ((5000+e[6])//10000) * 10
                self.model.set_params(param)
                s, _ = self.model.run_odeint(samp_freq=1000, samp_len=samp_len)
                v: np.ndarray = scipy.stats.zscore(s[5000:, 0])
                ca: np.ndarray = scipy.stats.zscore(s[5000:, -1])

                v_norm = []
                ca_norm = []
                for j in range(len(e)-1):
                    tlst = np.linspace(e[j], e[j+1], 9, dtype=int)
                    for k in range(len(tlst)-1):
                        v_norm.append(v[tlst[k]:tlst[k+1]].var(ddof=0))
                        # v_norm.append(v[tlst[k]:tlst[k+1]].std(ddof=0))
                        ca_norm.append(ca[tlst[k]:tlst[k+1]].mean())
                hm_df.iloc[i, :] = v_norm
                hm_ca_df.iloc[i, :] = ca_norm

        with open(res_p/f'{self.wavepattern}_{self.model_name}_mp.pickle', 'wb') as f:
            pickle.dump(hm_df, f)
        with open(res_p/f'{self.wavepattern}_{self.model_name}_ca.pickle', 'wb') as f:
            pickle.dump(hm_ca_df, f)
        
        plt.figure(figsize=(20, 20))
        sns.heatmap(hm_df.values.tolist(), cmap='jet')
        plt.savefig(res_p/f'{self.wavepattern}_{self.model_name}_mp_hm.png')

        plt.figure(figsize=(20, 20))
        sns.heatmap(hm_ca_df.values.tolist(), cmap='jet')
        plt.savefig(res_p/f'{self.wavepattern}_{self.model_name}_ca_hm.png')

    def time_bifurcation_rep(self, filename: str, channel: str) -> None:
        """ Calculate time points for 1st~6th burst firing for 
        the representative parameter set through bifurcation.

        Parameters
        ----------
        filename : str
            the name of file in which parameter sets are contained
        """
        p: Path = Path.cwd().parents[0]
        data_p: Path = p / 'results' / f'{self.wavepattern}_params' / self.model_name
        resname: str = f'{self.wavepattern}_{self.model_name}_{channel}_time.pickle'
        res_p: Path = p / 'results' / 'normalization_mp_ca' / 'bifurcation_rep' / resname
        with open(data_p/filename, 'rb') as f:
            param = pickle.load(f)

        res_df = pd.DataFrame([], columns=range(7), index=np.arange(900, 1101))
        for i in tqdm(res_df.index):
            if channel != 'g_kleak' and channel != 'g_naleak':
                p = copy(param)
                p[channel] = p[channel] * i / 1000
                g = None
                gl_name = None
            elif channel == 'g_kleak':
                self.model.leak.set_div()
                g_kl = self.model.leak.gkl
                g = copy(g_kl)
                g = g * i / 1000
                gl_name = 'k'
            elif channel == 'g_naleak':
                self.model.leak.set_div()
                g_nal = self.model.leak.gnal
                g = copy(g_nal)
                g = g * i / 1000
                gl_name = 'na'

            if self.wavepattern == 'SWS':
                res_df.loc[i, :] = self.norm_sws(p, g, gl_name)
            elif self.wavepattern == 'SPN':
                res_df.loc[i, :] = self.norm_spn(p, g, gl_name)
            else:
                raise NameError(f'Wavepattern {self.wavepattern} is unvalid.')

            if i%10 == 0:
                with open(res_p, 'wb') as f:
                    pickle.dump(res_df, f)
                print(f'Now i={i}, and pickled')

        with open(res_p, 'wb') as f:
            pickle.dump(res_df, f)

    def time_bifurcation_all(self, filename: str, 
                             channel: str, magnif: float) -> None:
        """ Calculate time points for 1st~6th burst firing for 
        the representative parameter set through bifurcation.

        Parameters
        ----------
        filename : str
            the name of file in which parameter sets are contained
        """
        p: Path = Path.cwd().parents[0]
        data_p: Path = p / 'results' / f'{self.wavepattern}_params' / self.model_name
        resname: str = f'{self.wavepattern}_{self.model_name}_{channel}_{magnif}_time.pickle'
        res_p: Path = p / 'results' / 'normalization_mp_ca' / 'bifurcation_all' / resname
        with open(data_p/filename, 'rb') as f:
            df = pickle.load(f)

        res_df = pd.DataFrame([], columns=range(7), index=range(len(df)))
        for i in tqdm(range(len(df))):
            param = df.iloc[i, :]
            if channel != 'g_kleak' and channel != 'g_naleak':
                p = copy(param)
                p[channel] = p[channel] * magnif
                g = None
                gl_name = None
            elif channel == 'g_kleak':
                self.model.leak.set_div()
                g_kl = self.model.leak.gkl
                g = copy(g_kl)
                g = g * magnif
                gl_name = 'k'
            elif channel == 'g_naleak':
                self.model.leak.set_div()
                g_nal = self.model.leak.gnal
                g = copy(g_nal)
                g = g * magnif
                gl_name = 'na'

            if self.wavepattern == 'SWS':
                res_df.loc[i, :] = self.norm_sws(p, g, gl_name)
            elif self.wavepattern == 'SPN':
                res_df.loc[i, :] = self.norm_spn(p, g, gl_name)
            else:
                raise NameError(f'Wavepattern {self.wavepattern} is unvalid.')

            if i%10 == 0:
                with open(res_p, 'wb') as f:
                    pickle.dump(res_df, f)
                print(f'Now i={i}, and pickled')
        with open(res_p, 'wb') as f:
            pickle.dump(res_df, f)


if __name__ == '__main__':
    arg: List = sys.argv
    model = arg[1]
    wavepattern = arg[2]
    filename = arg[3]
    method = arg[4]
    if model == 'RAN':
        channel_bool = [1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1]
        model_name = 'RAN'
        norm = analysistools.norm_fre_mp.Normalization(
            'X', wavepattern, channel_bool, model_name
            )
    elif model == 'Model2':
        channel_bool = [1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1]
        model_name = 'Model2'
        norm = analysistools.norm_fre_mp.Normalization(
            'X', wavepattern, channel_bool, model_name
            )
    else:
        norm = analysistools.norm_fre_mp.Normalization(model, wavepattern)
    
    if method == 'time':
        norm.time(filename)
    elif method == 'mp_ca':
        norm.mp_ca(filename)
    elif method == 'time_bifurcation_rep':
        channel = arg[5]
        norm.time_bifurcation_rep(filename, channel)
    elif method == 'time_bifurcation_all':
        channel = arg[5]
        magnif = float(arg[6])
        norm.time_bifurcation_all(filename, channel, magnif)
