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
from multiprocessing import Pool
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
    def __init__(self, model: str='AN', wavepattern: str=None, 
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
        elif self.model == 'SAN':
            self.model_name = 'SAN'
            self.model = anmodel.models.SANmodel(ion, concentration)
        elif self.model == 'RAN':
            self.model_name = 'RAN'
            ran_bool = [1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1]
            self.model = anmodel.models.Xmodel(channel_bool=ran_bool, ion=ion, concentration=concentration)
        elif self.model == "X":
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
                self.norm_sws(param=param, channel=channel, samp_len=samp_len+10)
            else:
                return [None] * 7

    def norm_sws(self, param: pd.Series, channel=None, channel2=None, 
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
        if channel is not None:
            self.model.set_params(param.drop(['g_kl', 'g_nal']))
            if channel != 'g_nal' and channel != 'g_kl' and channel2 != 'g_nal' and channel2 != 'g_kl':
                self.model.leak.reset_div()
            else:
                self.model.leak.set_gk(param['g_kl'])
                self.model.leak.set_gna(param['g_nal'])
        else:
            self.model.set_params(param)

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
                self.norm_sws(param=param, channel=channel, samp_len=samp_len+10)
            else:
                return [None] * 7

    def norm_spn(self, param: pd.Series, channel=None, channel2=None, 
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
        if 'g_nal' in param.index or 'g_kl' in param.index:
            self.model.set_params(param.drop(['g_kl', 'g_nal']))
        if channel != 'g_nal' and channel != 'g_kl' and channel2 != 'g_nal' and channel2 != 'g_kl':
            self.model.leak.reset_div()
            self.model.set_params(param)
        else:
            self.model.leak.set_div()
            self.model.leak.set_gk(param['g_kl'])
            self.model.leak.set_gna(param['g_nal'])

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
                self.norm_spn(param=param, channel=channel, channel2=channel2, samp_len=samp_len+10)
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
        # res_p: Path = p / 'results' / 'normalization_mp_ca' / f'{self.wavepattern}_{self.model_name}_time.pickle'
        res_p: Path = p / 'results' / 'normalization_mp_ca' / f'{filename}_time.pickle'
        with open(data_p/filename, 'rb') as f:
            df = pickle.load(f)
            df.index = range(len(df))

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
            param_df.index = range(len(param_df))
        with open(res_p/f'{self.wavepattern}_{self.model_name}_time.pickle', 'rb') as f:
            time_df = pickle.load(f).dropna(how='all')
            time_df.index = range(len(time_df))
        
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

        with open(res_p/f'{filename}_mp.pickle', 'wb') as f:
            pickle.dump(hm_df, f)
        with open(res_p/f'{filename}_ca.pickle', 'wb') as f:
            pickle.dump(hm_ca_df, f)
        
        plt.figure(figsize=(20, 20))
        sns.heatmap(hm_df.values.tolist(), cmap='jet')
        plt.savefig(res_p/f'{self.wavepattern}_{self.model_name}_mp_hm.png')

        plt.figure(figsize=(20, 20))
        sns.heatmap(hm_ca_df.values.tolist(), cmap='jet')
        plt.savefig(res_p/f'{self.wavepattern}_{self.model_name}_ca_hm.png')

    def time_bifurcation_rep(self, filename: str, channel: str, diff: int=100) -> None:
        """ Calculate time points for 1st~6th burst firing for 
        the representative parameter set through bifurcation.

        Parameters
        ----------
        filename : str
            the name of file in which parameter sets are contained
        """
        p: Path = Path.cwd().parents[0]
        data_p: Path = p / 'results' / f'{self.wavepattern}_params' / self.model_name
        resname: str = f'{filename}_{channel}_{diff}_time.pickle'
        res_p: Path = p / 'results' / 'normalization_mp_ca' / 'bifurcation_rep' / f'{self.model_name}'
        res_p.mkdir(parents=True, exist_ok=True)
        with open(data_p/filename, 'rb') as f:
            param = pickle.load(f)
        self.model.set_params(param)
        if channel == 'g_nal' or channel == 'g_kl':
            self.model.leak.set_div()
            param.loc['g_nal'] = self.model.leak.gnal
            param.loc['g_kl'] = self.model.leak.gkl

        start = 1000 - diff
        end = 1000 + diff + 1
        res_df = pd.DataFrame([], columns=range(7), index=np.arange(start, end))
        for i in tqdm(res_df.index):
            param_c = copy(param)
            param_c[channel] = param_c[channel] * i / 1000
            if self.wavepattern == 'SWS':
                res_df.loc[i, :] = self.norm_sws(param_c, channel)
            elif self.wavepattern == 'SPN':
                res_df.loc[i, :] = self.norm_spn(param_c, channel)
            else:
                raise NameError(f'Wavepattern {self.wavepattern} is unvalid.')

            if i%10 == 0:
                with open(res_p/resname, 'wb') as f:
                    pickle.dump(res_df, f)
                print(f'Now i={i}, and pickled')

        with open(res_p/resname, 'wb') as f:
            pickle.dump(res_df, f)
    
    def two_bifur_singleprocess(self, args) -> None:
        core, param_lst, r_df, channel1, channel2, res_p, resname = args
        for p_lst in param_lst:
            m, param_c = p_lst
            r_name = f'{resname}_{m}.pickle'
            for i in tqdm(r_df.columns):
                param_cc = copy(param_c)
                # param_cc[f'g_{channel2}'] = param_cc[f'g_{channel2}'] * i/1000
                param_cc[channel2] = param_cc[channel2] * i/1000
                if self.wavepattern == 'SWS':
                    # try:
                    #     r_df.loc[m, i] = 1000 / np.diff(self.norm_sws(param_cc, channel2, channel1)).mean()
                    # except:
                    #     r_df.loc[m, i] = None
                    pass
                elif self.wavepattern == 'SPN':
                    try:
                        r_df.loc[m, i] = 1000 / np.diff(self.norm_spn(param=param_cc, channel=channel1, channel2=channel2)).mean()
                    except:
                        r_df.loc[m, i] = None
                else:
                    raise NameError(f'Wavepattern {self.wavepattern} is unvalid.')

            with open(res_p/r_name, 'wb') as f:
                pickle.dump(r_df, f)

    def two_bifur_multi_singleprocess(self, ncore, filename, channel1, channel2, diff, interval):
        p: Path = Path.cwd().parents[0]
        data_p: Path = p / 'results' / f'{self.wavepattern}_params' / self.model_name
        res_p: Path = p / 'results' / 'normalization_mp_ca' / 'two_bifurcation' / f'{self.model_name}' / f'{channel1}_{channel2}'
        channel1 = 'g_' + channel1
        channel2 = 'g_' + channel2
        res_p.mkdir(parents=True, exist_ok=True)
        with open(data_p/filename, 'rb') as f:
            param = pickle.load(f)
        self.model.set_params(param)
        self.model.leak.set_div()
        param.loc['g_nal'] = self.model.leak.gnal
        param.loc['g_kl'] = self.model.leak.gkl

        start = 1000 - diff
        end = 1000 + diff + 1
        # index: channel_1, columns: channel_2
        magnif_lst = np.arange(start, end, interval)
        res_df = pd.DataFrame(index=magnif_lst, columns=magnif_lst)
        resname = f'{filename}_{diff}'

        args: List = []
        for core, m_lst in enumerate(np.array_split(magnif_lst, ncore)):
            param_lst = []
            for m in m_lst:
                param_c = copy(param)
                # param_c[f'g_{channel1}'] = param_c[f'g_{channel1}'] * m/1000
                param_c[channel1] = param_c[channel1] * m/1000
                param_lst.append([m, param_c])
            r_df = res_df.loc[m_lst, :]
            args.append((core, param_lst, r_df, channel1, channel2, res_p, resname))
        with Pool(processes=ncore) as pool:
            pool.map(self.two_bifur_singleprocess, args)

    def load_two_bifur(self, filename, ch1, ch2, diff, interval):
        p: Path = Path.cwd().parents[0]
        res_p: Path = p / 'results' / 'normalization_mp_ca' / 'two_bifurcation' / f'{self.model_name}' / f'{ch1}_{ch2}'
        start = 1000 - diff
        end = 1000 + diff + 1
        magnif_lst = np.arange(start, end, interval)
        self.res_df = pd.DataFrame(index=magnif_lst, columns=magnif_lst)
        for m in magnif_lst:
            resname = f'{filename}_{diff}_{m}.pickle'
            with open(res_p/resname, 'rb') as f:
                self.res_df.loc[m, :] = pickle.load(f).iloc[0, :]

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
        resname: str = f'{filename}_{channel}_{magnif}_time.pickle'
        res_p: Path = p / 'results' / 'normalization_mp_ca' / 'bifurcation_all' / f'{self.model_name}'
        res_p.mkdir(parents=True, exist_ok=True)
        with open(data_p/filename, 'rb') as f:
            df = pickle.load(f)

        res_df = pd.DataFrame([], columns=range(7), index=range(len(df)))
        for i in tqdm(range(len(df))):
            param = df.iloc[i, :]
            self.model.set_params(param)
            self.model.leak.set_div()
            param.loc['g_nal'] = self.model.leak.gnal
            param.loc['g_kl'] = self.model.leak.gkl
            param_c = copy(param)
            param_c[channel] = param_c[channel] * magnif
            if self.wavepattern == 'SWS':
                res_df.loc[i, :] = self.norm_sws(param_c, channel)
            elif self.wavepattern == 'SPN':
                res_df.loc[i, :] = self.norm_spn(param_c, channel)
            else:
                raise NameError(f'Wavepattern {self.wavepattern} is unvalid.')

            if i%10 == 0:
                with open(res_p/resname, 'wb') as f:
                    pickle.dump(res_df, f)
                print(f'Now i={i}, and pickled')
        with open(res_p/resname, 'wb') as f:
            pickle.dump(res_df, f)

    def load_time_bifurcation_all(self, dataname: str, diff: float=0.025):
        p: Path = Path.cwd().parents[0]
        res_p = p / 'results' / 'normalization_mp_ca' / 'bifurcation_all' / f'{self.model_name}'
        magnif_u = 1 + diff
        magnif_d = 1 - diff

        if self.wavepattern == 'SWS':
            with open(res_p/f'{dataname}_g_kvhh_1.0_time.pickle', 'rb') as f:
                self.norm_t = pickle.load(f)
                self.norm_fr = 1000 / self.norm_t.dropna().diff(axis=1).mean(axis=1)
        elif self.wavepattern == 'SPN':
            with open(res_p/f'{dataname}_g_kvsi_1.0_time.pickle', 'rb') as f:
                self.norm_t = pickle.load(f)
                self.norm_fr = 1000 / self.norm_t.dropna().diff(axis=1).mean(axis=1)

        with open(res_p/f'{dataname}_g_kleak_{magnif_u}_time.pickle', 'rb') as f:
            self.kl_t_u = pickle.load(f)
            self.kl_fr_u = 1000 / self.kl_t_u.dropna().diff(axis=1).mean(axis=1) / self.norm_fr
        with open(res_p/f'{dataname}_g_kleak_{magnif_d}_time.pickle', 'rb') as f:
            self.kl_t_d = pickle.load(f)
            self.kl_fr_d = 1000 / self.kl_t_d.dropna().diff(axis=1).mean(axis=1) / self.norm_fr
        with open(res_p/f'{dataname}_g_kvsi_{magnif_u}_time.pickle', 'rb') as f:
            self.kvsi_t_u = pickle.load(f)
            self.kvsi_fr_u = 1000 / self.kvsi_t_u.dropna().diff(axis=1).mean(axis=1) / self.norm_fr
        with open(res_p/f'{dataname}_g_kvsi_{magnif_d}_time.pickle', 'rb') as f:
            self.kvsi_t_d = pickle.load(f)
            self.kvsi_fr_d = 1000 / self.kvsi_t_d.dropna().diff(axis=1).mean(axis=1) / self.norm_fr
        with open(res_p/f'{dataname}_g_kca_{magnif_u}_time.pickle', 'rb') as f:
            self.kca_t_u = pickle.load(f)
            self.kca_fr_u = 1000 / self.kca_t_u.dropna().diff(axis=1).mean(axis=1) / self.norm_fr
        with open(res_p/f'{dataname}_g_kca_{magnif_d}_time.pickle', 'rb') as f:
            self.kca_t_d = pickle.load(f)
            self.kca_fr_d = 1000 / self.kca_t_d.dropna().diff(axis=1).mean(axis=1) / self.norm_fr
        with open(res_p/f'{dataname}_g_naleak_{magnif_u}_time.pickle', 'rb') as f:
            self.nal_t_u = pickle.load(f)
            self.nal_fr_u = 1000 / self.nal_t_u.dropna().diff(axis=1).mean(axis=1) / self.norm_fr
        with open(res_p/f'{dataname}_g_naleak_{magnif_d}_time.pickle', 'rb') as f:
            self.nal_t_d = pickle.load(f)
            self.nal_fr_d = 1000 / self.nal_t_d.dropna().diff(axis=1).mean(axis=1) / self.norm_fr
        with open(res_p/f'{dataname}_g_nap_{magnif_u}_time.pickle', 'rb') as f:
            self.nap_t_u = pickle.load(f)
            self.nap_fr_u = 1000 / self.nap_t_u.dropna().diff(axis=1).mean(axis=1) / self.norm_fr
        with open(res_p/f'{dataname}_g_nap_{magnif_d}_time.pickle', 'rb') as f:
            self.nap_t_d = pickle.load(f)
            self.nap_fr_d = 1000 / self.nap_t_d.dropna().diff(axis=1).mean(axis=1) / self.norm_fr
        with open(res_p/f'{dataname}_g_cav_{magnif_u}_time.pickle', 'rb') as f:
            self.cav_t_u = pickle.load(f)
            self.cav_fr_u = 1000 / self.cav_t_u.dropna().diff(axis=1).mean(axis=1) / self.norm_fr
        with open(res_p/f'{dataname}_g_cav_{magnif_d}_time.pickle', 'rb') as f:
            self.cav_t_d = pickle.load(f)
            self.cav_fr_d = 1000 / self.cav_t_d.dropna().diff(axis=1).mean(axis=1) / self.norm_fr
        with open(res_p/f'{dataname}_t_ca_{magnif_u}_time.pickle', 'rb') as f:
            self.tca_t_u = pickle.load(f)
            self.tca_fr_u = 1000 / self.tca_t_u.dropna().diff(axis=1).mean(axis=1) / self.norm_fr
        with open(res_p/f'{dataname}_t_ca_{magnif_d}_time.pickle', 'rb') as f:
            self.tca_t_d = pickle.load(f)
            self.tca_fr_d = 1000 / self.tca_t_d.dropna().diff(axis=1).mean(axis=1) / self.norm_fr
        data_dic = {
            'kleak': [self.kl_fr_u, self.kl_fr_d], 
            'kca': [self.kca_fr_u, self.kca_fr_d], 
            'naleak': [self.nal_fr_u, self.nal_fr_d], 
            'cav': [self.cav_fr_u, self.cav_fr_d], 
            'nap': [self.nap_fr_u, self.nap_fr_d], 
            'tca': [self.tca_fr_u, self.tca_fr_d], 
        }

        if self.model_name == 'AN':
            with open(res_p/f'{dataname}_g_nav_{magnif_u}_time.pickle', 'rb') as f:
                self.nav_t_u = pickle.load(f)
                self.nav_fr_u = 1000 / self.nav_t_u.dropna().diff(axis=1).mean(axis=1) / self.norm_fr
            with open(res_p/f'{dataname}_g_nav_{magnif_d}_time.pickle', 'rb') as f:
                self.nav_t_d = pickle.load(f)
                self.nav_fr_d = 1000 / self.nav_t_d.dropna().diff(axis=1).mean(axis=1) / self.norm_fr
            with open(res_p/f'{dataname}_g_kvhh_{magnif_u}_time.pickle', 'rb') as f:
                self.kvhh_t_u = pickle.load(f)
                self.kvhh_fr_u = 1000 / self.kvhh_t_u.dropna().diff(axis=1).mean(axis=1) / self.norm_fr
            with open(res_p/f'{dataname}_g_kvhh_{magnif_d}_time.pickle', 'rb') as f:
                self.kvhh_t_d = pickle.load(f)
                self.kvhh_fr_d = 1000 / self.kvhh_t_d.dropna().diff(axis=1).mean(axis=1) / self.norm_fr
            with open(res_p/f'{dataname}_g_kvsi_{magnif_u}_time.pickle', 'rb') as f:
                self.kvsi_t_u = pickle.load(f)
                self.kvsi_fr_u = 1000 / self.kvsi_t_u.dropna().diff(axis=1).mean(axis=1) / self.norm_fr
            with open(res_p/f'{dataname}_g_kvsi_{magnif_d}_time.pickle', 'rb') as f:
                self.kvsi_t_d = pickle.load(f)
                self.kvsi_fr_d = 1000 / self.kvsi_t_d.dropna().diff(axis=1).mean(axis=1) / self.norm_fr
            with open(res_p/f'{dataname}_g_kva_{magnif_u}_time.pickle', 'rb') as f:
                self.kva_t_u = pickle.load(f)
                self.kva_fr_u = 1000 / self.kva_t_u.dropna().diff(axis=1).mean(axis=1) / self.norm_fr
            with open(res_p/f'{dataname}_g_kva_{magnif_d}_time.pickle', 'rb') as f:
                self.kva_t_d = pickle.load(f)
                self.kva_fr_d = 1000 / self.kva_t_d.dropna().diff(axis=1).mean(axis=1) / self.norm_fr
            with open(res_p/f'{dataname}_g_kir_{magnif_u}_time.pickle', 'rb') as f:
                self.kir_t_u = pickle.load(f)
                self.kir_fr_u = 1000 / self.kir_t_u.dropna().diff(axis=1).mean(axis=1) / self.norm_fr
            with open(res_p/f'{dataname}_g_kir_{magnif_d}_time.pickle', 'rb') as f:
                self.kir_t_d = pickle.load(f)
                self.kir_fr_d = 1000 / self.kir_t_d.dropna().diff(axis=1).mean(axis=1) / self.norm_fr
            with open(res_p/f'{dataname}_g_ampar_{magnif_u}_time.pickle', 'rb') as f:
                self.ampar_t_u = pickle.load(f)
                self.ampar_fr_u = 1000 / self.ampar_t_u.dropna().diff(axis=1).mean(axis=1) / self.norm_fr
            with open(res_p/f'{dataname}_g_ampar_{magnif_d}_time.pickle', 'rb') as f:
                self.ampar_t_d = pickle.load(f)
                self.ampar_fr_d = 1000 / self.ampar_t_d.dropna().diff(axis=1).mean(axis=1) / self.norm_fr
            with open(res_p/f'{dataname}_g_nmdar_{magnif_u}_time.pickle', 'rb') as f:
                self.nmdar_t_u = pickle.load(f)
                self.nmdar_fr_u = 1000 / self.nmdar_t_u.dropna().diff(axis=1).mean(axis=1) / self.norm_fr
            with open(res_p/f'{dataname}_g_nmdar_{magnif_d}_time.pickle', 'rb') as f:
                self.nmdar_t_d = pickle.load(f)
                self.nmdar_fr_d = 1000 / self.nmdar_t_d.dropna().diff(axis=1).mean(axis=1) / self.norm_fr
            with open(res_p/f'{dataname}_g_gabar_{magnif_u}_time.pickle', 'rb') as f:
                self.gabar_t_u = pickle.load(f)
                self.gabar_fr_u = 1000 / self.gabar_t_u.dropna().diff(axis=1).mean(axis=1) / self.norm_fr
            with open(res_p/f'{dataname}_g_gabar_{magnif_d}_time.pickle', 'rb') as f:
                self.gabar_t_d = pickle.load(f)
                self.gabar_fr_d = 1000 / self.gabar_t_d.dropna().diff(axis=1).mean(axis=1) / self.norm_fr
            data_dic['nav'] = [self.nav_fr_u, self.nav_fr_d]
            data_dic['kvhh'] = [self.kvhh_fr_u, self.kvhh_fr_d]
            data_dic['kvsi'] = [self.kvsi_fr_u, self.kvsi_fr_d]
            data_dic['kva'] = [self.kva_fr_u, self.kva_fr_d]
            data_dic['kir'] = [self.kir_fr_u, self.kir_fr_d]
            data_dic['ampar'] = [self.ampar_fr_u, self.ampar_fr_d]
            data_dic['nmdar'] = [self.nmdar_fr_u, self.nmdar_fr_d]
            data_dic['gabar'] = [self.gabar_fr_u, self.gabar_fr_d]
        elif self.model_name == 'SAN':
            with open(res_p/f'{dataname}_g_kvhh_{magnif_u}_time.pickle', 'rb') as f:
                self.kvhh_t_u = pickle.load(f)
                self.kvhh_fr_u = 1000 / self.kvhh_t_u.dropna().diff(axis=1).mean(axis=1) / self.norm_fr
            with open(res_p/f'{dataname}_g_kvhh_{magnif_d}_time.pickle', 'rb') as f:
                self.kvhh_t_d = pickle.load(f)
                self.kvhh_fr_d = 1000 / self.kvhh_t_d.dropna().diff(axis=1).mean(axis=1) / self.norm_fr
            data_dic['kvhh'] = [self.kvhh_fr_u, self.kvhh_fr_d]
        elif self.model_name == 'RAN':
            with open(res_p/f'{dataname}_g_kvsi_{magnif_u}_time.pickle', 'rb') as f:
                self.kvsi_t_u = pickle.load(f)
                self.kvsi_fr_u = 1000 / self.kvsi_t_u.dropna().diff(axis=1).mean(axis=1) / self.norm_fr
            with open(res_p/f'{dataname}_g_kvsi_{magnif_d}_time.pickle', 'rb') as f:
                self.kvsi_t_d = pickle.load(f)
                self.kvsi_fr_d = 1000 / self.kvsi_t_d.dropna().diff(axis=1).mean(axis=1) / self.norm_fr
            data_dic['kvsi'] = [self.kvsi_fr_u, self.kvsi_fr_d]

        len_data = len(self.norm_fr)
        self.n_df = pd.DataFrame(index=data_dic.keys(), columns=['inc', 'dec'])
        self.diff_df = pd.DataFrame(index=data_dic.keys(), columns=['inc', 'dec'])
        self.diff_sig = pd.DataFrame(index=data_dic.keys(), columns=['inc', 'dec'])
        for ch in data_dic.keys():
            n_inc = 0
            n_dec = 0
            avg_inc = 0
            avg_dec = 0
            var_inc = 0
            var_dec = 0
            fr_u, fr_d = data_dic[ch]
            fr_u.index = range(len(fr_u))
            fr_d.index = range(len(fr_d))
            sh_idx = np.intersect1d(fr_u.dropna().index, fr_d.dropna().index)
            sh_idx = sh_idx.astype(int)
            for idx in sh_idx:
                if fr_d[idx] < 0.1 or fr_u[idx] > 2.0:
                    pass
                elif fr_d[idx] > 2.0 or fr_u[idx] < 0.1:
                    pass
                elif fr_d[idx] < 0.975 and fr_u[idx] > 1.025:
                    n_inc += 1
                    diff = fr_u[idx] - fr_d[idx]
                    avg_inc_min1 = copy(avg_inc)
                    avg_inc += (diff - avg_inc) / n_inc
                    var_inc += (diff - avg_inc_min1) * (diff - avg_inc)
                elif fr_d[idx] > 1.025 and fr_u[idx] < 0.975:
                    n_dec += 1
                    diff = fr_u[idx] - fr_d[idx]
                    avg_dec_min1 = copy(avg_dec)
                    avg_dec += (diff-avg_dec) / n_dec
                    var_dec += (diff - avg_dec_min1) * (diff - avg_dec_min1)
                else:
                    pass
            self.n_df.loc[ch] = [n_inc/len_data, n_dec/len_data]
            self.diff_df.loc[ch] = [avg_inc, avg_dec]
            self.diff_sig.loc[ch] = [np.sqrt(var_inc/(len_data-1)), np.sqrt(var_dec/(len_data-1))]
        self.n_df = pd.DataFrame(self.n_df.stack().reset_index())
        self.diff_df = pd.DataFrame(self.diff_df.stack().reset_index())
        self.diff_sig = pd.DataFrame(self.diff_sig.stack().reset_index())
        self.n_df.columns = ['channel', 'inc/dec', 'value']
        self.diff_df.columns = ['channel', 'inc/dec', 'value']
        self.diff_sig.columns = ['channel', 'inc/dec', 'value']

    def calc_cal(self, filename: str, t_filename: str, 
                 channel: str, tp: str):
        """ Calculate calcium max/min/mean for 1st~6th burst firing for 
        the all parameter sets.

        Parameters
        ----------
        filename: str
            the name of file in which parameter sets are contained
        tp: str
            type of the parameter sets (e.g., typical, atypical)
        """
        p: Path = Path.cwd().parents[0]
        data_p: Path = p / 'results' / 'normalization_mp_ca'
        param_p: Path = p / 'results' / f'{self.wavepattern}_params' / self.model_name
        with open(data_p/f'{t_filename}', 'rb') as f:
            t_df = pickle.load(f).dropna(how='all')
        with open(param_p/f'{filename}', 'rb') as f:
            df = pickle.load(f)
        t_df.index = range(len(t_df)) # reset index
        df.index = range(len(df)) # reset index
        if len(t_df) != len(df):
            print('The number of parameter sets is different between time file and parameter file!!!')
            return 0
        with open(data_p/'incdec_analysis'/'index'/ f'{self.model_name}' / f'{channel}_{tp}.pickle', 'rb') as f:
            idx = pickle.load(f)
        # extract parameter sets in interest
        t_df = t_df.loc[idx]
        df = df.loc[idx]
        resname: str = f'{filename}_{channel}_{tp}.pickle'
        res_p: Path = p / 'results' / 'normalization_mp_ca' / 'incdec_analysis' / 'calcium' / self.model_name
        res_p.mkdir(parents=True, exist_ok=True)
        ca_max_lst = []
        ca_min_lst = []
        ca_mean_lst = []
        for i in tqdm(range(len(t_df))):
            param = df.iloc[i, :]
            e = t_df.iloc[i, :]
            if e[0] == None:
                pass
            else:
                samp_len = 10 + ((5000+e[6])//10000) * 10
                self.model.set_params(param)
                s, _ = self.model.run_odeint(samp_freq=1000, samp_len=samp_len)
                ca: np.ndarray = s[5000:, -1]
                ca_max_loc = [] # local max
                ca_min_loc = [] # local min
                for i in range(6):
                    ca_max_loc.append(ca[e[i]:e[i+1]].max())
                    ca_min_loc.append(ca[e[i]:e[i+1]].min())
                ca_max_lst.append(np.mean(ca_max_loc))
                ca_min_lst.append(np.mean(ca_min_loc))
                ca_mean_lst.append(np.mean(ca[e[0]:e[6]]))
        tp_lst = [tp] * len(ca_max_lst)
        res_df = pd.DataFrame([ca_max_lst, ca_min_lst, ca_mean_lst, tp_lst], 
                              index=['max', 'min', 'mean', 'type']).T
        with open(res_p/resname, 'wb') as f:
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
        diff = int(arg[6])
        norm.time_bifurcation_rep(filename, channel, diff)
    elif method == 'two_bifur':
        channel1 = arg[5]
        channel2 = arg[6]
        diff = int(arg[7])
        interval = int(arg[8])
        ncore = int(arg[9])
        norm.two_bifur_multi_singleprocess(ncore, filename, channel1, channel2, diff, interval)
    elif method == 'time_bifurcation_all':
        channel = arg[5]
        magnif = float(arg[6])
        norm.time_bifurcation_all(filename, channel, magnif)
    elif method == 'calc_calcium':
        t_filename = arg[5]
        channel = arg[6]
        tp = arg[7]
        norm.calc_cal(filename, t_filename, channel, tp)
