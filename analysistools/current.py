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


class AN:
    def __init__(self, wavepattern: str, 
                 ion: bool=False, concentration: Dict=None) -> None:
        self.model = anmodel.models.ANmodel()
        self.cnst = anmodel.params.Constants()
        self.wavepattern = wavepattern

    def set_params(self, param: pd.Series) -> None:
        self.leak = anmodel.channels.Leak(g=float(param['g_leak']))
        self.leak.set_div()
        self.nav = anmodel.channels.NavHH(g=float(param['g_nav']))
        self.kvhh = anmodel.channels.KvHH(g=float(param['g_kvhh']))
        self.kva = anmodel.channels.KvA(g=float(param['g_kva']))
        self.kvsi = anmodel.channels.KvSI(g=float(param['g_kvsi']))
        self.cav = anmodel.channels.Cav(g=float(param['g_cav']))
        self.nap = anmodel.channels.NaP(g=float(param['g_nap']))
        self.kca = anmodel.channels.KCa(g=float(param['g_kca']))
        self.kir = anmodel.channels.KIR(g=float(param['g_kir']))
        self.ampar = anmodel.channels.AMPAR(g=float(param['g_ampar']))
        self.nmdar = anmodel.channels.NMDAR(g=float(param['g_nmdar']))
        self.gabar = anmodel.channels.GABAR(g=float(param['g_gabar']))

    def get(self, s: np.ndarray) -> List[np.ndarray]:
        i_leak = np.array([self.leak.i(v=s[i, 0]) for i in range(len(s))])
        i_kl = np.array([self.leak.ikl(v=s[i, 0]) for i in range(len(s))])
        i_nal = np.array([self.leak.inal(v=s[i, 0]) for i in range(len(s))])
        i_nav = np.array([self.nav.i(v=s[i, 0], h=s[i, 1]) for i in range(len(s))])
        i_kvhh = np.array([self.kvhh.i(v=s[i, 0], n=s[i, 2]) for i in range(len(s))])
        i_kva = np.array([self.kva.i(v=s[i, 0], h=s[i, 3]) for i in range(len(s))])
        i_kvsi = np.array([self.kvsi.i(v=s[i, 0], m=s[i, 4]) for i in range(len(s))])
        i_cav = np.array([self.cav.i(v=s[i, 0]) for i in range(len(s))])
        i_nap = np.array([self.nap.i(v=s[i, 0]) for i in range(len(s))])
        i_kca = np.array([self.kca.i(v=s[i, 0], ca=s[i, 9]) for i in range(len(s))])
        i_kir = np.array([self.kir.i(v=s[i, 0]) for i in range(len(s))])
        i_ampar = np.array([self.ampar.i(v=s[i, 0], s=s[i, 5]) for i in range(len(s))])
        i_nmdar = np.array([self.nmdar.i(v=s[i, 0], s=s[i, 7]) for i in range(len(s))])
        i_gabar = np.array([self.gabar.i(v=s[i, 0], s=s[i, 8]) for i in range(len(s))])
        i_lst = [
            i_leak, i_kl, i_nal, i_nav, i_kvhh, 
            i_kva, i_kvsi, i_cav, i_nap, i_kca, 
            i_kir, i_ampar, i_nmdar, i_gabar,
        ]
        return i_lst

    def get_p(self, s: np.ndarray) -> List[np.ndarray]:
        i_lst = self.get(s)
        i_leak = i_lst[0]
        i_kl = i_lst[1]
        i_nal = i_lst[2]
        i_nav = i_lst[3]
        i_kvhh = i_lst[4]
        i_kva = i_lst[5]
        i_kvsi = i_lst[6]
        i_cav = i_lst[7]
        i_nap = i_lst[8]
        i_kca = i_lst[9]
        i_kir = i_lst[10]
        i_ampar = i_lst[11]
        i_nmdar = i_lst[12]
        i_gabar = i_lst[13]

        i_ampar_in = [i_ampar[i] if s[i, 0]<self.cnst.vAMPAR else 0 for i in range(len(s))]
        i_ampar_out = [i_ampar[i] if s[i, 0]>self.cnst.vAMPAR else 0 for i in range(len(s))]
        i_nmdar_in = [i_nmdar[i] if s[i, 0]<self.cnst.vNMDAR else 0 for i in range(len(s))]
        i_nmdar_out = [i_nmdar[i] if s[i, 0]>self.cnst.vNMDAR else 0 for i in range(len(s))]
        i_gabar_in = [i_gabar[i] if s[i, 0]<self.cnst.vGABAR else 0 for i in range(len(s))]
        i_gabar_out = [i_gabar[i] if s[i, 0]>self.cnst.vGABAR else 0 for i in range(len(s))]

        i_out = i_kl + i_kvhh + i_kva + i_kvsi + i_kir + i_kca + i_ampar_out + i_nmdar_out + i_gabar_out
        i_in = i_nal + i_nav + i_cav + i_nap + i_ampar_in + i_nmdar_in + i_gabar_in
        i_kl_p = i_kl / i_out
        i_kvhh_p = i_kvhh / i_out
        i_kva_p = i_kva / i_out
        i_kvsi_p = i_kvsi / i_out
        i_kir_p = i_kir / i_out
        i_kca_p = i_kca / i_out
        i_ampar_out_p = i_ampar_out / i_out
        i_nmdar_out_p = i_nmdar_out / i_out
        i_gabar_out_p = i_gabar_out / i_out
        i_nal_p = i_nal / i_in
        i_nav_p = i_nav / i_in
        i_cav_p = i_cav / i_in
        i_nap_p = i_nap / i_in
        i_ampar_in_p = i_ampar_in / i_in
        i_nmdar_in_p = i_nmdar_in / i_in
        i_gabar_in_p = i_gabar_in / i_in
        
        ip_out = [i_kl_p, i_kvhh_p, i_kva_p, i_kvsi_p, i_kir_p, i_kca_p, i_ampar_out_p, i_nmdar_out_p, i_gabar_out_p]
        ip_in = [i_nal_p, i_nav_p, i_cav_p, i_nap_p, i_ampar_in_p, i_nmdar_in_p, i_gabar_in_p]
        return ip_out, ip_in

    def p_heatmap(self, filename: str):
        now = datetime.now()
        date = f'{now.year}_{now.month}_{now.day}'
        p: Path = Path.cwd().parents[0]
        res_p = p / 'results' / 'current' / 'AN' / f'{date}_{self.wavepattern}'
        res_p.mkdir(parents=True, exist_ok=True)

        data_p = p / 'results' / f'{self.wavepattern}_params' / 'AN' / filename
        time_p = p / 'results' / 'normalization_mp_ca' / f'{self.wavepattern}_AN_time.pickle'
        with open(data_p, 'rb') as f:
            param_df = pickle.load(f)
        with open(time_p, 'rb') as f:
            time_df = pickle.load(f).dropna(how='all')
        param_df.index = range(len(param_df))
        time_df.index = range(len(time_df))
        if len(param_df) != len(time_df):
            raise Exception
        
        p_res_dic = {
            'kleak': pd.DataFrame([], columns=range(48), index=param_df.index), 
            'kvhh': pd.DataFrame([], columns=range(48), index=param_df.index), 
            'kva': pd.DataFrame([], columns=range(48), index=param_df.index), 
            'kvsi': pd.DataFrame([], columns=range(48), index=param_df.index), 
            'kir': pd.DataFrame([], columns=range(48), index=param_df.index), 
            'kca': pd.DataFrame([], columns=range(48), index=param_df.index), 
            'ampar_out': pd.DataFrame([], columns=range(48), index=param_df.index), 
            'nmdar_out': pd.DataFrame([], columns=range(48), index=param_df.index), 
            'gabar_out': pd.DataFrame([], columns=range(48), index=param_df.index), 
            'naleak': pd.DataFrame([], columns=range(48), index=param_df.index), 
            'nav': pd.DataFrame([], columns=range(48), index=param_df.index), 
            'cav': pd.DataFrame([], columns=range(48), index=param_df.index), 
            'nap': pd.DataFrame([], columns=range(48), index=param_df.index),
            'ampar_in': pd.DataFrame([], columns=range(48), index=param_df.index), 
            'nmdar_in': pd.DataFrame([], columns=range(48), index=param_df.index), 
            'gabar_in': pd.DataFrame([], columns=range(48), index=param_df.index), 
        }
        for idx in tqdm(param_df.index):
            param = param_df.loc[idx, :]
            self.set_params(param)
            self.model.set_params(param)
            e = time_df.loc[idx, :]
            try:
                samp_len = 10 + ((5000+e[6])//10000) * 10
            except TypeError:
                continue
            s, _ = self.model.run_odeint(samp_len=samp_len)
            ip_out, ip_in = self.get_p(s[5000:, :])
            i_kl_p, i_kvhh_p, i_kva_p, i_kvsi_p, i_kir_p, i_kca_p, i_ampar_out_p, i_nmdar_out_p, i_gabar_out_p = ip_out
            i_nal_p, i_nav_p, i_cav_p, i_nap_p, i_ampar_in_p, i_nmdar_in_p, i_gabar_in_p = ip_in
            p_data_dic = {
            'kleak': i_kl_p, 
            'kvhh': i_kvhh_p, 
            'kva': i_kva_p, 
            'kvsi': i_kvsi_p, 
            'kir': i_kir_p, 
            'kca': i_kca_p, 
            'ampar_out': i_ampar_out_p, 
            'nmdar_out': i_nmdar_out_p, 
            'gabar_out': i_gabar_out_p, 
            'naleak': i_nal_p, 
            'nav': i_nav_p, 
            'cav': i_cav_p, 
            'nap': i_nap_p,
            'ampar_in': i_ampar_in_p, 
            'nmdar_in': i_nmdar_in_p, 
            'gabar_in': i_gabar_in_p, 
            }
            for j in range(len(e)-1):
                tlst = np.linspace(e[j], e[j+1], 9, dtype=int)
                for k in range(len(tlst)-1):
                    for channel in p_res_dic.keys():
                        p_res_dic[channel].loc[idx, j*8+k] = p_data_dic[channel][tlst[k]:tlst[k+1]].mean()

        for channel in p_res_dic.keys():
            with open(res_p/channel, 'wb') as f:
                pickle.dump(p_res_dic[channel], f)

    def load_p_heatmap(self, date):
        p: Path = Path.cwd().parents[0]
        res_p = p / 'results' / 'current' / 'AN' / date
        with open(res_p/'kleak', 'rb') as f:
            self.kl_hm = pickle.load(f)
        with open(res_p/'kvhh', 'rb') as f:
            self.kvhh_hm = pickle.load(f)
        with open(res_p/'kva', 'rb') as f:
            self.kva_hm = pickle.load(f)
        with open(res_p/'kvsi', 'rb') as f:
            self.kvsi_hm = pickle.load(f)
        with open(res_p/'kir', 'rb') as f:
            self.kir_hm = pickle.load(f)
        with open(res_p/'kca', 'rb') as f:
            self.kca_hm = pickle.load(f)
        with open(res_p/'ampar_out', 'rb') as f:
            self.ampar_out_hm = pickle.load(f)
        with open(res_p/'nmdar_out', 'rb') as f:
            self.nmdar_out_hm = pickle.load(f)
        with open(res_p/'gabar_out', 'rb') as f:
            self.gabar_out_hm = pickle.load(f)
        with open(res_p/'naleak', 'rb') as f:
            self.nal_hm = pickle.load(f)
        with open(res_p/'nav', 'rb') as f:
            self.nav_hm = pickle.load(f)
        with open(res_p/'cav', 'rb') as f:
            self.cav_hm = pickle.load(f)
        with open(res_p/'nap', 'rb') as f:
            self.nap_hm = pickle.load(f)
        with open(res_p/'ampar_in', 'rb') as f:
            self.ampar_in_hm = pickle.load(f)
        with open(res_p/'nmdar_in', 'rb') as f:
            self.nmdar_in_hm = pickle.load(f)
        with open(res_p/'gabar_in', 'rb') as f:
            self.gabar_in_hm = pickle.load(f)


class SAN:
    def __init__(self, ion: bool=False, concentration: Dict=None) -> None:
        self.model = anmodel.models.SANmodel()
        self.wavepattern = 'SWS'

    def set_params(self, param: pd.Series) -> None:
        self.leak = anmodel.channels.Leak(g=float(param['g_leak']))
        self.leak.set_div()
        self.kvhh = anmodel.channels.KvHH(g=float(param['g_kvhh']))
        self.cav = anmodel.channels.Cav(g=float(param['g_cav']))
        self.nap = anmodel.channels.NaP(g=float(param['g_nap']))
        self.kca = anmodel.channels.KCa(g=float(param['g_kca']))

    def get(self, s: np.ndarray) -> List[np.ndarray]:
        i_leak = np.array([self.leak.i(v=s[i, 0]) for i in range(len(s))])
        i_kl = np.array([self.leak.ikl(v=s[i, 0]) for i in range(len(s))])
        i_nal = np.array([self.leak.inal(v=s[i, 0]) for i in range(len(s))])
        i_kvhh = np.array([self.kvhh.i(v=s[i, 0], n=s[i, 1]) for i in range(len(s))])
        i_cav = np.array([self.cav.i(v=s[i, 0]) for i in range(len(s))])
        i_nap = np.array([self.nap.i(v=s[i, 0]) for i in range(len(s))])
        i_kca = np.array([self.kca.i(v=s[i, 0], ca=s[i, 2]) for i in range(len(s))])        
        i_lst = [
            i_leak, i_kl, i_nal, i_kvhh, 
            i_cav, i_nap, i_kca
        ]
        return i_lst

    def get_p(self, s: np.ndarray) -> List[np.ndarray]:
        i_lst = self.get(s)
        i_leak = i_lst[0]
        i_kl = i_lst[1]
        i_nal = i_lst[2]
        i_kvhh = i_lst[3]
        i_cav = i_lst[4]
        i_nap = i_lst[5]
        i_kca = i_lst[6]

        i_out = i_kl + i_kvhh + i_kca
        i_in = i_nal + i_cav + i_nap
        i_kl_p = i_kl / i_out
        i_kvhh_p = i_kvhh / i_out
        i_kca_p = i_kca / i_out
        i_nal_p = i_nal / i_in
        i_cav_p = i_cav / i_in
        i_nap_p = i_nap / i_in
        ip_out = [i_kl_p, i_kvhh_p, i_kca_p]
        ip_in = [i_nal_p, i_cav_p, i_nap_p]
        return ip_out, ip_in

    def p_heatmap(self, filename: str):
        now = datetime.now()
        date = f'{now.year}_{now.month}_{now.day}'
        p: Path = Path.cwd().parents[0]
        res_p = p / 'results' / 'current' / 'SAN' / date
        res_p.mkdir(parents=True, exist_ok=True)

        data_p = p / 'results' / f'{self.wavepattern}_params' / 'SAN' / filename
        time_p = p / 'results' / 'normalization_mp_ca' / f'{self.wavepattern}_SAN_time.pickle'
        with open(data_p, 'rb') as f:
            param_df = pickle.load(f)
        with open(time_p, 'rb') as f:
            time_df = pickle.load(f).dropna(how='all')
        param_df.index = range(len(param_df))
        time_df.index = range(len(time_df))
        if len(param_df) != len(time_df):
            raise Exception
        
        p_res_dic = {
            'kleak': pd.DataFrame([], columns=range(48), index=param_df.index), 
            'kvhh': pd.DataFrame([], columns=range(48), index=param_df.index), 
            'kca': pd.DataFrame([], columns=range(48), index=param_df.index), 
            'naleak': pd.DataFrame([], columns=range(48), index=param_df.index), 
            'cav': pd.DataFrame([], columns=range(48), index=param_df.index), 
            'nap': pd.DataFrame([], columns=range(48), index=param_df.index),
        }
        for idx in tqdm(param_df.index):
            param = param_df.loc[idx, :]
            self.set_params(param)
            self.model.set_params(param)
            e = time_df.loc[idx, :]
            try:
                samp_len = 10 + ((5000+e[6])//10000) * 10
            except TypeError:
                continue
            s, _ = self.model.run_odeint(samp_len=samp_len)
            ip_out, ip_in = self.get_p(s[5000:, :])
            i_kl_p, i_kvhh_p, i_kca_p = ip_out
            i_nal_p, i_cav_p, i_nap_p = ip_in
            p_data_dic = {
                'kleak': i_kl_p, 
                'kvhh': i_kvhh_p, 
                'kca': i_kca_p, 
                'naleak': i_nal_p, 
                'cav': i_cav_p, 
                'nap': i_nap_p,
            }
            for j in range(len(e)-1):
                tlst = np.linspace(e[j], e[j+1], 9, dtype=int)
                for k in range(len(tlst)-1):
                    for channel in p_res_dic.keys():
                        p_res_dic[channel].loc[idx, j*8+k] = p_data_dic[channel][tlst[k]:tlst[k+1]].mean()

        for channel in p_res_dic.keys():
            with open(res_p/channel, 'wb') as f:
                pickle.dump(p_res_dic[channel], f)

    def load_p_heatmap(self, date):
        p: Path = Path.cwd().parents[0]
        res_p = p / 'results' / 'current' / 'SAN' / date
        with open(res_p/'kleak', 'rb') as f:
            self.kl_hm = pickle.load(f)
        with open(res_p/'kvhh', 'rb') as f:
            self.kvhh_hm = pickle.load(f)
        with open(res_p/'kca', 'rb') as f:
            self.kca_hm = pickle.load(f)
        with open(res_p/'naleak', 'rb') as f:
            self.nal_hm = pickle.load(f)
        with open(res_p/'cav', 'rb') as f:
            self.cav_hm = pickle.load(f)
        with open(res_p/'nap', 'rb') as f:
            self.nap_hm = pickle.load(f)


class RAN:
    def __init__(self, ion: bool=False, concentration: Dict=None) -> None:
        channel_bool = [1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1]
        self.model = anmodel.models.Xmodel(channel_bool)
        self.wavepattern = 'SPN'

    def set_params(self, param: pd.Series) -> None:
        self.leak = anmodel.channels.Leak(g=float(param['g_leak']))
        self.leak.set_div()
        self.kvsi = anmodel.channels.KvSI(g=float(param['g_kvsi']))
        self.cav = anmodel.channels.Cav(g=float(param['g_cav']))
        self.nap = anmodel.channels.NaP(g=float(param['g_nap']))
        self.kca = anmodel.channels.KCa(g=float(param['g_kca']))

    def get(self, s: np.ndarray) -> List[np.ndarray]:
        i_leak = np.array([self.leak.i(v=s[i, 0]) for i in range(len(s))])
        i_kl = np.array([self.leak.ikl(v=s[i, 0]) for i in range(len(s))])
        i_nal = np.array([self.leak.inal(v=s[i, 0]) for i in range(len(s))])
        i_kvsi = np.array([self.kvsi.i(v=s[i, 0], m=s[i, 1]) for i in range(len(s))])
        i_cav = np.array([self.cav.i(v=s[i, 0]) for i in range(len(s))])
        i_nap = np.array([self.nap.i(v=s[i, 0]) for i in range(len(s))])
        i_kca = np.array([self.kca.i(v=s[i, 0], ca=s[i, 2]) for i in range(len(s))])
        i_lst = [
            i_leak, i_kl, i_nal, i_kvsi, 
            i_cav, i_nap, i_kca
        ]
        return i_lst

    def get_p(self, s: np.ndarray) -> List[np.ndarray]:
        i_lst = self.get(s)
        i_leak = i_lst[0]
        i_kl = i_lst[1]
        i_nal = i_lst[2]
        i_kvsi = i_lst[3]
        i_cav = i_lst[4]
        i_nap = i_lst[5]
        i_kca = i_lst[6]

        i_out = i_kl + i_kvsi + i_kca
        i_in = i_nal + i_cav + i_nap
        i_kl_p = i_kl / i_out
        i_kvsi_p = i_kvsi / i_out
        i_kca_p = i_kca / i_out
        i_nal_p = i_nal / i_in
        i_cav_p = i_cav / i_in
        i_nap_p = i_nap / i_in
        ip_out = [i_kl_p, i_kvsi_p, i_kca_p]
        ip_in = [i_nal_p, i_cav_p, i_nap_p]
        return ip_out, ip_in

    def p_heatmap(self, filename: str):
        now = datetime.now()
        date = f'{now.year}_{now.month}_{now.day}'
        p: Path = Path.cwd().parents[0]
        res_p = p / 'results' / 'current' / 'RAN' / date
        res_p.mkdir(parents=True, exist_ok=True)

        data_p = p / 'results' / f'{self.wavepattern}_params' / 'RAN' / filename
        time_p = p / 'results' / 'normalization_mp_ca' / f'{self.wavepattern}_RAN_time.pickle'
        with open(data_p, 'rb') as f:
            param_df = pickle.load(f)
        with open(time_p, 'rb') as f:
            time_df = pickle.load(f).dropna(how='all')
        param_df.index = range(len(param_df))
        time_df.index = range(len(time_df))
        if len(param_df) != len(time_df):
            raise Exception
        
        p_res_dic = {
            'kleak': pd.DataFrame([], columns=range(48), index=param_df.index), 
            'kvsi': pd.DataFrame([], columns=range(48), index=param_df.index), 
            'kca': pd.DataFrame([], columns=range(48), index=param_df.index), 
            'naleak': pd.DataFrame([], columns=range(48), index=param_df.index), 
            'cav': pd.DataFrame([], columns=range(48), index=param_df.index), 
            'nap': pd.DataFrame([], columns=range(48), index=param_df.index),
        }
        for idx in tqdm(param_df.index):
            param = param_df.loc[idx, :]
            self.set_params(param)
            self.model.set_params(param)
            e = time_df.loc[idx, :]
            try:
                samp_len = 10 + ((5000+e[6])//10000) * 10
            except TypeError:
                continue
            s, _ = self.model.run_odeint(samp_len=samp_len)
            ip_out, ip_in = self.get_p(s[5000:, :])
            i_kl_p, i_kvsi_p, i_kca_p = ip_out
            i_nal_p, i_cav_p, i_nap_p = ip_in
            p_data_dic = {
                'kleak': i_kl_p, 
                'kvsi': i_kvsi_p, 
                'kca': i_kca_p, 
                'naleak': i_nal_p, 
                'cav': i_cav_p, 
                'nap': i_nap_p,
            }
            for j in range(len(e)-1):
                tlst = np.linspace(e[j], e[j+1], 9, dtype=int)
                for k in range(len(tlst)-1):
                    for channel in p_res_dic.keys():
                        p_res_dic[channel].loc[idx, j*8+k] = p_data_dic[channel][tlst[k]:tlst[k+1]].mean()

        for channel in p_res_dic.keys():
            with open(res_p/channel, 'wb') as f:
                pickle.dump(p_res_dic[channel], f)

    def load_p_heatmap(self, date):
        p: Path = Path.cwd().parents[0]
        res_p = p / 'results' / 'current' / 'RAN' / date
        with open(res_p/'kleak', 'rb') as f:
            self.kl_hm = pickle.load(f)
        with open(res_p/'kvsi', 'rb') as f:
            self.kvsi_hm = pickle.load(f)
        with open(res_p/'kca', 'rb') as f:
            self.kca_hm = pickle.load(f)
        with open(res_p/'naleak', 'rb') as f:
            self.nal_hm = pickle.load(f)
        with open(res_p/'cav', 'rb') as f:
            self.cav_hm = pickle.load(f)
        with open(res_p/'nap', 'rb') as f:
            self.nap_hm = pickle.load(f)


if __name__ == '__main__':
    arg: List = sys.argv
    method = arg[1]
    model = arg[2]
    wavepattern = arg[3]
    filename = arg[4]
    if method == 'p_heatmap':
        if model == 'AN':
            analysistools.current.AN(wavepattern=wavepattern).p_heatmap(filename)
        elif model == 'SAN':
            analysistools.current.SAN().p_heatmap(filename)
        elif model == 'RAN':
            analysistools.current.RAN().p_heatmap(filename)
