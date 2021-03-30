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
        self.model_name = 'AN'
        self.cnst = anmodel.params.Constants()
        self.fs = anmodel.analysis.FreqSpike()
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
        # i_leak = i_lst[0]
        # i_kl = i_lst[1]
        # i_nal = i_lst[2]
        # i_nav = i_lst[3]
        # i_kvhh = i_lst[4]
        # i_kva = i_lst[5]
        # i_kvsi = i_lst[6]
        # i_cav = i_lst[7]
        # i_nap = i_lst[8]
        # i_kca = i_lst[9]
        # i_kir = i_lst[10]
        # i_ampar = i_lst[11]
        # i_nmdar = i_lst[12]
        # i_gabar = i_lst[13]
        i_ex_tot = np.sum(np.array(i_lst)+np.abs(np.array(i_lst)), axis=0)
        i_in_tot = np.sum(np.array(i_lst)-np.abs(np.array(i_lst)), axis=0)
        i_ex_p_lst = []
        i_in_p_lst = []
        for i, cur in enumerate(i_lst):
            i_ex_p = (cur+np.abs(cur)) / i_ex_tot
            i_in_p = (cur-np.abs(cur)) / i_in_tot
            i_ex_p_lst.append(i_ex_p)
            i_in_p_lst.append(i_in_p)
        return i_in_p_lst, i_ex_p_lst

        # i_ampar_in = [i_ampar[i] if s[i, 0]<self.cnst.vAMPAR else 0 for i in range(len(s))]
        # i_ampar_out = [i_ampar[i] if s[i, 0]>self.cnst.vAMPAR else 0 for i in range(len(s))]
        # i_nmdar_in = [i_nmdar[i] if s[i, 0]<self.cnst.vNMDAR else 0 for i in range(len(s))]
        # i_nmdar_out = [i_nmdar[i] if s[i, 0]>self.cnst.vNMDAR else 0 for i in range(len(s))]
        # i_gabar_in = [i_gabar[i] if s[i, 0]<self.cnst.vGABAR else 0 for i in range(len(s))]
        # i_gabar_out = [i_gabar[i] if s[i, 0]>self.cnst.vGABAR else 0 for i in range(len(s))]

        # i_out = i_kl + i_kvhh + i_kva + i_kvsi + i_kir + i_kca + i_ampar_out + i_nmdar_out + i_gabar_out
        # i_in = i_nal + i_nav + i_cav + i_nap + i_ampar_in + i_nmdar_in + i_gabar_in
        # i_kl_p = i_kl / i_out
        # i_kvhh_p = i_kvhh / i_out
        # i_kva_p = i_kva / i_out
        # i_kvsi_p = i_kvsi / i_out
        # i_kir_p = i_kir / i_out
        # i_kca_p = i_kca / i_out
        # i_ampar_out_p = i_ampar_out / i_out
        # i_nmdar_out_p = i_nmdar_out / i_out
        # i_gabar_out_p = i_gabar_out / i_out
        # i_nal_p = i_nal / i_in
        # i_nav_p = i_nav / i_in
        # i_cav_p = i_cav / i_in
        # i_nap_p = i_nap / i_in
        # i_ampar_in_p = i_ampar_in / i_in
        # i_nmdar_in_p = i_nmdar_in / i_in
        # i_gabar_in_p = i_gabar_in / i_in
        
        # ip_out = [i_kl_p, i_kvhh_p, i_kva_p, i_kvsi_p, i_kir_p, i_kca_p, i_ampar_out_p, i_nmdar_out_p, i_gabar_out_p]
        # ip_in = [i_nal_p, i_nav_p, i_cav_p, i_nap_p, i_ampar_in_p, i_nmdar_in_p, i_gabar_in_p]
        # return ip_out, ip_in

    def p_heatmap(self, filename: str):
        now = datetime.now()
        date = f'{now.year}_{now.month}_{now.day}'
        p: Path = Path.cwd().parents[0]
        res_p = p / 'results' / 'current' / 'p_heatmap' / 'AN' / f'{date}_{self.wavepattern}'
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
            with open(res_p/f'{channel}.pickle', 'wb') as f:
                pickle.dump(p_res_dic[channel], f)

    def load_p_heatmap(self, date):
        p: Path = Path.cwd().parents[0]
        res_p = p / 'results' / 'current' / 'p_heatmap' / 'AN' / f'{date}_{self.wavepattern}'
        with open(res_p/'kleak.pickle', 'rb') as f:
            self.kl_hm = pickle.load(f)
        with open(res_p/'kvhh.pickle', 'rb') as f:
            self.kvhh_hm = pickle.load(f)
        with open(res_p/'kva.pickle', 'rb') as f:
            self.kva_hm = pickle.load(f)
        with open(res_p/'kvsi.pickle', 'rb') as f:
            self.kvsi_hm = pickle.load(f)
        with open(res_p/'kir.pickle', 'rb') as f:
            self.kir_hm = pickle.load(f)
        with open(res_p/'kca.pickle', 'rb') as f:
            self.kca_hm = pickle.load(f)
        with open(res_p/'ampar_out.pickle', 'rb') as f:
            self.ampar_out_hm = pickle.load(f)
        with open(res_p/'nmdar_out.pickle', 'rb') as f:
            self.nmdar_out_hm = pickle.load(f)
        with open(res_p/'gabar_out.pickle', 'rb') as f:
            self.gabar_out_hm = pickle.load(f)
        with open(res_p/'naleak.pickle', 'rb') as f:
            self.nal_hm = pickle.load(f)
        with open(res_p/'nav.pickle', 'rb') as f:
            self.nav_hm = pickle.load(f)
        with open(res_p/'cav.pickle', 'rb') as f:
            self.cav_hm = pickle.load(f)
        with open(res_p/'nap.pickle', 'rb') as f:
            self.nap_hm = pickle.load(f)
        with open(res_p/'ampar_in.pickle', 'rb') as f:
            self.ampar_in_hm = pickle.load(f)
        with open(res_p/'nmdar_in.pickle', 'rb') as f:
            self.nmdar_in_hm = pickle.load(f)
        with open(res_p/'gabar_in.pickle', 'rb') as f:
            self.gabar_in_hm = pickle.load(f)

    def curr_trace(self, filename):
        now = datetime.now()
        date = f'{now.year}_{now.month}_{now.day}'
        p: Path = Path.cwd().parents[0]
        res_p = p / 'results' / 'current' / 'curr_trace' / 'AN' / date
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
            'kleak': pd.DataFrame([], columns=range(6000), index=param_df.index), 
            'kvhh': pd.DataFrame([], columns=range(6000), index=param_df.index), 
            'kva': pd.DataFrame([], columns=range(6000), index=param_df.index), 
            'kvsi': pd.DataFrame([], columns=range(6000), index=param_df.index), 
            'kir': pd.DataFrame([], columns=range(6000), index=param_df.index), 
            'kca': pd.DataFrame([], columns=range(6000), index=param_df.index), 
            'ampar_out': pd.DataFrame([], columns=range(6000), index=param_df.index), 
            'nmdar_out': pd.DataFrame([], columns=range(6000), index=param_df.index), 
            'gabar_out': pd.DataFrame([], columns=range(6000), index=param_df.index), 
            'naleak': pd.DataFrame([], columns=range(6000), index=param_df.index), 
            'nav': pd.DataFrame([], columns=range(6000), index=param_df.index), 
            'cav': pd.DataFrame([], columns=range(6000), index=param_df.index), 
            'nap': pd.DataFrame([], columns=range(6000), index=param_df.index),
            'ampar_in': pd.DataFrame([], columns=range(6000), index=param_df.index), 
            'nmdar_in': pd.DataFrame([], columns=range(6000), index=param_df.index), 
            'gabar_in': pd.DataFrame([], columns=range(6000), index=param_df.index), 
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
            v = s[5000:, 0]
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
                tlst = np.linspace(e[j], e[j+1], 1000, dtype=int)
                for ch in p_res_dic.keys():
                    p_res_dic[ch].loc[idx, 1000*j:1000*(j+1)-1] = p_data_dic[ch][tlst]
                    
        for channel in p_res_dic.keys():
            with open(res_p/f'{channel}.pickle', 'wb') as f:
                pickle.dump(p_res_dic[channel], f)
    
    def load_curr_trace(self, date):
        p: Path = Path.cwd().parents[0]
        res_p = p / 'results' / 'current' / 'curr_trace' / 'AN' / date
        with open(res_p/'kleak.pickle', 'rb') as f:
            self.kl_ct = pickle.load(f)
        self.kl_ct_mean = self.kl_ct.mean()
        self.kl_ct_std = self.kl_ct.std()
        with open(res_p/'kvsi.pickle', 'rb') as f:
            self.kvsi_ct = pickle.load(f)
        self.kvsi_ct_mean = self.kvsi_ct.mean()
        self.kvsi_ct_std = self.kvsi_ct.std()
        with open(res_p/'kca.pickle', 'rb') as f:
            self.kca_ct = pickle.load(f)
        self.kca_ct_mean = self.kca_ct.mean()
        self.kca_ct_std = self.kca_ct.std()
        with open(res_p/'naleak.pickle', 'rb') as f:
            self.nal_ct = pickle.load(f)
        self.nal_ct_mean = self.nal_ct.mean()
        self.nal_ct_std = self.nal_ct.std()
        with open(res_p/'cav.pickle', 'rb') as f:
            self.cav_ct = pickle.load(f)
        self.cav_ct_mean = self.cav_ct.mean()
        self.cav_ct_std = self.cav_ct.std()
        with open(res_p/'nap.pickle', 'rb') as f:
            self.nap_ct = pickle.load(f)
        self.nap_ct_mean = self.nap_ct.mean()
        self.nap_ct_std = self.nap_ct.std()

    def mp_ca_trace(self, filename: str):
        now = datetime.now()
        date = f'{now.year}_{now.month}_{now.day}'
        p: Path = Path.cwd().parents[0]
        res_p = p / 'results' / 'current' / 'mp_ca_trace' / 'AN' / f'{self.wavepattern}' / date
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
     
        mp_res = pd.DataFrame([], columns=range(6000), index=param_df.index)
        ca_res = pd.DataFrame([], columns=range(6000), index=param_df.index) 
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
            v: np.ndarray = scipy.stats.zscore(s[5000:, 0])
            ca: np.ndarray = scipy.stats.zscore(s[5000:, -1])
            for j in range(len(e)-1):
                tlst = np.linspace(e[j], e[j+1], 1000, dtype=int)
                mp_res.loc[idx, 1000*j:1000*(j+1)-1] = v[tlst]
                ca_res.loc[idx, 1000*j:1000*(j+1)-1] = ca[tlst]
        
        with open(res_p/'mp.pickle', 'wb') as f:
            pickle.dump(mp_res, f)
        with open(res_p/'ca.pickle', 'wb') as f:
            pickle.dump(ca_res, f)

    def load_mp_ca_trace(self, date):
        p: Path = Path.cwd().parents[0]
        res_p = p / 'results' / 'current' / 'mp_ca_trace' / 'AN' / date
        with open(res_p/'mp.pickle', 'rb') as f:
            self.mp = pickle.load(f)
        self.mp_mean = self.mp.mean()
        self.mp_std = self.mp.std()
        with open(res_p/'ca.pickle', 'rb') as f:
            self.ca = pickle.load(f)
        self.ca_mean = self.ca.mean()
        self.ca_std = self.ca.std()

    def b_s_ratio(self, filename: str):
        now = datetime.now()
        date = f'{now.year}_{now.month}_{now.day}'
        p: Path = Path.cwd().parents[0]
        res_p = p / 'results' / 'current' / 'b_s_ratio' / 'AN' / f'{date}_{self.wavepattern}'
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
        
        ch_lst = [
            'kleak', 
            'kvhh', 
            'kva', 
            'kvsi', 
            'kir', 
            'kca', 
            'ampar_out', 
            'nmdar_out', 
            'gabar_out', 
            'naleak', 
            'nav', 
            'cav', 
            'nap',
            'ampar_in', 
            'nmdar_in', 
            'gabar_in', 
        ]
        res_b_df = pd.DataFrame([], columns=ch_lst, index=param_df.index)
        res_s_df = pd.DataFrame([], columns=ch_lst, index=param_df.index)
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
            if self.wavepattern == 'SWS':
                v_sq = self.fs.square_wave(s[e[0]:e[6], 0], spike='peak')
            elif self.wavepattern == 'SPN':
                v_sq = self.fs.square_wave(s[e[0]:e[6], 0], spike='bottom')
            else:
                raise Exception
            ip_out, ip_in = self.get_p(s[e[0]:e[6], :])
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
            for ch in ch_lst:
                cur_p = p_data_dic[ch]
                cur_p_burst = cur_p[v_sq.astype(np.bool)]
                cur_p_silent = cur_p[np.logical_not(v_sq.astype(np.bool))]
                res_b_df.loc[idx, ch] = cur_p_burst.mean()
                res_s_df.loc[idx, ch] = cur_p_silent.mean()
        
        with open(res_p/'burst.pickle', 'wb') as f:
            pickle.dump(res_b_df, f)
        with open(res_p/'silent.pickle', 'wb') as f:
            pickle.dump(res_s_df, f)

    def load_b_s_ratio(self, date):
        p: Path = Path.cwd().parents[0]
        res_p = p / 'results' / 'current' / 'b_s_ratio' / 'AN' / f'{date}_{self.wavepattern}'
        with open(res_p/'burst.pickle', 'rb') as f:
            self.burst_ratio = pickle.load(f)
        with open(res_p/'silent.pickle', 'rb') as f:
            self.silent_ratio = pickle.load(f)
        out_ch = ['kleak', 'kvhh', 'kva', 'kvsi', 'kir', 'kca', 
                  'ampar_out', 'nmdar_out', 'gabar_out']
        in_ch = ['naleak', 'nav', 'cav', 'nap',
                 'ampar_in', 'nmdar_in', 'gabar_in', ]
        self.b_out = pd.DataFrame(self.burst_ratio.loc[:, out_ch].stack()).reset_index()
        self.b_in = pd.DataFrame(self.burst_ratio.loc[:, in_ch].stack()).reset_index()
        self.s_out = pd.DataFrame(self.silent_ratio.loc[:, out_ch].stack()).reset_index()
        self.s_in = pd.DataFrame(self.silent_ratio.loc[:, in_ch].stack()).reset_index()
        for bs_df in [self.b_out, self.b_in, self.s_out, self.s_in]:
            bs_df.columns = ['param_index', 'channel', 'value']


class SAN:
    def __init__(self, ion: bool=False, concentration: Dict=None) -> None:
        self.model = anmodel.models.SANmodel()
        self.model_name = 'SAN'
        self.fs = anmodel.analysis.FreqSpike()
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
        # i_leak = i_lst[0]
        # i_kl = i_lst[1]
        # i_nal = i_lst[2]
        # i_kvhh = i_lst[3]
        # i_cav = i_lst[4]
        # i_nap = i_lst[5]
        # i_kca = i_lst[6]
        i_ex_tot = np.sum(np.array(i_lst)+np.abs(np.array(i_lst)), axis=0)
        i_in_tot = np.sum(np.array(i_lst)-np.abs(np.array(i_lst)), axis=0)
        i_ex_p_lst = []
        i_in_p_lst = []
        for i, cur in enumerate(i_lst):
            i_ex_p = (cur+np.abs(cur)) / i_ex_tot
            i_in_p = (cur-np.abs(cur)) / i_in_tot
            i_ex_p_lst.append(i_ex_p)
            i_in_p_lst.append(i_in_p)
        return i_in_p_lst, i_ex_p_lst

        # i_out = i_kl + i_kvhh + i_kca
        # i_in = i_nal + i_cav + i_nap
        # i_kl_p = i_kl / i_out
        # i_kvhh_p = i_kvhh / i_out
        # i_kca_p = i_kca / i_out
        # i_nal_p = i_nal / i_in
        # i_cav_p = i_cav / i_in
        # i_nap_p = i_nap / i_in
        # ip_out = [i_kl_p, i_kvhh_p, i_kca_p]
        # ip_in = [i_nal_p, i_cav_p, i_nap_p]
        # return ip_out, ip_in

    def p_heatmap(self, filename: str):
        now = datetime.now()
        date = f'{now.year}_{now.month}_{now.day}'
        p: Path = Path.cwd().parents[0]
        res_p = p / 'results' / 'current' / 'p_heatmap' / 'SAN' / date
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
            with open(res_p/f'{channel}.pickle', 'wb') as f:
                pickle.dump(p_res_dic[channel], f)

    def load_p_heatmap(self, date):
        p: Path = Path.cwd().parents[0]
        res_p = p / 'results' / 'current' / 'p_heatmap' / 'SAN' / date
        with open(res_p/'kleak.pickle', 'rb') as f:
            self.kl_hm = pickle.load(f)
        with open(res_p/'kvhh.pickle', 'rb') as f:
            self.kvhh_hm = pickle.load(f)
        with open(res_p/'kca.pickle', 'rb') as f:
            self.kca_hm = pickle.load(f)
        with open(res_p/'naleak.pickle', 'rb') as f:
            self.nal_hm = pickle.load(f)
        with open(res_p/'cav.pickle', 'rb') as f:
            self.cav_hm = pickle.load(f)
        with open(res_p/'nap.pickle', 'rb') as f:
            self.nap_hm = pickle.load(f)

    def mp_ca_trace(self, filename: str):
        now = datetime.now()
        date = f'{now.year}_{now.month}_{now.day}'
        p: Path = Path.cwd().parents[0]
        res_p = p / 'results' / 'current' / 'mp_ca_trace' / 'SAN' / date
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
     
        mp_res = pd.DataFrame([], columns=range(6000), index=param_df.index)
        ca_res = pd.DataFrame([], columns=range(6000), index=param_df.index) 
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
            v: np.ndarray = scipy.stats.zscore(s[5000:, 0])
            ca: np.ndarray = scipy.stats.zscore(s[5000:, -1])
            for j in range(len(e)-1):
                tlst = np.linspace(e[j], e[j+1], 1000, dtype=int)
                mp_res.loc[idx, 1000*j:1000*(j+1)-1] = v[tlst]
                ca_res.loc[idx, 1000*j:1000*(j+1)-1] = ca[tlst]
        
        with open(res_p/'mp.pickle', 'wb') as f:
            pickle.dump(mp_res, f)
        with open(res_p/'ca.pickle', 'wb') as f:
            pickle.dump(ca_res, f)

    def load_mp_ca_trace(self, date):
        p: Path = Path.cwd().parents[0]
        res_p = p / 'results' / 'current' / 'mp_ca_trace' / 'SAN' / date
        with open(res_p/'mp.pickle', 'rb') as f:
            self.mp = pickle.load(f)
        self.mp_mean = self.mp.mean()
        self.mp_std = self.mp.std()
        with open(res_p/'ca.pickle', 'rb') as f:
            self.ca = pickle.load(f)
        self.ca_mean = self.ca.mean()
        self.ca_std = self.ca.std()

    def b_s_ratio(self, filename: str):
        now = datetime.now()
        date = f'{now.year}_{now.month}_{now.day}'
        p: Path = Path.cwd().parents[0]
        res_p = p / 'results' / 'current' / 'b_s_ratio' / 'SAN' / date
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
        
        ch_lst = [
            'kleak', 
            'kvhh', 
            'kca', 
            'naleak', 
            'cav', 
            'nap',
        ]
        res_b_df = pd.DataFrame([], columns=ch_lst, index=param_df.index)
        res_s_df = pd.DataFrame([], columns=ch_lst, index=param_df.index)
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
            v_sq = self.fs.square_wave(s[e[0]:e[6], 0], spike='peak')
            ip_out, ip_in = self.get_p(s[e[0]:e[6], :])
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
            for ch in ch_lst:
                cur_p = p_data_dic[ch]
                cur_p_burst = cur_p[v_sq.astype(np.bool)]
                cur_p_silent = cur_p[np.logical_not(v_sq.astype(np.bool))]
                res_b_df.loc[idx, ch] = cur_p_burst.mean()
                res_s_df.loc[idx, ch] = cur_p_silent.mean()
        
        with open(res_p/'burst.pickle', 'wb') as f:
            pickle.dump(res_b_df, f)
        with open(res_p/'silent.pickle', 'wb') as f:
            pickle.dump(res_s_df, f)

    def load_b_s_ratio(self, date):
        p: Path = Path.cwd().parents[0]
        res_p = p / 'results' / 'current' / 'b_s_ratio' / 'SAN' / date
        with open(res_p/'burst.pickle', 'rb') as f:
            self.burst_ratio = pickle.load(f)
        with open(res_p/'silent.pickle', 'rb') as f:
            self.silent_ratio = pickle.load(f)
        out_ch = ['kleak', 'kvhh', 'kca']
        in_ch = ['naleak', 'cav', 'nap']
        self.b_out = pd.DataFrame(self.burst_ratio.loc[:, out_ch].stack()).reset_index()
        self.b_in = pd.DataFrame(self.burst_ratio.loc[:, in_ch].stack()).reset_index()
        self.s_out = pd.DataFrame(self.silent_ratio.loc[:, out_ch].stack()).reset_index()
        self.s_in = pd.DataFrame(self.silent_ratio.loc[:, in_ch].stack()).reset_index()
        for bs_df in [self.b_out, self.b_in, self.s_out, self.s_in]:
            bs_df.columns = ['param_index', 'channel', 'value']
            bs_df.replace('kvhh', 'kvsi/kvhh', inplace=True)


class RAN:
    def __init__(self, ion: bool=False, concentration: Dict=None) -> None:
        channel_bool = [1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1]
        self.model = anmodel.models.Xmodel(channel_bool)
        self.model_name = 'RAN'
        self.fs = anmodel.analysis.FreqSpike()
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
        # i_leak = i_lst[0]
        # i_kl = i_lst[1]
        # i_nal = i_lst[2]
        # i_kvsi = i_lst[3]
        # i_cav = i_lst[4]
        # i_nap = i_lst[5]
        # i_kca = i_lst[6]
        i_ex_tot = np.sum(np.array(i_lst)+np.abs(np.array(i_lst)), axis=0)
        i_in_tot = np.sum(np.array(i_lst)-np.abs(np.array(i_lst)), axis=0)
        i_ex_p_lst = []
        i_in_p_lst = []
        for i, cur in enumerate(i_lst):
            i_ex_p = (cur+np.abs(cur)) / i_ex_tot
            i_in_p = (cur-np.abs(cur)) / i_in_tot
            i_ex_p_lst.append(i_ex_p)
            i_in_p_lst.append(i_in_p)
        return i_in_p_lst, i_ex_p_lst

        # i_out = i_kl + i_kvsi + i_kca
        # i_in = i_nal + i_cav + i_nap
        # i_kl_p = i_kl / i_out
        # i_kvsi_p = i_kvsi / i_out
        # i_kca_p = i_kca / i_out
        # i_nal_p = i_nal / i_in
        # i_cav_p = i_cav / i_in
        # i_nap_p = i_nap / i_in
        # ip_out = [i_kl_p, i_kvsi_p, i_kca_p]
        # ip_in = [i_nal_p, i_cav_p, i_nap_p]
        # return ip_out, ip_in

    def p_heatmap(self, filename: str):
        now = datetime.now()
        date = f'{now.year}_{now.month}_{now.day}'
        p: Path = Path.cwd().parents[0]
        res_p = p / 'results' / 'current' / 'p_heatmap' / 'RAN' / date
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
            with open(res_p/f'{channel}.pickle', 'wb') as f:
                pickle.dump(p_res_dic[channel], f)

    def load_p_heatmap(self, date):
        p: Path = Path.cwd().parents[0]
        res_p = p / 'results' / 'current' / 'p_heatmap' / 'RAN' / date
        with open(res_p/'kleak.pickle', 'rb') as f:
            self.kl_hm = pickle.load(f)
        with open(res_p/'kvsi.pickle', 'rb') as f:
            self.kvsi_hm = pickle.load(f)
        with open(res_p/'kca.pickle', 'rb') as f:
            self.kca_hm = pickle.load(f)
        with open(res_p/'naleak.pickle', 'rb') as f:
            self.nal_hm = pickle.load(f)
        with open(res_p/'cav.pickle', 'rb') as f:
            self.cav_hm = pickle.load(f)
        with open(res_p/'nap.pickle', 'rb') as f:
            self.nap_hm = pickle.load(f)

    def curr_trace(self, filename):
        now = datetime.now()
        date = f'{now.year}_{now.month}_{now.day}'
        p: Path = Path.cwd().parents[0]
        res_p = p / 'results' / 'current' / 'curr_trace' / 'RAN' / date
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
            'kleak': pd.DataFrame([], columns=range(6000), index=param_df.index), 
            'kvsi': pd.DataFrame([], columns=range(6000), index=param_df.index), 
            'kca': pd.DataFrame([], columns=range(6000), index=param_df.index), 
            'naleak': pd.DataFrame([], columns=range(6000), index=param_df.index), 
            'cav': pd.DataFrame([], columns=range(6000), index=param_df.index), 
            'nap': pd.DataFrame([], columns=range(6000), index=param_df.index),
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
            v = s[5000:, 0]
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
                tlst = np.linspace(e[j], e[j+1], 1000, dtype=int)
                for ch in p_res_dic.keys():
                    p_res_dic[ch].loc[idx, 1000*j:1000*(j+1)-1] = p_data_dic[ch][tlst]
                    
        for channel in p_res_dic.keys():
            with open(res_p/f'{channel}.pickle', 'wb') as f:
                pickle.dump(p_res_dic[channel], f)

    def load_curr_trace(self, date):
        p: Path = Path.cwd().parents[0]
        res_p = p / 'results' / 'current' / 'curr_trace' / 'RAN' / date
        with open(res_p/'kleak.pickle', 'rb') as f:
            self.kl_ct = pickle.load(f)
        self.kl_ct_mean = self.kl_ct.mean()
        self.kl_ct_std = self.kl_ct.std()
        with open(res_p/'kvsi.pickle', 'rb') as f:
            self.kvsi_ct = pickle.load(f)
        self.kvsi_ct_mean = self.kvsi_ct.mean()
        self.kvsi_ct_std = self.kvsi_ct.std()
        with open(res_p/'kca.pickle', 'rb') as f:
            self.kca_ct = pickle.load(f)
        self.kca_ct_mean = self.kca_ct.mean()
        self.kca_ct_std = self.kca_ct.std()
        with open(res_p/'naleak.pickle', 'rb') as f:
            self.nal_ct = pickle.load(f)
        self.nal_ct_mean = self.nal_ct.mean()
        self.nal_ct_std = self.nal_ct.std()
        with open(res_p/'cav.pickle', 'rb') as f:
            self.cav_ct = pickle.load(f)
        self.cav_ct_mean = self.cav_ct.mean()
        self.cav_ct_std = self.cav_ct.std()
        with open(res_p/'nap.pickle', 'rb') as f:
            self.nap_ct = pickle.load(f)
        self.nap_ct_mean = self.nap_ct.mean()
        self.nap_ct_std = self.nap_ct.std()

    def mp_ca_trace(self, filename: str):
        now = datetime.now()
        date = f'{now.year}_{now.month}_{now.day}'
        p: Path = Path.cwd().parents[0]
        res_p = p / 'results' / 'current' / 'mp_ca_trace' / 'RAN' / date
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
     
        mp_res = pd.DataFrame([], columns=range(6000), index=param_df.index)
        ca_res = pd.DataFrame([], columns=range(6000), index=param_df.index) 
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
            v: np.ndarray = scipy.stats.zscore(s[5000:, 0])
            ca: np.ndarray = scipy.stats.zscore(s[5000:, -1])
            for j in range(len(e)-1):
                tlst = np.linspace(e[j], e[j+1], 1000, dtype=int)
                mp_res.loc[idx, 1000*j:1000*(j+1)-1] = v[tlst]
                ca_res.loc[idx, 1000*j:1000*(j+1)-1] = ca[tlst]
        
        with open(res_p/'mp.pickle', 'wb') as f:
            pickle.dump(mp_res, f)
        with open(res_p/'ca.pickle', 'wb') as f:
            pickle.dump(ca_res, f)

    def load_mp_ca_trace(self, date):
        p: Path = Path.cwd().parents[0]
        res_p = p / 'results' / 'current' / 'mp_ca_trace' / 'RAN' / date
        with open(res_p/'mp.pickle', 'rb') as f:
            self.mp = pickle.load(f)
        self.mp_mean = self.mp.mean()
        self.mp_std = self.mp.std()
        with open(res_p/'ca.pickle', 'rb') as f:
            self.ca = pickle.load(f)
        self.ca_mean = self.ca.mean()
        self.ca_std = self.ca.std()

    def b_s_ratio(self, filename: str):
        now = datetime.now()
        date = f'{now.year}_{now.month}_{now.day}'
        p: Path = Path.cwd().parents[0]
        res_p = p / 'results' / 'current' / 'b_s_ratio' / 'RAN' / date
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
        
        ch_lst = [
            'kleak', 
            'kvsi', 
            'kca', 
            'naleak', 
            'cav', 
            'nap', 
        ]
        res_b_df = pd.DataFrame([], columns=ch_lst, index=param_df.index)
        res_s_df = pd.DataFrame([], columns=ch_lst, index=param_df.index)
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
            v_sq = self.fs.square_wave(s[e[0]:e[6], 0], spike='bottom')
            ip_out, ip_in = self.get_p(s[e[0]:e[6], :])
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
            for ch in ch_lst:
                cur_p = p_data_dic[ch]
                cur_p_burst = cur_p[v_sq.astype(np.bool)]
                cur_p_silent = cur_p[np.logical_not(v_sq.astype(np.bool))]
                res_b_df.loc[idx, ch] = cur_p_burst.mean()
                res_s_df.loc[idx, ch] = cur_p_silent.mean()
        
        with open(res_p/'burst.pickle', 'wb') as f:
            pickle.dump(res_b_df, f)
        with open(res_p/'silent.pickle', 'wb') as f:
            pickle.dump(res_s_df, f)

    def load_b_s_ratio(self, date):
        p: Path = Path.cwd().parents[0]
        res_p = p / 'results' / 'current' / 'b_s_ratio' / 'RAN' / date
        with open(res_p/'burst.pickle', 'rb') as f:
            self.burst_ratio = pickle.load(f)
        with open(res_p/'silent.pickle', 'rb') as f:
            self.silent_ratio = pickle.load(f)
        out_ch = ['kleak', 'kvsi', 'kca']
        in_ch = ['naleak', 'cav', 'nap']
        self.b_out = pd.DataFrame(self.burst_ratio.loc[:, out_ch].stack()).reset_index()
        self.b_in = pd.DataFrame(self.burst_ratio.loc[:, in_ch].stack()).reset_index()
        self.s_out = pd.DataFrame(self.silent_ratio.loc[:, out_ch].stack()).reset_index()
        self.s_in = pd.DataFrame(self.silent_ratio.loc[:, in_ch].stack()).reset_index()
        for bs_df in [self.b_out, self.b_in, self.s_out, self.s_in]:
            bs_df.columns = ['param_index', 'channel', 'value']
            bs_df.replace('kvsi', 'kvsi/kvhh', inplace=True)

    def current_bifurcation_rep(self, filename: str, channel: str, diff: int=100, 
                                mode: str='proportion') -> None:
        norm = analysistools.norm_fre_mp.Normalization(model='RAN', wavepattern='SPN')
        p: Path = Path.cwd().parents[0]
        data_p: Path = p / 'results' / f'{self.wavepattern}_params' / self.model_name
        res_b_name: str = f'{filename}_{channel}_{diff}_{mode}_burst.pickle'
        res_s_name: str = f'{filename}_{channel}_{diff}_{mode}_silent.pickle'
        res_p: Path = p / 'results' / 'current' / 'bifurcation_rep' / f'{self.model_name}'
        res_p.mkdir(parents=True, exist_ok=True)
        with open(data_p/filename, 'rb') as f:
            param = pickle.load(f)
        self.set_params(param)
        self.model.set_params(param)
        if channel == 'g_nal' or channel == 'g_kl':
            self.model.leak.set_div()
            param.loc['g_nal'] = self.model.leak.gnal
            param.loc['g_kl'] = self.model.leak.gkl
        else:
            self.model.leak.reset_div()
        ch_lst = ['kleak', 'kvsi', 'kca', 
                   'naleak', 'cav', 'nap']

        if mode == 'value':
            e: List[int] = norm.norm_spn(param, channel)
            s, _ = self.model.run_odeint()
            v_sq = self.fs.square_wave(s[e[0]:e[6], 0], spike='bottom')
            i_lst_o = self.get(s[e[0]:e[6], :])
            i_leak_o, i_kl_o, i_nal_o, i_kvsi_o, i_cav_o, i_nap_o, i_kca_o = i_lst_o
            original_dic = {
                'kleak': i_kl_o, 
                'kvsi': i_kvsi_o, 
                'kca': i_kca_o, 
                'naleak': i_nal_o, 
                'cav': i_cav_o, 
                'nap': i_nap_o, 
            }
            for ch in ch_lst:
                i_o = original_dic[ch]
                i_o_burst = i_o[v_sq.astype(np.bool)]
                i_o_silent = i_o[np.logical_not(v_sq.astype(np.bool))]
                original_dic[ch] = [i_o_burst, i_o_silent]
        
        start = 1000 - diff
        end = 1000 + diff + 1
        res_b_df = pd.DataFrame([], columns=ch_lst, index=np.arange(start, end))
        res_s_df = pd.DataFrame([], columns=ch_lst, index=np.arange(start, end))
        for i in tqdm(res_b_df.index):
            param_c = copy(param)
            param_c[channel] = param_c[channel] * i / 1000
            self.set_params(param_c.drop(['g_kl', 'g_nal']))
            self.model.set_params(param_c.drop(['g_kl', 'g_nal']))
            if channel == 'g_kl' or channel == 'g_nal':
                self.leak.set_gk(param_c['g_kl'])
                self.leak.set_gna(param_c['g_nal'])
                self.model.leak.set_gk(param_c['g_kl'])
                self.model.leak.set_gna(param_c['g_nal'])
            e: List[int] = norm.norm_spn(param_c, channel)
            try:
                samp_len = 10 + ((5000+e[6])//10000) * 10
            except TypeError:
                continue
            s, _ = self.model.run_odeint(samp_len=samp_len)
            v_sq = self.fs.square_wave(s[e[0]:e[6], 0], spike='bottom')
            if mode == 'proportion':  # proportion against whole inward/outward current
                ip_out, ip_in = self.get_p(s[e[0]:e[6], :])
                i_kl_p, i_kvsi_p, i_kca_p = ip_out
                i_nal_p, i_cav_p, i_nap_p = ip_in
                data_dic = {
                    'kleak': i_kl_p, 
                    'kvsi': i_kvsi_p, 
                    'kca': i_kca_p, 
                    'naleak': i_nal_p, 
                    'cav': i_cav_p, 
                    'nap': i_nap_p, 
                }
                for ch in ch_lst:
                    cur_p = data_dic[ch]
                    cur_p_burst = cur_p[v_sq.astype(np.bool)]
                    cur_p_silent = cur_p[np.logical_not(v_sq.astype(np.bool))]
                    res_b_df.loc[i, ch] = cur_p_burst.mean()
                    res_s_df.loc[i, ch] = cur_p_silent.mean()
            elif mode == 'value':  # proportion against initial current value
                i_lst = self.get(s[e[0]:e[6], :])
                i_leak, i_kl, i_nal, i_kvsi, i_cav, i_nap, i_kca = i_lst
                data_dic = {
                    'kleak': i_kl, 
                    'kvsi': i_kvsi, 
                    'kca': i_kca, 
                    'naleak': i_nal, 
                    'cav': i_cav, 
                    'nap': i_nap, 
                }
                for ch in ch_lst:
                    cur = data_dic[ch]
                    i_burst = cur[v_sq.astype(np.bool)]
                    i_silent = cur[np.logical_not(v_sq.astype(np.bool))]
                    i_o_burst, i_o_silent = original_dic[ch]
                    i_p_burst = i_burst.mean() / i_o_burst.mean()
                    i_p_silent = i_silent.mean() / i_o_silent.mean()
                    res_b_df.loc[i, ch] = i_p_burst
                    res_s_df.loc[i, ch] = i_p_silent

        with open(res_p/res_b_name, 'wb') as f:
            pickle.dump(res_b_df, f)
        with open(res_p/res_s_name, 'wb') as f:
            pickle.dump(res_s_df, f)

    def load_bifur_rep(self, filename: str, channel: str, diff: int=100, 
                       mode: str='proportion') -> None:
        p: Path = Path.cwd().parents[0]
        res_p: Path = p / 'results' / 'current' / 'bifurcation_rep' / f'{self.model_name}'
        res_b_name: str = f'{filename}_{channel}_{diff}_{mode}_burst.pickle'
        res_s_name: str = f'{filename}_{channel}_{diff}_{mode}_silent.pickle'
        with open(res_p/res_b_name, 'rb') as f:
            self.b_df = pickle.load(f)
        with open(res_p/res_s_name, 'rb') as f:
            self.s_df = pickle.load(f)


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
    elif method == 'curr_trace':
        if model == 'AN':
            analysistools.current.AN(wavepattern=wavepattern).curr_trace(filename)
        # elif model == 'SAN':
        #     analysistools.current.SAN().curr_trace(filename)
        elif model == 'RAN':
            analysistools.current.RAN().curr_trace(filename)
    elif method == 'mp_ca_trace':
        if model == 'AN':
            analysistools.current.AN(wavepattern=wavepattern).mp_ca_trace(filename)
        elif model == 'SAN':
            analysistools.current.SAN().mp_ca_trace(filename)
        elif model == 'RAN':
            analysistools.current.RAN().mp_ca_trace(filename)
    elif method == 'b_s_ratio':
        if model == 'AN':
            analysistools.current.AN(wavepattern=wavepattern).b_s_ratio(filename)
        elif model == 'SAN':
            analysistools.current.SAN().b_s_ratio(filename)
        elif model == 'RAN':
            analysistools.current.RAN().b_s_ratio(filename)
    elif method == 'cur_bifur_rep':
        channel = arg[5]
        diff = int(arg[6])
        mode = arg[7]
        if model == 'AN':
            pass
        elif model == 'SAN':
            pass
        elif model == 'RAN':
            analysistools.current.RAN().current_bifurcation_rep(filename, channel, diff, mode)
