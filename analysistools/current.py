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


class AN:
    def __init__(self, param: pd.Series,
                 ion: bool=False, concentration: Dict=None) -> None:
        self.model = anmodel.models.ANmodel()
        self.set_params(param)

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

        i_out = i_kl + i_kvhh + i_kva + i_kvsi + i_kir + i_kca + i_gabar
        i_in = i_nal + i_nav + i_cav + i_nap
        i_kl_p = i_kl / i_out
        i_kvhh_p = i_kvhh / i_out
        i_kva_p = i_kva / i_out
        i_kir_p = i_kir / i_out
        i_kca_p = i_kca / i_out
        i_gabar_p = i_gabar / i_out
        i_nal_p = i_nal / i_in
        i_nav_p = i_nav / i_in
        i_cav_p = i_cav / i_in
        i_nap_p = i_nap / i_in
        ip_out = [i_kl_p, i_kvhh_p, i_kva_p, i_kir_p, i_kca_p, i_gabar_p]
        ip_in = [i_nal_p, i_nav_p, i_cav_p, i_nap_p]
        return ip_out, ip_in


class SAN:
    def __init__(self, param: pd.Series,
                 ion: bool=False, concentration: Dict=None) -> None:
        self.model = anmodel.models.SANmodel()
        self.set_params(param)

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


class X:
    def __init__(self, param: pd.Series, channel_bool: List, 
                 ion: bool=False, concentration: Dict=None) -> None:
        self.model = anmodel.models.Xmodel(channel_bool)
        self.set_params(param)

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
