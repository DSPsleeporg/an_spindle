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
"""
LIMIT THE NUMBER OF THREADS!
change local env variables BEFORE importing numpy
"""
os.environ["OMP_NUM_THREADS"] = "1"  # 2nd likely
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"  # most likely

from copy import copy
from datetime import datetime
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import scipy.optimize
from scipy.stats import gaussian_kde
import sympy as sym
from tqdm import tqdm
from typing import Dict, List, Iterator, Optional

import anmodel
import analysistools


class Analysis:
    def __init__(self, param: pd.Series, 
                 model: str='AN', wavepattern: str='SWS', 
                 channel_bool: Optional[Dict]=None, 
                 model_name: Optional[str]=None, 
                 ion: bool=False, concentration: Dict=None) -> None:
        self.param = param
        self.model = model
        self.wavepattern = wavepattern
        if self.model == 'SAN':
            self.model_name = 'SAN'
            self.model = anmodel.models.SANmodel(ion, concentration)
        if self.model == "X":
            if channel_bool is None:
                raise TypeError('Designate channel in argument of X model.')
            self.model_name = model_name
            self.model = anmodel.models.Xmodel(channel_bool, ion, concentration)
        self.model.set_params(self.param)

        self.samp_freq = 10000
        self.l_pad = 0.1
        self.v_pad = 10

    def ode_bifur(self, channel: str, magnif: float) -> np.ndarray:
        p = copy(self.param)
        p[channel] = p[channel] * magnif
        self.model.set_params(p)
        s, _ = self.model.run_odeint(samp_freq=self.samp_freq)
        return s

    def set_s(self, s: np.ndarray, 
              st: int, en: int) -> None:
        self.s = s
        self.st = st
        self.en = en

    def nullcline(self, t: int, 
                  ax: plt.axes, flow=False) -> plt.axes:
        lmin = self.s[self.st:self.en, 1].min() - self.l_pad
        lmax = self.s[self.st:self.en, 1].max() + self.l_pad
        vmin = self.s[self.st:self.en, 0].min() - self.v_pad
        vmax = self.s[self.st:self.en, 0].max() + self.v_pad
        l_grid, v_grid = np.meshgrid(np.arange(lmin, lmax, 0.001), 
                                     np.arange(vmin, vmax, 0.1))
        if self.model_name == 'SAN':
            dldt = np.array([self.model.kvhh.dndt(v, n) for (v, n) in zip(v_grid.ravel(), l_grid.ravel())]).reshape(l_grid.shape)
            dvdt = self.model.dvdt([v_grid, l_grid, self.s[t, 2]])
        elif self.model_name == 'RAN':
            dldt = self.model.kvsi.dmdt(v_grid, l_grid)
            dvdt = self.model.dvdt({
                'v': v_grid, 
                'm_kvsi': l_grid, 
                'ca': self.s[t, 2]
            })
        ct1 = ax.contour(v_grid, l_grid, dldt, 
                         levels=[0], colors='steelblue')
        ct2 = ax.contour(v_grid, l_grid, dvdt, 
                         levels=[0], colors='forestgreen')
        ct1.collections[0].set_label('$dm/dt=0$')
        ct2.collections[0].set_label('$dv/dt=0$')
        if flow:
            ax.streamplot(np.arange(vmin, vmax, 0.1), 
                          np.arange(lmin, lmax, 0.001), 
                          dvdt.T, dldt.T, color='gray')
        return ax

    def diagram(self, ca_range: List, start_points: List[float], 
                ax: plt.axes, plot: bool=True, stability: bool=True, 
                legend: bool=False, ) -> plt.axes :
        eq_color = {
            'Stable node' : 'C0', 
            'Unstable node' : 'C1', 
            'Saddle' : 'C4', 
            'Stable focus' : 'magenta', 
            'Unstable focus' : 'C2', 
            'Center (Hopf)' : 'C5', 
            'Transcritical (Saddle-Node)' : 'C6'
        }
        eq_linestyle = {
            'Stable node' : 'dashed', 
            'Unstable node' : 'solid', 
            'Saddle' : 'dashdot', 
            'Stable focus' : 'dashed', 
            'Unstable focus' : 'solid', 
            'Center (Hopf)' : 'solid', 
            'Transcritical (Saddle-Node)' : 'solid'
        }
        ca_space = np.linspace(ca_range[0], ca_range[1], 1000)[::-1]
        
        def _findroot(func, init):
            sol, _, convergence, _ = scipy.optimize.fsolve(func, init, full_output=1)
            if convergence == 1:
                return sol
            return np.array([np.nan]*1)
        
        def _numerical_continuation(func, v_ini: float, ca_space: np.ndarray):
            eq = []
            for ca in ca_space:
                eq.append(_findroot(lambda x: func(x, ca), 
                                    eq[-1] if eq else v_ini))
            return eq

        def _func(v: float, ca: float) -> float:
            if self.model_name == 'SAN':
                l_inf = self.model.kvhh.n_inf(v=v)
                dvdt = self.model.dvdt([v, l_inf, ca])
            elif self.model_name == 'RAN':
                l_inf = self.model.kvsi.m_inf(v=v)
                dvdt = self.model.dvdt({
                    'v': v, 
                    'm_kvsi': l_inf, 
                    'ca': ca
                })
            return dvdt

        def _jacobian(v, ca):
            x, y = sym.symbols('x, y')
            if self.model_name == 'SAN':
                l = self.model.kvhh.n_inf(v)
                dfdx = sym.diff(self.model.dvdt([x, y, ca]), x)
                dfdy = sym.diff(self.model.dvdt([x, y, ca]), y)
                dgdx = sym.diff(self.model.kvhh.dndt(v=x, n=y), x)
                dgdy = sym.diff(self.model.kvhh.dndt(v=x, n=y), y)
            elif self.model_name == 'RAN':
                l = self.model.kvsi.m_inf(v=v)
                dfdx = sym.diff(self.model.dvdt({'v': x, 'm_kvsi': y, 'ca': ca}), x)
                dfdy = sym.diff(self.model.dvdt({'v': x, 'm_kvsi': y, 'ca': ca}), y)
                dgdx = sym.diff(self.model.kvsi.dmdt(v=x, m=y), x)
                dgdy = sym.diff(self.model.kvsi.dmdt(v=x, m=y), y)
            j = np.array([[np.float(dfdx.subs([(x, v), (y, l)])), 
                           np.float(dfdy.subs([(x, v), (y, l)]))], 
                          [np.float(dgdx.subs([(x, v), (y, l)])), 
                           np.float(dgdy.subs([(x, v), (y, l)]))]])
            return j

        def _stability(j) -> str:
            det = np.linalg.det(j)
            trace = np.matrix.trace(j)
            if np.isclose(trace, 0) and np.isclose(det, 0):
                nat = 'Center (Hopf)'
            elif np.isclose(det, 0):
                nat = 'Transcritical (Saddle-Node)'
            elif det < 0:
                nat = 'Saddle'
            else:
                nat = 'Stable' if trace < 0 else 'Unstable'
                nat += ' focus' if (trace**2 - 4 * det) < 0 else ' node'
            return nat
        
        def _get_branches(start_points):
            branches = []
            for init in start_points:
                eq = _numerical_continuation(_func, init, ca_space)
                nat = [_stability(_jacobian(v, ca))
                       for (v, ca) in zip(eq, ca_space)]
                branches.append((np.array([x for x in eq]), nat))
            return branches

        def _get_segments(nats: List['str']) -> Dict:
            st = 0
            seg = {}
            for i, val in enumerate(nats[1:], 1):
                if val != nats[st] or i == len(nats)-1:
                    seg[(st, i)] = nats[st]
                    st = i
            return seg

        if not plot:
            eq_lst = []
            for init in start_points:
                eq = _numerical_continuation(_func, init, ca_space)
                eq_lst.append(eq)
            return eq_lst

        if not stability:
            for init in start_points:
                eq = _numerical_continuation(_func, init, ca_space)
                ax.plot(ca_space, [x for x in eq], color='k')
            return ax

        branches = _get_branches(start_points)
        labels = frozenset()
        for eq, nat in branches:
            labels = labels.union(frozenset(nat))
            seg = _get_segments(nat)
            for idx, n in seg.items():
                ax.plot(ca_space[idx[0]:idx[1]], eq[idx[0]:idx[1]], 
                        color=eq_color[n] if n in eq_color else 'k', 
                        linestyle=eq_linestyle[n] if n in eq_linestyle else '-')
        if legend:
            ax.legend([mpatches.Patch(color=eq_color[n]) for n in labels],
                      labels,
                      bbox_to_anchor=(1.05, 1), loc='upper left', 
                      borderaxespad=0, fontsize=16)
        return ax


class WavePattern:
    def __init__(self, model: str='AN', wavepattern: str='SWS', 
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

        self.samp_freq=1000
        self.wc = anmodel.analysis.WaveCheck()
    
    def singleprocess(self, args: List) -> None:
        now, param, channel = args
        date: str = f'{now.year}_{now.month}_{now.day}'
        p: Path = Path.cwd().parents[0]
        res_p: Path = p / 'results' / 'bifurcation' / 'wavepattern' / f'{self.model_name}_{self.wavepattern}'
        res_p.mkdir(parents=True, exist_ok=True)
        save_p: Path = res_p / f'{date}_{channel}.pickle'

        magnif_arr = np.arange(0, 2.001, 0.001)
        df = pd.DataFrame(index=magnif_arr, columns=['WavePattern'])
        for magnif in tqdm(magnif_arr):
            if channel != 'g_kleak' and channel != 'g_naleak':
                p = copy(param)
                p[channel] = p[channel] * magnif
                self.model.set_params(p)
            elif channel == 'g_kleak':
                self.model.set_params(param)
                g_kl = self.model.leak.gkl
                g = copy(g_kl)
                self.model.leak.set_gk(g)
            elif channel == 'g_naleak':
                self.model.set_params(param)
                g_nal = self.model.leak.gnal
                g = copy(g_nal)
                self.model.leak.set_gnal(g)

            s, _ = self.model.run_odeint(samp_freq=self.samp_freq)
            if self.wavepattern == 'SWS':
                wp: anmodel.analysis.WavePattern = self.wc.pattern(s[5000:, 0])
            elif self.wavepattern == 'SPN':
                wp: anmodel.analysis.WavePattern = self.wc.pattern_spn(s[5000:, 0])
            df.loc[magnif] = wp
        with open(save_p, 'wb') as f:
            pickle.dump(df, f)

    def multi_singleprocess(self, filename) -> None:
        args = []
        now = datetime.now()
        p: Path = Path.cwd().parents[0]
        data_p: Path = p / 'results' / f'{self.wavepattern}_params' / self.model_name
        with open(data_p/filename, 'rb') as f:
            param = pickle.load(f)
        ch_lst = list(param.index)
        if 'g_leak' in ch_lst:
            ch_lst.remove('g_leak')
            ch_lst.extend(['g_kleak', 'g_naleak'])
        for channel in ch_lst:
            args.append((now, param, channel))
        with Pool(processes=len(param.index)) as pool:
            pool.map(self.singleprocess, args)


class Property:
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
        res_p: Path = p / 'results' / 'bifurcation'  / 'property' / f'{self.model_name}_{self.wavepattern}'
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
                s, _ = self.model.run_odeint(samp_freq=self.samp_freq, samp_len=samp_len)
                v: np.ndarray = s[e[0]:e[6], 0]
                infolst = self.getinfo(v)
                df.loc[i] = infolst
        with open(save_p, 'wb') as f:
            pickle.dump(df, f)


if __name__ == '__main__':
    arg: List = sys.argv
    method = arg[1]
    model = arg[2]
    wavepattern = arg[3]
    filename = arg[4]
    if method == 'wavepattern':
        if model == 'X':
            channel_bool = [1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1]
            model_name = 'RAN'
            wp = analysistools.bifurcation.WavePattern(
                model=model, 
                wavepattern=wavepattern, 
                channel_bool=channel_bool, 
                model_name=model_name, 
            )
        else:
            wp = analysistools.bifurcation.WavePattern(
                model=model, 
                wavepattern=wavepattern, 
            )
        wp.multi_singleprocess(filename)

    elif method == 'property':
        channel = arg[5]
        magnif = float(arg[6])
        if model == 'X':
            channel_bool = [1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1]
            model_name = 'RAN'
            prp = analysistools.bifurcation.Property(
                channel=channel, 
                magnif=magnif, 
                model=model, 
                wavepattern=wavepattern, 
                channel_bool=channel_bool, 
                model_name=model_name, 
            )
        else:
            prp = analysistools.bifurcation.Property(
                channel=channel,
                magnif=magnif, 
                model=model, 
                wavepattern=wavepattern, 
            )
        prp.main(filename=filename)