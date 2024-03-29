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
from scipy.integrate import odeint
import scipy.optimize
from scipy.stats import gaussian_kde
import sympy as sym
from tqdm import tqdm
from typing import Dict, List, Tuple, Iterator, Optional

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

        self.samp_freq = 100000
        self.l_pad = 0.1
        self.v_pad = 10

    def change_params(self, channel:str, magnif: float) -> None:
        if channel != 'g_nal' and channel != 'g_kl':
            p = copy(self.param)
            p[channel] = p[channel] * magnif
            self.model.set_params(p)
        else:
            self.model.set_params(self.param)
            self.model.leak.set_div()
            gnal = self.model.leak.gnal
            gkl = self.model.leak.gkl
            if channel == 'g_nal':
                gxl = copy(gnal)
                gxl = gxl * magnif
                self.model.leak.set_gna(gxl)
            elif channel == 'g_kl':
                gxl = copy(gkl)
                gxl = gxl * magnif
                self.model.leak.set_gk(gxl)

    def ode_bifur(self, channel: Optional[str]=None, 
                  magnif: Optional[float]=None) -> np.ndarray:
        if channel is not None:
            self.change_params(channel, magnif)
        s, _ = self.model.run_odeint(samp_freq=self.samp_freq)
        return s

    def set_s(self, st: int, en: int, 
              channel: Optional[str]=None, 
              magnif: Optional[float]=None) -> None:
        self.s = self.ode_bifur(channel, magnif)
        self.st = st
        self.en = en

    def nullcline(self, t: int, ax: plt.axes, 
                  mode='t', flow=False, density=1, 
                  channel: Optional[str]=None, 
                  magnif: Optional[float]=None) -> plt.axes:
        lmin = self.s[self.st:self.en, 1].min() - self.l_pad
        lmax = self.s[self.st:self.en, 1].max() + self.l_pad
        vmin = self.s[self.st:self.en, 0].min() - self.v_pad
        vmax = self.s[self.st:self.en, 0].max() + self.v_pad
        l_grid, v_grid = np.meshgrid(np.arange(lmin, lmax, 0.001), 
                                     np.arange(vmin, vmax, 0.1))
        if mode == 't':
            ca = self.s[t, 2]
        elif mode == 'ca':
            ca = t
        if self.model_name == 'SAN':
            dldt = np.array([self.model.kvhh.dndt(v, n) for (v, n) in zip(v_grid.ravel(), l_grid.ravel())]).reshape(l_grid.shape)
            dvdt = self.model.dvdt([v_grid, l_grid, ca])
        elif self.model_name == 'RAN':
            dldt = self.model.kvsi.dmdt(v_grid, l_grid)
            dvdt = self.model.dvdt({
                'v': v_grid, 
                'm_kvsi': l_grid, 
                'ca': ca, 
            })
        ct1 = ax.contour(v_grid, l_grid, dldt, 
                         levels=[0], colors='steelblue', # 4682b4
                         linestyles='-', linewidths=3)
        ct2 = ax.contour(v_grid, l_grid, dvdt, 
                         levels=[0], colors='forestgreen', # 228b22
                         linestyles='-', linewidths=3)
        # ct1.collections[0].set_label('$dm/dt=0$')
        # ct2.collections[0].set_label('$dv/dt=0$')
        if flow:
            ax.streamplot(np.arange(vmin, vmax, 0.1), 
                          np.arange(lmin, lmax, 0.001), 
                          dvdt.T, dldt.T, color='gray', density=density)
        return ax

    def diagram(self, ca_range: List, start_points: List[float], 
                ax: plt.axes, plot: bool=True, stability: bool=True, 
                legend: bool=False, color=True) -> plt.axes :
        eq_color = {
            'Stable node' : 'C0', 
            'Unstable node' : 'limegreen', # ff8c00
            'Saddle' : 'darkorange', # 556b2f
            'Stable focus' : 'turquoise', # 4169e1
            'Unstable focus' : 'slateblue', # c71585
            'Center (Hopf)' : 'C5', 
            'Transcritical (Saddle-Node)' : 'C6'
        }
        eq_linestyle = {
            'Stable node' : 'solid', 
            'Unstable node' : 'solid', 
            'Saddle' : 'solid', 
            'Stable focus' : 'solid', 
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
                # ax.plot(ca_space[idx[0]:idx[1]], eq[idx[0]:idx[1]], 
                #         color=eq_color[n] if n in eq_color else 'k', 
                #         linestyle=eq_linestyle[n] if n in eq_linestyle else '-')
                if color:
                    ax.plot(eq[idx[0]:idx[1]], ca_space[idx[0]:idx[1]], 
                            color=eq_color[n] if n in eq_color else 'k', 
                            linestyle=eq_linestyle[n] if n in eq_linestyle else '-', 
                            linewidth=4)
                else: 
                    ax.plot(eq[idx[0]:idx[1]], ca_space[idx[0]:idx[1]], 
                            color='gray', 
                            linestyle=eq_linestyle[n] if n in eq_linestyle else '-', 
                            linewidth=4)
        if legend:
            ax.legend([mpatches.Patch(color=eq_color[n]) for n in labels],
                      labels,
                      bbox_to_anchor=(1.05, 1), loc='upper left', 
                      borderaxespad=0, fontsize=16)
        return ax


class AttractorAnalysis:
    def __init__(self, model: str='AN', wavepattern: str='SWS', 
                 channel_bool: Optional[Dict]=None, 
                 model_name: Optional[str]=None, 
                 ion: bool=False, concentration: Dict=None) -> None:
        self.model = model
        self.wavepattern = wavepattern
        if self.model == 'SAN':
            self.model_name = 'SAN'
            self.model = anmodel.models.SANmodel(ion, concentration)
        if self.model == 'X':
            if channel_bool is None:
                raise TypeError('Designate channel in argument of X model.')
            self.model_name = model_name
            self.model = anmodel.models.Xmodel(channel_bool, ion, concentration)

        self.samp_freq = 10000

    def dvdt(self, args: List) -> float:
        if self.model_name == 'SAN':
            v, n = args
            return ((-10.0*self.model.params.area 
                * (self.model.kvhh.i(v, n=n) 
                + self.model.cav.i(v) 
                + self.model.kca.i(v, ca=self.ca) 
                + self.model.nap.i(v) 
                + self.model.leak.i(v))) 
                / (10.0*self.model.params.cm*self.model.params.area))
        elif self.model_name == 'RAN':
            v, m = args
            return ((-10.0*self.model.params.area 
                * (self.model.kvsi.i(v, m=m) 
                + self.model.cav.i(v) 
                + self.model.kca.i(v, ca=self.ca) 
                + self.model.nap.i(v) 
                + self.model.leak.i(v))) 
                / (10.0*self.model.params.cm*self.model.params.area))
    
    def diff_op(self, args: List, time: np.ndarray) -> List:
        if self.model_name == 'SAN':
            v, n = args
            dvdt = self.dvdt(args)
            dndt = self.model.kvhh.dndt(v=v, n=n)
            return [dvdt, dndt]
        elif self.model_name == 'RAN':
            v, m = args
            dvdt = self.dvdt(args)
            dmdt = self.model.kvsi.dmdt(v=v, m=m)
            return [dvdt, dmdt]

    def run_odeint(self, ini: List, samp_len: int=10):
        solvetime = np.linspace(1, 1000*samp_len, self.samp_freq*samp_len)
        s, _ = odeint(self.diff_op, ini, solvetime, 
                      atol=1.0e-5, rtol=1.0e-5, full_output=True)
        return s

    def change_param(self, channel: str, magnif: float) -> None:
        if channel != 'g_nal' and channel != 'g_kl':
            p = copy(self.param)
            p[channel] = p[channel] * magnif
            self.model.set_params(p)
        else:
            self.model.set_params(self.param)
            self.model.leak.set_div()
            gnal = self.model.leak.gnal
            gkl = self.model.leak.gkl
            if channel == 'g_nal':
                gxl = copy(gnal)
                gxl = gxl * magnif
                self.model.leak.set_gna(gxl)
            elif channel == 'g_kl':
                gxl = copy(gkl)
                gxl = gxl * magnif
                self.model.leak.set_gk(gxl)

    def find_attractor(self, start_points: Tuple) -> float:
        def _findroot(func, init: float) -> np.ndarray:
            sol, _, convergence, _ = scipy.optimize.fsolve(func, init, full_output=1)
            if convergence == 1:
                return sol
            return np.array([np.nan]*1)
        
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

        def _jacobian(v: float, ca: float) -> np.ndarray:
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

        def _stability(j: np.ndarray) -> str:
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
        
        for init in start_points:
            eq = _findroot(lambda x: _func(x, self.ca), init)[0]
            nat = _stability(_jacobian(eq, self.ca))
            if nat == 'Stable focus':
                return eq  # v value at stable focus attractor
        raise Exception('Stabel focus attractor was not found.')

    def singleprocess(self, args: Tuple) -> None:
        core, res_df, self.ca, v_start, res_p, resname, channel, magnif = args
        samp_len = 10
        vini_lst = res_df.columns
        lini_lst = res_df.index
        if channel is not None:
            self.change_param(channel, magnif)
        vatt = self.find_attractor(v_start)
        if self.model_name == 'SAN':
            latt = self.model.kvhh.n_inf(v=vatt)
        elif self.model_name == 'RAN':
            latt = self.model.kvsi.m_inf(v=vatt)

        def _recur(ini: List, samp_len: str) -> float:
            s = self.run_odeint(ini, samp_len)
            try:
                t_v = np.where(np.abs(s[:, 0]-vatt) < 1.0e-3)[0]
                t_l = np.where(np.abs(s[:, 1]-latt) < 1.0e-5)[0]
                t = np.intersect1d(t_v, t_l)[0]
                return t
            except IndexError:
                if samp_len > 100:
                    raise Exception
                samp_len = samp_len * 2
                _recur(ini, samp_len)

        for lini in lini_lst:
            for vini in tqdm(vini_lst):
                ini = [vini, lini]
                t = _recur(ini, samp_len)
                # t = np.max([t_v, t_l])
                res_df.loc[lini, vini] = t
        with open(res_p/resname, 'wb') as f:
            pickle.dump(res_df, f)
        
    def multi_singleprocess(self, ncore: int, filename: str, 
                            vargs: List, largs: List , 
                            ca: float, v_start: Tuple, 
                            channel: Optional[str]=None, 
                            magnif: Optional[float]=None) -> None:
        """
        Parameters
        ----------
        vargs : Tuple
            vmin, vmax, vint
        largs : Tuple
            l: n_kvhh in SAN model and m_kvsi in RAN model. lmin, lmax, lint
        """
        now = datetime.now()
        date: str = f'{now.year}_{now.month}_{now.day}'
        p: Path = Path.cwd().parents[0]
        data_p: Path = p / 'results' / f'{self.wavepattern}_params' / self.model_name
        if channel is None:
            dicname = f'{ca}_{vargs[0]}_{vargs[1]}_{largs[0]}_{largs[1]}'
        else:
            dicname = f'{ca}_{vargs[0]}_{vargs[1]}_{largs[0]}_{largs[1]}_{channel}_{magnif}'
        res_p: Path = p / 'results' / 'bifurcation' / 'attractor_time' / f'{self.model_name}' / dicname
        res_p.mkdir(parents=True, exist_ok=True)
        with open(data_p/filename, 'rb') as f:
            self.param = pickle.load(f)
        self.model.set_params(self.param)
        v_lst = np.linspace(vargs[0], vargs[1], vargs[2])
        l_lst = np.linspace(largs[0], largs[1], vargs[2])

        args: List = []
        for core, ll_lst in enumerate(np.array_split(l_lst, ncore)):
            res_df = pd.DataFrame(index=ll_lst, columns=v_lst) 
            resname = f'{date}_{filename}_{core}.pickle'
            args.append((core, res_df, ca, v_start, res_p, resname, channel, magnif))
        with Pool(processes=ncore) as pool:
            pool.map(self.singleprocess, args)

    def load_data(self, date: str, filename: str,  
                  ncore: int, ca: float, 
                  vrange: List, lrange: List, 
                  channel: Optional[str]=None, 
                  magnif: Optional[float]=None) -> None:
        p: Path = Path.cwd().parents[0]
        if channel is None:
            dicname = f'{ca}_{vrange[0]}_{vrange[1]}_{lrange[0]}_{lrange[1]}'
        else:
            dicname = f'{ca}_{vrange[0]}_{vrange[1]}_{lrange[0]}_{lrange[1]}_{channel}_{magnif}'
        res_p: Path = p / 'results' / 'bifurcation' / 'attractor_time' / f'{self.model_name}' / dicname
        with open(res_p/f'{date}_{filename}.pickle_0.pickle', 'rb') as f:
            self.res_df = pickle.load(f)
        for core in range(1, ncore):
            resname = f'{date}_{filename}.pickle_{core}.pickle'
            with open(res_p/resname, 'rb') as f:
                r_df = pickle.load(f)
            self.res_df = pd.concat([self.res_df, r_df]).sort_index(ascending=False)

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
                self.model.leak.set_div()
                self.model.set_params(param)
                g_kl = self.model.leak.gkl
                g = copy(g_kl)
                g = g * magnif
                self.model.leak.set_gk(g)
            elif channel == 'g_naleak':
                self.model.leak.set_div()
                self.model.set_params(param)
                g_nal = self.model.leak.gnal
                g = copy(g_nal)
                g = g * magnif
                self.model.leak.set_gna(g)

            s, _ = self.model.run_odeint(samp_freq=self.samp_freq)
            # if you want to detect the SWS firing pattern in the method that 
            # Tatsuki et al. or Yoshida et al. applied, you should use the code below.
            # if self.wavepattern == 'SWS':
            #     wp: anmodel.analysis.WavePattern = self.wc.pattern(s[5000:, 0])
            # elif self.wavepattern == 'SPN':
            #     wp: anmodel.analysis.WavePattern = self.wc.pattern_spn(s[5000:, 0])
            if self.wavepattern == 'SWS':
                spike = 'peak'
            elif self.wavepattern == 'SPN':
                spike = 'bottom'
            wp: anmodel.analysis.WavePattern = self.wc.pattern_spn(s[5000:, 0], spike)
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
        with Pool(processes=len(ch_lst)) as pool:
            pool.map(self.singleprocess, args)


class Simple(WavePattern):
    def __init__(self, model: str='AN', wavepattern: str='SWS', 
                 channel_bool: Optional[Dict]=None, 
                 model_name: Optional[str]=None, 
                 ion: bool=False, concentration: Dict=None) -> None:
        super().__init__(model, wavepattern, channel_bool, model_name, 
                         ion, concentration)

    def singleprocess(self, args: List) -> None:
        now, df, channel = args
        date: str = f'{now.year}_{now.month}_{now.day}'
        p: Path = Path.cwd().parents[0]
        res_p: Path = p / 'results' / 'bifurcation' / 'simple' / f'{self.model_name}_{self.wavepattern}'
        res_p.mkdir(parents=True, exist_ok=True)
        save_p: Path = res_p / f'{date}_{channel}.pickle'
        res_df = pd.DataFrame(index=range(len(df)), columns=[channel])

        def _judge() -> anmodel.analysis.WavePattern:
            s, _ = self.model.run_odeint(samp_freq=self.samp_freq)
            # if you want to detect the SWS firing pattern in the method that 
            # Tatsuki et al. or Yoshida et al. applied, you should use the code below.
            # if self.wavepattern == 'SWS':
            #     wp: anmodel.analysis.WavePattern = self.wc.pattern(s[5000:, 0])
            # elif self.wavepattern == 'SPN':
            #     wp: anmodel.analysis.WavePattern = self.wc.pattern_spn(s[5000:, 0])
            if self.wavepattern == 'SWS':
                spike = 'peak'
            elif self.wavepattern == 'SPN':
                spike = 'bottom'
            wp: anmodel.analysis.WavePattern = self.wc.pattern_spn(s[5000:, 0], spike)
            return wp

        if channel != 'g_kleak' and channel != 'g_naleak':
            for i in tqdm(range(len(df))):
                param = df.iloc[i, :]
                if channel != 't_ca':
                    param[channel] = param[channel] / 1000
                else:
                    param[channel] = param[channel] * 1000
                self.model.set_params(param)
                wp = _judge()
                res_df.iloc[i, 0] = wp
        elif channel == 'g_kleak':
            self.model.leak.set_div()
            for i in tqdm(range(len(df))):
                param = df.iloc[i, :]
                self.model.set_params(param)
                g_kl = self.model.leak.gkl
                self.model.leak.set_gk(g_kl/1000)
                wp = _judge()
                res_df.iloc[i, 0] = wp
        elif channel == 'g_naleak':
            self.model.leak.set_div()
            for i in tqdm(range(len(df))):
                param = df.iloc[i, :]
                self.model.set_params(param)
                g_nal = self.model.leak.gnal
                self.model.leak.set_gna(g_nal/1000)
                wp = _judge()
                res_df.iloc[i, 0] = wp
        with open(save_p, 'wb') as f:
            pickle.dump(res_df, f)

    def multi_singleprocess(self, filename, divleak=False) -> None:
        args = []
        now = datetime.now()
        p: Path = Path.cwd().parents[0]
        data_p: Path = p / 'results' / f'{self.wavepattern}_params' / self.model_name
        with open(data_p/filename, 'rb') as f:
            df = pickle.load(f)
        ch_lst = list(df.columns)
        if 'g_leak' in ch_lst:
            if divleak:
                ch_lst.remove('g_leak')
                ch_lst.extend(['g_kleak', 'g_naleak'])
            else:
                pass
        for channel in ch_lst:
            args.append((now, df, channel))
        with Pool(processes=len(ch_lst)) as pool:
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

    def main(self, filename: str, t_filename='normal'):
        now: datetime = datetime.now()
        date: str = f'{now.year}_{now.month}_{now.day}'
        p: Path = Path.cwd().parents[0]
        res_p: Path = p / 'results' / 'bifurcation'  / 'property' / f'{self.model_name}_{self.wavepattern}'
        res_p.mkdir(parents=True, exist_ok=True)
        save_p: Path = res_p /f'{filename}'
        save_p.mkdir(parents=True, exist_ok=True)

        data_p: Path = p / 'results' / f'{self.wavepattern}_params' / self.model_name
        time_p: Path = p / 'results' / 'normalization_mp_ca'
        if t_filename == 'normal':
            t_file = time_p / f'{self.wavepattern}_{self.model_name}_time.pickle'
        elif t_filename == 'bifur':
            t_file = time_p /'bifurcation_all'/f'{self.model_name}' / f'{filename}_{self.channel}_{self.magnif}_time.pickle'
        else:
            t_file = time_p / f'{t_filename}'

        with open(data_p/filename, 'rb') as f:
            param_df = pickle.load(f)
            param_df.index = range(len(param_df))
        with open(t_file, 'rb') as f:
            time_df = pickle.load(f).dropna(how='all')
            time_df.index = range(len(time_df))
        
        print(len(param_df))
        print(len(time_df))
        if len(param_df) != len(time_df):
            raise IndexError('Parameter dataframe and time dataframe do not match!!')

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
            if self.channel != 'g_kleak' and self.channel != 'g_naleak':
                param[self.channel] = param[self.channel] * self.magnif
                self.model.set_params(param)
            elif self.channel == 'g_kleak' or self.channel == 'g_naleak':
                self.model.set_params(param)
                self.model.leak.set_div()
                if self.channel == 'g_kleak':
                    g_kl = self.model.leak.gkl
                    g_kl = g_kl * self.magnif
                    self.model.leak.set_gk(g_kl)
                elif self.channel == 'g_naleak':
                    g_nal = self.model.leak.gnal
                    g_nal = g_nal * self.magnif
                    self.model.leak.set_gna(g_nal)

            e = time_df.iloc[i, :]
            try:
                samp_len = 10 + ((5000+e[6])//10000) * 10
            except TypeError:
                continue
            s, _ = self.model.run_odeint(samp_freq=self.samp_freq, samp_len=samp_len)
            v: np.ndarray = s[e[0]:e[6], 0]
            infolst = self.getinfo(v)
            df.loc[i] = infolst
        with open(save_p/f'{date}_{self.channel}_{self.magnif}.pickle', 'wb') as f:
            pickle.dump(df, f)

    def plot_singleprocess(self, args: List):
        _, df, channel, pct = args
        p: Path = Path.cwd().parents[0]
        res_p: Path = p / 'results' / 'bifurcation'  / 'plot' / f'{self.model_name}_{self.wavepattern}' / f'{channel}_{pct}'
        res_p.mkdir(parents=True, exist_ok=True)
        
        for idx in df.index:
            _, ax = plt.subplots(1, 3, figsize=(14, 4))
            param = df.loc[idx, :]
            self.model.set_params(param)
            s, _ = self.model.run_odeint()
            ax[1].plot(s[5000:, 0])

            if channel != 'g_kleak' and channel != 'g_naleak':
                param_sm = copy(param)
                param_lg = copy(param)
                param_sm[channel] = param_sm[channel] * (1-pct)
                param_lg[channel] = param_lg[channel] * (1+pct)
                self.model.set_params(param_sm)
                s_sm, _ = self.model.run_odeint()
                self.model.set_params(param_lg)
                s_lg, _ = self.model.run_odeint()
                s_lg, _ = self.model.run_odeint()
            elif channel == 'g_kleak' or channel == 'g_naleak':
                self.model.set_params(param)
                self.model.leak.set_div()
                if channel == 'g_kleak':
                    g_kl = self.model.leak.gkl
                    g_sm = copy(g_kl)
                    g_lg = copy(g_kl)
                    g_sm = g_sm * (1-pct)
                    g_lg = g_lg * (1+pct)
                    self.model.leak.set_gk(g_sm)
                    s_sm, _ = self.model.run_odeint()
                    self.model.leak.set_gk(g_lg)
                    s_lg, _ = self.model.run_odeint()
                elif channel == 'g_naleak':
                    g_nal = self.model.leak.gnal
                    g_sm = copy(g_nal)
                    g_lg = copy(g_nal)
                    g_sm = g_sm * (1-pct)
                    g_lg = g_lg * (1+pct)
                    self.model.leak.set_gna(g_sm)
                    s_sm, _ = self.model.run_odeint()
                    self.model.leak.set_gna(g_lg)
                    s_lg, _ = self.model.run_odeint()
                self.model.leak.reset_div()

            ax[0].plot(s_sm[5000:, 0])
            ax[2].plot(s_lg[5000:, 0])
            plt.savefig(res_p/f'index_{idx}')

    def plot_multisingleprocess(self, filename, pct, ncore):
        args = []
        p: Path = Path.cwd().parents[0]
        res_p = p / 'results' / f'{self.wavepattern}_params' / f'{self.model_name}' / filename
        with open(res_p, 'rb') as f:
            param_df = pickle.load(f)
        param_df.index = range(len(param_df))
        for core in range(ncore):
            group = pd.qcut(list(param_df.index), ncore, labels=False)
            df = param_df.loc[group==core, :]
            args.append((core, df, self.channel, pct))
        with Pool(processes=ncore) as pool:
            pool.map(self.plot_singleprocess, args)
            

if __name__ == '__main__':
    arg: List = sys.argv
    method = arg[1]
    model = arg[2]
    wavepattern = arg[3]
    filename = arg[4]
    if method == 'wavepattern':
        if model == 'RAN':
            channel_bool = [1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1]
            model_name = 'RAN'
            wp = analysistools.bifurcation.WavePattern(
                model='X', 
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

    elif method == 'simple':
        if model == 'RAN':
            channel_bool = [1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1]
            model_name = 'RAN'
            sm = analysistools.bifurcation.Simple(
                model='X', 
                wavepattern=wavepattern, 
                channel_bool=channel_bool, 
                model_name=model_name, 
            )
        else:
            sm = analysistools.bifurcation.Simple(
                model=model, 
                wavepattern=wavepattern, 
            )
        sm.multi_singleprocess(filename)

    elif method == 'property':
        t_filename = arg[5]
        channel = arg[6]
        magnif = float(arg[7])
        if model == 'RAN':
            channel_bool = [1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1]
            model_name = 'RAN'
            prp = analysistools.bifurcation.Property(
                channel=channel, 
                magnif=magnif, 
                model='X', 
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
        prp.main(filename=filename, t_filename=t_filename)

    elif method == 'attractor_time':
        ncore = int(arg[5])
        vargs = (float(arg[6]), float(arg[7]), int(arg[8]))
        largs = (float(arg[9]), float(arg[10]), int(arg[11]))
        ca = float(arg[12])
        v_start = (float(arg[13]), float(arg[14]))
        try:
            channel = arg[15]
            magnif = float(arg[16])
        except IndexError:
            channel = None
            magnif = None
        if model == 'RAN':
            channel_bool = [1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1]
            model_name = 'RAN'
            at = analysistools.bifurcation.AttractorAnalysis(
                model='X', 
                model_name=model_name, 
                channel_bool=channel_bool, 
                wavepattern=wavepattern, 
            )
        else:
            at = analysistools.bifurcation.AttractorAnalysis(
                model=model, 
                wavepattern=wavepattern,
            )
        at.multi_singleprocess(ncore, filename, vargs, largs, ca, v_start, channel, magnif)

    elif method == 'plot':
        channel = arg[5]
        magnif = None
        pct = float(arg[6])
        ncore = int(arg[7])
        if model == 'RAN':
            channel_bool = [1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1]
            model_name = 'RAN'
            prp = analysistools.bifurcation.Property(
                channel=channel, 
                magnif=magnif, 
                model='X', 
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
        prp.plot_multisingleprocess(filename=filename, pct=pct, ncore=ncore)
