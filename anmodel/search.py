# -*- coding: utf-8 -*-

"""
This is a parameter search module. With this module, a certain firing pattern 
can be searched randomly with AN model, SAN model and X model. In order to 
lessen dependence of models on parameters, it's important to make various 
parameter sets (models) and then extract common features among them.\
In this script, parameter sets that recapitulate slow wave sleep (SWS) firing
pattern can be searched with algorithms as described in Tatsuki et al., 2016 
and Yoshida et al., 2018.
"""

__author__ = 'Fumiya Tatsuki, Kensuke Yoshida, Tetsuya Yamada, \
              Takahiro Katsumata, Shoi Shi, Hiroki R. Ueda'
__status__ = 'Published'
__version__ = '1.0.0'
__date__ = '15 May 2020'


import os
import sys
"""
LIMIT THE NUMBER OF THREADS!
change local env variables BEFORE importing numpy
"""
os.environ["OMP_NUM_THREADS"] = "1"  # 2nd likely
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"  # most likely

from datetime import datetime
from multiprocessing import Pool
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from time import time
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

import analysis
import models
import readinfo


class NormalSearch:
    """ Normal Parameter Search (Parameter search from given parameter sets)

    Parameter search from given parameter sets in the form of pd.Dataframe. 
    This type of search is usefule when you want to check the firing states for
    parameter sets which you previously collected, or parameter sets that is
    modified from the original those (ex. bifurcaiton analysis).

    Parameters
    ----------
        model : str
        model in which parameter search is conducted
    pattern : str
        searched firing pattern
    channel_bool : list (bool) or None
        WHEN YOU USE X MODEL, YOU MUST DESIGNATE THIS LIST.\
        Channel lists that X model contains. True means channels incorporated 
        in the model and False means not. The order of the list is the same 
        as other lists or dictionaries that contain channel information in 
        AN model. Example: \
        channel_bool = [
            1,  # leak channel
            0,  # voltage-gated sodium channel
            1,  # HH-type delayed rectifier potassium channel
            0,  # fast A-type potassium channel
            0,  # slowly inactivating potassium channel
            1,  # voltage-gated calcium channel
            1,  # calcium-dependent potassium channel
            1,  # persistent sodium channel
            0,  # inwardly rectifier potassium channel
            0,  # AMPA receptor
            0,  # NMDA receptor
            0,  # GABA receptor
            1,  # calcium pump
        ]\
        This is SAN model, default None
    model_name : str or None
        name of the X model, default None
    ion : bool
        whether you make equiribrium potential variable or not, 
        default False
    concentration : dictionary or str or None
        dictionary of ion concentration, or 'sleep'/'awake' that
        designate typical ion concentrations, default None
    """
    def __init__(self, model: str, pattern: str='SWS',
                 samp_freq: int=1000, samp_len: int=10, 
                 channel_bool: Optional[List[bool]]=None, 
                 model_name: Optional[str]=None, 
                 ion: bool=False, concentration: Optional[Dict]=None) -> None:
        self.wave_check = analysis.WaveCheck(samp_freq=samp_freq)
        self.pattern = pattern
        self.samp_freq = samp_freq
        self.samp_len = samp_len

        if model == 'AN':
            self.model_name = 'AN'
            self.model = models.ANmodel(ion, concentration)
        if model == 'SAN':
            self.model_name = 'SAN'
            self.model = models.SANmodel(ion, concentration)
        if model == "X":
            if channel_bool is None:
                raise TypeError('Designate channel in argument of X model.')
            self.model_name = model_name
            self.model = models.Xmodel(channel_bool, ion, concentration)
    
    def singleprocess(self, df: pd.DataFrame, filename: str) -> None:
        """ Normal parameter search using single core.

        In normal parameter search (parameter search for a given parameter sets), 
        we don't run so many simulations. So single core processing enough works.

        Parameters
        ----------
        df : pd.DataFrame
            parameter sets in the form of pandas.DataFrame (usually gets from 
            self.read_df()).
        filename : str
            the name of the file for saving hit parameter sets.
        """
        p: Path = Path.cwd().parents[0]
        res_p: Path = p / 'results' / f'{self.pattern}_params' / 'NormalSearch'
        res_p.mkdir(parents=True, exist_ok=True)
        save_p: Path = res_p / f'{filename}.pickle'

        param_df: pd.DataFrame = pd.DataFrame([])
        nhit: int = 0
        nfail: int = 0
        for i in range(len(df)):
            params: Dict = df.to_dict(orient='index')[i]
            self.model.set_params(params=params)
            s: np.ndarray
            info: Dict
            s, info  = self.model.run_odeint()

            if info['message'] == 'Excess work done on this call (perhaps wrong Dfun type).':
                pass
            
            v: np.ndarray = s[self.samp_freq*self.samp_len//2:, 0]
            if self.pattern != 'SPN':
                pattern: analysis.WavePattern = self.wave_check.pattern(v=v)
            else:
                pattern: analysis.WavePattern = self.wave_check.pattern_spn(v=v)
            
            print(pattern.name)
            if pattern.name == self.pattern:
                print('Hit!')
                nhit += 1
                param_df = pd.concat([param_df, params])
            else:
                nfail += 1
        
        print(f'Among {len(df)} parameter sets, {nhit} parameter sets hit.')
        with open(str(save_p), "wb") as f:
             pickle.dump(param_df, f)


class RandomSearch():
    """ Random parameter search.
    
    Generate parameter sets randomly, and pick up those which recapitulate a cirtain
    firing pattern. 

    Parameters
    ----------
    model : str
        model in which parameter search is conducted
    pattern : str
        searched firing pattern
    ncore : int
        number of cores you are going to use
    hr : int or str
        how long to run parameter search (hour), default 48 (2 days)
    channel_bool : list (bool) or None
        WHEN YOU USE X MODEL, YOU MUST DESIGNATE THIS LIST.\
        Channel lists that X model contains. True means channels incorporated 
        in the model and False means not. The order of the list is the same 
        as other lists or dictionaries that contain channel information in 
        AN model. Example: \
        channel_bool = [
            1,  # leak channel
            0,  # voltage-gated sodium channel
            1,  # HH-type delayed rectifier potassium channel
            0,  # fast A-type potassium channel
            0,  # slowly inactivating potassium channel
            1,  # voltage-gated calcium channel
            1,  # calcium-dependent potassium channel
            1,  # persistent sodium channel
            0,  # inwardly rectifier potassium channel
            0,  # AMPA receptor
            0,  # NMDA receptor
            0,  # GABA receptor
            1,  # calcium pump
        ]\
        This is SAN model, default None
    model_name : str or None
        name of the X model, default None
    ion : bool
        whether you make equiribrium potential variable or not, 
        default False
    concentration : dictionary or str or None
        dictionary of ion concentration, or 'sleep'/'awake' that
        designate typical ion concentrations, default None

    Attributes
    ----------
    wave_check : object
        Keep attributes and helper functions needed for parameter search.
    pattern : str
        searched firing pattern
    time : int
        how long to run parameter search (hour)
    model_name : str
        model name
    model : object
        Simulation model object. See anmodel.models.py
    """
    def __init__(self, model: str, pattern: str='SWS', ncore: int=1, 
                 hr: int=48, samp_freq: int=1000, samp_len: int=10, 
                 channel_bool: Optional[List[bool]]=None, 
                 model_name: Optional[str]=None, 
                 ion: bool=False, concentration: Optional[Dict]=None) -> None:
        self.wave_check = analysis.WaveCheck(samp_freq=samp_freq)
        self.pattern = pattern
        self.ncore = ncore
        self.hr = int(hr)
        self.samp_freq = samp_freq
        self.samp_len = samp_len

        if model == 'AN':
            self.model_name = 'AN'
            self.model = models.ANmodel(ion, concentration)
        if model == 'SAN':
            self.model_name = 'SAN'
            self.model = models.SANmodel(ion, concentration)
        if model == "X":
            if channel_bool is None:
                raise TypeError('Designate channel in argument of X model.')
            self.model_name = model_name
            self.model = models.Xmodel(channel_bool, ion, concentration)

    def singleprocess(self, args: List) -> None:
        """ Random parameter search using single core.

        Search parameter sets which recapitulate a cirtain firing pattern randomly, 
        and save them every 1 hour. After self.time hours, this process terminates.

        Parameters
        ----------
        args : list
            core : int
                n th core of designated number of cores
            now : datetime.datetime
                datetime.datetime.now() when simulation starts
            time_start : float
                time() when simulation starts
            rand_seed : int
                random seed for generating random parameters. 0 ~ 2**32-1.
        """
        core, now, time_start, rand_seed = args
        date: str = f'{now.year}_{now.month}_{now.day}'
        p: Path = Path.cwd().parents[0]
        res_p: Path = p / 'results' / f'{self.pattern}_params' / f'{date}_{self.model_name}'
        res_p.mkdir(parents=True, exist_ok=True)
        save_p: Path = res_p / f'{self.pattern}_{date}_{core}.pickle'

        param_df: pd.DataFrame = pd.DataFrame([])
        niter: int = 0  # number of iteration
        nhit: int = 0
        nfail: int = 0
        st: float = time()  # start time : updated every 1 hour
        np.random.seed(rand_seed)

        while True:
            niter += 1
            new_params: pd.DataFrame = pd.DataFrame.from_dict(
                self.model.set_rand_params(), orient='index').T
            s: np.ndarray
            info: Dict
            s, info  = self.model.run_odeint()
            
            if info['message'] == 'Excess work done on this call (perhaps wrong Dfun type).':
                pass
            
            v: np.ndarray = s[self.samp_freq*self.samp_len//2:, 0]
            # if you want to detect the SWS firing pattern in the method that 
            # Tatsuki et al. or Yoshida et al. applied, you should use the code below.
            # if self.pattern != 'SPN':
            #     pattern: analysis.WavePattern = self.wave_check.pattern(v=v)
            # else:
            #     pattern: analysis.WavePattern = self.wave_check.pattern_spn(v=v)
            pattern: analysis.WavePattern = self.wave_check.pattern_spn(v=v)
            
            if pattern.name == self.pattern:
                print('Hit!')
                nhit += 1
                param_df = pd.concat([param_df, new_params])
            else:
                nfail += 1
            
            ## save parameters every 1 hour 
            md: float = time()
            if (md - st) > 60 * 60:  # 1 hour
                st: float = time()  # update start time
                with open(str(save_p), "wb") as f:
                    pickle.dump(niter, f)
                    pickle.dump(param_df, f)
                log: str = f'Core {core}: {len(param_df)} {self.pattern} parameter sets were pickled.'
                print(datetime.now(), log)
            
            ## finish random parameter search after "self.time" hours
            if (md - time_start) > 60 * 60 * self.hr:
                print(f'Core {core}: {self.hr} hours have passed, so parameter search has terminated.')
                break

    def multi_singleprocess(self) -> None:
        """ Random parameter search using multi cores.
        """
        args: List = []
        now: datetime = datetime.now()
        time_start: float = time()
        for core in range(self.ncore):
            args.append((core, now, time_start, np.random.randint(0, 2 ** 32 - 1, 1)))

        print(f'Random search: using {self.ncore} cores to explore {self.pattern}')
        with Pool(processes=self.ncore) as pool:
            pool.map(self.singleprocess, args)


if __name__ == '__main__':
    arg: List = sys.argv
    search_method: str = arg[1]
    date: str = arg[2]
    read: readinfo.Read = readinfo.Read(date=date)
    idf: pd.DataFrame = read.get_info()

    if not pd.isnull(idf.loc['channel_bool'][1]):
        channel_bool: Dict = read.channel_bool()
    else:
        channel_bool = None
    if not pd.isnull(idf.loc['ion'][1]):
        ion: bool = True
        concentration: Dict = read.concentration()
    else:
        ion: bool = False
        concentration = None

    if search_method == 'ns':  # Normal Search
        ns = NormalSearch(
            model=str(idf.loc['model'][1]), 
            pattern=str(idf.loc['pattern'][1]), 
            samp_freq=int(idf.loc['samp_freq'][1]), 
            samp_len=int(idf.loc['samp_len'][1]), 
            channel_bool=channel_bool, 
            model_name=str(idf.loc['model_name'][1]), 
            ion=ion, 
            concentration=concentration, 
        )
        filename: str = arg[3]
        df: pd.DataFrame = read.paramdf(filename=filename)
        df = read._setcolname(df=df, model=str(idf.loc['model'][1]), channel_bool=channel_bool)
        ns.singleprocess(df=df, filename=filename)

    elif search_method == 'rs':  # Random Search
        rps = RandomSearch(
            model=str(idf.loc['model'][1]), 
            pattern=str(idf.loc['pattern'][1]), 
            ncore=int(idf.loc['ncore'][1]), 
            hr=int(idf.loc['hr'][1]), 
            samp_freq=int(idf.loc['samp_freq'][1]), 
            samp_len=int(idf.loc['samp_len'][1]), 
            channel_bool=channel_bool, 
            model_name=str(idf.loc['model_name'][1]), 
            ion=ion, 
            concentration=concentration, 
        )
        rps.multi_singleprocess()
