# -*- coding: utf-8 -*-

"""
This is a module that support manual search for parameters that recapitulate
a cirtain firing pattern in AN model. First, we plot results of simulation 
by given parameter sets that were collected by random parameter search, then
we conduct manual search and record the number of parameter set that 
recapitulate well in the form of csv file. Finally, merge those parameter 
sets into a single data frame and save it for further analysis.
"""

__author__ = 'Tetsuya Yamada'
__status__ = 'Editing'
__version__ = '1.0.0'
__date__ = '23 May 2020'


import sys
import matplotlib.pyplot as plt
from multiprocessing import Pool
import pandas as pd
from pathlib import Path
import pickle
from typing import List, Dict, Tuple, Optional

import models
import readinfo


plt.switch_backend('agg')

class Plot:
    """ Plot parameter sets collected in RPS.
    
    Plot parameter sets collected by random parameter search in AN model for 
    manual detection.

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
                 samp_freq: int=1000, samp_len: int=10, 
                 channel_bool: Optional[List[bool]]=None, 
                 model_name: Optional[str]=None, 
                 ion: bool=False, concentration: Optional[Dict]=None) -> None:
        self.pattern = pattern
        self.ncore = ncore
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

    def singleprocess(self, args: Tuple[pd.DataFrame, Path]) -> None:
        """ Plot parameter sets in a single parameter dataframe.

        Parameters
        ----------
        args : Tuple[pd.DataFrame, Path]
            param_df : pd.DataFrame
                dataframe containing parameter sets to be plotted
            sdir_p : Path
                path for save directory
        """
        param_df, sdir_p = args
        for i in range(len(param_df)):
            file_name: str = f'index_{i}.png'
            save_p: Path = sdir_p / file_name
            param_dict: Dict = param_df.iloc[i].to_dict()
            self.model.set_params(param_dict)
            s, _ = self.model.run_odeint()
            plt.plot(s[self.samp_freq*self.samp_len//2:, 0])
            plt.xlabel('time (msec)')
            plt.ylabel('V (mV)')
            plt.savefig(save_p)
            plt.close()
    
    def multi_singleprocess(self, date: int) -> None:
        """ Plot parameter sets in multi parameter dataframe (using multi core).
        """
        p: Path = Path.cwd().parents[0]
        res_p = p / 'results' / f'{self.pattern}_params' / f'{date}_{self.model_name}'
        args: List = []
        tot_trial_num: int = 0
        for core in range(self.ncore):
            param_p: Path = res_p / f'{self.pattern}_{date}_{core}.pickle'
            with open(param_p, 'rb') as f:
                num: int = pickle.load(f)
                param_df: pd.DataFrame = pickle.load(f)
            tot_trial_num += num
            sdir_p = p / 'results' / 'SWS_soloplot' / f'{date}_{self.model_name}' / str(core)
            sdir_p.mkdir(parents=True, exist_ok=True)
            args.append((param_df, sdir_p))
        print(f'Simulation was conducted {tot_trial_num} times in total.')
        with Pool(processes=self.ncore) as pool:
            pool.map(self.singleprocess, args)


if __name__ == '__main__':
    arg: List = sys.argv
    date: str = arg[1]
    year: str = f'20{date[:2]}'
    month: str = str(int(date[2:4]))
    day: str = date[4:6]
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

    plot = Plot(
        model=str(idf.loc['model'][1]), 
        pattern=str(idf.loc['pattern'][1]), 
        ncore=int(idf.loc['ncore'][1]), 
        samp_freq=int(idf.loc['samp_freq'][1]), 
        samp_len=int(idf.loc['samp_len'][1]), 
        channel_bool=channel_bool, 
        model_name=str(idf.loc['model_name'][1]), 
        ion=ion, 
        concentration=concentration
    )
    plot.multi_singleprocess(f'{year}_{month}_{day}')