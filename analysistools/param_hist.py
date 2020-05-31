# -*- coding: utf-8 -*-

"""
This is a module for drawing parameter distribution histogram. 
"""

__author__ = 'Tetsuya Yamada'
__status__ = 'Editing'
__version__ = '1.0.0'
__date__ = '31 May 2020'


import os
import sys
sys.path.append('../')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from scipy.stats import gaussian_kde
from typing import Dict, List, Iterator

import anmodel
from analysistools import tools

class ParamHist:
    """ Draw parameter distribution histogram.

    TODO: implement in other than AN model

    Parameters
    ----------
    date : str
        date when the simulation was conducted
    """
    def __init__(self, date: str):
        read = anmodel.readinfo.Read(date=date)
        idf: pd.DataFrame = read.get_info()
        self.model: str = str(idf.loc['model'][1])
        self.pattern: str = str(idf.loc['pattern'][1])

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

        if self.model == 'AN':
            self.model_name = 'AN'
            self.model = anmodel.models.ANmodel(ion, concentration)
        if self.model == 'SAN':
            self.model_name = 'SAN'
            self.model = anmodel.models.SANmodel(ion, concentration)
        if self.model == "X":
            if channel_bool is None:
                raise TypeError('Designate channel in argument of X model.')
            self.model_name = self.model_name
            self.model = anmodel.models.Xmodel(channel_bool, ion, concentration)
    
        year: str = f'20{date[:2]}'
        month: str = str(int(date[2:4]))
        day: str = date[4:6]
        p: Path = Path.cwd().parents[0]
        res_p: Path = p / 'results' / f'{self.pattern}_params' / f'{year}_{month}_{day}_{self.model_name}'
        save_p = res_p / f'{self.pattern}_{year}_{month}_{day}_hitmerged.pickle'
        with open(save_p, 'rb') as f:
            self.param_df: pd.DataFrame = pickle.load(f)
        self.logparam_df: pd.DataFrame = self.param_df.applymap(np.log10)

        self.grid_args: Dict = tools.get_gridargs()
        self.bin_list: List = self.get_binlist()

        """
        TODO: implement in other than AN model
        """
    
    def get_binlist(self) -> List:
        """ Get binsize for each channel.

        Returns
        ----------
        bin_list : List[int]
            binsize for each channel
        """
        bin_list: List = []
        for c in list(self.grid_args.keys()):
            param_range: float = self.grid_args[c][1] - self.grid_args[c][0]
            channel_bin: int = int((self.logparam_df[c].max()-self.logparam_df[c].min())/param_range*100)
            bin_list.append(channel_bin)
        return bin_list

    def plot(self):
        """ Plot histogram and fitting curve.
        """
        _, ax = plt.subplots(3, 5, figsize=(16, 10))
        for i, c in enumerate(self.grid_args.keys()):
            x_grid = np.linspace(self.grid_args[c][0], self.grid_args[c][1], 100)
            kde_model = gaussian_kde(self.logparam_df.loc[:, c])
            y = kde_model(x_grid)
            ax[i//5, i%5].plot(x_grid, y)
            ax[i//5, i%5].hist(self.logparam_df.loc[:, c], alpha=0.3, bins=self.bin_list[i], normed=1)
            ax[i//5, i%5].set_xlim(self.grid_args[c][0], self.grid_args[c][1])
            ax[i//5, i%5].set_title(c)        
