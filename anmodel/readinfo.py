# -*- coding: utf-8 -*-

"""
This is a module for reading information csv file. Each method returns 
dataframe containing necessary information for each analysis or simulation.
"""

__author__ = 'Tetsuya Yamada'
__status__ = 'Editing'
__version__ = '1.0.0'
__date__ = '22 May 2020'


import pandas as pd
from pathlib import Path
from typing import Dict

class Read:
    """ Read information files.

    Parameters
    ----------
    date : str
        date when simulation was ran. ex) 200522
    
    Attributes
    ----------
    p : Path
        path for current directory
    date : str
        date when simulation was ran
    info_df : pd.DataFrame
        information dataframe about the simulation
    """
    def __init__(self, date: str) -> None:
        self.p: Path = Path.cwd().parents[0]
        info_p: Path = self.p / 'info' / f'{date}.csv'
        self.date = date
        self.info_df: pd.DataFrame = pd.read_csv(info_p, header=None, index_col=0)
        
    def get_info(self) -> pd.DataFrame:
        return self.info_df

    def channel_bool(self) -> Dict:
        cb_p: Path = self.p / 'info' / f'{self.date}_channel.csv'
        cb_df: pd.DataFrame = pd.read_csv(cb_p, header=None, index_col=0)
        cb_dic: Dict = cb_df.to_dict()[1]
        return cb_dic
        
    def concentration(self) -> Dict:
        c_p: Path = self.p / 'info' / f'{self.date}_concentration.csv'
        c_df: pd.DataFrame = pd.read_csv(c_p, header=None, index_col=0)
        c_dic: Dict = c_df.to_dict()[1]
        return c_dic