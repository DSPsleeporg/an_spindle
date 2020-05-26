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
import pickle
from pathlib import Path
from typing import List, Dict, Optional

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

    def channel_bool(self) -> List[bool]:
        cb_p: Path = self.p / 'info' / f'{self.date}_channel.csv'
        cb_df: pd.DataFrame = pd.read_csv(cb_p, header=None, index_col=0)
        channel_bool: List[bool] = cb_df.loc[0].values.astype(bool).tolist()
        return channel_bool
        
    def concentration(self) -> Dict:
        c_p: Path = self.p / 'info' / f'{self.date}_concentration.csv'
        c_df: pd.DataFrame = pd.read_csv(c_p, header=None, index_col=0)
        c_dic: Dict = c_df.to_dict()[1]
        return c_dic

    def paramdf(self, filename: str) -> pd.DataFrame:
        """ Read dataframe that contains parameter sets.

        Parameters
        ----------
        filename : str
            name of the pickle file that contains parameter sets

        Returns
        ----------
        pd.DataFrame
            parameter sets in the form of pandas.DataFrame
        """
        df_p: Path = self.p / 'info' / 'search_df' / f'{filename}.pickle'
        with open(df_p, mode='rb') as f:
            df: pd.DataFrame = pickle.load(f)
        df = df.reset_index()
        return df
    
    def _setcolname(self, df: pd.DataFrame, 
                    model: str, channel_bool: Optional[List]=None):
        colname: List = [
            'g_leak', 'g_nav', 'g_kvhh', 'g_kva', 'g_kvsi', 
            'g_cav', 'g_kca', 'g_nap', 'g_kir', 
            'g_ampar', 'g_nmdar', 'g_gabar', 't_ca',
        ]
        if model == 'SAN':
            colname: List = [
                'g_leak', 'g_kvhh', 'g_cav', 'g_kca', 'g_nap', 't_ca', 
            ]
        elif model == 'X':
            colname: List = [colname[i] for i in range(len(colname)) if channel_bool[i]]
        colname.insert(0, 'index')
        df.columns = colname
        return df