# -*- coding: utf-8 -*-

__author__ = 'Tetsuya Yamada'
__status__ = 'Editing'
__version__ = '1.0.0'
__date__ = '31 May 2020'


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from scipy.stats import gaussian_kde
import sklearn
import sklearn.decomposition
from typing import Dict, List, Iterator, Optional, Tuple

import anmodel
from analysistools import tools

class PCA:
    def __init__(self, model: str='AN', 
                 channel_bool: Optional[Dict]=None, 
                 model_name: Optional[str]=None, 
                 ion: bool=False, concentration: Dict=None) -> None:
        self.model = model
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

        self.pca = sklearn.decomposition.PCA()

    def main(self, param_df: pd.DataFrame, pc: int=2) -> Tuple:
        logparam_df = param_df.applymap(np.log10)
        normparam_df = self.normalize(logparam_df)
        pc_df, pc_cov = self.run_pca(normparam_df)
        maxidx = self.get_repidx(pc_df, pc=pc)
        repparam = param_df.iloc[maxidx]
        reppc = pc_df.iloc[maxidx]
        return (pc_df, pc_cov, repparam, reppc)

    def plot_pc12(self, param_df: pd.DataFrame,
                  ax: mpl.axes.Axes) -> mpl.axes.Axes:
        pc_df, _, _, reppc = self.main(param_df)
        x, y = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
        pos = np.vstack([x.ravel(), y.ravel()])
        kernel_pc12 = gaussian_kde(pc_df[['pc1', 'pc2']].T)
        z = np.reshape(kernel_pc12.pdf(pos), x.shape)
        ax.scatter(pc_df['pc1'], pc_df['pc2'], s=9, alpha=0.7, color='skyblue')
        ax.scatter(reppc['pc1'], reppc['pc2'], s=9, color='red')
        ax.contour(x, y, z, alpha=1, cmap='OrRd')
        return ax

    def normalize(self, logparam_df: pd.DataFrame) -> pd.DataFrame:
        grid_args: Dict = tools.get_gridargs()
        ga_df: pd.DataFrame = pd.DataFrame(grid_args)
        start: pd.Series = ga_df.iloc[0]
        end: pd.Series = ga_df.iloc[1]
        normparam_df = (logparam_df-start) / (end-start)
        normparam_df = normparam_df.dropna(how='all', axis=1)
        return normparam_df

    def run_pca(self, normparam_df: pd.DataFrame) -> (pd.DataFrame, np.ndarray):
        self.pca.fit(normparam_df)
        pc = self.pca.transform(normparam_df)
        pc_df = pd.DataFrame(pc)
        pc_df.columns = [f'pc{i+1}' for i in range(len(normparam_df.columns))]
        pc_cov = np.cov(pc_df.T)
        return (pc_df, pc_cov)

    def get_repidx(self, pc_df: pd.DataFrame, pc: int=2) -> int:
        pc_lst = []
        for i in range(pc):
            pc_lst.append(f'pc{i+1}')
        kernel = gaussian_kde(pc_df[pc_lst].T)
        kde_score = kernel.pdf(pc_df[pc_lst].T)
        maxidx = np.where(kde_score == kde_score.max())
        return maxidx

    def plot_exp_ratio(self, ax: mpl.axes.Axes):
        ax.plot(np.cumsum(self.pca.explained_variance_ratio_), marker='.', color='m')
        return ax
