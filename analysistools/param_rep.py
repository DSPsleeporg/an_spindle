# -*- coding: utf-8 -*-

__author__ = 'Tetsuya Yamada'
__status__ = 'Editing'
__version__ = '1.0.0'
__date__ = '31 May 2020'


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from scipy.stats import gaussian_kde
import sklearn
import sklearn.decomposition
from typing import Dict, List

import anmodel
from analysistools import tools

class PCA:
    def __init__(self, date):
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
        self.normparam_df: pd.DataFrame = self.normalize()

        self.pca = sklearn.decomposition.PCA()

    def normalize(self):
        grid_args: Dict = tools.get_gridargs()
        ga_df: pd.DataFrame = pd.DataFrame(grid_args)
        start: pd.Series = ga_df.iloc[0]
        end: pd.Series = ga_df.iloc[1]
        normparam_df = (self.logparam_df-start) / (end-start)
        return normparam_df

    def run_pca(self) -> None:
        self.pca.fit(self.normparam_df)
        self.pc = self.pca.transform(self.normparam_df)
        self.pc_df = pd.DataFrame(self.pc)
        self.pc_df.columns = [f'pc{i+1}' for i in range(len(self.param_df.columns))]
        self.pc_cov = np.cov(self.pc_df.T)

    def get_reparam_pc12(self):
        self.kernel_pc12 = gaussian_kde(self.pc_df[['pc1', 'pc2']].T)
        kde_score = self.kernel_pc12.pdf(self.pc_df[['pc1', 'pc2']].T)
        maxidx = np.where(kde_score == kde_score.max())
        reparam_pc12 = self.param_df.iloc[maxidx]
        reppc_pc12 = self.pc_df.iloc[maxidx]
        return reparam_pc12, reppc_pc12

    def get_reparam_pc123(self):
        self.kernel_pc123 = gaussian_kde(self.pc_df[['pc1', 'pc2', 'pc3']].T)
        kde_score = self.kernel_pc123.pdf(self.pc_df[['pc1', 'pc2', 'pc3']].T)
        maxidx = np.where(kde_score == kde_score.max())
        reparam_pc123 = self.param_df.iloc[maxidx]
        reppc_pc123 = self.pc_df.iloc[maxidx]
        return reparam_pc123, reppc_pc123

    def param_plot(self):
        self.run_pca()
        reparam_pc12, reppc_pc12 = self.get_reparam_pc12()
        _, ax = plt.subplots(1, 1, figsize=(6, 6))
        x, y = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
        pos = np.vstack([x.ravel(), y.ravel()])
        z = np.reshape(self.kernel_pc12.pdf(pos), x.shape)
        ax.scatter(self.pc_df['pc1'], self.pc_df['pc2'], s=3, alpha=0.7, color='skyblue')
        ax.scatter(reppc_pc12['pc1'], reppc_pc12['pc2'], color='red')
        ax.contour(x, y, z, alpha=1, cmap='OrRd')
        return ax
