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

from itertools import chain
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import scipy.stats
from tqdm import tqdm
from typing import Dict, List, Iterator, Optional

import anmodel
import analysistools


class Burstinfo:
    def __init__(self, model: str='AN', wavepattern: str='SPN', 
                 channel_bool: Optional[Dict]=None, 
                 model_name: Optional[str]=None, 
                 ion: bool=False, concentration: Dict=None)-> None:
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

    def get(self, param: pd.Series) -> Dict:
        self.model.set_params(param)
        s, _ = self.model.run_odeint()
        v = s[5000:, 0]

        als = anmodel.analysis.FreqSpike()
        burstidx, ave_spike_per_burst, num_burst = als.get_burstinfo(v)
        
