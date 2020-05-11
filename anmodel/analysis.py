# -*- coding: utf-8 -*-

__author__ = 'Fumiya Tatsuki, Kensuke Yoshida, Tetsuya Yamada, Shoi Shi, Hiroki R. Ueda'
__status__ = 'in prep'
__version__ = '1.0.0'
__date__ = '11 May 2020'


import os
import sys
"""
LIMIT THE NUMBER OF THREADS!
change local env variables BEFORE importing numpy
"""
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from enum import Flag, auto
import numpy as np
import pandas as pd
from scipy.signal import periodogram
from scipy import signal


class WavePattern(Flag):
    SWS = 'SWS'
    SWS_FEW_SPIKES = 'SWS_FEW_SPIKES'
    AWAKE = 'AWAKE'
    RESTING = 'RESTING'
    EXCLUDED = 'EXCLUDED'
    ERROR = 'ERROR'


class WaveCheck:
    def __init__(self, samp_freq: int=1000) -> None:
        self.wave_pattern = WavePattern
        self.samp_freq = samp_freq
        self.freq_spike = FreqSpike(samp_freq=samp_freq)
    
    def pattern(self, v: np.ndarray) -> WavePattern:
        if np.isinf(v) or np.isnan(v):
            return self.wave_pattern.EXCLUDED
        detv = signal.detrend(v)
        max_potential = max(detv)
        f, spw = periodogram(detv, fs=self.samp_freq)
        maxamp = max(spw)
        nummax = spw.tolist().index(maxamp)
        maxfre = f[nummax]
        numfire = self.freq_spike.count_spike(v)

        if 200 < max_potential:
            return self.wave_pattern.EXCLUDED
        elif (maxfre < 0.2) or (numfire < 5 * 2):
            return self.wave_pattern.RESTING
        elif (0.2 < maxfre < 10.2) and (numfire > 5 * maxfre - 1):
            return self.wave_pattern.SWS
        elif (0.2 < maxfre < 10.2) and (numfire <= 5 * maxfre - 1):
            return self.wave_pattern.SWS_FEW_SPIKES
        elif 10.2 < maxfre:
            return self.wave_pattern.AWAKE
        else:
            return self.wave_pattern.EXCLUDED


class FreqSpike:
    def __init__(self, samp_freq: int) -> None:
        self.samp_freq = samp_freq

    def count_spike(self, v: np.ndarray) -> int:
        """ Count how many times a neuron fired.

        If neuron traverse -20 mV in a very short time range (1ms), 
        traverse count is added 1. Here, spike count is calculated as 
        traverse count // 2. 

        Parameter
        ---------
        v : np.ndarray
            membrane potential of a neuron
        
        Return
        ---------
        int
            spike count
        """
        ntraverse = 0
        ms = self.samp_freq / 1000
        for i in range(len(v)):
            if (v[i]+20) * (v[i+ms]+20) < 0:
                ntraverse += 1
        nspike = int(ntraverse//2)
        return nspike
