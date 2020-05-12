# -*- coding: utf-8 -*-

""" 
This is the analysis module for Averaged Neuron (AN) model. In this module, 
you can analyze firing patterns from AN model, mainly using frequency and
spike analysis. 
"""

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
from scipy.signal import periodogram
from scipy import signal


class WavePattern(Flag):
    """ Enumeration class that distinguish different wave pattern.
    """
    SWS = 'SWS'
    SWS_FEW_SPIKES = 'SWS_FEW_SPIKES'
    AWAKE = 'AWAKE'
    RESTING = 'RESTING'
    EXCLUDED = 'EXCLUDED'
    ERROR = 'ERROR'


class WaveCheck:
    """ Check which wave pattern the neuronal firing belong to.

    Parameters
    ----------
    samp_freq : int
        sampling frequency of neuronal recordings (Hz)
    
    Attributes
    ----------
    wave_patters : WavePattern
        choices of wave pattern: enumeration objects
    samp_freq : int
        sampling frequency of neuronal recordings (Hz)
    freq_spike : FreqSpike
        contains helper functions for analyzing firing pattern
        using those frequency and spikes
    """
    def __init__(self, samp_freq: int=1000) -> None:
        self.wave_pattern = WavePattern
        self.samp_freq = samp_freq
        self.freq_spike = FreqSpike(samp_freq=samp_freq)
    
    def pattern(self, v: np.ndarray) -> WavePattern:
        """

        Parameters
        ----------
        v : np.ndarray
            membrane potential over time

        Returns
        ----------
        WavePattern
            which wave pattern `v` belong to 
        """
        if np.any(np.isinf(v)) or np.any(np.isnan(v)):
            return self.wave_pattern.EXCLUDED
        detv: np.ndarray = signal.detrend(v)
        max_potential: float = max(detv)
        f: np.ndarray  # Array of sample frequencies
        spw: np.ndarray  # Array of power spectral density or power spectrum
        f, spw = periodogram(detv, fs=self.samp_freq)
        maxamp: float = max(spw)
        nummax: int = spw.tolist().index(maxamp)
        maxfre: float = f[nummax]
        numfire: int = self.freq_spike.count_spike(v)

        if 200 < max_potential:
            return self.wave_pattern.EXCLUDED
        elif (maxfre < 0.2) or (numfire < 5 * 2):
            return self.wave_pattern.RESTING
        elif (0.2 < maxfre < 10.2) and (numfire > 5*5*maxfre - 1):
            return self.wave_pattern.SWS
        elif (0.2 < maxfre < 10.2) and (numfire <= 5*5*maxfre - 1):
            return self.wave_pattern.SWS_FEW_SPIKES
        elif 10.2 < maxfre:
            return self.wave_pattern.AWAKE
        else:
            return self.wave_pattern.EXCLUDED


class FreqSpike:
    """ 

    Parameters
    ----------
    samp_freq : int
        sampling frequency of neuronal recordings (Hz)

    Attributes
    ----------
    samp_freq : int
        sampling frequency of neuronal recordings (Hz)
    """
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
        ntraverse: int = 0
        ms: int = int(self.samp_freq / 1000)
        for i in range(len(v)-1):
            if (v[i]+20) * (v[i+ms]+20) < 0:
                ntraverse += 1
        nspike: int = int(ntraverse//2)
        return nspike
