# -*- coding: utf-8 -*-

""" 
This is the analysis module for Averaged Neuron (AN) model. In this module, 
you can analyze firing patterns from AN model, mainly using frequency and
spike analysis. 
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
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from enum import Flag, auto
import numpy as np
import pandas as pd
from scipy.signal import periodogram
from scipy import signal
from typing import List


class WavePattern(Flag):
    """ Enumeration class that distinguish different wave pattern.
    """
    SWS = auto()
    SWS_FEW_SPIKES = auto()
    AWAKE = auto()
    RESTING = auto()
    EXCLUDED = auto()
    ERROR = auto()
    SPN = auto()


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
        """ analyse the firing pattern of the result of the simulation.

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
        elif (maxfre < 0.2) or (numfire < 5*2):
            return self.wave_pattern.RESTING
        elif (0.2 < maxfre < 10.2) and (numfire > 5*3*maxfre - 1):
            return self.wave_pattern.SWS
        elif (0.2 < maxfre < 10.2) and (numfire <= 5*3*maxfre - 1):
            return self.wave_pattern.SWS_FEW_SPIKES
        elif maxfre > 10.2:
            return self.wave_pattern.AWAKE
        else:
            return self.wave_pattern.EXCLUDED

    def pattern_spn(self, v: np.ndarray) -> WavePattern:
        """ analyse the firing pattern of the result of the simulation (spindle ver.).

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
        ave_revspike_per_burst: float = self.freq_spike.get_burstinfo(v=v, spike='bottom')[1]
        v_sq: np.ndarray = self.freq_spike.square_wave(v=v, spike='bottom')
        v_group: pd.DataFrame = pd.DataFrame([v, v_sq]).T.groupby(1)
        if np.any(v_sq==0) and np.any(v_sq==1):
            vmin_silent: float = float(v_group.min().iloc[0])
            vmin_burst: float = float(v_group.min().iloc[1])
        else:
            vmin_silent = vmin_burst = float(v_group.min().iloc[0])

        # if vmin_silent > vmin_burst: # doesn't need .iloc[0]?
        #     return self.wave_pattern.SPN

        if 200 < max_potential:
            return self.wave_pattern.EXCLUDED
        elif (maxfre < 0.2) or (numfire < 5*2):
            return self.wave_pattern.RESTING
        elif (0.2 < maxfre < 10.2) and (ave_revspike_per_burst>2):
            if vmin_silent > vmin_burst:
                return self.wave_pattern.SPN
            return self.wave_pattern.SWS
        elif (0.2 < maxfre < 10.2) and (numfire <= 5*maxfre - 1):
            return self.wave_pattern.SWS_FEW_SPIKES
        elif maxfre > 10.2:
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
    def __init__(self, samp_freq: int=1000) -> None:
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
        
        Returns
        ---------
        int
            spike count
        """
        peaktime: np.ndarray = signal.argrelmax(v, order=1)[0]
        spikeidx: np.ndarray = np.where(v[peaktime]>-20)[0]
        spiketime: np.ndarray = peaktime[spikeidx]
        nspike: int = len(spiketime)
        return nspike

    def get_spikeinfo(self, v: np.ndarray) -> np.ndarray:
        """ Get time index when spike occurs from the result of the simulation.

        Parameters
        ----------
        v : np.ndarray
            membrane potential of a neuron

        Returns
        ----------
        np.ndarray
            spike time index
        """
        peaktime: np.ndarray = signal.argrelmax(v, order=1)[0]
        spikeidx: np.ndarray = np.where(v[peaktime]>-20)[0]
        spiketime: np.ndarray = peaktime[spikeidx]
        nspike: int = len(spiketime)
        return nspike, spiketime

    def get_ahpinfo(self, v:np.ndarray) -> np.ndarray:
        """ Get time index when afterhyperpolarization (AHP) occurs.

        Parameters
        ----------
        v : np.ndarray
            membrane potential of a neuron

        Returns
        ----------
        np.ndarray
            spike time index
        """
        peaktime: np.ndarray = signal.argrelmin(v, order=1)[0]
        spikeidx: np.ndarray = np.where(v[peaktime]<-80)[0]
        spiketime: np.ndarray = peaktime[spikeidx]
        nspike: int = len(spiketime)
        return nspike, spiketime

    def get_burstinfo(self, v: np.ndarray, 
                      isi_thres: int=50, 
                      spike_thres:int =2, 
                      spike: str = 'peak'
        ) -> (List, float, int):
        """ Get information around burst firing from the result of the simulation.

        Parameters
        ----------
        v : np.ndarray
            membrane potential of a neuron
        isi_thres : int
            interspike interval threshold. over this threshold, spikes are 
            regarded as being in the separated event.
        spike_thres : int
            spike threshold. under this threshold, each event doesn't 
            regarded as a "burst" event.

        Returns
        ----------
        burstidx : List
            list of time index in which burst firing occurs.
        ave_spike_per_burst : float
            average number of spike during single burst event.
        num_burst : int
            number of burst event in the given simulation result.
        """
        isi_thres: int = isi_thres * self.samp_freq / 1000
        if spike == 'peak':
            _, spiketime = self.get_spikeinfo(v)
        elif spike == 'bottom':
            _, spiketime = self.get_ahpinfo(v)
        isi: np.ndarray = np.diff(spiketime)
        grouped_events: np.ndarray = np.split(spiketime, np.where(isi>isi_thres)[0]+1)
        burst_events: np.ndarray = [x for x in grouped_events if len(x)>=spike_thres]
        num_burst: int = len(burst_events)
        if num_burst == 0:  # no burst events
            ave_spike_per_burst: float = 0.
            burstidx: List = []
            return burstidx, ave_spike_per_burst, num_burst
        else:
            ave_spike_per_burst: float = len(np.concatenate(burst_events)) / num_burst
            padding: List = [np.diff(x).mean() for x in burst_events]
            burstidx: List = []
            for i in range(len(burst_events)):
                idx: List = [j for j in range(len(v)) if burst_events[i][0]-padding[i]<j<burst_events[i][-1]+padding[i]]
                burstidx.append(idx)
            return burstidx, ave_spike_per_burst, num_burst
        
    def square_wave(self, v: np.ndarray, spike: str='peak') -> np.ndarray:
        """ approximate firing pattern of the given parameter set into square wave.

        Parameters
        ----------
        v : np.ndarray
            membrane potential of a neuron
        
        Returns
        ---------
        v_sq : np.ndarray
            array contains 0 or 1, 0 during silent phase and 1 during burst phase.
        """
        burstidx: List = self.get_burstinfo(v=v, spike=spike)[0]
        v_sq: np.ndarray = np.zeros(len(v))
        for bidx in burstidx:
            v_sq[bidx] = 1.
        return v_sq
