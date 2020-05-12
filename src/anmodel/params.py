# -*- coding: utf-8 -*-

"""
This is parameter module for Averaged Neuron (AN) model.
These parameters are identical to those in Tatsuki et al., 2016
and Yoshida et al., 2018.
"""

__author__ = 'Fumiya Tatsuki, Kensuke Yoshida, Tetsuya Yamada, Shoi Shi, Hiroki R. Ueda'
__status__ = 'in prep'
__version__ = '1.0.0'
__date__ = '11 May 2020'


class Constants:
    """ Constant values needed for AN model.

    These constant values are based on previous researches.  See Tatsuki et al., 2016, 
    Yoshida et al., 2018 and Compte et al., 2006.

    Attributes
    ----------
    cm : float
        membrane capacitance (uF/cm2)
    area : float
        area of neuron (mm2)
    tauA : float
        time constant of inactivation variable for fast A-type potassium channel
    s_a_ampar : float
        coefficient of f(V)
    s_tau_ampar : float
        time constant of gating variable differential equation of AMPAR
    x_a_nmdar : float
        coefficient of f(V)
    x_tau_nmdar : float
        time constant for differential equation of second-order gating variable x
    s_a_nmdar : float
        coefficient of (1 - s)
    s_tau_nmdar : float
        time constant for differential equation of gating variable s
    s_a_gabar : float
        coefficient of f(V)
    s_tau_gabar : float
        time constant for differential equation of gating variable s
    vL : float
        equilibrium potential of leak channel (mV)
    vNaL : float
        equilibrium potential of leak sodium channel (mV)
    vNa : float
        equilibrium potential of sodium ion (mV)
    vK : float
        equilibrium potential of potassium ion (mV)
    vCa : float
        equilibrium potential of calcium channel (mV)
    vAMPAR : float
        equilibrium potential of AMPA receptor (mV)
    vNMDAR : float
        equilibrium potential of NMDA receptor (mV)
    vGABAR : float
        equilibrium potential of GABA receptor (mV)
    an_ini : list (float)
        initial parameters for differential equations of AN model:
            v : membrane potential
            h_nav : inactivation variable of voltage-gated sodium channel
            n_kvhh : activation variable of HH-type voltage-gated 
                     potassium channel
            h_kva : inactivation variable of fast A-type potassium channel
            m_kvsi : activation variable of slowly inactivating potassium channel
            s_ampar : gating variable of AMPA recptor
            x_nmdar : second-order gating variable of NMDA receptor
            s_nmdar : gating variable of NMDA receptor
            s_gabar : gating variable of GABA receptor
            ca : intracellular calcium concentration
    san_ini : list (float)
        initial parameters for differential equations of SAN model:
            v : membrane potential
            n_kvhh : activation variable of HH-type voltage-gated 
                     potassium channel
            ca : intracellular calcium concentration
    """
    def __init__(self) -> None:
        self.cm = 1.0
        self.area = 0.02

        self.a_ca = 0.5
        self.kd_ca = 30.0
        self.tau_a = 15.0
        self.s_a_ampar = 3.48
        self.s_tau_ampar = 2.0
        self.s_a_nmdar = 0.5
        self.s_tau_nmdar = 100.0
        self.x_a_nmdar = 3.48
        self.x_tau_nmdar = 2.0
        self.s_a_gabar = 1.0
        self.s_tau_gabar = 10.0

        self.vL = -60.95
        self.vNaL = 0.
        self.vNa = 55.0
        self.vK = -100.0
        self.vCa = 120.0
        self.vAMPAR = 0.
        self.vNMDAR = 0.
        self.vGABAR = -70.0
        self.an_ini = [
            -45.,   # 0 : v
            0.045,  # 1 : h_nav
            0.54,   # 2 : n_kvhh
            0.045,  # 3 : h_kva
            0.34,   # 4 : m_kvsi
            0.01,   # 5 : s_ampar
            0.01,   # 6 : x_nmdar
            0.01,   # 7 : s_nmdar
            0.01,   # 8 : s_gabar
            1.,     # 9 : Ca
            ]
        self.san_ini = [
            -45.,   # 0 : v
            0.54,   # 1 : n_kvhh
            1.,     # 2 : Ca
        ]


class Ion:
    """ Constant values needed for AN model with ion and typical ion concentrations.

    These constant values are based on Ramussen et al., 2017.

    Attributes
    ---------
    r : float
        gas constant (J/K/mol)
    t : float
        body temprature (K)
    f : float
        Faraday constant (C/mol)
    awake_ion : dictionary (float)
        typical ion concentrations which recapitulates awake firing pattern
    sleep_ion : dictionary (float)
        typical ion concentrations which recapitulates sleep (SWS) firing pattern
    """
    def __init__(self) -> None:
        self.r = 8.314472
        self.t = 310.
        self.f = 9.64853399 * 10000

        self.awake_ion = {
            'ex_na': 140,
            'in_na': 7.0,
            'ex_k': 4.4,
            'in_k': 140,
            'ex_cl': 140,
            'in_cl': 7.0,
            'ex_ca': 1.2,
            'in_ca': 0.001,
            'ex_mg': 0.7,
        }
        self.sleep_ion = {
            'ex_na': 140.0,
            'in_na': 7.0,
            'ex_k': 3.9,
            'in_k': 140.0,
            'ex_cl': 140.0,
            'in_cl': 7.0,
            'ex_ca': 1.35,
            'in_ca': 0.001,
            'ex_mg': 0.8,
        }


class TypicalParam:
    """ Typical parameter set that recapitulate a cirtain firing pattern.

    Attributes
    ----------
    an_sws : dictionary (float)
        typical parameter set which recapitulate SWS firing pattern in AN model
        See : Tatsuki et al., 2016
    san_sws : dictionary (float)
        typical parameter set which recapitulate SWS firing patter in SAN model
        See : Yoshida et al., 2018 figure 1L.
    """
    def __init__(self) -> None:
        self.an_sws = {
            'g_leak': 0.03573,
            'g_nav': 12.2438,
            'g_kvhh': 2.61868,
            'g_kva': 1.79259,
            'g_kvsi': 0.0350135,
            'g_cav': 0.0256867,
            'g_kca': 2.34906,
            'g_nap': 0.0717984,
            'g_kir': 0.0166454,
            'g_ampar': 0.513425,
            'g_nmdar': 0.00434132,
            'g_gabar': 0.00252916,
            't_ca': 121.403,
        }
        self.san_sws = {
            'g_leak': 0.016307,
            'g_kvhh': 19.20436,
            'g_cav': 0.1624,
            'g_kca': 0.7506,
            'g_nap': 0.63314,
            't_ca': 739.09,
        }
