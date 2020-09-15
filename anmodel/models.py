# -*- coding: utf-8 -*-

"""
This is the model module for Averaged Neuron (AN) model. In this module, you 
can simulate previous models (AN model: Tatsuki et al., 2016 and SAN model: 
Yoshida et al., 2018), and various models with 'X model' based on channel 
and parameter modules. Also, intracellular and extracellular ion concentration
can be taken into consideration using Nernst equation (See Ramussen et al., 2017)\\

Models\\
AN model : \\
This model contains 9 types of channels, 3 types of neurotransmitter receptors, 
and a type of pumps. It has 10 differential equations. See Tatsuki et al., 2016 
and Yoshida et al., 2018.\\
SAN model : \\
This model contains 5 types of channels and a type of pumps. It has only three 
differential equations. This is the simplest model which can recapitulate slow 
wave sleep (SWS) firing pattern. See Yoshida et al., 2018.\\
X model : \\
We can create X model by choosing arbitrary channels, neurotransmitter receptors 
and pumps based on channels module. This model would help you a lot when you want 
to simplify AN model which recapitulate a certain firing pattern.\\

ENJOY YOUR SIMULATION WITH AVERAGED NEURON MODEL!!!!
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

from collections import deque
import itertools
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from typing import Dict, Iterator, List, Optional, Union

import channels
import params


class ANmodel:
    """ Averaged Neuron (AN) model.

    This model contains 9 types of channels, 3 types of neurotransmitter 
    receptors, and a type of pumps. It has 10 differential equations.
    See Tatsuki et al., 2016 and Yoshida et al., 2018.

    Parameters
    ----------
    ion : bool
        whether you make equiribrium potential variable or not, 
        default False
    concentration : dictionary or str or None
        dictionary of ion concentration, or 'sleep'/'awake' that
        designate typical ion concentrations, default None

    Attributes
    ----------
    const : anmodel.params.Constants
        contains constants needed for AN model
    ini : list (float)
        initial parameters for differential equations of AN model
    leak : anmodel.channels.Leak
        leak channel object
    nav : anmodel.channels.NavHH
        Hodgkin-Huxley type voltage-gated sodium channel object
    kvhh : anmodel.channels.KvHH
        Hodgkin-Huxley type delayed rectifier potassium channel object
    kva : anmodel.channels.KvA
        Fast A-type voltage-geted potassium channel object
    cav : anmodel.channels.Cav
        voltage-gated calcium channel object
    nap : anmodel.channels.NaP
        persistent sodium channel object
    kca : anmodel.channels.KCa
        calcium-dependent potassium channel object
    kir : anmodel.channels.KIR
        inwardly rectifier potassium channel object
    ampar : anmodel.channels.AMPAR
        AMPA receptor object
    nmdar : anmodel.channels.NMDAR
        NMDA receptor object
    gabar : anmodel.channels.GABAR
        GABA receptor object
    ion : bool
        whether you make equiribrium potential variable or not
    concentration : dictionary or str or None
        dictionary of ion concentrations, default None
    ion_const : object
        contains constants needed when you take ions into consideration
    """
    def __init__(self, ion: bool=False, 
                 concentration: Optional[Union[Dict, str]]=None) -> None:
        self.params = params.Constants()
        self.ini = self.params.an_ini
        self.leak = channels.Leak()
        self.nav = channels.NavHH()
        self.kvhh = channels.KvHH()
        self.kva = channels.KvA()
        self.kvsi = channels.KvSI()
        self.cav = channels.Cav()
        self.nap = channels.NaP()
        self.kca = channels.KCa()
        self.kir = channels.KIR()
        self.ampar = channels.AMPAR()
        self.nmdar = channels.NMDAR()
        self.gabar = channels.GABAR()

        self.ion = ion
        if ion:
            self.ion_params = params.Ion()
            if type(concentration) is not str:
                self.concentration = concentration
            else:
                if concentration == 'awake':
                    self.concentration = self.ion_params.awake_ion
                if concentration == 'sleep':
                    self.concentration = self.ion_params.sleep_ion
            self.set_equil_potential(concentration)
            self.nmdar = channels.NMDAR(ion=True, ex_mg=concentration['ex_mg'])

    def set_equil_potential(self, concentration: Dict) -> None:
        """ set equilibrium potential using Nernst equation

        When considering intracellular and extracellular ion concentration,
        equiribrium potential for each ion can be calculated by Nernst equation.
        In our model, intracellular and extracellular ion concentrations 
        don't change over time except calcium, those equiribrium potentials
        don't change over time too. Regarding calcium, its equilibrium potential
        is needed to be updated over time. 

        Parameters
        ----------
        concentrations : list
            dictionary of ion concentrations
        """
        r: float = self.ion_params.r
        t: float = self.ion_params.t
        f: float = self.ion_params.f
        ex_na: float = concentration['ex_na']
        in_na: float = concentration['in_na']
        ex_k: float = concentration['ex_k']
        in_k: float = concentration['in_k']
        ex_cl: float = concentration['ex_cl']
        in_cl: float = concentration['in_cl']
        ex_ca: float = concentration['ex_ca']
        in_ca: float = concentration['in_ca']

        def __v(pk: float, pna: float, pcl: float, pca: float) -> float:
            """ calculate equilibrium potential with multiple kinds of ions

            This is a hidden function.

            Parameters
            ----------
            pk : float
                ratio of potassium ion to sodium and chloride ion
            pna : float
                ratio of sodium ion to potassium and chloride ion
            pcl : float
                ratio of chloride ion to potassium and sodium ion

            Returns
            ----------
            float
                equiribrium potential based on Nernst equation
            """
            ex_ion = pk * ex_k + pna * ex_na + pcl * in_cl + pca * ex_ca
            in_ion = pk * in_k + pna * in_na + pcl * ex_cl + pca * in_ca
            v = r * t / f * np.log(ex_ion/in_ion) * 1000
            return v

        vNa: float = r * t / f * np.log(ex_na/in_na) * 1000
        vK: float = r * t / f * np.log(ex_k/in_k) * 1000
        vCa: float = r * t / (f * 2) * np.log(ex_ca / in_ca) * 1000
        vL: float = __v(pk=1., pna=0.08, pcl=0.1, pca=0.)
        vAMPA: float = __v(pk=1., pna=1., pcl=0., pca=0.)
        vNMDA: float = __v(pk=1., pna=1., pcl=0., pca=1.)
        vGABA: float = r * t / f * np.log(ex_cl/in_cl) * 1000

        self.leak.set_e(new_e=vL)
        self.nav.set_e(new_e=vNa)
        self.kvhh.set_e(new_e=vK)
        self.kva.set_e(new_e=vK)
        self.kvsi.set_e(new_e=vK)
        self.cav.set_e(new_e=vCa)
        self.kca.set_e(new_e=vK)
        self.kir.set_e(new_e=vK)
        self.ampar.set_e(new_e=vAMPA)
        self.nmdar.set_e(new_e=vNMDA)
        self.gabar.set_e(new_e=vGABA)

    def set_vCa(self, in_ca: float) -> None:
        """ set equiribrium potential for calcium ion.

        concentration of intracellular calcium ion changes over time, 
        so its equilibrium potential also changes over time.

        Parameter
        ----------
        in_ca : float
            concentration of intracellular calcium
        """
        r: float = self.ion_params.r
        t: float = self.ion_params.t
        f: float = self.ion_params.f
        ex_ca: float = self.concentration['ex_na']
        vCa: float = r * t / (f * 2) * np.log(ex_ca / in_ca) * 1000
        self.cav.set_e(new_e=vCa)

    def get_e(self) -> Dict:
        e_dict: Dict = {}
        e_dict['vL'] = self.leak.get_e()
        e_dict['vNa'] = self.nav.get_e()
        e_dict['vK'] = self.kvhh.get_e()
        e_dict['vCa'] = self.cav.get_e()
        e_dict['vAMPAR'] = self.ampar.get_e()
        e_dict['vNMDAR'] = self.nmdar.get_e()
        e_dict['vGABAR'] = self.gabar.get_e()
        return e_dict

    def gen_params(self) -> Dict:
        """ generate parameters randomly for AN model.

        generate parameters randomly in logarithmic scale, and then choose
        parameter sets which recapitulate a certain firing pattern.
        channels : 10^-2 ~ 10^2
        neurotransmitter receptors : 10^-3 ~ 10^1
        time constant of pump : 10^1 ~ 10^3

        Returns
        ----------
        dictionary
            parameter dictionary

        See Also
        ----------
        anmodel.search : random parameter search is implemented
        """
        param_dict: Dict = {}

        gX_name: List[str] = ['g_leak', 'g_nav', 'g_kvhh', 'g_kva', 'g_kvsi', 
                         'g_cav', 'g_kca', 'g_nap', 'g_kir']
        gX_log: np.ndarray = 4 * np.random.rand(9) - 2  # from -2 to 2
        gX: np.ndarray = (10 * np.ones(9)) ** gX_log  # 0.01 ~ 100
        gX_itr: Iterator = zip(gX_name, gX)

        gR_name: List[str] = ['g_ampar', 'g_nmdar', 'g_gabar']
        gR_log: np.ndarray = 4 * np.random.rand(3) - 3  # from -3 to 1
        gR: np.ndarray = (10 * np.ones(3)) ** gR_log  # 0.001 ~ 10
        gR_itr: Iterator = zip(gR_name, gR)

        tCa_log: np.ndarray = 2 * np.random.rand(1) + 1  # from 1 to 3
        tCa: np.ndarray = 10 ** tCa_log    # 10 ~ 1000
        tCa_dict: Dict = {'t_ca': float(tCa)}

        param_dict.update(gX_itr)
        param_dict.update(gR_itr)
        param_dict.update(tCa_dict)
        return param_dict

    def set_params(self, params: Dict) -> None:
        """ Set parameters to the channels

        Parameters
        ----------
        params : dictionary
            channel, receptor and pump parameters
        """
        self.leak.set_g(float(params['g_leak']))
        self.nav.set_g(float(params['g_nav']))
        self.kvhh.set_g(float(params['g_kvhh']))
        self.kva.set_g(float(params['g_kva']))
        self.kvsi.set_g(float(params['g_kvsi']))
        self.cav.set_g(float(params['g_cav']))
        self.kca.set_g(float(params['g_kca']))
        self.nap.set_g(float(params['g_nap']))
        self.kir.set_g(float(params['g_kir']))
        self.ampar.set_g(float(params['g_ampar']))
        self.nmdar.set_g(float(params['g_nmdar']))
        self.gabar.set_g(float(params['g_gabar']))
        self.tau_ca = float(params['t_ca'])
        self.leak.set_div()

    def set_rand_params(self) -> Dict:
        """ Set random parameters to the channels

        Returns
        ----------
        dictionary
            parameter dictionary

        See Also
        ----------
        anmodel.search : random parameter search is implemented
        """
        new_params: Dict = self.gen_params()
        self.set_params(new_params)
        return new_params

    def set_sws_params(self) -> None:
        """ Set typical parameter that recapitulate SWS firing pattern.
        """
        typ_params: Dict = params.TypicalParam().an_sws
        self.set_params(typ_params)

    def get_params(self) -> Dict:
        """ Get current channel conductances.

        Returns
        ----------
        dict
            parameter dictionary
        """
        params: Dict = {}
        params['g_leak']: float = self.leak.get_g()
        params['g_nav']: float = self.nav.get_g()
        params['g_kvhh']: float = self.kvhh.get_g()
        params['g_kva']: float = self.kva.get_g()
        params['g_kvsi']: float = self.kvsi.get_g()
        params['g_cav']: float = self.cav.get_g()
        params['g_kca']: float = self.kca.get_g()
        params['g_nap']: float = self.nap.get_g()
        params['g_kir']: float = self.kir.get_g()
        params['g_ampar']: float = self.ampar.get_g()
        params['g_nmdar']: float = self.nmdar.get_g()
        params['g_gabar']: float = self.gabar.get_g()
        params['t_ca']: float = self.tau_ca
        return params

    def dvdt(self, args: List[float]) -> float:
        """ Calculate dv/dt for given parameters.

        Membrane potential changes over time dependent on currents that
        flow each channels and neurotransmitter receptors.

        Parameters
        ----------
        args : list (float)
            valuable list in a certain time

        Results
        ----------
        float
            dv/dt for given parameters
        """
        v, h_nav, n_kvhh, h_kva, m_kvsi, s_ampar, _, s_nmdar, s_gabar, ca = args
        return ((-10.0*self.params.area 
                * (self.leak.i_div(v)
                + self.nav.i(v, h=h_nav) 
                + self.kvhh.i(v, n=n_kvhh)
                + self.kva.i(v, h=h_kva)
                + self.kvsi.i(v, m=m_kvsi)
                + self.cav.i(v)
                + self.kca.i(v, ca=ca)
                + self.nap.i(v)
                + self.kir.i(v))
                - (self.ampar.i(v, s=s_ampar)
                + self.nmdar.i(v, s=s_nmdar)
                + self.gabar.i(v, s=s_gabar))) 
                / (10.0*self.params.cm*self.params.area))
    
    def dCadt(self, args: List[float]) -> float:
        """ Calculate dCa/dt for given parameters.

        Intracellular calcium changes over time dependent on CaV channel, 
        NMDA receptor, and calcium pump.

        Parameters
        ----------
        args : list (float)
            valuable list in a certain time

        Returns
        ----------
        float
            dCa/dt
        """
        v, s_nmdar, ca = args
        a_ca: float = self.params.a_ca
        area: float = self.params.area
        tau_ca: float= self.tau_ca
        dCadt: float = (- a_ca * (10.0*area*self.cav.i(v))
                        - a_ca * self.nmdar.i(v, s=s_nmdar)
                        - ca / tau_ca)
        return dCadt

    def diff_op(self, args: List[float], time: float) -> List[float]:
        """ Differential equations to be solved.

        Parameters
        ----------
        args : list (float)
            valuable list in a certain time
        time : float
            each time point to solve differential equation
        Returns
        ----------
        list (float)
            list of variables differentiated by t
        """
        v, h_nav, n_kvhh, h_kva, m_kvsi, s_ampar, x_nmdar, s_nmdar, s_gabar, ca = args
        ca_args: List[float] = [v, s_nmdar, ca]

        if self.ion:
            self.set_vCa(in_ca=ca)

        dvdt: float = self.dvdt(args=args)
        dhNadt: float = self.nav.dhdt(v=v, h=h_nav)
        dnKdt: float = self.kvhh.dndt(v=v, n=n_kvhh)
        dhAdt: float = self.kva.dhdt(v=v, h=h_kva)
        dmKSdt: float = self.kvsi.dmdt(v=v, m=m_kvsi)
        dsAMPAdt: float = self.ampar.dsdt(v=v, s=s_ampar)
        dxNMDAdt: float = self.nmdar.dxdt(v=v, x=x_nmdar)
        dsNMDAdt: float = self.nmdar.dsdt(v=v, s=s_nmdar, x=x_nmdar)
        dsGABAdt: float = self.gabar.dsdt(v=v, s=s_gabar)
        dCadt: float = self.dCadt(args=ca_args)
        return [dvdt, 
                dhNadt, 
                dnKdt, 
                dhAdt, 
                dmKSdt,
                dsAMPAdt, 
                dxNMDAdt, 
                dsNMDAdt, 
                dsGABAdt, 
                dCadt,
                ]

    def run_odeint(self, samp_freq: int=1000, samp_len: int=10) -> (np.ndarray, Dict):
        """ Solve differential equations of diff_op.

        Parameters
        ----------
        samp_freq : int
            Sampling frequency (Hz)
        samp_len : int
            How long you record the activity of model neuron (sec)

        Results:
        ----------
        np.ndarray (float) : (samp_freq*samp_len, number of ODE)
            solution for the each differential equation
        dictionary
            dictionary containing additional output information. 
            see scipy.integrate.odeint() documentation.
        """
        solvetime: np.ndarray = np.linspace(1, 1000*samp_len, samp_freq*samp_len)
        s: np.ndarray
        info: Dict
        s, info = odeint(self.diff_op, self.ini, solvetime, 
                         atol=1.0e-5, rtol=1.0e-5, full_output=True)
        '''
        trendy and faster implementation to solve ODE: 
        from scipy.integrate import solve_ivp
        s = solve_ivp(self.diff_op, (0, 10000), self.ini, t_eval=solvetime)
        '''
        return s, info


class SANmodel(ANmodel):
    """ Simplified Averaged Neuron (SAN) model.

    This model contains 5 types of channels and a type of pumps. It has
    only three differential equations. This is the simplest model which 
    can recapitulate slow wave sleep (SWS) firing pattern.
    See Yoshida et al., 2018

    Parameters
    ----------
    ion : bool
        whether you make equiribrium potential variable or not, 
        default False
    concentration : dictionary
        dictionary of ion concentration, default None

    Attributes
    ----------
    ini : list (float)
        initial parameters for differential equations of SAN model.
        update from initial parameters of AN model
    """
    def __init__(self, ion: bool=False, concentration: Optional[Dict]=None) -> None:
        super().__init__(ion=ion, concentration=concentration)
        self.ini = self.params.san_ini

    def gen_params(self) -> Dict:
        """ Generate parameters randomly.

        Generate parameters randomly in logarithmic scale, and then choose
        parameter sets which recapitulate a certain firing pattern. Updated
        from ANmodel.gen_params() for SAN model.
        channels : 10^-2 ~ 10^2
        time constant of pump : 10^1 ~ 10^3

        Returns
        ----------
        dictionary
            parameter dictionary

        See Also
        ----------
        anmodel.search : random parameter search is implemented
        """
        param_dict: Dict = {}

        gX_name: List[str] = ['g_leak', 'g_kvhh', 'g_cav', 'g_kca', 'g_nap']
        gX_log: np.ndarray = 4 * np.random.rand(5) - 2  # from -2 to 2
        gX: np.ndarray = (10 * np.ones(5)) ** gX_log  # 0.01 ~ 100
        gX_itr: Iterator = zip(gX_name, gX)

        tCa_log: np.ndarray = 2 * np.random.rand(1) + 1  # from 1 to 3
        tCa: np.ndarray = 10 ** tCa_log    # 10 ~ 1000
        tCa_dict: Dict = {'t_ca': float(tCa)}

        param_dict.update(gX_itr)
        param_dict.update(tCa_dict)
        return param_dict

    def set_params(self, params: Dict) -> None:
        """ Set parameters to the channels

        Updated from ANmodel.set_param() for SAN model.

        Parameters
        ----------
        params : dictionary
            channel and pump parameters
        """
        self.leak.set_g(float(params["g_leak"]))
        self.kvhh.set_g(float(params["g_kvhh"]))
        self.cav.set_g(float(params["g_cav"]))
        self.kca.set_g(float(params["g_kca"]))
        self.nap.set_g(float(params["g_nap"]))
        self.tau_ca = float(params["t_ca"])
        self.leak.set_div()

    def set_sws_params(self) -> None:
        """ Set typical parameter that recapitulate SWS firing pattern. 
        Updated from ANmodel.set_sws_params() for SAN model.
        """
        typ_params: Dict = params.TypicalParam().san_sws
        self.set_params(typ_params)

    def get_params(self) -> Dict:
        """ Get current channel conductances.

        Returns
        ----------
        dict
            parameter dictionary
        """
        params: Dict = {}
        params['g_leak'] = self.leak.get_g()
        params['g_kvhh'] = self.kvhh.get_g()
        params['g_cav'] = self.cav.get_g()
        params['g_kca'] = self.kca.get_g()
        params['g_nap'] = self.nap.get_g()
        params['t_ca'] = self.tau_ca
        return params

    def dvdt(self, args: List[float]) -> float:
        """ Calculate dv/dt for given parameters.

        Membrane potential changes over time dependent on currents that
        flow each channels and neurotransmitter receptors. Updated from
        ANmodel.dvdt() for SAN model.

        Parameters
        ----------
        args : list (float)
            valuable list in a certain time

        Results
        ----------
        float
            dv/dt for given parameters
        """
        v, n_kvhh, ca = args
        return ((-10.0*self.params.area 
                * (self.kvhh.i(v, n=n_kvhh) 
                + self.cav.i(v) 
                + self.kca.i(v, ca=ca) 
                + self.nap.i(v) 
                + self.leak.i_div(v))) 
                / (10.0*self.params.cm*self.params.area))

    def dCadt(self, args: List[float]) -> float:
        """ Calculate dCa/dt for given parameters.

        Intracellular calcium changes over time dependent on CaV channel, 
        NMDA receptor, and calcium pump. Updated from ANmodel.dCadt() 
        for SAN model.

        Parameters
        ----------
        args : list (float)
            valuable list in a certain time

        Returns
        ----------
        float
            dCa/dt
        """
        v, ca = args
        a_Ca: float = self.params.a_ca
        area: float = self.params.area
        tau_Ca: float = self.tau_ca
        dCadt: float = -a_Ca * (10.0*area*self.cav.i(v)) - ca/tau_Ca
        return dCadt

    def diff_op(self, args: List[float], time: float) -> List[float]:
        """ Differential equations to be solved.

        Parameters
        ----------
        args : list (float)
            valuable list in a certain time
        time : float
            each time point to solve differential equation

        Returns
        ----------
        list
            list of variables differentiated by t
        """
        v, nK, ca = args
        ca_args: List[float] = [v, ca]
        dvdt: float = self.dvdt(args=args)
        dnKdt: float = self.kvhh.dndt(v=v, n=nK)
        dCadt: float = self.dCadt(args=ca_args)
        return [dvdt, dnKdt, dCadt]


class Xmodel(ANmodel):
    """ X model (arbitrary model) based on AN model.

    We can create X model by choosing arbitrary channels, neurotransmitter
    receptors and pumps based on channels module. This model would help you
    a lot when you want to simplify AN model which recapitulate a certain 
    firing pattern.

    Parameters
    ----------
    ion : bool
        whether you make equiribrium potential variable or not, 
        default False
    concentration : dictionary or str or None
        dictionary of ion concentration, or 'sleep'/'awake' that
        designate typical ion concentrations, default None

    Attributes
    ----------
    ini : list (float)
        Initial parameters for differential equations of SAN model.
        update from initial parameters of AN model
    channel_bool : list (bool)
        True means channels incorporated in the model and False means not.
        The order of the list is the same as other lists or dictionaries
        that contain channel information in AN model. Example: \
        channel_bool = [
            1,  # leak channel
            0,  # voltage-gated sodium channel
            1,  # HH-type delayed rectifier potassium channel
            0,  # fast A-type potassium channel
            0,  # slowly inactivating potassium channel
            1,  # voltage-gated calcium channel
            1,  # calcium-dependent potassium channel
            1,  # persistent sodium channel
            0,  # inwardly rectifier potassium channel
            0,  # AMPA receptor
            0,  # NMDA receptor
            0,  # GABA receptor
            1,  # calcium pump
        ]\
        This is SAN model.
    channel : list (object)
        List of all channel objects.
    ode_list : list (str)
        List of variables that changes over time (has differential 
        equation).
    """
    def __init__(self, channel_bool: List, ion: bool=False, 
                 concentration: Optional[Dict]=None) -> None:
        super().__init__(ion, concentration)
        channel_name: List[str] = ['leak', 'nav', 'kvhh', 'kva', 'kvsi', 
                                   'cav', 'nap', 'kca', 'kir', 
                                   'ampar', 'nmdar', 'gabar', 'ca']
        channel_object: List = [self.leak, self.nav, self.kvhh, self.kva, self.kvsi,
                                self.cav, self.nap, self.kca, self.kir, 
                                self.ampar, self.nmdar, self.gabar]
        self.channel_bool: Dict = dict(zip(channel_name, channel_bool))
        self.channel: Dict = dict(zip(channel_name[:-1], channel_object))

        self.ini: Dict = self.params.an_ini
        self.ode_list: List[str] = ['v', 'h_nav', 'n_kvhh', 'h_kva', 'm_kvsi',
                                    's_ampar', 'x_nmdar', 's_nmdar', 's_gabar', 'ca']
        for i, ode in enumerate(self.ode_list[1:-1]):
            if not self.channel_bool[ode[2:]]:
                self.ini[i+1] = None
                self.ode_list[i+1] = None 
        if not self.channel_bool['cav'] and not self.channel_bool['nmdar'] and not self.channel_bool['ca']:
            self.ini[9] = None  # ca
            self.ode_list[9] = None
        self.ini = [x for x in self.ini if x is not None]
        self.ode_list = [x for x in self.ode_list if x is not None]

    def gen_params(self) -> Dict:
        """ Generate parameters randomly.

        Generate parameters randomly in logarithmic scale, and then choose
        parameter sets which recapitulate a certain firing pattern. Updated
        from ANmodel.gen_params() for SAN model.\
        channels : 10^-2 ~ 10^2\
        receptors : 10^-3 ~ 10\
        time constant of pump : 10^1 ~ 10^3

        Returns
        ----------
        dictionary
            parameter dictionary

        See Also
        ----------
        anmodel.search : random parameter search is implemented
        """
        param_dict: Dict = {}

        gX_name: List[str] = ['g_leak', 'g_nav', 'g_kvhh', 'g_kva', 'g_kvsi', 
                              'g_cav', 'g_kca', 'g_nap', 'g_kir']
        gX_name: List[str] = list(itertools.compress(gX_name, list(self.channel_bool.values())[:9]))
        gX_log: np.ndarray = 4 * np.random.rand(len(gX_name)) - 2  # from -2 to 2
        gX: np.ndarray = (10 * np.ones(len(gX_name))) ** gX_log  # 0.01 ~ 100
        gX_itr: Iterator = zip(gX_name, gX)

        gR_name: List[str] = ['g_ampar', 'g_nmdar', 'g_gabar']
        gR_name: List[str] = list(itertools.compress(gR_name, list(self.channel_bool.values())[9:12]))
        gR_log: np.ndarray = 4 * np.random.rand(len(gR_name)) - 3  # from -3 to 1
        gR: np.ndarray = (10 * np.ones(len(gR_name))) ** gR_log  # 0.001 ~ 10
        gR_itr: Iterator = zip(gR_name, gR)

        param_dict.update(gX_itr)
        param_dict.update(gR_itr)

        if self.channel_bool['ca']:
            tCa_log: np.ndarray = 2 * np.random.rand(1) + 1  # from 1 to 3
            tCa: np.ndarray = 10 ** tCa_log    # 10 ~ 1000
            tCa_dict: Dict = {'t_ca': float(tCa)}
            param_dict.update(tCa_dict)

        return param_dict

    def set_params(self, params: Dict) -> None:
        """ Set parameters to the channels

        Updated from ANmodel.set_param() for X model.

        Parameters
        ----------
        params : dictionary
            channel and pump parameters

        Raise
        ----------
        AttributeError
            This error occurs when channels you designated when creating X model
            is different from those designated in params.
        """
        channel_param: str
        for channel_param in list(params.keys())[:-1]:
            channel_name: str = channel_param[2:]
            if self.channel_bool[channel_name]:
                self.channel[channel_name].set_g(float(params[channel_param]))
            else:
                raise AttributeError('Model does not match parameter sets')

        if 't_ca' == list(params.keys())[-1]:  # or if 't_ca' in params.keys():
            self.tau_ca = float(params['t_ca'])
        else:
            self.tau_ca = float('inf')
        self.leak.set_div()

    def get_params(self) -> Dict:
        """ Get current channel conductances.

        Returns
        ----------
        dict
            parameter dictionary
        """
        params: Dict = {}
        channel: str
        for channel in list(self.channel.keys()):
            if self.channel_bool[channel]:
                params[channel] = self.channel[channel].get_g()
        if self.tau_ca != float('inf'):
            params['t_ca'] = self.tau_ca
        return params
        
    def dvdt(self, args: Dict) -> float:
        """ Calculate dv/dt for given parameters.

        Membrane potential changes over time dependent on currents that
        flow each channels and neurotransmitter receptors. Updated from
        ANmodel.dvdt() for X model.

        Parameters
        ----------
        args : dictionary
            keys : str\\
                names of variables for the differential equations\\
            values : float\\
                values of variables for the differential equations\\

        Results
        ----------
        float
            dv/dt for given parameters
        """
        if self.channel_bool['leak']:
            i_leak: float = self.leak.i_div(args['v'])
        else:
            i_leak: float = 0.
        
        if self.channel_bool['nav']:
            i_nav: float = self.nav.i(args['v'], h=args['h_nav'])
        else:
            i_nav: float = 0.

        if self.channel_bool['kvhh']:
            i_kvhh: float = self.kvhh.i(args['v'], n=args['n_kvhh'])
        else:
            i_kvhh: float = 0.

        if self.channel_bool['kva']:
            i_kva: float = self.kva.i(args['v'], h=args['h_kva'])
        else:
            i_kva: float = 0.

        if self.channel_bool['kvsi']:
            i_kvsi: float = self.kvsi.i(args['v'], m=args['m_kvsi'])
        else:
            i_kvsi: float = 0.

        if self.channel_bool['cav']:
            i_cav: float = self.cav.i(args['v'])
        else:
            i_cav: float = 0.

        if self.channel_bool['kca']:
            i_kca: float = self.kca.i(args['v'], ca=args['ca'])
        else:
            i_kca: float = 0.
        
        if self.channel_bool['nap']:
            i_nap: float = self.nap.i(args['v'])
        else:
            i_nap: float = 0.

        if self.channel_bool['kir']:
            i_kir: float = self.kir.i(args['v'])
        else:
            i_kir: float = 0.

        if self.channel_bool['ampar']:
            i_ampar: float = self.ampar.i(args['v'], s=args['s_ampar'])
        else:
            i_ampar: float = 0.

        if self.channel_bool['nmdar']:
            i_nmdar: float = self.nmdar.i(args['v'], s=args['s_nmdar'])
        else:
            i_nmdar: float = 0.

        if self.channel_bool['gabar']:
            i_gabar: float = self.gabar.i(args['v'], s=args['s_gabar'])
        else:
            i_gabar: float = 0.

        return ((-10.0*self.params.area 
                * (i_leak
                + i_nav 
                + i_kvhh 
                + i_kva 
                + i_kvsi 
                + i_cav 
                + i_kca 
                + i_nap 
                + i_kir) 
                - (i_ampar 
                + i_nmdar 
                + i_gabar))
                / (10.0*self.params.cm*self.params.area))

    def dCadt(self, args: Dict) -> float:
        """ Calculate dCa/dt for given parameters.

        Intracellular calcium changes over time dependent on CaV channel, 
        NMDA receptor, and calcium pump. Updated from ANmodel.dCadt() 
        for X model.

        Parameters
        ----------
        args : dictionary\\
            keys : str\\
                names of variables for the differential equations\\
            values : float\\
                values of variables for the differential equations\\

        Returns
        ----------
        float
            dCa/dt
        """
        if self.channel_bool['cav']:
            i_cav: float = self.cav.i(args['v'])
        else:
            self.I_cav = 0

        if self.channel_bool['nmdar']:
            i_nmdar: float = self.nmdar.i(args['v'], s=args['s_nmdar'])
        else:
            i_nmdar: float = 0

        ca: float = args['ca']
        a_ca: float = self.params.a_ca
        area: float = self.params.area
        tau_ca: float = self.tau_ca
        dCadt: float = -a_ca * (10.0*area*i_cav) - a_ca*i_nmdar - ca/tau_ca
        return dCadt

    def diff_op(self, args: List, time: float) -> List:
        """ Differential equations to be solved.

        Parameters
        ----------
        args : list (float)
            valuable list in a certain time
        time : float
            each time point to solve differential equation

        Returns
        ----------
        list
            list of variables differentiated by t
        """
        ode_args: Dict = dict(zip(self.ode_list, args))
        dvdt: float = self.dvdt(args=ode_args)
        ode: List[float] = [dvdt]

        if self.channel_bool['nav']:
            dhNadt: float = self.nav.dhdt(v=ode_args['v'], h=ode_args['h_nav'])
            ode.append(dhNadt)        
        if self.channel_bool['kvhh']:
            dnKdt: float = self.kvhh.dndt(v=ode_args['v'], n=ode_args['n_kvhh'])
            ode.append(dnKdt)
        if self.channel_bool['kva']:
            dhAdt: float = self.kva.dhdt(v=ode_args['v'], h=ode_args['h_kva'])
            ode.append(dhAdt)
        if self.channel_bool['kvsi']:
            dmKSdt: float = self.kvsi.dmdt(v=ode_args['v'], m=ode_args['m_kvsi'])
            ode.append(dmKSdt)
        if self.channel_bool['ampar']:
            dsAMPAdt: float = self.ampar.dsdt(v=ode_args['v'], s=ode_args['s_ampar'])
            ode.append(dsAMPAdt)
        if self.channel_bool['nmdar']:
            dxNMDAdt: float = self.nmdar.dxdt(v=ode_args['v'], x=ode_args['x_nmdar'])
            dsNMDAdt: float = self.nmdar.dsdt(v=ode_args['v'], s=ode_args['s_nmdar'], x=ode_args['x_nmdar'])
            ode.append(dxNMDAdt)
            ode.append(dsNMDAdt)
        if self.channel_bool['gabar']:
            dsGABAdt: float = self.gabar.dsdt(v=ode_args['v'], s=ode_args['s_gabar'])
            ode.append(dsGABAdt)
        if self.channel_bool['cav'] or self.channel_bool['nmdar'] or self.channel_bool['ca']:
            ca_args: Dict = {
                'v' : ode_args.get('v', None),
                's_nmdar' : ode_args.get('s_nmdar', None),
                'ca' : ode_args.get('ca', None),
            }
            dCadt: float = self.dCadt(ca_args)
            ode.append(dCadt)
        
        return ode
