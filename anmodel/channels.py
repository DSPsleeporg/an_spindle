# -*- coding: utf-8 -*-

"""
This is the channel module for Averaged Neuron (AN) model. 
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

import numpy as np
from typing import Optional

import params


params = params.Constants()


class Base:
    """ Keep basic attributes and helper functions for each channel.

    Parameters
    ----------
    g : float
        channel condactance
    e : float
        equilibrium (reversal) potential for a channel
    
    Attributes
    ----------
    g : float
        channel conductance
    e : float
        equilibrium (reversal) potential for a channel
    """

    def __init__(self, g: float, e: float) -> None:
        self.g = g
        self.e = e
    
    def set_g(self, new_g: float) -> None:
        """ Set a new conductance for a channel.

        Parameters
        ----------
        new_g : float
            new conductance set for a channel
        """
        self.g = new_g

    def get_g(self) -> float:
        ''' Get current channel conductance value.

        Returns
        ----------
        float
            current conductance
        '''
        return self.g

    def set_e(self, new_e: float) -> None:
        """ Set a new equiribrium potential for a channel

        Parameters
        ----------
            new equiribrium potential for a channel
        """
        self.e = new_e


class Leak(Base):
    """ Leak channel (sodium / potassium).

    Leak channel can be divided into leak sodium channel and leak
    potassium channel. Usually it doesn't have to be divided. If 
    it is separated, you can conduct more detailed analysis.

    Parameters
    ----------
    g : float
    channel conductance
    e : float
        equiribrium potential for the channel

    Attributes
    ----------
    g : float
        the channel conductance
    e : float
        equiribrium potential for the channel
    """
    def __init__(self, g: Optional[int]=None, e: float=params.vL) -> None:
        super().__init__(g, e)

    def i(self, v:float) -> float:
        """ Calculate current that flows through the channel.

        I = g * (v - e).
        v : membrane potential
        e : equiribrium potential (for sodium ion)

        Parameters
        ----------
        v : float
            membrane potential

        Returns
        ----------
        float
            current that flows through the channel
        """
        return self.g * (v - self.e)

    def set_div(self, vnal: float=params.vNaL, vkl: float=params.vK) -> None:
        """ Setting about deviding leak channel into Na leak and K leak.

        Conductances of leak potassium channel and leak sodium channel are 
        defined as:
        gkl = gleak * (vleak - vnal) / (vkl - vnal)
        gnal = gleak * (vleak - vkl) / (vnal - vkl).
        These definition sattisfies gleak=gkl+gnal.

        Parameters
        ----------
        vk : float
            equilibrium potential for leak potassium channel
        vnal : float
            equilibrium potential for leak sodium channel
        """
        self.vnal = vnal
        self.vkl = vkl
        self.gnal = self.g * (self.e - self.vkl) / (self.vnal - self.vkl)
        self.gkl = self.g * (self.e - self.vnal) / (self.vkl - self.vnal)

    def set_gna(self, new_gnal: float) -> None:
        """ Set a new conductance for a leak sodium channel.

        Parameters
        ----------
        new_gna : float
            new conductance set for a leak sodium channel
        """
        self.gnal = new_gnal
    
    def set_gk(self, new_gkl: float) -> None:
        """ Set a new conductance for a leak potassium channel.

        Parameters
        ----------
        new_gk : float
            new conductance set for a leak potassium channel
        """
        self.gkl = new_gkl

    def ikl(self, v: float) -> float:
        """ Calculate current that flows through the channel.

        I = g * (v - e).
        v : membrane potential
        e : equiribrium potential (for sodium ion)

        Parameters
        ----------
        v : float
            membrane potential

        Returns
        ----------
        float
            current that flows through the channel
        """
        return self.gkl * (v - self.vkl)

    def inal(self, v: float) -> float:
        """ Calculate current that flows through the channel.

        I = g * (v - e).
        v : membrane potential
        e : equiribrium potential (for sodium ion)

        Parameters
        ----------
        v : float
            membrane potential

        Returns
        ----------
        float
            current that flows through the channel
        """
        return self.gnal * (v - self.vnal)
    
    def i_div(self, v: float) -> float:
        return self.inal(v) + self.ikl(v)


class NavHH(Base):
    """ Hodgkin-Huxley type volatage gated sodium channel.

    Note
    ----------
    This formulation is from Compte et al., 2006

    Parameters
    ----------
    g : float or None
        channel conductance, default None
    e : float
        equiribrium potential for the channel, 
        default anmodel.params.Constatns.VNa.

    Attributes
    ----------
    g : float
        HH type sodium channel conductance
    e : float
        equiribrium potential for the channel
        (in most cases, sodium equiribrium potential)
    """

    def __init__(self, g: Optional[float]=None, e: float=params.vNa) -> None:
        super().__init__(g, e)

    def am(self, v: float) -> float:
        """ Calculate voltage-dependent transition rate for activation states.

        In the two state model, activation variable m can be discribed as,
            dm/dt = a(1-m) + bm.
        In this method, 'a' for m can be calculated.

        Parameters
        ----------
        v : float
            membrane potential

        Returns
        ----------
        float
            transition rate for activation states
        """
        if v == -33.:
            return 1.
        else:
            return 0.1 * (v+33.0) / (1.0-np.exp(-(v+33.0)/10.0))

    def bm(self, v: float) -> float:
        """ Calculate voltage-dependent transition rate for activation states.

        'b' in the two state model for m can be calculated.

        Parameters
        ----------
        v : float
            membrane potential

        Returns
        ----------
        float
            transition rate for activation states
        """
        return 4.0 * np.exp(-(v+53.7)/12.0)

    def m_inf(self, v: float) -> float:
        """ Calculate activation variable for steady state.

        In the steady state two state model for activation variable, dm/dt = 0.
        Now, m can be calculated from 0 = a(1-m) + bm. 

        Parameters
        ----------
        v : float
            membrane potential

        Results
        ----------
        float
            activation variable for the channel
        """
        return self.am(v) / (self.am(v) + self.bm(v))

    def ah(self, v: float) -> float:
        """ Calculate voltage-dependent transition rate for inactivation states.

        'a' in the two state model for inactivation variable h can be calculated.

        Parameters
        ----------
        v : float
            membrane potential

        Results
        ----------
        float
            transition rate for inactivation states
        """
        return 0.07 * np.exp(-(v+50.0)/10.0)

    def bh(self, v: float) -> float:
        """ Calculate voltage-dependent transition rate for inactivation states.

        'b' in the two state model for h can be calculated.

        Parameters
        ----------
        v : float
            membrane potential
        
        Results
        ----------
        float
            transition rate for inactivation states
        """
        return 1.0 / (1.0 + np.exp(-(v+20.0)/10.0))

    def h_inf(self, v: float) -> float:
        """ Calculate inactivation rate in steady state.

        In the steady state two state model for activation variable, dh/dt = 0.
        Now, m can be calculated from 0 = a(1-h) + bh. 

        Note
        ----------
        This isn't used in the AN model (not steady state, in this case).

        Parameters
        ----------
        v : float
            membrane potential

        Returns
        ----------
        float
            inactivation variable for the channel
        """
        return self.ah(v) / (self.ah(v) + self.bh(v))

    def h_tau(self, v: float) -> float:
        return 1 / 4 * (self.ah(v)+self.bh(v))

    def dhdt(self, v: float, h: float) -> float:
        """ Differential equation for inactiavtion variable (not in steady state).

        In the two state model, inactivation variable can be formulated as
        dh/dt = a(1-h) + bh.
        In this case, right side is multiplied by constant.

        Parameters
        ----------
        v : float
            membrane potential
        h : float
            inactivation variable

        Returns
        ----------
        float
            dh/dt
        """
        return 4.0 * (self.ah(v)*(1-h) - self.bh(v)*h)

    def i(self, v: float, h: float) -> float:
        """ Calculate current that flows through the channel.

        I = g * m^3 * h * (v - e).
        m : activation variable (steady state)
        h : inactiavtion variable
        v : membrane potential
        e : equiribrium potential (for sodium ion)

        Parameters
        ----------
        v : float
            membrane potential
        h : float
            inactivation variable

        Returns
        ----------
        float
            current that flows through the channel
        """
        return self.g * (self.m_inf(v)**3) * h * (v-self.e)


class KvHH(Base):
    """ Hodgkin-Huxley type voltage gated potassium channel (delayed rectifier).

    Note
    ----------
    This formulation is from Compte et al., 2006

    Parameters
    ----------
    g : float or None
        channel conductance, default None
    e : float
        equiribrium potential for the channel, 
        default anmodel.params.Constants.VK.

    Attributes
    ----------
    g : float
        HH type potassium channel conductance
    e : float
        equiribrium potential for the channel
        (in most cases, potassium equiribrium potential)
    """

    def __init__(self, g: Optional[float]=None, e: float=params.vK) -> None:
        super().__init__(g, e)

    def an(self, v: float) -> float:
        """ Calculate voltage-dependent transition rate for activation states.

        'a' in the two state model for activation state n can be calculated.

        Parameters
        ----------
        v : float
            membrane potential

        Results
        ----------
        float
            transition rate for activation states
        """
        if v == -34.:
            return 0.1
        else:
            return 0.01 * (v+34.0) / (1.0-np.exp(-(v+34.0)/10.0))

    def bn(self, v: float) -> float:
        """ Calculate voltage-dependent transition rate for activation states.

        'b' in the two state model for activation state n can be calculated.

        Parameters
        ----------
        v : float
            membrane potential

        Results
        ----------
        float
            transition rate for activation states
        """
        return 0.125 * np.exp(-(v+44.0)/25.0)

    def n_inf(self, v: float) -> float:
        """ Calculate activation variable in steady state.

        In the steady state two state model for activation state, dn/dt = 0.
        Now, n can be calculated from 0 = a(1-n) + bn. 

        Note
        ----------
        This isn't used in the AN model (not steady state, in this case).

        Parameters
        ----------
        v : float
            membrane potential

        Results
        ----------
        float
            activation variable for the channel
        """
        return self.an(v) / (self.an(v)+self.bn(v))

    def n_tau(self, v: float) -> float:
        return 1 / (4 * (self.an(v) + self.bn(v)))

    def dndt(self, v: float, n: float) -> float:
        """ Differential equation for actiavtion variable (not in steady state).

        In the two state model, activation variable can be formulated as
        dn/dt = a(1-n) + bn.
        In this case, right side is multiplied by constant.

        Parameters
        ----------
        v : float
            membrane potential
        n : float
            activation variable

        Returns
        ----------
        float
            dn/dt
        """
        return 4.0 * (self.an(v)*(1-n)-self.bn(v)*n)

    def i(self, v: float, n: float) -> float:
        """ Calculate current that flows through the channel.

        I = g * n^4 * (v - e).
        n : activation variable (steady state)
        v : membrane potential
        e : equiribrium potential (for potassium ion)

        Parameters
        ----------
        v : float
            membrane potential
        h : float
            inactivation variable

        Returns
        ----------
        float
            current that flows through the channel
        """
        return self.g * n**4 * (v-self.e)


class KvA(Base):
    """ Fast A-type potassium channel.

    Note
    ----------
    This formulation is from Compte et al., 2006

    Parameters
    ----------
    g : float or None
        channel conductance, default None
    e : float
        equiribrium potential for the channel, 
        default anmodel.params.Constants.VK.
    tau : float
        time constant for the differential equation dh/dt, 
        default anmodel.params.Constants.tauA

    Attribute
    ----------
    g : float
        the channel conductance
    e : float
        equiribrium potential for the channel
        (in most cases, potassium equiribrium potential)
    tau : float
        time constant for the differential equation dh/dt
    """
    def __init__(self, g: Optional[float]=None, e: float=params.vK, 
                 tau: float=params.tau_a) -> None:
        super().__init__(g, e)
        self.tau = tau

    def m_inf(self, v: float) -> float:
        """ Calculate activation variable in steady state.

        In this case, we use fitted formulation, not two state model.

        Parameters
        ----------
        v : float
            membrane potential

        Results
        ----------
        float
            activation variable for the channel
        """
        return 1.0 / (1.0 + np.exp(-(v+50.0)/20.0))

    def h_inf(self, v: float) -> float:
        """ Calculate inactivation variable in steady state.

        We use fitted formulation, not two state model.

        Note
        ----------
        This isn't used in the AN model (not steady state, in this case).

        Parameters
        ----------
        v : float
            membrane potential

        Results
        ----------
        float
            inactivation variable for the channel
        """
        return 1.0 / (1.0 + np.exp((v+80.0)/6.0))

    def dhdt(self, v: float, h: float) -> float:
        """ Differential equation for inactiavtion variable (not in steady state).

        dh/dt = (h_inf - h) / tau

        Parameters
        ----------
        v : float
            membrane potential
        h : float
            inactivation variable

        Returns
        ----------
        float
            dh/dt
        """
        return (self.h_inf(v)-h) / self.tau

    def i(self, v: float, h: float) -> float:
        """ Calculate current that flows through the channel.

        I = g * m^3 * h * (v - e).
        m : activation variable (steady state)
        h : inactivation variable
        v : membrane potential
        e : equiribrium potential (for potassium ion)

        Parameters
        ----------
        v : float
            membrane potential
        h : float
            inactivation variable

        Returns
        ----------
        float
            current that flows through the channel
        """
        return self.g * (self.m_inf(v)**3) * h * (v-self.e)


class KvSI(Base):
    """ Slowly inactivating potassium channel (a kind of delayed rectifier).

    Slowly inactivating potassium channel doesn't have inactivation variable.

    Note
    ----------
    This formulation is from Compte et al., 2006

    Parameters
    ----------
    g : float
    channel conductance
    e : float
        equiribrium potential for the channel

    Attributes
    ----------
    g : float
        the channel conductance
    e : float
        equiribrium potential for the channel
        (in most cases, potassium equiribrium potential)
    """
    def __init__(self, g: Optional[float]=None, e: float=params.vK) -> None:
        super().__init__(g, e)

    def m_inf(self, v: float) -> float:
        """ Calculate activation variable in steady state.

        In this case, we use fitted formulation, not two state model.

        Parameters
        ----------
        v : float
            membrane potential

        Results
        ----------
        float
            activation variable for the channel
        """
        return 1.0 / (1.0 + np.exp(-(v+34.0)/6.5))

    def m_tau(self, v: float) -> float:
        """ Calculate time constant for the differential equation dm/dt

        Parameters
        ----------
        v : float
            membrane potential

        Results
        ----------
        float
            time constant for the differential equation dm/dt
        """
        return 8.0 / (np.exp(-(v+55.0)/30.0) + np.exp((v+55.0)/30.0))

    def dmdt(self, v: float, m: float) -> float:
        """ Differential equation for actiavtion variable (not in steady state).

        dm/dt = (m_inf - m) / tau

        Parameters
        ----------
        v : float
            membrane potential
        m : float
            activation variable

        Returns
        ----------
        float
            dm/dt
        """
        return (self.m_inf(v)-m) / self.m_tau(v)

    def i(self, v: float, m: float) -> float:
        """ Calculate current that flows through the channel.

        I = g * m  * (v - e).
        m : activation variable (steady state)
        v : membrane potential
        e : equiribrium potential (for potassium ion)

        Parameters
        ----------
        v : float
            membrane potential
        h : float
            inactivation variable

        Returns
        ----------
        float
            current that flows through the channel
        """
        return self.g * m * (v-self.e)


class Cav(Base):
    """ Voltage-gated calcium channel.

    Parameters
    ----------
    g : float
    channel conductance
    e : float
        equiribrium potential for the channel

    Attributes
    ----------
    g : float
        the channel conductance
    e : float
        equiribrium potential for the channel
        (in most cases, calcium equiribrium potential)
    """
    def __init__(self, g: Optional[float]=None, e: float=params.vCa) -> None:
        super().__init__(g, e)

    def m_inf(self, v: float) -> float:
        """ Calculate activation variable in steady state.

        In this case, we use fitted formulation, not two state model.

        Parameters
        ----------
        v : float
            membrane potential

        Results
        ----------
        float
            activation variable for the channel
        """
        return 1.0 / (1.0 + np.exp(-(v+20.0)/9.0))

    def i(self, v: float) -> float:
        """ Calculate current that flows through the channel.

        I = g * m^2  * (v - e).
        m : activation variable (steady state)
        v : membrane potential
        e : equiribrium potential (for potassium ion)

        Parameters
        ----------
        v : float
            membrane potential

        Returns
        ----------
        float
            current that flows through the channel
        """
        return self.g * self.m_inf(v)**2 * (v-self.e)


class NaP(Base):
    """ Persistent sodium channel.

    Note
    ----------
    This formulation is from Compte et al., 2006

    Parameters
    ----------
    g : float
    channel conductance
    e : float
        equiribrium potential for the channel

    Attributes
    ----------
    g : float
        the channel conductance
    e : float
        equiribrium potential for the channel
        (in most cases, sodium equiribrium potential)
    """
    def __init__(self, g: Optional[float]=None, e: float=params.vNa) -> None:
        super().__init__(g, e)

    def m_inf(self, v: float) -> float:
        """ Calculate activation variable in steady state.

        In this case, we use fitted formulation, not two state model.

        Parameters
        ----------
        v : float
            membrane potential

        Results
        ----------
        float
            activation variable for the channel
        """
        return 1.0 / (1.0 + np.exp(-(v+55.7)/7.7))

    def i(self, v: float) -> float:
        """ Calculate current that flows through the channel.

        I = g * m^3  * (v - e).
        m : activation variable (steady state)
        v : membrane potential
        e : equiribrium potential (for potassium ion)

        Parameters
        ----------
        v : float
            membrane potential

        Returns
        ----------
        float
            current that flows through the channel
        """
        return self.g * self.m_inf(v)**3 * (v-self.e)


class KCa(Base):
    """ Calcium-dependent potassium channel.

    Parameters
    ----------
    g : float
    channel conductance
    e : float
        equiribrium potential for the channel
    kd_ca : float
        dissociation constant of calcium-dependent pottasium channels 

    Attributes
    ----------
    g : float
        the channel conductance
    e : float
        equiribrium potential for the channel
        (in most cases, calcium equiribrium potential)
    kd_ca : float
        dissociation constant of calcium-dependent pottasium channels 
    """
    def __init__(self, g: Optional[float]=None, e: float=params.vK, 
                 kd_ca: float=params.kd_ca) -> None:
        super().__init__(g, e)
        self.kd_ca = kd_ca

    def m_inf(self, ca: float) -> float:
        """ Calculate activation variable in steady state.

        In this case, we use fitted formulation, not two state model.

        Parameters
        ----------
        ca : float
            intracellular calcium concentration

        Results
        ----------
        float
            activation variable for the channel
        """
        return 1.0 / (1.0 + (self.kd_ca/ca)**(3.5))

    def i(self, v: float, ca: float) -> float:
        """ Calculate current that flows through the channel.

        I = g * m^3  * (v - e).
        m : activation variable (steady state)
        v : membrane potential
        e : equiribrium potential (for potassium ion)

        Parameters
        ----------
        v : float
            membrane potential
        ca : float
            intracellular calcium concentration

        Returns
        ----------
        float
            current that flows through the channel
        """
        return self.g * self.m_inf(ca) * (v-self.e)


class KIR(Base):
    """ Inwardly rectifying potassium channel.

    Note
    ----------
    This formulation is from Compte et al., 2006

    Parameters
    ----------
    g : float or None
    channel conductance, default None
    e : float
        equiribrium potential for the channel

    Attributes
    ----------
    g : float
        the channel conductance
    e : float
        equiribrium potential for the channel
        (in most cases, calcium equiribrium potential)
    """
    def __init__(self, g: Optional[float]=None, e: float=params.vK) -> None:
        super().__init__(g, e)

    def h_inf(self, v: float) -> float:
        """ Calculate inactivation variable in steady state.

        We use fitted formulation, not two state model.

        Parameters
        ----------
        v : float
            membrane potential

        Results
        ----------
        float
            inactivation variable for the channel
        """
        return 1.0/(1.0 + np.exp((v + 75.0)/4.0))

    def i(self, v: float) -> float:
        """ Calculate current that flows through the channel.

        I = g * h  * (v - e).
        h : inactivation variable (steady state)
        v : membrane potential
        e : equiribrium potential (for potassium ion)

        Parameters
        ----------
        v : float
            membrane potential

        Returns
        ----------
        float
            current that flows through the channel
        """
        return self.g * self.h_inf(v) * (v-self.e)


class AMPAR(Base):
    """ AMPA receptor.

    Note
    ----------
    This formulation is from Compte et al., 2006

    Parameters
    ----------
    g : float or None
    receptor conductance, default None
    e : float
        equiribrium potential for the receptor
    s_a : float
        coefficient of f(V)
    tau_a : float
        time constant for differential equation of gating variable s

    Attributes
    ----------
    g : float
        the receptor conductance
    e : float
        equiribrium potential for the receptor
    s_a : float
        coefficient of f(V)
    tau_a : float
        time constant for differential equation of gating variable s
    """
    def __init__(self, g: Optional[float]=None, e: float=params.vAMPAR, 
                 s_a: float=params.s_a_ampar, s_tau: float=params.s_tau_ampar) -> None:
        super().__init__(g, e)
        self.s_a = s_a
        self.tau_a = s_tau

    def f(self, v: float) -> float:
        """ Function that converts membrane potential into firing rate.

        Parameters
        ----------
        v : float
            membrane potential

        Returns
        ----------
        float
            f(v) : firing rate
        """
        return 1.0 / (1.0 + np.exp(-(v-20.0)/2.0))

    def dsdt(self, v: float, s: float) -> float:
        """ Differential equation for gating variable s.

        ds/dt = af(Vpre) - s/tau

        Parameters
        ----------
        v : float
            membrane potential
        s : float
            gating variable

        Returns
        ----------
        float
            ds/dt
        """
        return self.s_a * self.f(v) - s/self.tau_a

    def i(self, v: float, s: float) -> float:
        """ Calculate current that flows by the receptor.

        I = g * s  * (v - e).
        g : gating variable
        v : membrane potential
        e : equiribrium potential (for potassium ion)

        Parameters
        ----------
        v : float
            membrane potential
        s : float
            gating varriable

        Returns
        ----------
        float
            current that flows by the receptor
        """
        return self.g * s * (v - self.e)


class NMDAR(Base):
    """ NMDA receptor.

    The gating variable of NMDA receptor follows a two first-order 
    kinetic scheme so that EPSC has a slower rise phase and saturates
    at high presynaptic firing rates.

    Note
    ----------
    This formulation is from Compte et al., 2006 and Wang, 1999

    Parameters
    ----------
    g : float or None
    receptor conductance, default None
    e : float
        equiribrium potential for the receptor
    s_a : float
        coefficient of (1 - s)
    s_tau : float
        time constant for differential equation of gating variable s
    x_a : float
        coefficient of f(V)
    x_tau : float
        time constant for differential equation of second-order
        gating variable x

    Attributes
    ----------
    g : float
        the receptor conductance
    e : float
        equiribrium potential for the receptor
    s_a : float
        coefficient of f(V)
    tau_a : float
        time constant for differential equation of gating variable
    """
    def __init__(self, g: Optional[float]=None, e: float=params.vNMDAR, 
                 s_a: float=params.s_a_nmdar, s_tau: float=params.s_tau_nmdar, 
                 x_a: float=params.x_a_nmdar, x_tau: float=params.x_tau_nmdar,
                 ion: bool=False, ex_mg: Optional[float]=None) -> None:
        super().__init__(g, e)
        self.s_a = s_a
        self.s_tau = s_tau
        self.x_a = x_a
        self.x_tau = x_tau
        self.ion = ion
        self.ex_mg = ex_mg

    def f(self, v: float) -> float:
        """ Function that converts membrane potential into firing rate.

        Parameters
        ----------
        v : float
            membrane potential

        Returns
        ----------
        float
            f(v) : firing rate
        """
        return 1.0 / (1.0 + np.exp(-(v-20.0)/2.0))

    def dxdt(self, v: float, x: float) -> float:
        """ Differential equation for second-order gating variable x.

        dx/dt = af(Vpre) - x/tau

        Parameters
        ----------
        v : float
            membrane potential
        x : float
            second-order gating variable

        Returns
        ----------
        float
            dx/dt
        """
        return self.x_a * self.f(v) - x/self.x_tau

    def dsdt(self, v: float, s: float, x: float) -> float:
        """ Differential equation for gating variable s.

        ds/dt = ax(1 - s) - s/tau

        Parameters
        ----------
        v : float
            membrane potential
        s : float
            gating variable
        x : float
            second-order gating variable

        Returns
        ----------
        float
            ds/dt
        """
        return self.s_a * x * (1-s) - s/self.s_tau

    def i(self, v: float, s: float) -> float:
        """ Calculate current that flows by the receptor.

        I = a * g * s  * (v - e).
        a : scaling variable, 1 (not considering ion concentration) or
            1.1 / (1+Mg/8) (considering ion concentration)
        g : gating variable
        v : membrane potential
        e : equiribrium potential (for potassium ion)

        Parameters
        ----------
        v : float
            membrane potential
        s : float
            gating varriable

        Returns
        ----------
        float
            current that flows by the receptor
        """
        if self.ion:
            return 1.1 / (1.0+self.ex_mg/8.0) * self.g * s * (v-self.e)
        else:
            return self.g * s * (v-self.e)


class GABAR(Base):
    """ GABA receptor.

    Note
    ----------
    This formulation is from Compte et al., 2006

    Parameters
    ----------
    g : float or None
    receptor conductance, default None
    e : float
        equiribrium potential for the receptor
    s_a : float
        coefficient of f(V)
    tau_a : float
        time constant for differential equation of gating variable s

    Attributes
    ----------
    g : float
        the receptor conductance
    e : float
        equiribrium potential for the receptor
    s_a : float
        coefficient of f(V)
    tau_a : float
        time constant for differential equation of gating variable s
    """
    def __init__(self, g: Optional[float]=None, e: float=params.vGABAR, 
                 s_a: float=params.s_a_gabar, s_tau: float=params.s_tau_gabar) -> None:
        super().__init__(g, e)
        self.s_a = s_a
        self.s_tau = s_tau

    def f(self, v: float) -> float:
        """ Function that converts membrane potential into firing rate.

        Parameters
        ----------
        v : float
            membrane potential

        Returns
        ----------
        float
            f(v) : firing rate
        """
        return 1.0 / (1.0 + np.exp(-(v-20.0)/2.0))

    def dsdt(self, v: float, s: float) -> float:
        """ Differential equation for gating variable s.

        ds/dt = af(Vpre) - s/tau

        Parameters
        ----------
        v : float
            membrane potential
        s : float
            gating variable

        Returns
        ----------
        float
            ds/dt
        """
        return self.s_a * self.f(v) - s/self.s_tau

    def i(self, v: float, s: float) -> float:
        """ Calculate current that flows by the receptor.

        I = g * s  * (v - e).
        g : gating variable
        v : membrane potential
        e : equiribrium potential (for potassium ion)

        Parameters
        ----------
        v : float
            membrane potential
        s : float
            gating varriable

        Returns
        ----------
        float
            current that flows by the receptor
        """
        return self.g * s * (v-self.e)
