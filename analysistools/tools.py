# -*- coding: utf-8 -*-

from typing import Dict, List, Iterator

def get_gridargs() -> Dict:
    """ Get parameter range for histogram.

    Returns
    ----------
    grid_args : Dict
        parameter value range for each channel
    """
    grid_args: Dict = {}
    gX_name: List[str] = ['g_leak', 'g_nav', 'g_kvhh', 'g_kva', 'g_kvsi', 
                          'g_cav', 'g_kca', 'g_nap', 'g_kir']
    gX_range: List[List] = [[-2., 2., 0.01]] * len(gX_name)
    gX_itr: Iterator = zip(gX_name, gX_range)
    grid_args.update(gX_itr)
    gR_name: List[str] = ['g_ampar', 'g_nmdar', 'g_gabar']
    gR_range: List[List] = [[-3., 1., 0.01]] * len(gR_name)
    gR_itr: Iterator = zip(gR_name, gR_range)
    grid_args.update(gR_itr)
    tCa_dict: Dict = {'t_ca': [1., 3., 0.01]}
    grid_args.update(tCa_dict)
    return grid_args