#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import multiprocessing as mp
import os
import utils

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
target_list=pd.read_csv(f'{PACKAGEDIR}/pandora_targets.csv')
params=['st_rad','st_teff','st_rotp','pl_orbper','pl_dens']


def mp_doit(ins):
    
    test_set=utils.get_list(ins[-1],target_list)
    test_dis=utils.calc_dis(test_set,ins[:-1],target_list)
    
    return [test_set, test_dis]

def short_list(n: int,
               t: int = 20,
               tun: float = 2.0,
               target_list: pd.DataFrame = target_list,
               params: list = params
               ):
    """ Creates a short list based on target diversity in a set of parameters.
        
    Parameters
    ----------
    n:              int
                    Number of potential short lists to consider/number of iterations
    t:              int
                    Number of targets in the short list
    tun:            float
                    Tuning parameter for weights used in diversity calculations.
                    Default is 2, meaning targets with a parameter >= 2 standard deviations
                    apart are considered diverse (diversity score >= 0.8).
    target_list:    pandas.DataFrame
                    DataFrame including the targets, a tsm labeled 'TSM', and at least the 
                    columns in params.
    params:         list
                    A list of the parameters to be considered. Each one should be a string
                    that corresponds to a column in target_list
    Returns
    -------
    pandas.DataFrame
        A slice of target_list including only the targets of the most diverse list considered.
    float
        A number indicating the average diversity score for the final short list.
    """
    
    ins=utils.create_weight(target_list,params,tun)
    ins.append(t)
    
    p=mp.Pool(mp.cpu_count())
    res=p.map_async(mp_doit, np.ones((n,len(ins)))*ins)
    results=[r for r in res.get()]
    diss=[r[1] for r in results]
    final=results[np.where(diss==np.max(diss))[0][0]]
    sl=target_list.loc[final[0]]
    
    return sl, final[1]



#Future development:
#how unique is each target within this list? within the overall list?
#system approach -> add the tsm up for each planet in the system = total tsm
