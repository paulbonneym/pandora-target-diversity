#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
TARGETLIST=pd.read_csv(f'{PACKAGEDIR}/pandora_targets.csv')
PARAMS=['st_rad','st_teff','st_rotp','pl_orbper','pl_dens']


def short_list(n: int,
               t: int = 20,
               tun: float = 2.0,
               target_list: pd.DataFrame = TARGETLIST,
               params: list = PARAMS,
               systems: bool = False,
               return_best: bool = True):
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
    systems:        bool
                    Toggle whether to consider the target list from a system perspective
                    (True) or from an individual target perspective (False; default)
    Returns
    -------
    If return_best == True:
        pandas.DataFrame
            A slice of target_list including only the targets of the most diverse list considered
            and only the columns defined in params.
    Else:
        If systems == True:
            list
                A list of n lists indicating the location for all of the systems sampled from 
                target_list.
            list
                A list of n lists indicating the location for all of the planets sampled from 
                target_list.
            numpy.ndarray
                A numpy array of shape (5, n) containing the average diversity score for each 
                parameter for each iteration.
            numpy.ndarray
                A numpy array of shape (n) containing the average tsm score for each iteration.
                The tsm score is calculated by first summing the TSM for each systems' planets
                and then normalizing all of these values to between 0 and 1. Then the final score
                for each iteration is taken as the average of these summed scores for the sampled
                systems.
            pandas.Dataframe
                A slice of target_list including all targets but only the columns defined in params.
        If systems = False:
            list
                A list of n lists indicating the location for all of the planets sampled from 
                target_list.
            numpy.ndarray
                A numpy array of shape (5, n) containing the average diversity score for each 
                parameter for each iteration.
            numpy.ndarray
                A numpy array of shape (n) containing the average tsm score for each iteration.
                The tsm score is calculated by first normalizing all of the TSM values to between 
                0 and 1. Then the final score for each iteration is taken as the average of these 
                normalized scores for the sampled planets.
            pandas.Dataframe
                A slice of target_list including all targets but only the columns defined in params.
            
    """
    
    df=target_list[["hostname", *params, "TSM"]].sort_values("TSM", ascending=False)
    tsm=df.TSM
    
    u = tun * np.exp(np.std(np.log(df[params]), axis=0))
    p_med = np.exp(np.nanmedian(np.log(df[params]), axis=0))
    
    wt = np.log(0.8) / np.log(np.abs(u / (p_med + p_med + u)))
    
    Xs = []
    for param in params:
        x = np.nan_to_num(np.asarray(df[param]))
        Xs.append(np.abs((x[:, None] - x) / (x[:, None] + x)) ** wt[param])
    Xs = np.nan_to_num(np.asarray(Xs))
    
    sample_locs=[]
    
    if not systems:
        #normalize tsm to between 0-1 for scoring purposes
        tsm_norm=tsm-np.min(tsm)
        tsm_norm/=np.max(tsm_norm)
        for i in range(n):
            sample_locs.append(np.random.choice(np.arange(len(df)), size=t, p=tsm / tsm.sum(), replace=False))
        sample_locs=np.array(sample_locs)
        
        A = np.array([Xs[:, i, :][:, :, i] for i in sample_locs])
        param_div_score = (A[:,:,np.triu_indices(t, k=1)[0],np.triu_indices(t, k=1)[1]].sum(axis=(2)) / len(np.triu_indices(t, k=1)[0]))
        div_score = param_div_score.sum(axis=1) / len(params)
        tsm_score = np.array([ tsm_norm[i].mean() for i in sample_locs])
        
        if return_best:
            return df.iloc[sample_locs[np.where(div_score+tsm_score == np.max(div_score+tsm_score))[0][0]]]
        
        return sample_locs, param_div_score, tsm_score, df
        
    else:
        sample_locs=[]
        sample_host_locs=[]
        sample_tsms=[]
        #compress list and sum TSM
        it_name=np.array(list(set(df['hostname'])))
        it_mask=[df['hostname']==t for t in it_name]
        tsm_list=[]
        for m in it_mask:
            tsm_list.append(np.sum(tsm[m]))
        
        #new dataframe for compressed tsm
        tsm_comp=np.array(tsm_list).T
        #normalize this one as well
        tsm_norm=tsm_comp-np.min(tsm_comp)
        tsm_norm/=np.max(tsm_norm)
        
        for i in range(n):
        
            #choose the locations based on compressed tsm + get hostnames
            sample = np.random.choice(np.arange(len(it_name)),
                                           size=t,
                                           p=tsm_norm / tsm_norm.sum(),
                                           replace=False)
            sample_tsms.append(tsm_norm[sample])
            hosts=it_name[sample]
            
            #get location of all planets with these hostnames and append to sample_locs
            sample_locs.append(np.where(np.sum([df['hostname']==t for t in hosts], axis=0)==1)[0])
            #retain sample host locs to compare system parameter diversity
            sample_host_locs.append([np.where(df['hostname']==t)[0][0] for t in hosts])
        
        sample_tsms = np.array(sample_tsms)
        
        param_div_score=[]
        for p in range(len(params)):
            if params[p].startswith('st'):
                A = [Xs[p][i, :][:, i] for i in sample_host_locs]
                param_div_score.append([A[a][np.triu_indices(t, k=1)[0],np.triu_indices(t, k=1)[1]].mean() for a in range(n)])
                
            else:
                A = [Xs[p][i, :][:, i] for i in sample_locs]
                param_div_score.append([A[a][np.triu_indices(len(A[a]), k=1)[0],np.triu_indices(len(A[a]), k=1)[1]].mean() for a in range(n)])
                
        param_div_score = np.array(param_div_score)
        
        div_score = param_div_score.sum(axis=0) / len(params)
        tsm_score = sample_tsms.mean(axis=1)
        
        if return_best:
            return df.iloc[sample_locs[np.where(div_score+tsm_score == np.max(div_score+tsm_score))[0][0]]]
        
        return sample_host_locs, sample_locs, param_div_score, tsm_score, df
    
