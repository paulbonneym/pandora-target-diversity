#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import itertools
import multiprocessing as mp


#create weights from the standard deviation of each of the parameters for the entire target list
#i.e., if two values are more than 2 std away from each other, they are sufficiently different
#tun is a tuning parameter in case we want to make this distance larger or smaller
def create_weight(target_list:pd.DataFrame,
                  params:list,
                  tun:float):
    """ Creates the weights to be used in calculating the diversity of a target set.
    This is achieved by calculation the weight needed to calculate a diversity score
    of 0.8 if two parameters are "sufficiently diverse." By default, this means they
    are exactly 2 standard deviations apart from each other. Standard deviations for
    this calculation are calculated from the whole of target_list for each parameter.
        
    Parameters
    ----------
    target_list:    pandas.DataFrame
                    DataFrame including the targets, a tsm labeled 'TSM', and at least the 
                    columns in params.
    params:         list
                    A list of the parameters to be considered. Each one should be a string
                    that corresponds to a column in target_list
    tun:            float
                    Tuning parameter for weights used in diversity calculations.
                    Default is 2, meaning targets with a parameter >= 2 standard deviations
                    apart are considered diverse (diversity score >= 0.8).
    Returns
    -------
    list
        List of the weights to be used in order of the specified parameters.
    """
    
    wts=[]
    
    for p in params:
        param_list=target_list[p]
    
        u=tun*np.exp(np.std(np.log(param_list)))
        p_med=np.exp(np.nanmedian(np.log(param_list)))
        
        wts.append(np.log(0.8)/np.log(np.abs(u/(p_med+p_med+u))))
        
    return wts


#MC picker based on log(tsm)
def get_list(t:int,
             target_list:pd.DataFrame):
    """ Creates a short list randomly base on the transmission spectroscopy metric.
        
    Parameters
    ----------
    t:              int
                    Number of targets in the short list
    target_list:    pandas.DataFrame
                    DataFrame including the targets, a tsm labeled 'TSM', and at least the 
                    columns in params.
    Returns
    -------
    list
        A list of indicies in target_list that form the short list to be considered.
    """
    
    tsm=target_list['TSM']
    log_tsm=np.log(tsm) #use log(tsm) instead because it varys more smoothly
    norm_log_tsm=(log_tsm-np.min(log_tsm))/(np.max(log_tsm)-np.min(log_tsm)) #Normalize log_tsm to [0,1] for pseudo-MCMC picker
    index_list=np.arange(len(tsm)).tolist() #make a list to pop later, avoids choosing the same target twice
    
    rng = np.random.default_rng()
    acc_set=rng.random(int(t)) #Random set of numbers for TSM acceptance
    
    target_set=[]
    
    for i in acc_set:
        
        
        high=len(np.array(index_list)[np.array(index_list) < len(norm_log_tsm[norm_log_tsm>i])])
        
        #this avoids a case where the only possible target(s) to pick have already been chosen
        while high == 0:
            rng = np.random.default_rng()
            i = rng.random()
            high=len(np.array(index_list)[np.array(index_list) < len(norm_log_tsm[norm_log_tsm>i])])
        j=int(np.random.uniform(low=0,high=high))
        target_set.append(index_list.pop(j))
        
    return target_set

#dissimilarity function, weighted variation of the Manhattan Distance
dis = lambda val1, val2, wt: np.abs((val1-val2)/(val1+val2))**wt
""" Diversity calculation function.
    
Parameters
----------
val1:           float
                Number of potential short lists to consider/number of iterations
val2:           float
                Number of targets in the short list
wt:             float
                Tuning parameter for weights used in diversity calculations.
                Default is 2, meaning targets with a parameter >= 2 standard deviations
                apart are considered diverse (diversity score >= 0.8).
Returns
-------
float
    A number indicating the dissimilarity between the two values.
    
Notes
-----
The function is a weighted variation of the Manhattan distance based on the
Bray-Curtis dissimilarity (Bray & Curtis, 1957; doi:10.2307/1942268) and the Earth
Similarity Index (Schulze-Makuch et al., 2011; doi:10.1089/ast.2010.0592), though
the usage here is more similar to the original usage in ecology.
"""
        
#diversity calc
def calc_dis(target_set:list,
             wts:list,
             target_list:pd.DataFrame):
    """ Calculates the average diversity score for a given target_set within target_list.
        
    Parameters
    ----------
    target_set:     list
                    A list of indicies in target_list that form the short list to be considered.
    wts:            list
                    List of the weights to be used in order of the specified parameters.
    target_list:    pandas.DataFrame
                    DataFrame including the targets, a tsm labeled 'TSM', and at least the 
                    columns in params.
    Returns
    -------
    float
        A number indicating the average diversity score for the final short list.
    
    Notes
    -----
    This function has the parameters hard coded into it. If changes are made to params,
    those changes need to be reflected in this code. See the README for more information.
    """
    
    dis_arr=[]
    
    #get parameters from dataframe
    st_rad=target_list['st_rad'][target_set]
    st_teff=target_list['st_teff'][target_set]
    st_rotp=target_list['st_rotp'][target_set]
    pl_orbper=target_list['pl_orbper'][target_set]
    pl_dens=target_list['pl_dens'][target_set]
    
    #create weights
    st_rad_wt=wts[0]
    st_teff_wt=wts[1]
    st_rotp_wt=wts[2]
    pl_orbper_wt=wts[3]
    pl_dens_wt=wts[4]
    
    for i,j in itertools.combinations(target_set,2):
        
        #dissimilarity calculations
        #if dissimilarity is a nan, replace with 0 as a value is likely missing
        d=dis(st_rad[i],st_rad[j],st_rad_wt)
        if np.isnan(d):
            d=0
        dis_arr.append(d)
        d=dis(st_teff[i],st_teff[j],st_teff_wt)
        if np.isnan(d):
            d=0
        dis_arr.append(d)
        d=dis(st_rotp[i],st_rotp[j],st_rotp_wt)
        if np.isnan(d):
            d=0
        dis_arr.append(d)
        d=dis(pl_orbper[i],pl_orbper[j],pl_orbper_wt)
        if np.isnan(d):
            d=0
        dis_arr.append(d)
        d=dis(pl_dens[i],pl_dens[j],pl_dens_wt)
        if np.isnan(d):
            d=0
        dis_arr.append(d)
        
    #take the average dissimilarity and return
    av_dis=np.average(dis_arr)
    return av_dis