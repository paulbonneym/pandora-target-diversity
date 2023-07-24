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
                    that corresponds to a column in target_list.
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
             target_list:pd.DataFrame,
             systems=False):
    """ Creates a short list randomly base on the transmission spectroscopy metric.
        
    Parameters
    ----------
    t:              int
                    Number of targets in the short list.
    target_list:    pandas.DataFrame
                    DataFrame including the targets, and at least a tsm column labeled 'TSM'.
    systems:        bool
                    Toggle whether to consider the target list from a system perspective
                    (True) or from an individual target perspective (False; default).
    Returns
    -------
    list
        A list of indicies in target_list that form the short list to be considered.
    """
    tsm=target_list['TSM']
    
    if systems:
        #removes duplicates
        it_name=list(set(target_list['hostname']))
        it_mask=[target_list['hostname']==t for t in it_name]
        
        tsm_list=[]
        for n in range(len(it_name)):
            tsm_list.append(np.sum(tsm[it_mask[n]]))
        
        #new dataframe for compressed tsm
        tsm=pd.DataFrame(np.array([tsm_list]).T, columns=['TSM'])
    
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
        
        if systems:
            host=it_name[index_list.pop(j)]
            for targ in target_list[target_list['hostname']==host].index:
                target_set.append(targ)
        else:
            target_set.append(index_list.pop(j))
        
        #if systems:
        #instead of popping indicies, pop names and append all the indicies that correspond to that name
        #still chould get out a list that included all indicies from the input target set
        
    return target_set


dis = lambda val1, val2, wt: np.abs((val1-val2)/(val1+val2))**wt
""" Diversity calculation function.
    
Parameters
----------
val1:           float
                The first parameter to consider.
val2:           float
                The second parameter to consider.
wt:             float
                Weight used for the calculation.
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
def calc_dis(test_set:pd.DataFrame,
             wts:list,
             systems: bool = False):
    """ Calculates the average diversity score for a given target_set within target_list.
        
    Parameters
    ----------
    test_set        pandas.DataFrame
                    DataFrame that is to be tested for diversity including the targets,
                    a tsm labeled 'TSM', and at least the columns in params.
    wts:            list
                    List of the weights to be used in order of the specified parameters.
    systems:        bool
                    Toggle whether to consider the target list from a system perspective
                    (True) or from an individual target perspective (False; default).
    Returns
    -------
    list
        A list tracking the two targets compared for each iteration. It contains the inidicies
        in target_list for each pair of targets.
    list
        A list of the diversity score for each pair of targets.
    
    Notes
    -----
    This function has the parameters hard coded into it. If changes are made to params,
    those changes need to be reflected in this code. See the README for more information.
    """
    
    dis_arr=[]
    it_arr=[]
    
    #get parameters from dataframe
    st_rad=test_set['st_rad']
    st_teff=test_set['st_teff']
    st_rotp=test_set['st_rotp']
    pl_orbper=test_set['pl_orbper']
    pl_dens=test_set['pl_dens']
    pl_name=test_set['pl_name']
    
    #create weights
    st_rad_wt=wts[0]
    st_teff_wt=wts[1]
    st_rotp_wt=wts[2]
    pl_orbper_wt=wts[3]
    pl_dens_wt=wts[4]
    
    #for considering systems as targets
    if systems:
        #removes duplicates
        it_name=list(set(test_set['hostname']))
        it_mask=[test_set['hostname']==t for t in it_name]
    
        for i,j in itertools.combinations(range(len(it_name)),2):
            dis_ar=[]
            #dissimilarity calculations
            #if dissimilarity is a nan, replace with 0 as a value is likely missing
            #stellar parameters are compared star to star
            d=dis(st_rad.loc[it_mask[i]].iloc[0],st_rad.loc[it_mask[j]].iloc[0],st_rad_wt)
            if np.isnan(d):
                d=0
            dis_ar.append(d)
            d=dis(st_teff[it_mask[i]].iloc[0],st_teff[it_mask[j]].iloc[0],st_teff_wt)
            if np.isnan(d):
                d=0
            dis_ar.append(d)
            d=dis(st_rotp[it_mask[i]].iloc[0],st_rotp[it_mask[j]].iloc[0],st_rotp_wt)
            if np.isnan(d):
                d=0
            dis_ar.append(d)
            
            #planetary parameters are compared between all planets in each system and the average over all is used
            plan_it=[]
            for a in range(len(test_set[it_mask[i]])):
                orbper_a=pl_orbper.loc[it_mask[i]].iloc[a]
                dens_a=pl_dens.loc[it_mask[i]].iloc[a]
                
                for b in range(len(test_set[it_mask[j]])):
                    orbper_b=pl_orbper.loc[it_mask[j]].iloc[b]
                    dens_b=pl_dens.loc[it_mask[j]].iloc[b]
                    
                    plan_it.append([pl_name.loc[it_mask[i]].iloc[a],pl_name.loc[it_mask[j]].iloc[b]])
                    
                    d=dis(orbper_a,orbper_b,pl_orbper_wt)
                    if np.isnan(d):
                        d=0
                    dis_ar.append(d)
                    d=dis(dens_a,dens_b,pl_dens_wt)
                    if np.isnan(d):
                        d=0
                    dis_ar.append(d)
                    
            dis_arr.append(dis_ar)
            it_arr.append([it_name[i],it_name[j],plan_it])
    
    #for considering individual planets as targets
    else:
        for i,j in itertools.combinations([i for i in test_set.index],2):
            dis_ar=[]
            #dissimilarity calculations
            #if dissimilarity is a nan, replace with 0 as a value is likely missing
            d=dis(st_rad[i],st_rad[j],st_rad_wt)
            if np.isnan(d):
                d=0
            dis_ar.append(d)
            d=dis(st_teff[i],st_teff[j],st_teff_wt)
            if np.isnan(d):
                d=0
            dis_ar.append(d)
            d=dis(st_rotp[i],st_rotp[j],st_rotp_wt)
            if np.isnan(d):
                d=0
            dis_ar.append(d)
            d=dis(pl_orbper[i],pl_orbper[j],pl_orbper_wt)
            if np.isnan(d):
                d=0
            dis_ar.append(d)
            d=dis(pl_dens[i],pl_dens[j],pl_dens_wt)
            if np.isnan(d):
                d=0
            dis_ar.append(d)
            
            dis_arr.append(dis_ar)
            it_arr.append([test_set['pl_name'].loc[i],test_set['pl_name'].loc[j]])
    
    return it_arr, dis_arr

