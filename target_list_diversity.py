#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 10:40:48 2023

@author: paul
"""

import numpy as np
import pandas as pd
import itertools
import multiprocessing as mp

target_list=pd.read_csv('/mnt/ede2dab4-59b1-4f32-9008-20b08df035c8/Code (Laptop)/pandora_targets.csv')
params=['st_rad','st_teff','st_rotp','pl_orbper','pl_dens']
n=20

#create weights from the standard deviation of each of the parameters for the entire target list
#i.e., if two values are more than 2 std away from each other, they are sufficiently different
#tun is a tuning parameter in case we want to make this distance larger or smaller
def create_weight(target_list=target_list,params=params,tun=2):
    
    wts=[]
    
    for p in params:
        param_list=target_list[p]
    
        u=tun*np.exp(np.std(np.log(param_list)))
        p_med=np.exp(np.nanmedian(np.log(param_list)))
        
        wts.append(np.log(0.8)/np.log(np.abs(u/(p_med+p_med+u))))
    
    #Previous iteration below, retained in case it is needed/wanted
    
    # val1=target_list[f'{param}'][i]
    # val2=target_list[f'{param}'][j]
    
    # if np.isnan(val1):
    #     print(param)
    #     print(i)
    # if np.isnan(val2):
    #     print(param)
    #     print(j)
    
    # if val1 > val2:
    #     er1=np.abs(target_list[f'{param}err2'][i])
    #     if np.isnan(er1):
    #         er1=val1
    #     er2=np.abs(target_list[f'{param}err1'][j])
    #     if np.isnan(er2):
    #         er2=val2
    #     er=5*np.average([er1,er2])
    # else:
    #     er1=np.abs(target_list[f'{param}err1'][i])
    #     if np.isnan(er1):
    #         er1=val1
    #     er2=np.abs(target_list[f'{param}err2'][j])
    #     if np.isnan(er2):
    #         er2=val2
    #     er=5*np.average([er1,er2])
    
    # wt=np.log(0.8)/np.log(np.abs(er/(val1+val2)))
    
    return wts

wts=create_weight(target_list,params)

#MC picker based on log(tsm)
def get_list(n,target_list=target_list):
    
    tsm=target_list['TSM']
    log_tsm=np.log(tsm) #use log(tsm) instead because it varys more smoothly
    norm_log_tsm=(log_tsm-np.min(log_tsm))/(np.max(log_tsm)-np.min(log_tsm)) #Normalize log_tsm to [0,1] for pseudo-MCMC picker
    index_list=np.arange(len(tsm)).tolist() #make a list to pop later, avoids choosing the same target twice
    
    rng = np.random.default_rng()
    acc_set=rng.random(int(n)) #Random set of numbers for TSM acceptance
    
    target_set=[]
    
    for i in acc_set:
        
        
        high=len(np.array(index_list)[np.array(index_list) < len(norm_log_tsm[norm_log_tsm>i])])
        
        #this avoids a case where the 
        while high == 0:
            rng = np.random.default_rng()
            i = rng.random()
            high=len(np.array(index_list)[np.array(index_list) < len(norm_log_tsm[norm_log_tsm>i])])
        j=int(np.random.uniform(low=0,high=high))
        target_set.append(index_list.pop(j))
        
    return target_set




#dissimilarity function, weighted variation of the Manhattan Distance
dis = lambda val1, val2, wt: np.abs((val1-val2)/(val1+val2))**wt
        
#diversity calc
def calc_dis(target_set,wts=wts,target_list=target_list):
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


def mp_doit(t):
    
    test_set=get_list(t)
    test_dis=calc_dis(test_set)
    
    return [test_set, test_dis]

#n = how many tries for the algorithm
def short_list(n,t=20,target_list=target_list,params=params):
    
    p=mp.Pool(mp.cpu_count())
    res=p.map_async(mp_doit, np.ones(n)*t)
    results=[r for r in res.get()]
    diss=[r[1] for r in results]
    final=results[np.where(diss==np.max(diss))[0][0]]
    sl=target_list.loc[final[0]]
    
    return sl, final[1]



#Future development:
#how unique is each target within this list? within the overall list?
#system approach -> add the tsm up for each planet in the system = total tsm


