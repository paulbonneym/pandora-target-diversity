#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import os
import utils

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
target_list=pd.read_csv(f'{PACKAGEDIR}/pandora_targets.csv')
params=['st_rad','st_teff','st_rotp','pl_orbper','pl_dens']
sy_param_num=3 #number of system parameters, shoulf go before individual planet parameters in params


#For systems=True:
#at the end, the number to divide by to average for av_df should be:
    #(sy_param_num * (len(list(set(target_list['hostname'])))-1)) + ((len(params)-sy_param_num) * (len(target_list)-1))
    #         ^ this quantity is all of the system comparisons             ^ this number is all of the individual planet-planet comparisons

#for the ind_df, the first sy_param_num should be divided by len(list(set(target_list['hostname'])))-1 and the others by len(target_list)-1


def target_uniqueness(target_list=target_list, params=params, tun=2, parameter_diversity=False, systems=False):
    
    target_list=target_list.reset_index() #in case this is a slice of the long list. This does preserve the original indicies if needed
    
    wts=utils.create_weight(target_list,params,tun)
    it_arr,dis_arr=utils.calc_dis(target_list,wts,systems=systems)
    
    #set up dataframes to hold diversity scores to average
    av_df=pd.concat([target_list['hostname'],target_list['pl_name'],pd.DataFrame(np.zeros(len(target_list)).T, columns=['List Diversity']).astype(float)],axis=1)
    ind_df=pd.concat([target_list['hostname'],target_list['pl_name'],pd.DataFrame(np.zeros((len(target_list),len(params))), columns=[f'{p} Diversity' for p in params]).astype(float)],axis=1)
    
    if systems:
        
        for i in range(len(it_arr)):
            pls=it_arr[i][2]
            divs=dis_arr[i]
            
            host_mask=(av_df['hostname']==it_arr[i][0])+(av_df['hostname']==it_arr[i][1])
            planet_masks=[(av_df['pl_name']==pls[p][0])+(av_df['pl_name']==pls[p][1]) for p in range(len(pls))]
            
            #add system diversities into List Diversity
            av_df.loc[host_mask,('List Diversity')]+=np.sum(divs[:sy_param_num])
            #add planetary diversities into List Diversity
            for p in range(len(pls)):
                av_df.loc[planet_masks[p], ('List Diversity')]+=np.sum(divs[sy_param_num:][p*2:(p*2)+2])
                
            if parameter_diversity:
                #add system diversities into their individual diversities
                for j in range(sy_param_num):
                    ind_df.loc[host_mask,(f'{params[j]} Diversity')]+=divs[j]
                #add planetary diversities into their individual diversities
                for j in range(sy_param_num,len(params)):
                    for p in range(len(pls)):
                        ind_df.loc[planet_masks[p],(f'{params[j]} Diversity')]+=divs[j+(2*p)]
                        
        #divide by total number of diversity scores to get an average
        av_df['List Diversity']/=(sy_param_num * (len(list(set(target_list['hostname'])))-1)) + ((len(params)-sy_param_num) * (len(target_list)-1))
        if parameter_diversity:
            for j in range(sy_param_num):
                ind_df[f'{params[j]} Diversity']/=len(list(set(target_list['hostname'])))-1
            for j in range(sy_param_num,len(params)):
                ind_df[f'{params[j]} Diversity']/=len(target_list)-1
            
            return pd.concat([av_df,ind_df], axis=1)
        
        return av_df
    
    #Behavior if systems=False:
    for i in range(len(it_arr)):
        
        divs=dis_arr[i]
        
        host_mask=(av_df['hostname']==it_arr[i][0])+(av_df['hostname']==it_arr[i][1])
        
        #iteratively add the average diversity score for each pair of targets
        av_df.loc[host_mask,('List Diversity')]+=np.average(divs)
        
        #option to add each individual parameter diversity as well
        if parameter_diversity:
            for p in range(len(params)):
                ind_df.loc[host_mask,('params[p] Diversity')]+=divs[p]
            
    
    #divide by the number of combinations for each target, 1 less than the total amount in the list
    #this is the average diversity for the target within the list
    av_df['List Diversity']/=(len(target_list)-1)
    if parameter_diversity:
        for p in range(len(params)):
            ind_df['params[p] Diversity']/=(len(target_list)-1)
        
        return pd.concat([av_df,ind_df], axis=1)
    
    return av_df