# pandora-target-diversity
Tools to ensure the target list is diverse. The main usage of this package is to narrow down a list of potential targets into a short list based on observability (transmission spectroscopy metric) and diversity in the parameter space. 

### Example Usage

Below is an example of generating a short list using 10000 iterations:

'''python
from pandora-target-diversity import short_list
sl, div = short_list(10000)
'''

If you want to create a short list of a different length (20 by default), use the t argument:

'''python
from pandora-target-diversity import short_list
sl_longer, div_longer = short_list(10000,t=30)
'''

The diversity score can also be tuned to change the acceptable level of diversity, which is 2 standard deviations by default:

'''python
from pandora-target-diversity import short_list

#Create a list with less strict acceptable diversity
sl_less, div_less = short_list(10000,tun=1.5)

#Create a list with more strict acceptable diversity
sl_more, div_more = short_list(10000,tun=2.5)
'''

### Changing Considered Parameters

By default the package calculates diversity in stellar radius ('st_rad'), stellar effective temperature ('st_teff'), stellar rotation period ('st_rotp'), planetary orbital period ('pl_orbper'), and planetary density ('pl_dens'). If you wish to consider different parameters, please fork the repository with an appropriate name to your project as some code needs to be changed within utils.py. As an example, consider removing stellar effective temperature and planetary orbital period and adding planetary eqilibrium temperature ('pl_eqt'). As the diversity calculations are hard coded for each parameter set for speed considerations, we now need to change the main body of the calc_dis function as well as specify a new params variable. Assume for the example that params now looks like:

'''python
params_new=['st_rad','st_teff','pl_dens','pl_eqt']
'''


The default code looks like so:

'''python
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
'''

Weights calculations are independent of the parameters used, but the order matters for this function. Thus, the code would need to be changed to:

'''python
dis_arr=[]
    
#get parameters from dataframe
st_rad=target_list['st_rad'][target_set]
st_teff=target_list['st_teff'][target_set]
pl_dens=target_list['pl_dens'][target_set]
pl_eqt=target_list['pl_eqt'][target_set]

#create weights
st_rad_wt=wts[0]
st_teff_wt=wts[1]
pl_dens_wt=wts[2]
pl_eqt_wt=wts[3]

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
    d=dis(pl_dens[i],pl_dens[j],pl_dens_wt)
    if np.isnan(d):
        d=0
    dis_arr.append(d)
    d=dis(pl_eqt[i],pl_eqt[j],pl_eqt_wt)
    if np.isnan(d):
        d=0
    dis_arr.append(d)
'''

The main function can now be called like so (using any other keyword changed you wish):

'''python
from pandora-target-diversity import short_list
params_new=['st_rad','st_teff','pl_dens','pl_eqt']
sl_new, div_new = short_list(10000, params=params_new)
'''
