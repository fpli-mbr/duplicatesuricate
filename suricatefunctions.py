
# coding: utf-8
import numpy as np
import pandas as pd
import neatcleanstring as ncs
#library needed: geopy for the latlng distance calculations
#%%
def compare_values(source,target,colname,threshold=0.0):
    if colname == 'latlng':
        return geodistance(source,target)
    elif colname in ['streetaddress_wostopwords','companyname_wostopwords','cityname']:
        return ncs.fuzzyscore(source,target,tokenthreshold=0.7,countthreshold=0.5,mintokenlength=5)
    else:
        return ncs.compare_twostrings(source,target,minlength=0,threshold=0.8)

#%%
def exactmatch(a,b):
    if pd.isnull(a) or pd.isnull(b):
        return None
    else:
        if str(a).isdigit() or str(b).isdigit():
            try:
                a=str(int(a))
                b=str(int(b))
            except:
                a=str(a)
                b=str(b)
        return int((a == b))

#%%
def geodistance(source,target,threshold=100):
    '''
    calculate the distance between the source and the target
    return a normalized the score up to a certain threshold
    the greater the distance the lower the score, until threshold =0
    :param source: latlng coordinate, in lat,lng string format
    :param target: latlng coordinate, in lat,lng string format
    :param threshold: int, maximum distance in km
    :return: score between 0 (distance greater than threshold) and 1 (distance =0)
    '''
    if pd.isnull(source) or pd.isnull(target):
        return None
    else:
        from geopy.distance import vincenty
        sourcelat=float(source.split(',')[0])
        sourcelng=float(source.split(',')[1])
        targetlat=float(target.split(',')[0])
        targetlng=float(target.split(',')[1])
        dist=vincenty((sourcelat,sourcelng),(targetlat,targetlng)).km
        if dist <threshold:
            #return a score between 1 (distance = 0) and 0 (distance = threshold)
            return (threshold-dist)/threshold
        else:
            return 0

def compare_acronyme(a,b,minaccrolength=3):
    """
    Retrouve des acronymes dans les données
    :param a: string
    :param b: string
    :param minaccrolength: int, default 3, minimum length of accronym
    :return: float, between 0 and 1
    """
    if pd.isnull(a) or pd.isnull(b):
        return None
    else:
        a_acronyme = ncs.acronym(a)
        b_acronyme = ncs.acronym(b)
        if min(len(a_acronyme),len(b_acronyme))>= minaccrolength:
            a_score_acronyme=ncs.compare_tokenized_strings(a_acronyme,b,mintokenlength=minaccrolength)
            b_score_acronyme=ncs.compare_tokenized_strings(b_acronyme,a,mintokenlength=minaccrolength)
            if any(pd.isnull([a_score_acronyme,b_score_acronyme])):
                return None
            else:
                max_score=np.max([a_score_acronyme,b_score_acronyme])
                return max_score
        else:
            return None

#%%
identitycols=['companyname_wostopwords','dunsnumber']
locationcols=['cityname','postalcode','streetaddress_wostopwords']
        