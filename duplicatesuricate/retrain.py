import pandas as pd
import numpy as np
from itertools import combinations

def paired_from_gid(df,on_col):
    """
    This function, which takes as input a dataframe and the name of the column of possible matches
    Returns a dataframe with the list of possible combinations
    """
    #group by on on_col
    gb=df.groupby(on_col)
    #create a list out of the groups index
    groups=pd.Series(gb.groups)
    groups=groups.apply(lambda r:list(r))
    # from these list create all the paired combination
    groups=groups.apply(lambda r:list(combinations(r,2)))
    # transform into a DataFrame
    groups=groups.apply(pd.Series)
    groups = groups.rename(columns = lambda x : 'pair_' + str(x))
    groups=pd.DataFrame(groups)
    # copy the index
    groups[on_col]=groups.index
    # Melt the dataframe to get all the pairs in the same column
    groups=pd.melt(groups,value_name='pair',id_vars=[on_col],var_name='pairnumber')
    groups=groups.drop(['pairnumber','gid'],axis=1)
    # transform the pair into two columns
    x=groups['pair'].apply(lambda r:pd.Series(r).sort_values(ascending=True))
    x.columns=['ix_source','ix_target']
    groups=groups.join(x,how='left')
    groups=groups.dropna(subset=['pair'])
    groups=groups.drop(['pair'],axis=1)
    return groups

def appair(ix1,ix2):
    """
    This function creates a unique pair id from two ids (not sensitive to the order)
    """
    m=[ix1,ix2]
    m=sorted(m)
    m='_'.join([str(c) for c in m])
    return m

def unique_pairs(y_source,y_target,y_true=None):
    y1=np.array(y_source)
    y2=np.array(y_target)
    df=pd.DataFrame(y1)
    df.columns=['ix_source']
    df['ix_target']=y2
    df=df.loc[df['ix_source']!=df['ix_target']]
    df['pair']=df.apply(lambda r:appair(r['ix_source'],r['ix_target']),axis=1)
    if y_true is not None:
        df['y_true']=y_true
    df.drop_duplicates(subset=['pair'],inplace=True)
    df.drop(['pair'],axis=1,inplace=True)
    return df
