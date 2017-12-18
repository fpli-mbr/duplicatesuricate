import pandas as pd
import neatmartinet as nm
from duplicatesuricate import Launcher,RecordLinker
from duplicatesuricate.evaluation import FuncEvaluationModel
import duplicatesuricate.preprocessing.companycleaning as cc


# # Load Data


filepath='C:/_PYDAI/1-Data/3-Deduplication P11-F40/'
filename_input='input_F40.xlsx'
filename_target='p11_md2.csv'
filename_out='f40_to_p11_' #+timeextract

df_input = pd.read_excel(filepath+filename_input,index_col=0)
df_target=pd.read_csv(filepath+filename_target,index_col=0)



print(df_target.columns)
df_target.head()


query={'KRAUS': '537369803',
 'LAND1': 'DE',
 'LIFNR': 'ab',
 'NAME1': 'ComConsult',
 'ORT01': 'Aachen',
 'PSTLZ': '52076',
 'STCEG': 'DE154818876',
 'STRAS': 'Pascalstr.',
 'SYSID': 'df_target',
    'STCD1':None,
       'STCD2':None,
 'VBUND': None}
query=pd.Series(query,name='A1')
#df_input=pd.DataFrame(query).transpose()


# # Clean Data



coldict={'duns':'KRAUS',
 'name':'NAME1',
 'street':'STRAS',
        'countrycode':'LAND1'}
namecol=coldict['name']
streetcol=coldict['street']

id_cols=['STCD1','STCD2','STCEG','KRAUS','VBUND']
id_scores=[c+'_exactscore' for c in id_cols]
formatasstr=id_cols


def cleanfunc(df,coldict):
    for c in formatasstr:
        df[c]=df[c].apply(nm.format_int_to_str)
    c = coldict['duns']
    df[c]=df[c].apply(cc.cleanduns)
    c = namecol
    df[c]=df[c].apply(nm.format_ascii_lower)
    df[c+'_wostopwords']=df[c].apply(cc.rmv_companystopwords)
    #df[c+'_len']=df[c].apply(cc.name_len)
    c = streetcol
    df[c]=df[c].apply(nm.format_ascii_lower)
    df[c+'_wostopwords']=df[c].apply(cc.rmv_streetstopwords)
    return df


target_records=cleanfunc(df_target,coldict)


input_records=cleanfunc(df_input,coldict)


#%%
# # Start Deduplication



#define first filter
filter_all_any={'all':[coldict['countrycode']],'any':id_cols}
#Define intermediate threshold before further filtering
int_threshold={streetcol+'_wostopwords_fuzzyscore':0.5,namecol+'_wostopwords_fuzzyscore':0.5,'aggfunc':'any'}

#Define evaluation model, do not forget the used scores
streetscores=[streetcol+'_wostopwords_fuzzyscore']
namesscores=[namecol+'_wostopwords_fuzzyscore']

def evaluation_func(row):
    has_sameid=row[id_scores].max()
    has_samestreet=row[streetscores].mean()
    has_samename=row[namesscores].mean()
    if has_sameid==1:
        if has_samestreet>0.5:
            return max(has_samestreet,has_samename)
        else:
            return min(has_samestreet,has_samename)
    else:
        return 0
used_cols=streetscores+id_scores+namesscores
mymodel = FuncEvaluationModel(used_cols=used_cols,eval_func=evaluation_func)

#%%

rl = RecordLinker(df=df_target,filterdict=filter_all_any,
                  intermediate_thresholds=int_threshold,evaluator=mymodel,decision_threshold=0.5)

Lch = Launcher(input_records=df_input,target_records=df_target,linker=rl)
#%%
results=Lch.start_linkage(sample_size='all')

#%%
displaycols = ['LIFNR', 'NAME1', 'STRAS', 'PSTLZ', 'ORT01', 'LAND1', 'KRAUS', 'STCD2',
               'STCD1', 'STCEG', 'VBUND']
if len(results)>0:
    pd.DataFrame(pd.Series(results)).to_excel(filepath+'results.xlsx')
    df = Lch.format_results(res=results, fuzzy=[namecol, streetcol],
                            display=displaycols,ids=id_cols)
    df.to_excel(filepath + 'f40p11sidebyside.xlsx')
else:
    print('no duplicates')

