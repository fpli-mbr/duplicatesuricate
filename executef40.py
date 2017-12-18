import pandas as pd
import neatmartinet as nm
from duplicatesuricate import Launcher,RecordLinker
from duplicatesuricate.evaluation import FuncEvaluationModel
import duplicatesuricate.preprocessing.companycleaning as cc


# # Load Data


filepath='C:/_PYDAI/1-Data/3-Deduplication P11-F40/'
filename_target='input_f40.xlsx'
filename_out='f40_doublons 20171207.xlsx'

df_target = pd.read_excel(filepath+filename_target)
df_target['companysysid']='F40'+df_target['LIFNR']
df_target.set_index('companysysid',inplace=True)

print(df_target.columns)


#%%
coldict={'duns':'KRAUS',
 'name':'NAME1',
 'street':'STRAS',
        'countrycode':'LAND1'}

namecol=coldict['name']
streetcol=coldict['street']

id_cols=['STCD1','STCD2','STCEG','KRAUS','VBUND']
id_scores=[c+'_exactscore' for c in id_cols]
formatasstr=id_cols+['PSTLZ']



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


#%%
# # Start Deduplication
# ## Define filtering and evaluation model

# ## Define  evaluation model


#define first filter
filter_all_any={'all':[coldict['countrycode']],'any':id_cols}
#Define intermediate threshold before further filtering
int_threshold={streetcol+'_wostopwords_fuzzyscore':0.6,
               namecol+'_wostopwords_fuzzyscore':0.6,
            streetcol+'_fuzzyscore':0.6,
            namecol+'_fuzzyscore':0.6,
               'aggfunc':'any'}

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



rl = RecordLinker(df=df_target,filterdict=filter_all_any,
                  intermediate_thresholds=int_threshold,evaluator=mymodel,decision_threshold=0.5)

#%%
results=pd.Series()
unexplored=rl.df.index
#%%
for ix in unexplored:
    if ix in unexplored:
        unexplored=unexplored.drop(ix)
        if len(unexplored)>1:
            query=rl.df.loc[ix]
            r = rl.return_good_matches(query,on_index=unexplored)
            if r is not None:
                print('results',r,type(r))
                results.loc[ix]=r[0]
                if r[0] in unexplored:
                    unexplored=unexplored.drop(r[0])
print('deduplication achieved')
displaycols = ['LIFNR', 'NAME1', 'STRAS', 'PSTLZ', 'ORT01', 'LAND1', 'KRAUS', 'STCD2',
               'STCD1', 'STCEG', 'VBUND']
if results.shape[0]>0:
    pd.DataFrame(results).to_excel(filepath+filename_out)
    Lch = Launcher(input_records=df_target, target_records=df_target, linker=rl)
    rsd = results.to_dict()
    assert isinstance(rsd, dict)
    df = Lch.format_results(res=rsd, fuzzy=[namecol, streetcol],
                            display=displaycols)
    df.to_excel(filepath + 'f40sidebyside.xlsx')
    df_target[list(set(rl.compared_cols + displaycols))].drop(index=results.index).to_excel(
        filepath + 'newf40.xlsx')
else:
    print('no duplicates')
    df_target[list(set(rl.compared_cols + displaycols))].to_excel(
    filepath + 'newf40.xlsx')


#%%
#y=rl.predict_proba(query)
#print(y.max())
#print(y.loc[y>rl.decision_threshold])

# finalmodel = Launcher(input_records=df_input,target_records=df_target,linker=rl,cleanfunc=None)
#
# matches=finalmodel.start_linkage()
#
#
# displaycols=[namecol]+[streetcol]+id_cols
# displayfuzzy=[namecol]+[streetcol]
#
#
# results=finalmodel.format_results(matches,display=displaycols,fuzzy=displayfuzzy)
#
#
#
## timeextract=pd.datetime.now().strftime('%Y%m%d-%H{}%M'.format('h'))
# filename_out=filename_out+timeextract+'.xlsx' timeextract=pd.datetime.now().strftime('%Y%m%d-%H{}%M'.format('h'))
# filename_out=filename_out+timeextract+'.xlsx'
#
# results.to_excel(filename_out,index=True)