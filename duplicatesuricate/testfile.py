import pandas as pd
import numpy as np

path_sample='/Users/paulogier/Documents/8-PythonProjects/1-AD/4-Schaufuss/'
source=pd.read_csv(path_sample+'f40_data.csv',index_col=0)
print(source.shape[0])
target=pd.read_csv(path_sample+'p11_data.csv',index_col=0,nrows=10000)
print(target.shape[0])

# %%
from duplicatesuricate.deduplication import Launcher
suricate=Launcher(input_records=source,target_records=target)
suricate.clean_records()

# %%
from duplicatesuricate.recordlinkage import RecordLinker
irl = RecordLinker(n_estimators=500)
irl.train()

suricate.add_model(model=irl)

# %%
# hp_ix=130r070
# hp_q=suricate.input_records.loc[hp_ix]
# results=irl.return_good_matches(query=hp_q,target_records=suricate.target_records)
# print(suricate.target_records.loc[results])
print(pd.datetime.now())
nmax=10
test_index=np.random.permutation(source.index)[:nmax]
results=suricate.link_input_to_target(in_index=test_index,n_matches_max=2)
results.to_csv(path_sample+'results_deduplication.csv',sep=',',encoding='utf-8')
print(pd.datetime.now())