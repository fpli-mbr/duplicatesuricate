import pandas as pd
import numpy as np

path_sample='/Users/paulogier/Documents/8-PythonProjects/02-test_suricate/sampledata.xlsx'
source=pd.read_excel(path_sample,sheetname='F40',index_col=0)
print(source.shape[0])
target=pd.read_excel(path_sample,sheetname='P11',index_col=0)
print(target.shape[0])

# %%
from duplicatesuricate.deduplication import Launcher
suricate=Launcher(input_records=source,target_records=target)
suricate.clean_records()

# %%
from duplicatesuricate.recordlinkage import RecordLinker
irl = RecordLinker(n_estimators=1000)
irl.train()

#suricate.add_model(model=irl)

# %%
# hp_ix=130r070
# hp_q=suricate.input_records.loc[hp_ix]
# results=irl.return_good_matches(query=hp_q,target_records=suricate.target_records)
# print(suricate.target_records.loc[results])

#results=suricate.link_input_to_target()