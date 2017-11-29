# %%
# author : amber ocelot
# coding=utf-8
import neatmartinet as nm
import numpy as np
import pandas as pd

import duplicatesuricate.preprocessing.companydata

companystopwords = duplicatesuricate.preprocessing.companydata.companystopwords_list
streetstopwords = duplicatesuricate.preprocessing.companydata.streetstopwords_list
endingwords = duplicatesuricate.preprocessing.companydata.endingwords_list
bigcities = duplicatesuricate.preprocessing.companydata.bigcities
airbusnames = duplicatesuricate.preprocessing.companydata.airbus_names


# convert all duns number as strings with 9 chars
def cleanduns(s):
    # remove bad duns like DE0000000
    s = nm.format_int_to_str(s, zeropadding=9)
    if pd.isnull(s):
        return None
    else:
        s = str(s).rstrip('00000')
        if len(s) <= 5 or s[:3] == 'NDM':
            return None
        else:
            return s

id_cols=['registerid', 'registerid1', 'registerid2', 'taxid', 'kapisid']
cleandict = {
    'dunsnumber': cleanduns,
    'name': nm.format_ascii_lower,
    'street': nm.format_ascii_lower,
    'city': nm.format_ascii_lower,
    'name_wostopwords': (lambda r: nm.rmv_stopwords(r, stopwords=companystopwords), 'name'),
    'street_wostopwords': (lambda r: nm.rmv_stopwords(r, stopwords=streetstopwords), 'street'),
    'name_acronym': (lambda r: nm.acronym(r), 'name'),
    'postalcode': nm.format_int_to_str,
    'postalcode_1stdigit': (lambda r: None if pd.isnull(r) else str(r)[:1], 'postalcode'),
    'postalcode_2digits': (lambda r: None if pd.isnull(r) else str(r)[:2], 'postalcode'),
    'name_len': (lambda r: len(r), 'name'),
    'hasairbusname': (lambda r: 0 if pd.isnull(r) else int(any(w in r for w in airbusnames)), 'name'),
    'isbigcity': (lambda r: 0 if pd.isnull(r) else int(any(w in r for w in bigcities)), 'city')

}

for c in id_cols:
    cleandict[c] = nm.format_int_to_str


def clean_db(df, cleandict=cleandict):
    companystopwords = duplicatesuricate.preprocessing.companydata.companystopwords_list
    streetstopwords = duplicatesuricate.preprocessing.companydata.streetstopwords_list
    endingwords = duplicatesuricate.preprocessing.companydata.endingwords_list

    # Create an alert if the index is not unique
    if pd.Series(df.index).unique().shape[0] != df.shape[0]:
        raise KeyError('Error: index is not unique')

    # # check if columns is in the existing database, other create a null one
    # for c in [duplicatesuricate.preprocessing.companydata.idcol,
    #           duplicatesuricate.preprocessing.companydata.queryidcol]:
    #     if c not in df.columns:
    #         df[c] = None

    for k in cleandict.keys():
        newcol = k
        if type(cleandict[k])==tuple:
            oncol=cleandict[k][1]
            myfunc=cleandict[k][0]
        else:
            oncol=k
            myfunc=cleandict[k]
        df[newcol]=df[oncol].apply(myfunc)

    return df
