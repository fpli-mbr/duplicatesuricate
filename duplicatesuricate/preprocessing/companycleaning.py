# %%
# author : amber ocelot
# coding=utf-8
import neatmartinet as nm
import numpy as np
import pandas as pd

import duplicatesuricate.preprocessing.companydata


def clean_db(df,on_cols=None):
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

    # normalize the strings
    for c in np.intersect1d(on_cols,['companyname', 'streetaddress', 'cityname']):
        df[c] = df[c].apply(nm.format_ascii_lower)

    # convert all duns number as strings with 9 chars
    def cleanduns(s):
        # remove bad duns like DE0000000
        s = nm.format_int_to_str(s,zeropadding=9)
        if pd.isnull(s):
            return None
        else:
            s = str(s).rstrip('00000')
            if len(s) <= 5 or s[:3]=='NDM':
                return None
            else:
                return s
    if 'dunsnumber' in on_cols:
        df['dunsnumber'] = df['dunsnumber'].apply(cleanduns)

    # convert all postal codes to strings
    if 'postalcode' in on_cols:
        df['postalcode'] = df['postalcode'].apply(lambda r: nm.format_int_to_str(r))

    # convert all taxid and registerid to string
    for c in np.intersect1d(on_cols,['taxid', 'registerid']):
        if c in df.columns:
            df[c] = df[c].astype(str).replace(nm.nadict)
        else:
            df[c] = None

    # remove stopwords from company names
    if 'companyname_wostopwords' in on_cols:
        df['companyname_wostopwords'] = df['companyname'].apply(
            lambda r: nm.rmv_stopwords(r, stopwords=companystopwords))

    if 'companyname_acronym' in on_cols:
        # create acronyms of company names
        df['companyname_acronym'] = df['companyname'].apply(nm.acronym)

    # remove stopwords from street addresses
    df['streetaddress_wostopwords'] = df['streetaddress'].apply(
        lambda r: nm.rmv_stopwords(r, stopwords=streetstopwords, endingwords=endingwords))

    # Take the first digits and the first two digits of the postal code
    df['postalcode_1stdigit'] = df['postalcode'].apply(
        lambda r: np.nan if pd.isnull(r) else str(r)[:1]
    )
    df['postalcode_2digits'] = df['postalcode'].apply(
        lambda r: np.nan if pd.isnull(r) else str(r)[:2]
    )

    # Calculate length of strings
    for c in ['companyname']:
        mycol = c + '_len'
        df[mycol] = df[c].apply(lambda r: None if pd.isnull(r) else len(r))
        max_length = df[mycol].max()
        df.loc[df[mycol].isnull() == False, mycol] = df.loc[df[
                                                                mycol].isnull() == False, mycol] / max_length


    # Calculate frequency of city used
    df['cityfrequency'] = nm.calculate_cat_frequency(df['cityname'])

    # Define the list of big cities
    df['isbigcity'] = df['cityname'].isin(duplicatesuricate.preprocessing.companydata.bigcities).astype(int)

    # Define the list of airbus names and equivalents

    df['has_airbusequiv'] = df['companyname_wostopwords'].apply(
        lambda r: 0 if pd.isnull(r) else any(w in r for w in
                                             duplicatesuricate.preprocessing.companydata.airbus_names)).astype(
        int)

    return df
