# %%
# author : amber ocelot
# coding=utf-8

import numpy as np
import pandas as pd
import neatmartinet as nm
from duplicatesuricate import config



def clean_db(df):
    companystopwords = config.companystopwords_list
    streetstopwords = config.streetstopwords_list
    endingwords = config.endingwords_list

    df['Index'] = df.index

    # Create an alert if the index is not unique
    if df['Index'].unique().shape[0] != df.shape[0]:
        raise ('Error: index is not unique')

    # check if columns is in the existing database, other create a null one
    for c in [config.idcol, config.queryidcol, 'latlng', 'state']:
        if c not in df.columns:
            df[c] = None

    # normalize the strings
    for c in ['companyname', 'streetaddress', 'cityname']:
        df[c] = df[c].apply(nm.normalizechars)

    # remove bad possible matches
    df.loc[df[config.idcol] == 0, config.idcol] = np.nan

    # convert all duns number as strings with 9 chars
    df['dunsnumber'] = df['dunsnumber'].apply(lambda r: nm.convert_int_to_str(r, 9))

    def cleanduns(s):
        # remove bad duns like DE0000000
        if pd.isnull(s):
            return None
        else:
            s = str(s).rstrip('00000')
            if len(s) <= 5 or s[:3]=='NDM':
                return None
            else:
                return s

    df['dunsnumber'] = df['dunsnumber'].apply(cleanduns)

    # convert all postal codes to strings
    df['postalcode'] = df['postalcode'].apply(lambda r: nm.convert_int_to_str(r))

    # convert all taxid and registerid to string
    for c in ['taxid', 'registerid']:
        if c in df.columns:
            df[c] = df[c].astype(str).replace(nm.nadict)
        else:
            df[c] = None

    # remove stopwords from company names
    # df['companyname_wostopwords'] = df['companyname'].apply(
    #     lambda r: nm.rmv_stopwords(r, stopwords=companystopwords))

    # create acronyms of company names
    # df['companyname_acronym'] = df['companyname'].apply(nm.acronym)

    # remove stopwords from street addresses
    # df['streetaddress_wostopwords'] = df['streetaddress'].apply(
    #     lambda r: nm.rmv_stopwords(r, stopwords=streetstopwords, endingwords=endingwords))

    # Calculate word use frequency in company names
    # df['companyname_wostopwords_wordfrequency'] = nm.calculate_token_frequency(
    #     df['companyname_wostopwords'])

    # Take the first digits and the first two digits of the postal code
    # df['postalcode_1stdigit'] = df['postalcode'].apply(
    #     lambda r: np.nan if pd.isnull(r) else str(r)[:1]
    # )
    # df['postalcode_2digits'] = df['postalcode'].apply(
    #     lambda r: np.nan if pd.isnull(r) else str(r)[:2]
    # )
    #
    # # Calculate length of strings
    # for c in ['companyname', 'companyname_wostopwords', 'streetaddress']:
    #     mycol = c + '_len'
    #     df[mycol] = df[c].apply(lambda r: None if pd.isnull(r) else len(r))
    #     max_length = df[mycol].max()
    #     df.loc[df[mycol].isnull() == False, mycol] = df.loc[df[
    #                                                             mycol].isnull() == False, mycol] / max_length
    #
    # # Calculate number of tokens in string
    # for c in ['companyname_wostopwords']:
    #     mycol = c + '_ntokens'
    #     df[mycol] = df[c].apply(lambda r: None if pd.isnull(r) else len(r.split(' ')))
    #     max_value = df[mycol].max()
    #     df.loc[df[mycol].isnull() == False, mycol] = df.loc[df[
    #                                                             mycol].isnull() == False, mycol] / max_value
    #
    # # Calculate frequency of city used
    # df['cityfrequency'] = nm.calculate_cat_frequency(df['cityname'])
    #
    # # Define the list of big cities
    # df['isbigcity'] = df['cityname'].isin(config.bigcities).astype(int)
    #
    # # Define the list of airbus names and equivalents
    #
    # df['has_airbusequiv'] = df['companyname_wostopwords'].apply(
    #     lambda r: 0 if pd.isnull(r) else any(w in r for w in config.airbus_names)).astype(
    #     int)

    return None
