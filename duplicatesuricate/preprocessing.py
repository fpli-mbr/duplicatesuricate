import neatmartinet as nm
import pandas as pd

companystopwords_list = ['aerospace',
                         'ag',
                         'and',
                         'co',
                         'company',
                         'consulting',
                         'corporation',
                         'de',
                         'deutschland',
                         'dr',
                         'electronics',
                         'engineering',
                         'europe',
                         'formation',
                         'france',
                         'gmbh',
                         'group',
                         'hotel',
                         'inc',
                         'ingenierie',
                         'international',
                         'kg',
                         'la',
                         'limited',
                         'llc',
                         'ltd',
                         'ltda',
                         'management',
                         'of',
                         'oy',
                         'partners',
                         'restaurant',
                         'sa',
                         'sarl',
                         'sas',
                         'service',
                         'services',
                         'sl',
                         'software',
                         'solutions',
                         'srl',
                         'systems',
                         'technologies',
                         'technology',
                         'the',
                         'uk',
                         'und']
streetstopwords_list = ['avenue', 'calle', 'road', 'rue', 'str', 'strasse', 'strae']
endingwords_list = ['strasse', 'str', 'strae']
bigcities = ['munich',
             'paris',
             'madrid',
             'hamburg',
             'toulouse',
             'berlin',
             'bremen',
             'london',
             'ulm',
             'stuttgart', 'blagnac']
airbus_names = ['airbus', 'casa', 'eads', 'cassidian', 'astrium', 'eurocopter']
idcol = 'groupid'
queryidcol='queryname'
companystopwords = companystopwords_list
streetstopwords = streetstopwords_list
endingwords = endingwords_list


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


def format_id(n):
    """
    Format an id to a str, removing separators like [, . / - ], leading zeros
    Args:
        n: id to be formatted

    Returns:
        str
    """

    if n is None:
        return None
    else:
        n = str(n)
        n = n.lstrip('0')
        for sep in ['-', '.', ' ', '/']:
            n = n.replace(sep, '')
        if len(n) == 0:
            return None
        else:
            return n


rmv_companystopwords = lambda r: nm.rmv_stopwords(r, stopwords=companystopwords)
rmv_streetstopwords = lambda r: nm.rmv_stopwords(r, stopwords=streetstopwords, endingwords=endingwords)
extract_postalcode_1digit = lambda r: None if pd.isnull(r) else str(r)[:1]
extract_postalcode_2digits = lambda r: None if pd.isnull(r) else str(r)[:2]
hasairbusname = lambda r: None if pd.isnull(r) else int(any(w in r for w in airbus_names))
isbigcity = lambda r: None if pd.isnull(r) else int(any(w in r for w in bigcities))
name_len = lambda r: None if pd.isnull(r) else len(r)
id_cols = ['registerid', 'registerid1', 'registerid2', 'taxid', 'kapisid']
cleandict = {
    'duns': cleanduns,
    'name': nm.format_ascii_lower,
    'street': nm.format_ascii_lower,
    'city': nm.format_ascii_lower,
    'name_wostopwords': (lambda r: nm.rmv_stopwords(r, stopwords=companystopwords), 'name'),
    'street_wostopwords': (lambda r: nm.rmv_stopwords(r, stopwords=streetstopwords, endingwords=endingwords), 'street'),
    'name_acronym': (lambda r: nm.acronym(r), 'name'),
    'postalcode': nm.format_int_to_str,
    'postalcode_1stdigit': (lambda r: None if pd.isnull(r) else str(r)[:1], 'postalcode'),
    'postalcode_2digits': (lambda r: None if pd.isnull(r) else str(r)[:2], 'postalcode'),
    'name_len': (lambda r: len(r), 'name'),
    'hasairbusname': (lambda r: 0 if pd.isnull(r) else int(any(w in r for w in airbus_names)), 'name'),
    'isbigcity': (lambda r: 0 if pd.isnull(r) else int(any(w in r for w in bigcities)), 'city')

}


def clean_db(df, cleandict=cleandict):
    companystopwords = companystopwords_list
    streetstopwords = streetstopwords_list
    endingwords = endingwords_list

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
        if type(cleandict[k]) == tuple:
            oncol = cleandict[k][1]
            myfunc = cleandict[k][0]
        else:
            oncol = k
            myfunc = cleandict[k]
        df[newcol] = df[oncol].apply(myfunc)

    return df