# %%
# author : paul ogier
# coding=utf-8
# Various string cleaning functions, not always orthodox
# Treatment of na values
# Various type conversion (int to str, str to date, split...)
# Various comparison functions using fuzzywuzzy package (python levehstein)

import numpy as np
import pandas as pd

navalues = ['#', None, np.nan, 'None', '-', 'nan', 'n.a.', ' ', '', '#REF!', '#N/A', '#NAME?', '#DIV/0!', '#NUM!',
            'NaT']
nadict = {}
for c in navalues:
    nadict[c] = None

separatorlist = [' ', ',', '/', '-', ':', "'", '(', ')', '|', '°', '!', '\n', '_']
motavecS = ['après', 'français', 'francais', 'sous', 'plus', 'repas', 'souris']
accentdict = {'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
              'à': 'a', 'ä': 'a', 'â': 'a', 'á': 'a',
              'ü': 'u', 'ù': 'u', 'ú': 'u', 'û': 'u',
              'ö': 'o', 'ò': 'o', 'ó': 'o', 'ô': 'o', 'ß': 'ss', '@': '(at)'}
specialutf8 = {'\u2026': '', '\u017e': 'z', '\u2018': '', '\u0192': 'f', '\u2013': ' ', '\u0161': 's', '\u2021': '',
               '\u201d': '', '\u201c': '',
               '\u00E6': 'ae'}
removedchars = ['#', '~', '&', '$', '%', '*', '+', '?', '.']


# %%
def convert_int_to_str(n, fillwithzeroes=None):
    """

    :param n: number to be converted from int to str
    :param fillwithzeroes: dafualt None, length of number to be padded iwth zeroes, default = None(remove zeroes)
    :return: str
    """

    if pd.isnull(n) or n in navalues:
        return None
    else:
        n = str(n)
        n = n.lstrip().rstrip()
        n = n.lstrip('0')
        n = n.split('.')[0]
        if fillwithzeroes is not None:
            n = n.zfill(fillwithzeroes)
        return n


# %%
def split(mystring, seplist=separatorlist):
    """

    Args:
        mystring (str): string to be split
        seplist (list): by default separatorlist, list of separators

    Returns:
        list

    """
    if mystring in navalues:
        return None
    else:
        if seplist is None:
            seplist = separatorlist
        for sep in seplist:
            mystring = mystring.replace(sep, ' ')
        mystring = mystring.replace('  ', ' ')
        mylist = mystring.split(' ')
        mylist = list(filter(lambda x: x not in navalues, mylist))
        mylist = list(filter(lambda x: len(x) > 0, mylist))
        return mylist


# %%
def normalizeunicode(mystring):
    """
    :param mystring: str
    :return: str, normalized as unicode
    """
    if mystring in navalues:
        return None
    else:
        mystring = str(mystring)
        import unicodedata
        mystring = unicodedata.normalize('NFKD', mystring)
        mystring = mystring.encode('ASCII', 'ignore').decode('ASCII')
        return mystring


# %%
def normalizechars(mystring, removesep=False, encoding='utf-8', lower=True, min_length=1, remove_s=False):
    """
    :param mystring: str
    :param removesep: boolean default False, remove separator
    :param encoding: default 'utf-8'
    :param lower: boolean, default True, lowercase of strings
    :param min_length: int, default 1, min length of string to be kept (inclusive)
    :param remove_s: boolean, default False, remove s at the end of words
    :return: str
    """
    if mystring in navalues:
        return None
    else:
        mystring = str(mystring)
        if lower:
            mystring = mystring.lower()
        mystring = mystring.encode(encoding).decode(encoding)
        for a in removedchars:
            mystring = mystring.replace(a, '')
        if removesep is True:
            for a in separatorlist:
                mystring = mystring.replace(a, ' ')
        if encoding == 'utf-8':
            for mydict in [accentdict, specialutf8]:
                for a in mydict.keys():
                    mystring = mystring.replace(a, mydict[a])
        mystring = mystring.replace('  ', ' ')
        mystring = normalizeunicode(mystring)
        if mystring is None:
            return None
        mystring = mystring.encode(encoding).decode(encoding)
        mystring = mystring.replace('  ', ' ').lstrip().rstrip()
        if remove_s is True:
            if mystring not in motavecS:
                mystring = mystring.rstrip('s')
        if len(mystring) >= min_length:
            return mystring
        else:
            return None


# %%
def word_count(myserie):
    """
    counts the occurence of words in a panda.Series
    :param myserie: panda.Series containing string values
    :return: a panda.Series with words as index and occurences as data
    """
    import itertools
    myserie = myserie.replace(nadict).dropna().apply(split)
    return pd.Series(list(itertools.chain(*myserie))).value_counts(dropna=True)


# %%
def convert_str_to_date(myserie, datelim=None, dayfirst=True, sep=None):
    """
    convert string to date
    :param myserie: pandas.Series, column to be converted
    :param datelim: datetime, default Today, date that is superior to all dates in the Serie. Is used to check whether the conversion
            is successful or not.
    :param dayfirst: boolean, default True, in the event that the automatic cheks are not able to arbitrate between day and month column
            is used to nominally select the correct group of digits as day (in this case, the first)
    :param sep: string, default None  If not given  the function will look for '-' , '/' , '.'
    :return: pandas.Series

    The date must be in the first 10 digits of the string, the first part separated by a space
    '2016.10.11 13:04' --> ok
    to check whether the conversion is correct:
    the date max must be lower than the datelim
    the variance of the month shall be lower than the variance of the days column

    """

    from datetime import datetime
    # clean a bit the string
    myserie = pd.Series(myserie).astype(str)
    myserie=myserie.replace(nadict)
    # check datelim
    if datelim is None:
        datelim = pd.datetime.now()

    # try automatic conversion via pandas.to_datetime
    try:
        methodepandas = pd.to_datetime(myserie)
        if methodepandas.max() <= datelim and methodepandas.dt.month.std() < methodepandas.dt.day.std():
            return methodepandas
    except:
        pass
    # if not working, try the hard way
    y = pd.DataFrame(index=myserie.index, columns=['ChaineSource'])
    y['ChaineSource'] = myserie
    y.dropna(inplace=True)
    if y.shape[0] == 0:
        myserie = pd.to_datetime(np.nan)
        return myserie

    # find separator
    if sep is None:
        # look for the separator used in the first row of the serie
        extrait = str(y['ChaineSource'].iloc[0])

        extrait = extrait[:min(len(extrait.split(' ')[0]),
                               10)]  # on sélectionne les 10 premiers caractères ou les premiers séparés par un espace

        if '/' in extrait:
            sep = '/'
        elif '-' in extrait:
            sep = '-'
        elif '.' in extrait:
            sep = '.'
        else:
            print(myserie.name, ':sep not found,Extrait: ', y['ChaineSource'].dropna().iloc[0])

    # split the first 10 characters (or the first part of the string separted by a blankspace using the separator
    # The split is done is three columns
    y['ChaineTronquee'] = y['ChaineSource'].apply(lambda r: r.split(' ')[0][:min(len(r.split(' ')[0]), 10)])
    y['A'] = y['ChaineTronquee'].apply(lambda r: int(r.split(sep)[0]))
    y['B'] = y['ChaineTronquee'].apply(lambda r: int(r.split(sep)[1]))
    y['C'] = y['ChaineTronquee'].apply(lambda r: int(r.split(sep)[2]))
    localList = ['A', 'B', 'C']

    year = None
    for i in localList:
        if y[i].max() >= 1970:
            year = i
    if year is None:
        print(myserie.name, ':Year not found')
        myserie = pd.to_datetime(np.nan)
        return myserie
    localList.remove(year)

    day = None
    month = None

    i0 = localList[0]
    i1 = localList[1]
    # méthode par mois max
    if y[i0].max() > 12:
        month = i1
        day = i0
    elif y[i1].max() > 12:
        month = i0
        day = i1
    else:
        tempdayi0 = y.apply(lambda r: datetime(year=r[year], month=r[i1], day=r[i0]), axis=1)
        tempdayi1 = y.apply(lambda r: datetime(year=r[year], month=r[i0], day=r[i1]), axis=1)

        # méthode par datelimite
        if tempdayi0.max() > datelim:
            day = i1
            month = i0
        elif tempdayi1.max() > datelim:
            day = i0
            month = i1
        # méthode par variance:
        else:
            if tempdayi0.dt.day.std() > tempdayi0.dt.month.std():
                day = i0
                month = i1
            elif tempdayi1.dt.day.std() > tempdayi1.dt.month.std():
                day = i1
                month = i0
            # méthode par hypothèse:
            else:
                # Cas YYYY - MM -DD
                if year == 'A':
                    print(myserie.name, 'utilisation hypothèse,YYYY - MM -DD')
                    day = 'C'
                    month = 'B'
                # Cas DD - MM - YYYY
                elif year == 'C' and dayfirst:
                    print(myserie.name, 'utilisation hypothèse,DD - MM - YYYY')
                    day = 'A'
                    month = 'B'
                # Cas DD - MM - YYYY
                elif year == 'C' and dayfirst == False:
                    print(myserie.name, 'utilisation hypothèse,MM - DD - YYYY')
                    day = 'A'
                    month = 'B'
                # Cas DD - YYYY - MM ?
                elif year == 'B':
                    print(myserie.name, 'utilisation hypothèse,DD - YYYY - MM')
                    day = 'A'
                    month = 'C'

    y['return'] = y.apply(lambda r: datetime(year=r[year], month=r[month], day=r[day]), axis=1)
    y.loc[y['return'].dt.year == 1970, 'return'] = pd.to_datetime(np.nan)
    myserie.loc[myserie.index] = pd.to_datetime(np.nan)
    myserie.loc[y.index] = pd.to_datetime(y['return'])
    myserie = pd.to_datetime(myserie)
    return myserie


# %%
def rmv_end_str(w, s):
    """
    remove str at the end of tken
    :param w: str, token to be cleaned
    :param s: str, string to be removed
    :return: str
    """
    if w.endswith(s):
        w = w[:-len(s)]
    return w


def rmv_end_list(w, mylist):
    """
    removed string at the end of tok
    :param w: str, word to be cleaned
    :param mylist: list, ending words to be removed
    :return: str
    """
    if type(mylist) == list:
        mylist.sort(key=len)
        for s in mylist:
            w = rmv_end_str(w, s)
    return w


# %%
def replace_list(mylist, mydict):
    """
    replace values in a list
    :param mylist: list to be replaced
    :param mydict: dictionary of correct values
    :return: list
    """
    newlist = []
    for m in mylist:
        if m in mydict.keys():
            newlist.append(mydict[m])
        else:
            newlist.append(m)
    return newlist


# %%

def rmv_stopwords(myword, stopwords=None, endingwords=None, replacedict=None):
    """
    remove stopwords, ending words, replace words
    :param myword: str,word to be cleaned
    :param stopwords: list, default None, list of words to be removed
    :param endingwords: list, default None, list of words to be removed at the end of tokens
    :param replacedict: dict, default None, dict of words to be replaced
    :return: str, cleaned string
    """
    if pd.isnull(myword):
        return None
    elif len(myword) == 0:
        return None
    else:
        mylist = split(myword)

        mylist = [m for m in mylist if not m in stopwords]

        if endingwords is not None:
            newlist = []
            for m in mylist:
                newlist.append(rmv_end_list(m, endingwords))
            mylist = list(set(newlist)).copy()

        if replacedict is not None:
            mylist = list(set(replace_list(mylist, replacedict)))

        myword = ' '.join(mylist)
        myword = myword.replace('  ', ' ')
        myword = myword.lstrip().rstrip()

        if len(myword) == 0:
            return None
        else:
            return myword


# %%
def calculate_token_frequency(myserie):
    """
    calculate the frequency a token is used in a particular column
    :param myserie: pandas.Series, column to be evaluated
    :return: pandas.Series of float in [0,1]
    """
    wordlist = word_count(myserie)

    def countoccurences(r, wordlist):
        if pd.isnull(r):
            return None
        else:
            mylist = r.split(' ')
            count = 0
            for m in mylist:
                if m in wordlist.index:
                    count += wordlist.loc[m]
            return count

    x = myserie.apply(lambda r: countoccurences(r, wordlist=wordlist))
    x.fillna(x.max(), inplace=True)
    x = x / x.max()
    return x

def calculate_cat_frequency(myserie):
    """
    calculate the frequency a category is used in a particular column
    Args:
        myserie: pandas.Series, column to be evaluated
    Return:
        pandas.Series of float in [0,1]
    """
    catlist = myserie.value_counts()

    def countcat(r, catlist):
        if pd.isnull(r):
            return None
        else:
            if r in catlist.index:
                return catlist.loc[r]

    x = myserie.apply(lambda r: countcat(r, catlist=catlist))
    x.fillna(x.max(), inplace=True)
    x = x / x.max()
    return x

def acronym(s):
    """
    create an acronym from a string based on split function from this module
    :param s:string 
    :return: string, first letter of each token in the string
    """
    m = split(s)
    if m is None:
        return None
    else:
        a = ''.join([s[0] for s in m])
        return a

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


def clean_db(df, idcol, queryidcol):
    companystopwords = companystopwords_list
    streetstopwords = streetstopwords_list
    endingwords = endingwords_list

    df['Index'] = df.index

    # Create an alert if the index is not unique
    if df['Index'].unique().shape[0] != df.shape[0]:
        raise ('Error: index is not unique')

    # check if columns is in the existing database, other create a null one
    for c in [idcol, queryidcol, 'latlng', 'state']:
        if c not in df.columns:
            df[c] = None

    # normalize the strings
    for c in ['companyname', 'streetaddress', 'cityname']:
        df[c] = df[c].apply(normalizechars)

    # remove bad possible matches
    df.loc[df[idcol] == 0, idcol] = np.nan

    # convert all duns number as strings with 9 chars
    df['dunsnumber'] = df['dunsnumber'].apply(lambda r: convert_int_to_str(r, 9))

    def cleanduns(s):
        # remove bad duns like DE0000000
        if pd.isnull(s):
            return None
        else:
            s = str(s).rstrip('00000')
            if len(s) <= 5:
                return None
            else:
                return s

    df['dunsnumber'] = df['dunsnumber'].apply(cleanduns)

    # convert all postal codes to strings
    df['postalcode'] = df['postalcode'].apply(lambda r: convert_int_to_str(r))

    # convert all taxid and registerid to string
    for c in ['taxid', 'registerid']:
        if c in df.columns:
            df[c] = df[c].astype(str).replace(nadict)
        else:
            df[c] = None

    # remove stopwords from company names
    df['companyname_wostopwords'] = df['companyname'].apply(
        lambda r: rmv_stopwords(r, stopwords=companystopwords))

    # create acronyms of company names
    df['companyname_acronym'] = df['companyname'].apply(acronym)

    # remove stopwords from street addresses
    df['streetaddress_wostopwords'] = df['streetaddress'].apply(
        lambda r: rmv_stopwords(r, stopwords=streetstopwords, endingwords=endingwords))

    # Calculate word use frequency in company names
    df['companyname_wostopwords_wordfrequency'] = calculate_token_frequency(
        df['companyname_wostopwords'])

    # Take the first digits and the first two digits of the postal code
    df['postalcode_1stdigit'] = df['postalcode'].apply(
        lambda r: np.nan if pd.isnull(r) else str(r)[:1]
    )
    df['postalcode_2digits'] = df['postalcode'].apply(
        lambda r: np.nan if pd.isnull(r) else str(r)[:2]
    )

    # Calculate length of strings
    for c in ['companyname', 'companyname_wostopwords', 'streetaddress']:
        mycol = c + '_len'
        df[mycol] = df[c].apply(lambda r: None if pd.isnull(r) else len(r))
        max_length = df[mycol].max()
        df.loc[df[mycol].isnull() == False, mycol] = df.loc[df[
                                                                mycol].isnull() == False, mycol] / max_length

    # Calculate number of tokens in string
    for c in ['companyname_wostopwords']:
        mycol = c + '_ntokens'
        df[mycol] = df[c].apply(lambda r: None if pd.isnull(r) else len(r.split(' ')))
        max_value = df[mycol].max()
        df.loc[df[mycol].isnull() == False, mycol] = df.loc[df[
                                                                mycol].isnull() == False, mycol] / max_value

    # Calculate frequency of city used
    df['cityfrequency'] = calculate_cat_frequency(df['cityname'])

    # Define the list of big cities
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
    df['isbigcity'] = df['cityname'].isin(bigcities).astype(int)

    # Define the list of big companies
    for c in ['thales', 'zodiac', 'ge', 'safran']:
        df['has_' + c] = df['companyname_wostopwords'].apply(
            lambda r: 0 if pd.isnull(r) else c in r).astype(int)

    # Define the list of airbus names and equivalents
    airbus_names = ['airbus', 'casa', 'eads', 'cassidian', 'astrium', 'eurocopter']
    df['has_airbusequiv'] = df['companyname_wostopwords'].apply(
        lambda r: 0 if pd.isnull(r) else any(w in r for w in airbus_names)).astype(
        int)

    return None
