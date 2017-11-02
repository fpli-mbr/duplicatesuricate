
# coding: utf-8
#library needed: geopy for the latlng distance calculations
## Various comparison functions using fuzzywuzzy package (python levehstein)

import numpy as np
import pandas as pd

#%%
def exactmatch(a,b):
    if pd.isnull(a) or pd.isnull(b):
        return None
    else:
        if str(a).isdigit() or str(b).isdigit():
            try:
                a=str(int(a))
                b=str(int(b))
            except:
                a=str(a)
                b=str(b)
        return int((a == b))

#%%
def geodistance(source,target,threshold=100):
    '''
    calculate the distance between the source and the target
    return a normalized the score up to a certain threshold
    the greater the distance the lower the score, until threshold =0
    :param source: latlng coordinate, in lat,lng string format
    :param target: latlng coordinate, in lat,lng string format
    :param threshold: int, maximum distance in km
    :return: score between 0 (distance greater than threshold) and 1 (distance =0)
    '''
    if pd.isnull(source) or pd.isnull(target):
        return None
    else:
        from geopy.distance import vincenty
        sourcelat=float(source.split(',')[0])
        sourcelng=float(source.split(',')[1])
        targetlat=float(target.split(',')[0])
        targetlng=float(target.split(',')[1])
        dist=vincenty((sourcelat,sourcelng),(targetlat,targetlng)).km
        if dist <threshold:
            #return a score between 1 (distance = 0) and 0 (distance = threshold)
            return (threshold-dist)/threshold
        else:
            return 0

def compare_acronyme(a,b,minaccrolength=3):
    """
    Retrouve des acronymes dans les données
    :param a: string
    :param b: string
    :param minaccrolength: int, default 3, minimum length of accronym
    :return: float, between 0 and 1
    """
    if pd.isnull(a) or pd.isnull(b):
        return None
    else:
        a_acronyme = acronym(a)
        b_acronyme = acronym(b)
        if min(len(a_acronyme),len(b_acronyme))>= minaccrolength:
            a_score_acronyme=compare_tokenized_strings(a_acronyme,b,mintokenlength=minaccrolength)
            b_score_acronyme=compare_tokenized_strings(b_acronyme,a,mintokenlength=minaccrolength)
            if any(pd.isnull([a_score_acronyme,b_score_acronyme])):
                return None
            else:
                max_score=np.max([a_score_acronyme,b_score_acronyme])
                return max_score
        else:
            return None

#%%
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
    myserie = pd.Series(myserie).astype(str).replace(nadict)
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
def compare_twostrings(a, b, minlength=3, threshold=0.0):
    """
    compare two strings using fuzzywuzzy.fuzz.ratio
    :param a: str, first string
    :param b: str, second string
    :param minlength: int, default 3, min  length for ratio
    :param threshold: float, default 0, threshold vor non-null value
    :return: float, number between 0 and 1
    """
    from fuzzywuzzy.fuzz import ratio
    if pd.isnull(a) or pd.isnull(b):
        return None
    elif min([len(a), len(b)]) < minlength:
        return 0
    else:
        r = ratio(a, b) / 100
        if r >= threshold:
            return r
        else:
            return 0.0


# %%
def compare_tokenized_strings(a, b, tokenthreshold=0.0, countthreshold=0.0, mintokenlength=3):
    """
    compare two strings by splitting them in tokens and comparing them tokens to tokens, using fuzzywuzzy.fuzz.ratio
    :param a: str, first string
    :param b: str, second string
    :param tokenthreshold: float, default 0.0, threshold for a match on a token
    :param countthreshold: float, default 0.0, threshold for a match on the two strings
    :param mintokenlength: int, default 0, 
    :return: float, number between 0 and 1
    """
    if pd.isnull(a) or pd.isnull(b):
        return None
    else:
        # exact match
        if a == b:
            return 1
        # split strings by tokens and calculate score on each token
        else:
            # split the string
            a_tokens = [s for s in a.split(' ') if len(s) >= mintokenlength]
            b_tokens = [s for s in b.split(' ') if len(s) >= mintokenlength]
            if len(a_tokens) == 0 or len(b_tokens) == 0:
                return None
            elif len(a_tokens) >= len(b_tokens):
                long_tokens = a_tokens
                short_tokens = b_tokens
            else:
                long_tokens = b_tokens
                short_tokens = a_tokens
            count = 0.0
            for t_short in short_tokens:
                if t_short in long_tokens:
                    count += 1
                else:
                    t_match_max = 0.0
                    for t_long in long_tokens:
                        t_match = compare_twostrings(t_short, t_long, threshold=tokenthreshold)
                        if t_match > t_match_max:
                            t_match_max = t_match
                    count += t_match_max

        percenttokensmatching = count / len(short_tokens)
        if percenttokensmatching >= countthreshold:
            return percenttokensmatching
        else:
            return 0.0


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
    :param myserie: pandas.Series, column to be evaluated
    :return: pandas.Series of float in [0,1]
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


# %%
def makeliststopwords(myserie, minlength=1, threshold=50, rmvwords=None, addwords=None, rmvdigits=True):
    """
    calculate the most common tokens found in a column
    :param myserie: pandas.Series, column to be evaluated
    :param minlength: int, default 1, min length of the token
    :param threshold: int, default 50, length of the first extract of stopwords, not counting words removed or words added
    :param rmvwords: list, default None, list of words to be removed from the list of stopwords
    :param addwords: list, default None, list of words to be added to the list of stopwords
    :param rmvdigits: boolean, default True, use digits in stopwords or not
    :return: list, list of stopwords
    """

    stopwords = word_count(myserie).index[:threshold].tolist()
    stopwords = [s for s in stopwords if len(s) >= minlength]
    if rmvdigits:
        stopwords = [s for s in stopwords if s.isdigit() == False]
    if rmvwords is not None:
        stopwords = [s for s in stopwords if not s in rmvwords]
    # noinspection PyAugmentAssignment
    if addwords is not None:
        stopwords += list(addwords)
    stopwords = list(set(stopwords))
    return stopwords

#%%
# author : paul ogier
# coding=utf-8
# Various string cleaning functions, not always orthodox
# Treatment of na values
# Various type conversion (int to str, str to date, split...)
# Various comparison functions using fuzzywuzzy package (python levehstein)


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
    myserie = pd.Series(myserie).astype(str).replace(nadict)
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
def compare_twostrings(a, b, minlength=3, threshold=0.0):
    """
    compare two strings using fuzzywuzzy.fuzz.ratio
    :param a: str, first string
    :param b: str, second string
    :param minlength: int, default 3, min  length for ratio
    :param threshold: float, default 0, threshold vor non-null value
    :return: float, number between 0 and 1
    """
    from fuzzywuzzy.fuzz import ratio
    if pd.isnull(a) or pd.isnull(b):
        return None
    elif min([len(a), len(b)]) < minlength:
        return 0
    else:
        r = ratio(a, b) / 100
        if r >= threshold:
            return r
        else:
            return 0.0


# %%
def compare_tokenized_strings(a, b, tokenthreshold=0.0, countthreshold=0.0, mintokenlength=3):
    """
    compare two strings by splitting them in tokens and comparing them tokens to tokens, using fuzzywuzzy.fuzz.ratio
    :param a: str, first string
    :param b: str, second string
    :param tokenthreshold: float, default 0.0, threshold for a match on a token
    :param countthreshold: float, default 0.0, threshold for a match on the two strings
    :param mintokenlength: int, default 0, 
    :return: float, number between 0 and 1
    """
    if pd.isnull(a) or pd.isnull(b):
        return None
    else:
        # exact match
        if a == b:
            return 1
        # split strings by tokens and calculate score on each token
        else:
            # split the string
            a_tokens = [s for s in a.split(' ') if len(s) >= mintokenlength]
            b_tokens = [s for s in b.split(' ') if len(s) >= mintokenlength]
            if len(a_tokens) == 0 or len(b_tokens) == 0:
                return None
            elif len(a_tokens) >= len(b_tokens):
                long_tokens = a_tokens
                short_tokens = b_tokens
            else:
                long_tokens = b_tokens
                short_tokens = a_tokens
            count = 0.0
            for t_short in short_tokens:
                if t_short in long_tokens:
                    count += 1
                else:
                    t_match_max = 0.0
                    for t_long in long_tokens:
                        t_match = compare_twostrings(t_short, t_long, threshold=tokenthreshold)
                        if t_match > t_match_max:
                            t_match_max = t_match
                    count += t_match_max

        percenttokensmatching = count / len(short_tokens)
        if percenttokensmatching >= countthreshold:
            return percenttokensmatching
        else:
            return 0.0


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
    :param myserie: pandas.Series, column to be evaluated
    :return: pandas.Series of float in [0,1]
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


# %%
def makeliststopwords(myserie, minlength=1, threshold=50, rmvwords=None, addwords=None, rmvdigits=True):
    """
    calculate the most common tokens found in a column
    :param myserie: pandas.Series, column to be evaluated
    :param minlength: int, default 1, min length of the token
    :param threshold: int, default 50, length of the first extract of stopwords, not counting words removed or words added
    :param rmvwords: list, default None, list of words to be removed from the list of stopwords
    :param addwords: list, default None, list of words to be added to the list of stopwords
    :param rmvdigits: boolean, default True, use digits in stopwords or not
    :return: list, list of stopwords
    """

    stopwords = word_count(myserie).index[:threshold].tolist()
    stopwords = [s for s in stopwords if len(s) >= minlength]
    if rmvdigits:
        stopwords = [s for s in stopwords if s.isdigit() == False]
    if rmvwords is not None:
        stopwords = [s for s in stopwords if not s in rmvwords]
    # noinspection PyAugmentAssignment
    if addwords is not None:
        stopwords += list(addwords)
    stopwords = list(set(stopwords))
    return stopwords
