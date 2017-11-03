# coding: utf-8
# Various comparison functions using fuzzywuzzy package (python levehstein)
# last checked with PyCharm

import numpy as np
import pandas as pd

# %%
def exactmatch(a, b):
    if pd.isnull(a) or pd.isnull(b):
        return None
    else:
        if str(a).isdigit() or str(b).isdigit():
            try:
                a = str(int(a))
                b = str(int(b))
            except:
                a = str(a)
                b = str(b)
        return int((a == b))


def compare_acronyme(a, b, minaccrolength=3):
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
        if min(len(a_acronyme), len(b_acronyme)) >= minaccrolength:
            a_score_acronyme = compare_tokenized_strings(a_acronyme, b, mintokenlength=minaccrolength)
            b_score_acronyme = compare_tokenized_strings(b_acronyme, a, mintokenlength=minaccrolength)
            if any(pd.isnull([a_score_acronyme, b_score_acronyme])):
                return None
            else:
                max_score = np.max([a_score_acronyme, b_score_acronyme])
                return max_score
        else:
            return None


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
def compare_tokenized_strings(a, b, tokenthreshold=0.8, countthreshold=0.0, mintokenlength=3):
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


separatorlist = [' ', ',', '/', '-', ':', "'", '(', ')', '|', '°', '!', '\n', '_']

navalues = ['#', None, np.nan, 'None', '-', 'nan', 'n.a.', ' ', '', '#REF!', '#N/A', '#NAME?', '#DIV/0!', '#NUM!',
            'NaT']


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
