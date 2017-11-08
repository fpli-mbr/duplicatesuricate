# coding: utf-8
# Various comparison functions using fuzzywuzzy package (python levehstein)
# last checked with PyCharm

import numpy as np
import pandas as pd
from neatmartinet import compare_tokenized_strings,compare_twostrings,acronym,split,navalues,separatorlist
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
