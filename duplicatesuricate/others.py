import numpy as np
import pandas as pd
from neatmartinet import acronym, compare_tokenized_strings


def check_column_same(a, b):
    import numpy as np
    if set(a) == set(b):
        return True
    else:
        common_set = np.intersect1d(a, b)
        missing_a_columns = list(filter(lambda x: x not in common_set, b))
        if len(missing_a_columns) > 0:
            print('unknown columns from', b.name, 'not in', a.name, ':', missing_a_columns)
        missing_b_columns = list(filter(lambda x: x not in common_set, a))
        if len(missing_b_columns) > 0:
            print('unknown columns from', a.name, 'not in', b.name, ':', missing_b_columns)
        return False


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


def find_missing_keys_in_index(keys, ref_list, verbose=True):
    """
    Takes as input the a list of keys, and check if they are present in a reference list
    For example, make sure that all of the keys are in the index before launching a loop
    Args:
        keys (iterable): list of keys to be checked
        ref_list (iterable): list of reference keys
        verbose (bool): whether or not to print the statements

    Returns:
        bool: If True, then keys are missing
    """
    incorrect_keys = list(filter(lambda x: x not in ref_list, keys))
    if len(incorrect_keys) > 0:
        if verbose:
            print('those keys are missing in the index:', incorrect_keys)
        return True
    else:
        return False