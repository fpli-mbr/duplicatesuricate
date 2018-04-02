import numpy as np
import pandas as pd
from fuzzywuzzy.fuzz import ratio, token_set_ratio
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType

class ScoreDict(dict):
    def _unpack(self):
        outputcols = []
        inputcols = []

        for d in ['all', 'any']:
            if self.get(d) is not None:
                for c in self[d]:
                    inputcols.append(c)
                    outputcols.append(c + '_exactscore')
        if self.get('attributes') is not None:
            for c in self['attributes']:
                inputcols.append(c)
                outputcols.append(c + '_source')
                outputcols.append(c + '_target')
        for k in _scorename.keys():
            if self.get(k) is not None:
                for c in self[k]:
                    inputcols.append(c)
                    outputcols.append(c + _scorename[k])
        return inputcols, outputcols
    def compared(self):
        compared_cols = set(self._unpack()[0])
        return compared_cols
    def scores(self):
        score_cols = set(self._unpack()[1])
        return score_cols

def _convert_fuzzratio(x):
    """
    convert a ratio between 0 and 100 to a ratio between 1 and -1
    Args:
        x (float):

    Returns:
        float
    """
    score = x / 50 - 1
    return score


def _fuzzyscore(a, b):
    """
    fuzzyscore using fuzzywuzzy.ratio
    Args:
        a (str):
        b (str):

    Returns:
        float score between -1 and 1
    """
    if pd.isnull(a) or pd.isnull(b):
        return 0.0
    else:
        score = _convert_fuzzratio(ratio(a, b))
        return score


_fuzzy_udf = udf(lambda a, b: _fuzzyscore(a, b), FloatType())


def _tokenscore(a, b):
    """
    fuzzyscore using fuzzywuzzy.token_set_ratio
    Args:
        a (str):
        b (str):

    Returns:
        float score between -1 and 1
    """
    if pd.isnull(a) or pd.isnull(b):
        return 0.0
    else:
        score = _convert_fuzzratio(token_set_ratio(a, b))
        return score


_token_udf = udf(lambda a, b: _tokenscore(a, b), FloatType())


def _exactmatch(a, b):
    if pd.isnull(a) or pd.isnull(b):
        return 0.0
    else:
        if a == b:
            return 1.0
        else:
            return -1.0


_exact_udf = udf(lambda a, b: _exactmatch(a, b), FloatType())


def _acronym(s):
    """
    make an acronym of the string: take the first line of each token
    Args:
        s (str):

    Returns:
        str
    """
    m = s.split(' ')
    if m is None:
        return None
    else:
        a = ''.join([s[0] for s in m])
        return a


def _compare_acronym(a, b, minaccrolength=3):
    """
    compare the acronym of two strings
    Args:
        a (str):
        b (str):
        minaccrolength (int): minimum length of accronym

    Returns:
        float : number between 0 and 1
    """
    if pd.isnull(a) or pd.isnull(b):
        return 0.0
    else:
        a_acronyme = _acronym(a)
        b_acronyme = _acronym(b)
        if min(len(a_acronyme), len(b_acronyme)) >= minaccrolength:
            a_score_acronyme = _tokenscore(a_acronyme, b)
            b_score_acronyme = _tokenscore(a, b_acronyme)
            if all(pd.isnull([a_score_acronyme, b_score_acronyme])):
                return 0.0
            else:
                max_score = np.max([a_score_acronyme, b_score_acronyme])
                return max_score
        else:
            return 0.0


_acronym_udf = udf(lambda a, b: _compare_acronym(a, b), FloatType())
_scorename = {'fuzzy': '_fuzzyscore',
              'token': '_tokenscore',
              'exact': '_exactscore',
              'acronym': '_acronymscore'}
_scorefuncs = {'fuzzy': _fuzzyscore,
               'token': _tokenscore,
               'exact': _exactmatch,
               'acronym': _compare_acronym}
_scoringkeys = list(_scorename.keys())


def _rmv_end_str(w, s):
    """
    remove str at the end of tken
    :param w: str, token to be cleaned
    :param s: str, string to be removed
    :return: str
    """
    if w.endswith(s):
        w = w[:-len(s)]
    return w

def build_similarity_table(query, targets,  scoredict=None):
        """
        Return the similarity features between the query and the rows in the required index, with the selected comparison functions.
        They can be fuzzy, token-based, exact, or acronym.
        The attribute request creates two column: one with the value for the query and one with the value for the row

        Args:
            query (pd.Series): attributes of the query
            targets (pd.DataFrame):
            scoredict (dict):

        Returns:
            pd.DataFrame:

        Examples:
            scoredict={'attributes':['name_len'],
                        'fuzzy':['name','street']
                        'token':'name',
                        'exact':'id'
                        'acronym':'name'}
            returns a comparison table with the following column names (and the associated scores):
                ['name_len_query','name_len_row','name_fuzzyscore','street_fuzzyscore',
                'name_tokenscore','id_exactscore','name_acronymscore']
        """
        if scoredict is None:
            return None
        on_index = targets.index
        table_score = pd.DataFrame(index=on_index)

        attributes_cols = scoredict.get('attributes')
        if attributes_cols is not None:
            for c in attributes_cols:
                table_score[c + '_source'] = query[c]
                table_score[c + '_target'] = targets[c]

        for c in _scoringkeys:
            table = _compare(query, targets, on_cols=scoredict.get(c), func=_scorefuncs[c],
                                  suffix=_scorename[c])
            table_score = pd.concat([table_score, table], axis=1)

        return table_score

def _compare(query, targets, on_cols, func, suffix):
    """
    compare the query to the target records on the selected row, with the selected cols,
    with a function. returns a pd.DataFrame with colum names the names of the columns compared and a suffix.
    if the query is null for the given column name, it retuns an empty column
    Args:
        query (pd.Series): query
        on_cols (list): list of columns on which to compare
        func (func): comparison function
        suffix (str): string to be added after column name

    Returns:
        pd.DataFrame

    Examples:
        table = self._compare(query,on_index=index,on_cols=['name','street'],func=fuzzyratio,sufix='_fuzzyscore')
        returns column names ['name_fuzzyscore','street_fuzzyscore']]
    """
    on_index = targets.index
    table = pd.DataFrame(index=on_index)

    if on_cols is None:
        return table

    compared_cols = on_cols.copy()
    if type(compared_cols) == str:
        compared_cols = [compared_cols]
    assert isinstance(compared_cols, list)

    for c in compared_cols:
        assert isinstance(c, str)
        colname = c + suffix
        if pd.isnull(query[c]):
            table[colname] = None
        else:
            table[colname] = targets[c].apply(lambda r: func(r, query[c]))
    return table