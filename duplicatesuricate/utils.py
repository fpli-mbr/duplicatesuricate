import numpy as np
import pandas as pd
from fuzzywuzzy.fuzz import ratio, token_set_ratio
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType, StructField, StructType, StringType, BooleanType, IntegerType

from pyspark.sql import functions as F
from pyspark.sql import types as T

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
    def to_dict(self):
        m=dict()
        for k in self.keys():
            m[k] = self[k]
        return m

    @classmethod
    def from_cols(cls,scorecols):
        """
        Args:
            scorecols (set): list of scoring cols
        Returns:

        """
        x_col = set(scorecols)
        m_dic = {}

        def _findscoreinfo(colname):
            if colname.endswith('_target'):
                k = 'attributes'
                u = _rmv_end_str(colname, '_target')
                return k, u
            elif colname.endswith('_source'):
                k = 'attributes'
                u = _rmv_end_str(colname, '_source')
                return k, u
            elif colname.endswith('score'):
                u = _rmv_end_str(colname, 'score')
                for k in ['fuzzy', 'token', 'exact', 'acronym']:
                    if u.endswith('_' + k):
                        u = _rmv_end_str(u, '_' + k)
                        return k, u
            else:
                return None

        for c in x_col:
            result = _findscoreinfo(c)
            if result is not None:
                method, column = result[0], result[1]
                if m_dic.get(method) is None:
                    m_dic[method] = [column]
                else:
                    m_dic[method] = list(set(m_dic[method] + [column]))
        if len(m_dic) > 0:
            return ScoreDict(m_dic)
        else:
            return None


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


def fuzzyscore(a, b):
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


_fuzzy_udf = udf(lambda a, b: fuzzyscore(a, b), FloatType())

def fuzzyspark(df, on, value):
    q_val = value
    df = df.select([on])
    # TODO: Rework that part
    df = df.withColumn('query', F.lit(q_val).cast(F.StringType()))
    df = df.withColumn('len', F.min(F.length(on), F.lit(len(q_val)).cast(T.IntegerType())))
    df = df.withColumn('levenshtein', F.levenshtein(on, 'query'))
    df = df.withColumn('score', F.col('levenshtein') / F.col('len'))
    df = df.select(['score'])
    return df

def token_score(df, on, value):
    q_val = value
    df = df.select([on])
    df = df.withColumn('query', F.lit(q_val).cast(F.StringType()))
    # TODO: implement the pattent
    pattern = ','
    df = df.withColumn('tokens1', F.split(F.col('left'), pattern))
    df = df.withColumn('tokens2', F.split(F.col('right'), pattern))
    # intersection = tokens1.intersection(tokens2)
    # diff1to2 = tokens1.difference(tokens2) = pure token 1
    # diff2to1 = tokens2.difference(tokens1) = pure token 2
    # TODO: implement an intersect and a diff method
    df = df.withColumn('intersection', F.intersect('tokens1', 'tokens2'))
    df = df.withColumn('diff1to2', F.diff('tokens1', 'tokens2'))
    df = df.withColumn('diff2to1', F.diff('tokens2', 'tokens1'))
    # sorted_sect = " ".join(sorted(intersection))
    # sorted_1to2 = " ".join(sorted(diff1to2))
    # sorted_2to1 = " ".join(sorted(diff2to1))
    # TODO: implement a concat for an array
    df = df.withColumn('sorted_sect', F.concat_ws(' ', F.sort_array('intersection')))
    df = df.withColumn('sorted_1to2 ', F.concat_ws(' ', F.sort_array('diff1to2')))
    df = df.withColumn('sorted_2to1', F.concat_ws(' ', F.sort_array('diff2to1')))
    # combined_1to2 = sorted_sect + " " + sorted_1to2 = chain 1 that has been sorted
    # combined_2to1 = sorted_sect + " " + sorted_2to1 = chain 2 that has been sorted
    # TODO: no, i'm joking
    df = df.withColumn('combined_1to2', F.concat_ws(' ', ['sorted_sect', 'sorted_1to2']))
    df = df.withColumn('combined_1to2', F.concat_ws(' ', ['sorted_sect', 'sorted_2to1']))
    # strip
    # sorted_sect = sorted_sect.strip()
    # combined_1to2 = combined_1to2.strip()
    # combined_2to1 = combined_2to1.strip()
    for c in ['sorted_sect', 'combined_1to2', 'combined_2to1']:
        df = df.withColumn(c, F.trim(c))
    # TODO: create a function spark_ratio
    df = df.withColumn('ratio1', spark_ratio(F.col('sorted_sect', F.col('combined_1to2'))))
    df = df.withColumn('ratio2', spark_ratio(F.col('sorted_sect', F.col('combined_2to1'))))
    df = df.withColumn('ratio3', spark_ratio(F.col('combined_2to1', F.col('combined_1to2'))))
    # pairwise = [
    #     ratio_func(sorted_sect, combined_1to2),
    #     ratio_func(sorted_sect, combined_2to1),
    #     ratio_func(combined_1to2, combined_2to1)
    # ]
    df = df.withColumn('max_ratio', F.max(['ratio1', 'ratio2', 'ratio3']))
    df = df.withColumnRenamed('max_ratio', 'token_fuzzy')
    df = df.select(['token_fuzzy'])
    return df

def spark_ratio(left, right):
    # TODO: sparkify this function
    df = df(['left', 'right'])
    df = df.withColumn('len', F.min(F.length('left'), F.length('right')))
    df = df.withColumn('levenshtein', F.levenshtein('left', 'right'))
    df = df.withColumn('inv_edit_distance', F.col('len') - F.col('levenshtein'))
    df = df.withColumn('ratio', F.col('inv_edit_distance') / F.col('len'))
    df = df.withColumnRenamed('ratio', 'fuzzy')
    df = df.select(['fuzzy'])
    return df

def tokenscore(a, b):
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


_token_udf = udf(lambda a, b: tokenscore(a, b), FloatType())


def exactmatch(a, b):
    if pd.isnull(a) or pd.isnull(b):
        return 0.0
    else:
        if a == b:
            return 1.0
        else:
            return -1.0


_exact_udf = udf(lambda a, b: exactmatch(a, b), FloatType())


def acronym(s):
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


def compare_acronym(a, b, minaccrolength=3):
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
        a_acronyme = acronym(a)
        b_acronyme = acronym(b)
        if min(len(a_acronyme), len(b_acronyme)) >= minaccrolength:
            a_score_acronyme = tokenscore(a_acronyme, b)
            b_score_acronyme = tokenscore(a, b_acronyme)
            if all(pd.isnull([a_score_acronyme, b_score_acronyme])):
                return 0.0
            else:
                max_score = np.max([a_score_acronyme, b_score_acronyme])
                return max_score
        else:
            return 0.0


_acronym_udf = udf(lambda a, b: compare_acronym(a, b), FloatType())
_scorename = {'fuzzy': '_fuzzyscore',
              'token': '_tokenscore',
              'exact': '_exactscore',
              'acronym': '_acronymscore'}
_scorefuncs = {'fuzzy': fuzzyscore,
               'token': tokenscore,
               'exact': exactmatch,
               'acronym': compare_acronym}
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


_sparktypedict = dict()
_sparktypedict[np.dtype('O')] = StringType()
_sparktypedict[np.dtype('int64')] = IntegerType()
_sparktypedict[np.dtype('float64')] = FloatType()
_sparktypedict[np.dtype('bool')] = BooleanType()

_sparktypedict[str] = StringType()
_sparktypedict[int] = IntegerType()
_sparktypedict[float] = FloatType()
_sparktypedict[bool] = BooleanType()


# Thank you


def _transform_pandas_spark(sqlContext, df, drop_index=False):
    """
    Takes a pandas DataFrame as an entry. Convert it to a Spark DF, using the pandas Schema and index
    Args:
        sqlContext (pyspark.sql.context.SQLContext)
        df (pd.DataFrame):
        drop_index (bool): if True, the index will not be saved. If False, the index will be a separate column

    Returns:
        pyspark.sql.dataframe.DataFrame
    """
    schema = []

    if drop_index is False:
        # add index column to the schema
        mytype = _sparktypedict[df.index.dtype]
        name = df.index.name
        if name is None:
            name = 'index'
        schema.append(StructField(name=name, dataType=mytype, nullable=False))

    # add compared_columns to the schema
    for c in df.columns.tolist():
        mytype = _sparktypedict[df[c].dtype]
        mycol = StructField(name=c, dataType=mytype, nullable=True)
        schema.append(mycol)
    # create schema
    schema = StructType(schema)

    if drop_index is False:
        # add index to the dataframe
        x = df.reset_index(drop=False)
    else:
        x = df

    # create dataframe
    ds = sqlContext.createDataFrame(x, schema=schema)
    return ds