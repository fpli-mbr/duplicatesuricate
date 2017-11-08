# Step between sting comparison and record linkage: creating similarity features and filtering

import pandas as pd
import numpy as np
from duplicatesuricate.recordlinkage import comparisonfunctions as cfunc


def filter_df(df, query, id_cols, loc_col):
    """
    returns the probability it could be a potential match for further examination
    Args:
        df (pd.DataFrame): attributes of the record
        query (pd.Series): attributes of the query
        id_cols (list): names of the columns which contain an id
        loc_col (str): names of the column of the location (often, country)

    Returns:
        pd.Series: boolean of being a potential match
    """
    # filter_bool = df.apply(lambda row: vector_to_vector_filter(row,
    #                                                             query,
    #                                                             id_cols,
    #                                                             loc_col), axis=1)
    s=pd.Series(index=df.index).fillna(False)
    if query.name in s.index:
        s.loc[query.name]=True

    for c in id_cols:
        if pd.isnull(query[c]) is False:
            s.loc[df[c]==query[c]] = True

    if pd.isnull(query[loc_col]) is False:
        s.loc[df[loc_col]==query[loc_col]] = True

    s=s.astype(bool)
    return s

def vector_to_vector_filter(row, query, id_cols, loc_col):
    """
    check if the row could be a potential match or Not.
    Args:
        row (pd.Series): attributes of the record
        query (pd.Series): attributes of the query
        id_cols (list): names of the columns which contain an id
        loc_col (str): name of the column of the location (often, country)
        fuzzy_filter_cols (list): names of the columns we want to filter_df on name

    Returns:
        bool: probability of being a potential match
    
    it goes like this:
    - if names of the row is an exact match --> return 1
    - if one of the id column is an exact match --> return 1
    - if the location column is not an exact match --> return 0
    """
    comparable_attributes = np.intersect1d(row.dropna().index, query.dropna().index)

    if row.name == query.name:
        # if we compare two exact same indexes (in the case of a deduplication on itself)
        # we would then immediately take it as a potential match - even if it is absurd
        return True
    else:
        # if some id (as defined in the id_cols) match
        for c in id_cols:
            if c in comparable_attributes:
                if query[c] == row[c]:
                    # we would then immediately take it as a potential match
                    return True

        if loc_col in comparable_attributes:
            if query[loc_col] != row[loc_col]:
                # if they are not in the same location (here, country), we reject as a potential match
                return False
            else:
                # else if they are in the same location, they are a potential candidate for further filtering
                # we filter_df further with fuzzy matching on the fuzzy_filter_cols
                # we would return the maximum similarity score
                return True
        else:
            return True


def build_similarity_table(df,
                           query,
                           feature_cols,
                           fuzzy_feature_cols,
                           tokens_feature_cols,
                           exact_feature_cols,
                           acronym_col):
    """
    Return the similarity features between the query and the row
    Args:
        df (pd.DataFrame): attributes of the record
        query (pd.Series): attributes of the query
        feature_cols (list): feature cols
        fuzzy_feature_cols (list): fuzzy scoring cols
        tokens_feature_cols (list): token scoring cols
        exact_feature_cols (list): exact scoring cols
        acronym_col (str): acronym scoring col
        traincols (list): ordered list of columns found in the training columns

    Returns:
        pd.DataFrame: 

    """

    # table_score = df.apply(lambda row: vector_to_vector_similarity(
    #                                                                 row=row,
    #                                                                query=query,
    #                                                                feature_cols=feature_cols,
    #                                                                fuzzy_feature_cols=fuzzy_feature_cols,
    #                                                                tokens_feature_cols=tokens_feature_cols,
    #                                                                exact_feature_cols=exact_feature_cols,
    #                                                                acronym_col=acronym_col),
    #                        axis=1)


    table_score=pd.DataFrame(index=df.index)
    if feature_cols is not None:
        for c in feature_cols:
            table_score[c+'_query']=query[c]
            table_score[c+'_row']=df[c]

    if fuzzy_feature_cols is not None:
        for c in fuzzy_feature_cols:
            colname = c+'_fuzzyscore'
            if pd.isnull(query[c]):
                table_score[colname]=None
            else:
                table_score[colname]=df[c].apply(lambda r:cfunc.compare_twostrings(r,query[c]))

    if tokens_feature_cols is not None:
        for c in tokens_feature_cols:
            colname = c+'_tokenscore'
            if pd.isnull(query[c]):
                table_score[colname]=None
            else:
                table_score[colname]=df[c].apply(lambda r:cfunc.compare_tokenized_strings(r,query[c]))

    if exact_feature_cols is not None:
        for c in exact_feature_cols:
            colname = c+'_exactscore'
            if pd.isnull(query[c]):
                table_score[colname]=None
            else:
                table_score[colname]=df[c].apply(lambda r:cfunc.exactmatch(r,query[c]))

    c = acronym_col
    if c is not None and len(c)>0:
        colname = c + '_acronym_fuzzyscore'
        if pd.isnull(query[c]):
            table_score[colname] = None
        else:
            table_score[colname] = df[c].apply(lambda r: cfunc.compare_acronyme(r,query[c]))

    return table_score


def vector_to_vector_similarity(
                                row,
                                query,
                                feature_cols,
                                fuzzy_feature_cols,
                                tokens_feature_cols,
                                exact_feature_cols,
                                acronym_col,
                                ):
    """
    Return the similarity features between the query and the row
    Args:
        row (pd.Series): attributes of the record
        query (pd.Series): attributes of the query
        feature_cols (list): feature cols
        fuzzy_feature_cols (list): fuzzy scoring cols
        tokens_feature_cols (list): token scoring cols
        exact_feature_cols (list): exact scoring cols
        acronym_col (str): acronym scoring col

    Returns:
        pd.Series: 

    """
    comparable_attributes = np.intersect1d(row.dropna().index, query.dropna().index)
    calculated_features=pd.Series(name=row.name)
    if feature_cols is not None:
        for c in feature_cols:
            if c in row.index:
                # fill the table with the features specific to each row
                calculated_features[c + '_row'] = row[c]
                # fill the table with the features of the query
                calculated_features[c + '_query'] = query[c]

    if fuzzy_feature_cols is not None:
        for c in fuzzy_feature_cols:
            colname=c+'_fuzzyscore'
            if colname not in calculated_features:
                if c in comparable_attributes:
                    calculated_features[colname] = cfunc.compare_twostrings(row[c], query[c])
                else:
                    calculated_features[colname] = None
    if tokens_feature_cols is not None:
        for c in tokens_feature_cols:
            colname = c + '_tokenscore'
            if c in comparable_attributes:
                calculated_features[colname] = cfunc.compare_tokenized_strings(row[c], query[c])
            else:
                calculated_features[colname] = None
    if exact_feature_cols is not None:
        for c in exact_feature_cols:
            colname = c + '_exactscore'
            if c in comparable_attributes:
                calculated_features[colname] = cfunc.exactmatch(row[c], query[c])
            else:
                calculated_features[colname] = None

    c = acronym_col
    if c is not None and len(c)>0:
        colname = c + '_acronym_fuzzyscore'
        if c in comparable_attributes:
            calculated_features[colname] = cfunc.compare_acronyme(row[c], query[c])
        else:
            calculated_features[colname] = None

    return calculated_features
