# Step between sting comparison and record linkage: creating similarity features and filtering

import pandas as pd
import numpy as np
from duplicatesuricate.recordlinkage import comparisonfunctions as cfunc


def filter_df(df, query, id_cols, loc_col, fuzzy_filter_cols):
    """
    returns the probability it could be a potential match for further examination
    Args:
        df (pd.DataFrame): attributes of the record
        query (pd.Series): attributes of the query
        id_cols (list): names of the columns which contain an id
        loc_col (list): names of the column of the location (often, country)
        fuzzy_filter_cols (list): names of the columns we want to filter_df on name

    Returns:
        pd.Series: probability of being a potential match
    """
    filter_score = df.apply(lambda row: vector_to_vector_filter(row,
                                                                query,
                                                                id_cols,
                                                                loc_col,
                                                                fuzzy_filter_cols), axis=1)
    return filter_score


def vector_to_vector_filter(row, query, id_cols, loc_col, fuzzy_filter_cols):
    """
    check if the row could be a potential match or Not.
    Args:
        row (pd.Series): attributes of the record
        query (pd.Series): attributes of the query
        id_cols (list): names of the columns which contain an id
        loc_col (str): name of the column of the location (often, country)
        fuzzy_filter_cols (list): names of the columns we want to filter_df on name

    Returns:
        float: probability of being a potential match
    
    it goes like this:
    - if names of the row is an exact match --> return 1
    - if one of the id column is an exact match --> return 1
    - if the location column is not an exact match --> return 0
    - if the location column is an exact match, return the maximum value of the fuzzy string comparison
      for the fuzzy filter_df cols
    """
    comparable_attributes = np.intersect1d(row.dropna().index, query.dropna().index)

    if row.name == query.name:
        # if we compare two exact same indexes (in the case of a deduplication on itself)
        # we would then immediately take it as a potential match
        return 1
    else:
        # if some id (as defined in the id_cols) match
        for c in id_cols:
            if c in comparable_attributes:
                if query[c] == row[c]:
                    # we would then immediately take it as a potential match
                    return 1

        if loc_col in comparable_attributes:
            if query[loc_col] != row[loc_col]:
                # if they are not in the same location (here, country), we reject as a potential match
                return 0
            else:
                # else if they are in the same location, they are a potential candidate for further filtering
                # we filter_df further with fuzzy matching on the fuzzy_filter_cols
                # we would return the maximum similarity score
                max_score = 0
                for c in np.intersect1d(fuzzy_filter_cols, comparable_attributes):
                    score = cfunc.compare_twostrings(row[c], query[c])
                    if pd.isnull(score) is False and score > max_score:
                        max_score = score
                return max_score
        else:
            return 0


def build_similarity_table(df,
                           query,
                           feature_cols,
                           fuzzy_feature_cols,
                           tokens_feature_cols,
                           exact_feature_cols,
                           acronym_col,
                           traincols):
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
    table_score = df.apply(lambda row: vector_to_vector_similarity(row=row,
                                                                   query=query,
                                                                   feature_cols=feature_cols,
                                                                   fuzzy_feature_cols=fuzzy_feature_cols,
                                                                   tokens_feature_cols=tokens_feature_cols,
                                                                   exact_feature_cols=exact_feature_cols,
                                                                   acronym_col=acronym_col,
                                                                   traincols=traincols),
                           axis=1)
    table_score=table_score.fillna(-1)
    return table_score


def vector_to_vector_similarity(row,
                                query,
                                feature_cols,
                                fuzzy_feature_cols,
                                tokens_feature_cols,
                                exact_feature_cols,
                                acronym_col,
                                traincols):
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
        traincols (list): ordered list of columns found in the training columns

    Returns:
        pd.Series: 

    """
    comparable_attributes = np.intersect1d(row.dropna().index, query.dropna().index)
    calculated_features = pd.Series(name=row.name)

    for c in feature_cols:
        # fill the table with the features specific to each row
        calculated_features[c + '_row'] = row[c]
        # fill the table with the features of the query
        calculated_features[c + '_query'] = query[c]

    for c in fuzzy_feature_cols:
        if c in comparable_attributes:
            calculated_features[c + '_fuzzyscore'] = cfunc.compare_twostrings(row[c], query[c])
        else:
            calculated_features[c] = None

    for c in tokens_feature_cols:
        if c in comparable_attributes:
            calculated_features[c + '_tokenscore'] = cfunc.compare_tokenized_strings(row[c], query[c])
        else:
            calculated_features[c] = None

    for c in exact_feature_cols:
        if c in comparable_attributes:
            calculated_features[c + '_exactscore'] = cfunc.exactmatch(row[c], query[c])
        else:
            calculated_features[c] = None

    c = acronym_col
    if c!= '':
        if c in comparable_attributes:
            calculated_features[c + '_acronym_fuzzyscore'] = cfunc.compare_acronyme(row[c], query[c])
        else:
            calculated_features[c] = None

    # re-arrange order columns
    calculated_features = calculated_features[traincols]

    calculated_features.fillna(-1, inplace=True)

    return calculated_features
