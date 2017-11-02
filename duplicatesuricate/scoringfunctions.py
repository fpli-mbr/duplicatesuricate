import pandas as pd
import duplicatesuricate.comparisonfunctions as cfunc
from duplicatesuricate import Suricate


def parallel_filter(sur, row_index):
    '''
    check if the row could be a potential match or Not.
    Args:
        sur: Suricate()
        row_index: index of the row

    Returns:
    boolean True if potential match, False otherwise
    '''

    # if we compare two exact same indexes
    if row_index == sur.query.name:
        return True
    else:
        # else if some ID match
        for c in ['dunsnumber', 'taxid', 'registerid']:
            if pd.isnull(sur.query[c]) == False:
                if sur.query[c] == sur.df.loc[row_index, c]:
                    return True
        # if they are not the same country we reject
        c = 'country'
        if sur.query[c] != sur.df.loc[row_index, c]:
            return False
        else:
            for c in ['streetaddress', 'companyname']:
                filterscore = cfunc.compare_twostrings(
                    sur.query[c], sur.df.loc[row_index, c])
                if filterscore is not None:
                    if filterscore >= sur._filterthreshold_:
                        return True
    return False

def parallel_calculate_comparison_score(sur, row_index):
    '''
    Return the score vector between the query index and the row index
    The query has been saved in sur.query for easier access
    Args:
        sur:Suricate()
        row_index: index of the row to be compared
    Returns:
         pd.Series() comparison score
    '''
    score = pd.Series()
    # fill the table with the features calculated
    for c in sur.featurecols:
        # fill the table with the features specific to each row
        score[c + '_feat_row'] = sur.df.loc[row_index, c].copy()
        # fill the table with the features of the query
        score[c + '_feat_query'] = sur.query[c]

    # calculate the various distances
    for c in ['companyname', 'companyname_wostopwords', 'companyname_acronym',
              'streetaddress', 'streetaddress_wostopwords', 'cityname', 'postalcode']:
        score[c + '_fuzzyscore'] = cfunc.compare_twostrings(sur.df.loc[row_index, c],
                                                              sur.query[c])
    # token scores
    for c in ['companyname_wostopwords', 'streetaddress_wostopwords']:
        score[c + '_tokenscore'] = cfunc.compare_tokenized_strings(
            sur.df.loc[row_index, c], sur.query[c], tokenthreshold=0.5, countthreshold=0.5)

    # acronym score
    c = 'companyname'
    score['companyname_acronym_tokenscore'] = cfunc.compare_acronyme(sur.query[c],
                                                                       sur.df.loc[row_index, c])

    # exactscore
    for c in ['country', 'state', 'dunsnumber', 'postalcode_1stdigit', 'postalcode_2digits', 'taxid',
              'registerid']:
        score[c + '_exactscore'] = cfunc.exactmatch(sur.query[c], sur.df.loc[row_index, c])

    # fill na values
    score = score.fillna(-1)

    # re-arrange order columns
    score = score[sur.traincols]
    return score
