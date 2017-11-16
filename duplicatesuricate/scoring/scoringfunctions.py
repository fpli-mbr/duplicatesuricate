# Step between sting comparison and record linkage: creating similarity features and filtering
import duplicatesuricate.others
import pandas as pd
import neatmartinet as nm


# from neatmartinet import compare_tokenized_strings as tokenscore
# from neatmartinet import compare_twostrings as fuzzyscore
# from duplicatesuricate.others import compare_acronyme as acronymscore
# from duplicatesuricate.others import exactmatch as exactscore


class Scorer:
    def __init__(self, df, filterdict=None, score1dict=None, score1func=None, score2dict=None):
        """

        Args:
            df (pd.DataFrame: reference records
            filterdict (dict):
            score1dict (dict):
            score1func (func):
            score2dict (dict):
        """
        self.df = df
        self.scoringkeys = ['attributes', 'fuzzy', 'token', 'exact', 'acronym']
        self.existingcols=[]

        self.filterdict = _checkdict(filterdict, mandatorykeys=['all', 'any'],existinginput=self.existingcols)
        self.existingcols = _unpack_scoredict(self.filterdict)

        self.score1dict = _checkdict(score1dict, mandatorykeys=self.scoringkeys,existinginput=self.existingcols)
        self.existingcols = self.existingcols + _unpack_scoredict(self.score1dict)

        self.score2dict = _checkdict(score2dict, mandatorykeys=self.scoringkeys, existinginput=self.existingcols)
        self.existingcols = self.existingcols + _unpack_scoredict(score2dict)

        self.score1func = score1func
        pass




    def filter_all_any(self, query, on_index):
        """
        returns the probability it could be a potential match for further examination
        Args:
            query (pd.Series): query
            on_index (pd.Index): index
            match_any_cols (list): a match on any of these columns returns True otherwise False
            match_all_cols (list):  a match on all of these columns return True otherwise False

        Returns:
            pd.DataFrame: first scoring
        """
        match_any_cols=self.filterdict['any']
        match_all_cols=self.filterdict['all']

        table_score_filter = pd.DataFrame(index=on_index)

        if match_all_cols is None and match_any_cols is None:
            return table_score_filter

        df = self.df.loc[on_index]

        if match_any_cols is not None:
            match_any_df = pd.DataFrame(index=on_index)
            for c in match_any_cols:
                match_any_df[c + '_exactscore'] = df[c].apply(
                    lambda r: duplicatesuricate.others.exactmatch(r, query[c]))
            anycriteriasmatch = (match_any_df == 1).any(axis=1)
            table_score_filter = pd.concat([table_score_filter, match_any_df], axis=1)
        else:
            anycriteriasmatch = pd.Series(index=on_index).fillna(False)

        if match_all_cols is not None:
            match_all_df = pd.DataFrame(index=on_index)
            for c in match_all_cols:
                match_all_df[c + '_exactscore'] = df[c].apply(
                    lambda r: duplicatesuricate.others.exactmatch(r, query[c]))
            allcriteriasmatch = (match_all_df == 1).all(axis=1)
            table_score_filter = pd.concat([table_score_filter, match_all_df], axis=1)
        else:
            allcriteriasmatch = pd.Series(index=on_index).fillna(False)

        results = (allcriteriasmatch | anycriteriasmatch)

        return table_score_filter.loc[results]

    def build_similarity_table(self,
                               query,
                               on_index,
                               scoredict):
        """
        Return the similarity features between the query and the row
        Args:
            query (pd.Series): attributes of the query
            on_index (pd.Index):
            scoredict (dict):

        Returns:
            pd.DataFrame:

        """
        attributes_cols=scoredict['attributes']
        fuzzyscore_cols=scoredict['fuzzy']
        tokenscore_cols=scoredict['token']
        exactscore_cols=scoredict['exact']
        acronymscore_cols=scoredict['acronym']

        table_score = pd.DataFrame(index=on_index)
        df = self.df.loc[on_index]

        if attributes_cols is not None:
            for c in attributes_cols:
                table_score[c + '_query'] = query[c]
                table_score[c + '_row'] = df[c]

        if fuzzyscore_cols is not None:
            for c in fuzzyscore_cols:
                colname = c + '_fuzzyscore'
                if pd.isnull(query[c]):
                    table_score[colname] = None
                else:
                    table_score[colname] = df[c].apply(lambda r: nm.compare_twostrings(r, query[c]))

        if tokenscore_cols is not None:
            for c in tokenscore_cols:
                colname = c + '_tokenscore'
                if pd.isnull(query[c]):
                    table_score[colname] = None
                else:
                    table_score[colname] = df[c].apply(lambda r: nm.compare_tokenized_strings(r, query[c]))

        if exactscore_cols is not None:
            for c in exactscore_cols:
                colname = c + '_exactscore'
                if pd.isnull(query[c]):
                    table_score[colname] = None
                else:
                    table_score[colname] = df[c].apply(lambda r: duplicatesuricate.others.exactmatch(r, query[c]))

        if acronymscore_cols is not None:
            for c in acronymscore_cols:
                colname = c + '_acronymscore'
                if pd.isnull(query[c]):
                    table_score[colname] = None
                else:
                    table_score[colname] = df[c].apply(lambda r: duplicatesuricate.others.compare_acronyme(r, query[c]))

        return table_score

    def filter_compare(self, query):
        """
        to do
        Args:
            query:

        Returns:

        """

        query = query.copy()

        # pre filter the records for further scoring: 0 if not possible choice, 0.5 if possible choice, 1 if sure choice
        workingindex = self.df.index
        table_score_filter = self.filter_all_any(query=query,
                                                 on_index=workingindex
                                                 )
        workingindex = table_score_filter.index
        if table_score_filter.shape[0] == 0:
            return None
        else:
            # do further scoring on the possible choices and the sure choices
            table_score_1 = self.build_similarity_table(query=query,
                                                        on_index=workingindex,scoredict=self.score1dict
                                                        )

            table_score_complete = table_score_filter.join(table_score_1, how='left')
            y_decision1 = table_score_complete.apply(lambda r: self.score1func(r), axis=1)
            table_score_complete = table_score_complete.loc[y_decision1]
            workingindex = table_score_complete.index

            if table_score_complete.shape[0] == 0:
                return None
            else:
                # we perform further analysis on the filtered index:
                # we complete the fuzzy score with additional columns
                # TO DO: add a check to make sure no duplicates columns

                table_score_2 = self.build_similarity_table(query=query, on_index=workingindex,scoredict=self.score2dict)

                # we join the two tables to have a complete view of the score
                table_score_complete = table_score_complete.join(table_score_2, how='left')

                return table_score_complete

def _unpack_scoredict(scoredict):
    """

    Args:
        scoredict (mydict):

    Returns:
        list
    """
    mylist=[]
    for d in ['all','any']:
        if scoredict.get(d) is not None:
            for c in scoredict[d]:
                mylist.append(c+'_exactscore')
    if scoredict.get('attributes') is not None:
        for c in scoredict['attributes']:
            mylist.append(c+'_query')
            mylist.append(c+'_row')
    for k in ['fuzzy','token','exact','acronym']:
        if scoredict.get(k) is not None:
            for c in scoredict[k]:
                mylist.append(c+'_'+k+'score')
    return mylist

def _checkdict(inputdict, mandatorykeys, existinginput=None):
    """

    Args:
        inputdict (dict):
        mandatorykeys (list):
        existinginput (list):

    Returns:

    """
    mydict = {}
    if inputdict is None:
        for c in mandatorykeys:
            mydict[c] = None
    else:
        for c in mandatorykeys:
            if inputdict.get(c) is None:
                mydict[c] = None
            else:
                if existinginput is None:
                    mydict[c]=inputdict[c].copy()
                else:
                    mydict[c]=[v for v in inputdict[c] if not v in existinginput]
    return mydict