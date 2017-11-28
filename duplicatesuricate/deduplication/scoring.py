# Step between sting comparison and record linkage: creating similarity features and filtering
import neatmartinet as nm
import pandas as pd

scorename = {'fuzzy': '_fuzzyscore',
             'token': '_tokenscore',
             'exact': '_exactmatch',
             'acronym': '_acronymscore'}

scorefuncs = {'fuzzy': nm.compare_twostrings,
              'token': nm.compare_tokenized_strings,
              'exact': nm.exactmatch,
              'acronym': nm.compare_acronyme}
scoringkeys=list(scorename.keys())


class Scorer:
    def __init__(self, df, filterdict=None, score_intermediate=None, decision_intermediate=None, score_further=None,
                 fillna=-1):
        """

        Args:
            df (pd.DataFrame: reference records
            filterdict (dict):
            score_intermediate (dict):
            decision_intermediate (func):
            score_further (dict):
            fillna (float): Value used to fill the na values
        """
        #TODO : docstring of scorer class
        self.df = df
        self.scorecols = []
        self.compared_cols=[]

        self.filterdict = _checkdict(filterdict, mandatorykeys=['all', 'any'], existinginput=self.scorecols)
        incols,outcols= _unpack_scoredict(self.filterdict)
        self.compared_cols+=incols
        self.scorecols+=outcols

        self.intermediate_score = _checkdict(score_intermediate, mandatorykeys=scoringkeys,
                                             existinginput=self.scorecols)
        incols,outcols= _unpack_scoredict(self.intermediate_score)
        self.compared_cols+=incols
        self.scorecols+=outcols

        self.further_score = _checkdict(score_further, mandatorykeys=scoringkeys, existinginput=self.scorecols)
        incols,outcols= _unpack_scoredict(self.further_score)
        self.compared_cols+=incols
        self.scorecols+=outcols

        for c in self.compared_cols,self.scorecols:
            c=list(set(c))

        self.intermediate_func = decision_intermediate

        self.navalue = fillna

        pass

    def filter_all_any(self, query, on_index, filterdict):
        """
        returns a pre-filtered table score calculated on the column names provided in the filterdict.
        in the values for 'any': a match on any of these columns ensure the row is kept for further analysis
        in the values for 'all': a match on all of these columns ensure the row is kept for further analysis
        if the row does not have any exact match for the 'any' columns, or if it has one bad match for the 'all' columns,
        it is filtered out
        Args:
            query (pd.Series): query
            on_index (pd.Index): index
            filterdict(dict): dictionnary two lists of values: 'any' and 'all'

        Returns:
            pd.DataFrame: a DataFrame with the exact score of the columns provided in the filterdict
        """
        table = pd.DataFrame(index=on_index)

        if filterdict is None:
            return table

        match_any_cols = filterdict.get('any')
        match_all_cols = filterdict.get('all')

        if match_all_cols is None and match_any_cols is None:
            return table

        df = self.df.loc[on_index]

        if match_any_cols is not None:
            match_any_df = pd.DataFrame(index=on_index)
            for c in match_any_cols:
                match_any_df[c + '_exactscore'] = df[c].apply(
                    lambda r: nm.exactmatch(r, query[c]))

            y = (match_any_df == 1)
            assert isinstance(y,pd.DataFrame)
            anycriteriasmatch = y.any(axis=1)
            table = pd.concat([table, match_any_df], axis=1)
        else:
            anycriteriasmatch = pd.Series(index=on_index).fillna(False)

        if match_all_cols is not None:
            match_all_df = pd.DataFrame(index=on_index)
            for c in match_all_cols:
                match_all_df[c + '_exactscore'] = df[c].apply(
                    lambda r: nm.exactmatch(r, query[c]))
            y = (match_all_df == 1)
            assert isinstance(y,pd.DataFrame)
            allcriteriasmatch = y.all(axis=1)

            table = pd.concat([table, match_all_df], axis=1)
        else:
            allcriteriasmatch = pd.Series(index=on_index).fillna(False)

        results = (allcriteriasmatch | anycriteriasmatch)

        assert isinstance(table,pd.DataFrame)

        return table.loc[results]

    def build_similarity_table(self,
                               query,
                               on_index,
                               scoredict):
        """
        Return the similarity features between the query and the rows in the required index, with the selected comparison functions.
        They can be fuzzy, token-based, exact, or acronym.
        The attribute request creates two column: one with the value for the query and one with the value for the row

        Args:
            query (pd.Series): attributes of the query
            on_index (pd.Index):
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

        table_score = pd.DataFrame(index=on_index)

        attributes_cols = scoredict.get('attributes')
        if attributes_cols is not None:
            for c in attributes_cols:
                table_score[c + '_query'] = query[c]
                table_score[c + '_row'] = self.df.loc[on_index, c]

        for c in scoringkeys:
            table = self._compare(query, on_index=on_index, on_cols=scoredict.get(c), func=scorefuncs[c],
                                  suffix=scorename[c])
            table_score = pd.concat([table_score, table], axis=1)

        return table_score

    def filter_compare(self, query):
        """
        Simultaneously create a similarity table and filter the data.
        It works in three steps:
        - filter with a logic (exact match on any of these cols OR exact match on all of these columns)
        - intermediate score with dedicated comparison methods on selected columns
        - filter with an intermediate decision function
        - further score with dedicated comparison methods on selected columns
        - returns the final similarity table which is the concatenation of all of the scoring functions above on the rows
            that have been filtered
        Args:
            query (pd.Series): query

        Returns:
            pd.DataFrame similarity table
        """

        # pre filter the records for further scoring based on an all / any exact match
        workingindex = self.df.index

        table_score_complete = self.filter_all_any(query=query,
                                                   on_index=workingindex,
                                                   filterdict=self.filterdict
                                                   )
        workingindex = table_score_complete.index

        if table_score_complete.shape[0] == 0:
            return None

        else:
            # do further scoring on the possible choices and the sure choices
            table_intermediate = self.build_similarity_table(query=query,
                                                             on_index=workingindex,
                                                             scoredict=self.intermediate_score)

            table_score_complete = table_score_complete.join(table_intermediate, how='left')
            del table_intermediate

            y_intermediate = table_score_complete.apply(lambda r: self.intermediate_func(r), axis=1)
            y_intermediate=y_intermediate.astype(bool)

            assert isinstance(y_intermediate,pd.Series)
            assert (y_intermediate.dtype == bool)

            table_score_complete = table_score_complete.loc[y_intermediate]

            workingindex = table_score_complete.index

            if table_score_complete.shape[0] == 0:
                return None
            else:
                # we perform further analysis on the filtered index:
                # we complete the fuzzy score with additional columns

                table_additional = self.build_similarity_table(query=query, on_index=workingindex,
                                                               scoredict=self.further_score)

                # check to make sure no duplicates columns
                duplicatecols = list(filter(lambda x:x in table_score_complete.columns,table_additional.columns))
                if len(duplicatecols)>0:
                    table_additional.drop(duplicatecols,axis=1,inplace=True)

                # we join the two tables to have a complete view of the score
                table_score_complete = table_score_complete.join(table_additional, how='left')

                del table_additional

                table_score_complete = table_score_complete.fillna(self.navalue)

                return table_score_complete

    def _compare(self, query, on_index, on_cols, func, suffix):
        """
        compare the query to the target records on the selected row, with the selected cols,
        with a function. returns a pd.DataFrame with colum names the names of the columns compared and a suffix.
        if the query is null for the given column name, it retuns an empty column
        Args:
            query (pd.Series): query
            on_index (pd.Index): index on which to compare
            on_cols (list): list of columns on which to compare
            func (func): comparison function
            suffix (str): string to be added after column name

        Returns:
            pd.DataFrame

        Examples:
            table = self._compare(query,on_index=index,on_cols=['name','street'],func=fuzzyratio,sufix='_fuzzyscore')
            returns column names ['name_fuzzyscore','street_fuzzyscore']]
        """
        table = pd.DataFrame(index=on_index)

        if on_cols is None:
            return table

        compared_cols=on_cols.copy()
        if type(compared_cols) == str:
            compared_cols=[compared_cols]
        assert isinstance(compared_cols,list)

        for c in compared_cols:
            assert isinstance(c,str)
            colname = c + suffix
            if pd.isnull(query[c]):
                table[colname] = None
            else:
                table[colname] = self.df.loc[on_index, c].apply(lambda r: func(r, query[c]))
        return table


def _unpack_scoredict(scoredict):
    """
    a list of scoring functions to be applied
    Args:
        scoredict (dict):

    Returns:
        list,list : input_cols, output_cols
    """
    #TODO : Add docstring
    outputcols = []
    inputcols=[]

    for d in ['all', 'any']:
        if scoredict.get(d) is not None:
            for c in scoredict[d]:
                inputcols.append(c)
                outputcols.append(c + '_exactscore')
    if scoredict.get('attributes') is not None:
        for c in scoredict['attributes']:
            inputcols.append(c)
            outputcols.append(c + '_query')
            outputcols.append(c + '_row')
    for k in scorename.keys():
        if scoredict.get(k) is not None:
            for c in scoredict[k]:
                inputcols.append(c)
                outputcols.append(c + scorename[k])
    return inputcols,outputcols


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
                    mydict[c] = inputdict[c].copy()
                else:
                    mydict[c] = [v for v in inputdict[c] if not v in existinginput]
    return mydict

def _calculatescoredict(existing_cols,used_cols):
    """
    From a set of existing comparison columns and needed columns, calculate the scoring dict that is needed to
    close the gap
    Args:
        existing_cols (list):
        used_cols (list):

    Returns:
        dict: scoredict-type
    """
    x=list(filter(lambda x: x not in existing_cols,used_cols))
    m={}
    def _findscoreinfo(c):
        if c.endswith('_row') or c.endswith('_query'):
            k='attributes'
            u=nm.rmv_end_str(c,'_row')
            return k,u
        elif c.endswith('score'):
            u=nm.rmv_end_str(c,'score')
            for k in ['fuzzy','token','exact','acronym']:
                if u.endswith('_'+k):
                    u=nm.rmv_end_str(u,'_'+k)
                    return k,u
        else:
            return None
    for c in x:
        result=_findscoreinfo(c)
        if result is not None:
            method,column = result[0],result[1]
            if m.get(method) is None:
                m[method]=[column]
            else:
                m[method]=list(set(m[method]+[column]))
    if len(m)>0:
        return m
    else:
        return None


