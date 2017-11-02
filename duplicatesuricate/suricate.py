# coding=utf-8
""" deduplication methods
# How to use it
# init a DeduplicationDatabase instance
# init a DeduplicationModel instance
# Build a training set to fit the deduplication model
# Fit the deduplicationModel
"""
import numpy as np
import pandas as pd
from duplicatesuricate import comparisonfunctions, cleaningfunctions,scoringfunctions


class Suricate:
    """
    A class that uses a pandas.DataFrame to store the data (self.df) and special methods (sklearn models) to eliminate doublons
    """

    def __init__(self, df, warmstart=True, idcol='groupid',
                 queryidcol='queryid', verbose=True):
        """

        Args:
            df (pd.DataFrame): Input table for deduplication
            warmstart (bool): if True, does not apply the cleaning function
            idcol (str): name of the column where to store the deduplication results
            queryidcol (str): name of the column used to store the original match
            verbose (bool): Turns on or off prints
        """
        start = pd.datetime.now()
        self.df = df
        self.featurecols = ['companyname_wostopwords_wordfrequency',
                            'companyname_len', 'companyname_wostopwords_len', 'streetaddress_len',
                            'companyname_wostopwords_ntokens', 'cityfrequency', 'isbigcity',
                            'has_thales', 'has_zodiac', 'has_ge', 'has_safran', 'has_airbusequiv']
        self.idcol = idcol
        self.queryidcol = queryidcol
        self.verbose = verbose
        self.model = None

        self.displaycols = [self.idcol, 'companyname', 'streetaddress', 'cityname', 'postalcode', 'country',
                            'dunsnumber', 'taxid', 'registerid']
        if self.verbose:
            print('Inputdatabase shape', df.shape)
            start = pd.datetime.now()

        if warmstart is False:
            self.clean_db()

        if self.verbose:
            end = pd.datetime.now()
            duration = (end - start).total_seconds()
            print('cleaning time', duration, 'seconds')
            print('output database shape', df.shape)

        # scoring threshold for fuzzyscore filtering
        self._filterthreshold_ = 0.8
        # scoring threshold using model predict_proba
        self._decisionthreshold_ = 0.6

        self.query = None

    def clean_db(self):
        """
        Clean the database
        Returns: None
        """
        self.df = cleaningfunctions.clean_db(df=self.df,idcol=self.idcol,queryidcol=self.queryidcol)

    def launch_calculation(self, nmax=None, in_index=None):
        """
        launches the deduplication process
        Args:
            nmax (int): maximum number of occurences
            in_index (list): deduplicate the following index

        Returns:
            None
        """
        if nmax is None:
            if in_index is not None:
                nmax = len(in_index)
            else:
                raise ('No maximum number given or index given')

        if self.verbose:
            print('deduplication started at ', pd.datetime.now())

        # loop on the rows to deduplicate
        for countdown in range(nmax):
            # find a row to deduplicate
            query_index = self._generate_query_index_(in_index)

            if query_index is None:
                print('no valid query available')
                break
            else:
                if self.verbose:
                    print('countdown', countdown + 1, 'of', nmax, ':')
                # deduplicate that row
                self._deduplicate_row_(query_index)

        if self.verbose:
            print('deduplication finished at ', pd.datetime.now())

        return None

    def _generate_query_index_(self, in_index=None):
        """
        this function returns a random index with no group id to start the deduplication process
        Args:
            in_index: index or list, default None the query should be in the selected index
    
        Returns:
            pd.Series:a row of the dataframe
        """

        if in_index is None:
            in_index = self.df.index

        x = self.df.loc[in_index]
        possiblechoices = x.loc[(x[self.idcol] == 0) | (x[self.idcol].isnull())].index
        if possiblechoices.shape[0] == 0:
            del x
            return None
        else:
            a = np.random.choice(possiblechoices)
            del x, possiblechoices
            return a

    def _deduplicate_row_(self, query_index):
        '''
        Deduplicate a row (search for duplicates in the database and update the groupid col)
        Args:
            query_index: index of the row to be deduplicated

        Returns:
            None
        '''
        if self.verbose:
            start = pd.datetime.now()

        # return the good matches as calculated by the model
        goodmatches_index = self._return_goodmatches_(query_index=query_index)

        # attribute/update the groupid of those matches
        n_deduplicated = self._update_idcol_(goodmatches_index, query_index)

        if self.verbose:
            end = pd.datetime.now()
            duration = (end - start).total_seconds()
            print('record', query_index, 'n_deduplicated', n_deduplicated, 'duration', duration)

        return None

    def _return_goodmatches_(self, query_index):
        '''
        return the index of the good matches as judged by the deduplication algorithm
        Args:
            query_index: index of the query which serves as the search for duplicates

        Returns:
            pd.Series().index
        '''

        # return the probability of being a match (as a Series, score 0->1)
        predictions = self._return_predictions_(query_index)

        # Select only the matches where the probability is greater than the decision threshold
        goodmatches = predictions.loc[predictions > self._decisionthreshold_].index

        return goodmatches

    def _return_predictions_(self, query_index):
        """
        return the index of the positive matches of the query
        Args:
            query_index: index of the query

        Returns:
            pandas.Series().index: positive matches

        """
        self.query = self.df.loc[query_index].copy()

        filtered_index = self.step_one_filter()

        tablescore = self.step_two_score(filtered_index)

        predictions = self.step_three_predict(tablescore)

        return predictions

    def _update_idcol_(self, goodmatches_index, query_index):
        """
        attribute a new matching id to the group of matches
        Modify the self.df[self.idcol] column of the database
        Args:
            goodmatches_index(list): index of the group of matches
            query_index: index to be used to save the original id of the matching record
    
        Returns:
        integer, number of ids deduplicated
        """
        # Take the maximum existing group id
        database_matching_id_max = self.df[self.idcol].max()
        if pd.isnull(database_matching_id_max):
            database_matching_id_max = 0

        # if we have a non empty dataframe
        if len(goodmatches_index) > 0:
            # if some group ids already exist take the first one
            if self.df.loc[goodmatches_index, self.idcol].max() > 0:
                goodmatches_matching_id = self.df.loc[goodmatches_index, self.idcol].dropna().iloc[0]
            else:
                # otherwise create a new one
                goodmatches_matching_id = database_matching_id_max + 1

            # take all lines who don't have a group id and give them the new id and save the index of the query
            nonallocatedmatches_index = self.df.loc[goodmatches_index].loc[
                self.df.loc[goodmatches_index, self.idcol].isnull()].index

            self.df.loc[nonallocatedmatches_index, self.idcol] = goodmatches_matching_id
            self.df.loc[nonallocatedmatches_index, self.queryidcol] = query_index
            n_deduplicated = len(nonallocatedmatches_index)
        else:
            n_deduplicated = None
        return n_deduplicated

    def showgroup(self, groupid, cols=None):
        '''
        show all the records that share the same groupid
        Args:
            groupid (int): int, id of he group to be displayed
            cols (list), list, default displaycols, columns ot be displayed
        Returns:
            pd.DataFrame
        '''
        if cols is None:
            cols = self.displaycols
        x = self.df.loc[self.df[self.idcol] == groupid, cols]
        return x

    def showpossiblematches(self, query_index):
        '''
        show the possible results from the deduplication
        Args:
            query_index (index): index on which to check possible matches
        Returns:
            pd.DataFrame
        '''

        y_proba = self._return_predictions_(query_index)

        x = self.df.loc[y_proba.index, self.displaycols].copy()
        x['score'] = y_proba
        x.sort_values(by='score', ascending=False, inplace=True)

        return x

    def extract_possible_pair(self, query_index, on_col='systemid', on_value='P11'):
        '''
        extract the closest match, from the group id, by filtering on a column and on a value
        Examples :
        name    city    systemid    groupid
        foo     munich  p11         1
        FOO     munich  f40         1
        bar     munich  p11         2
        if we apply that function to the second line, it will return the closest match possible (from the groupid list)
        by filtering out those where systemid is different from p11 (on_col = systemid, on_value='p11')
        Args:
            query_index: index of the query you want to test
            on_col: filter in a column
            on_value: 

        Returns:
        index value
        '''
        if self.df.loc[query_index, self.idcol] == 0:
            self._deduplicate_row_(query_index)
        groupid = self.df.loc[query_index, self.idcol]
        filtered_index = self.df.loc[self.df[self.idcol] == groupid].index
        if len(filtered_index) > 1:
            a = self.df.loc[filtered_index]
            a = a.loc[a[on_col] == on_value]
            if a.shape[0] > 0:
                filtered_index = a.index
                tablescore = self.step_two_score(filtered_index)
                predictions = self.step_three_predict(tablescore)
                return predictions.sort_values(ascending=False).index[0]
            else:
                return None
        else:
            return None

    def build_training_table_from_row(self, query_index, drop_undecided=True):
        '''
        Returns a labelled dataframe for a row which has been verified
        The problem is the transitivity of the relations a ~ b and b ~ c but (a priori a !~ c)
        - same groupid, same query at the origin: True
        - same groupid, different query : undecided (dropped or not)
        - different groupid: False
        Args:
            query_index: row that has been verified
            drop_undecided (bool): if True, the undecided are dropped

        Returns:
            pd.DataFrame, traincols + 'ismatch'
        '''

        self.query = self.df.loc[query_index].copy()

        filtered_index = self.step_one_filter()

        tablescore = self.step_two_score(filtered_index)

        # we calculate a vector to chekc if the groupid match or not
        labelled_results = (self.df.loc[filtered_index, self.idcol] == self.query[self.idcol])

        if drop_undecided:
            # we filter out the lines where we are not sure (different query origin)
            z = (self.df.loc[labelled_results, self.queryidcol] != query_index)
            z = z.loc[z]
            labelled_results.drop(z.index, inplace=True)

        tablescore = tablescore.loc[labelled_results.index]
        tablescore['ismatch'] = labelled_results

        return tablescore

    def build_training_table_from_list(self, in_index, drop_undecided=True):
        '''
        return a training table to append
        Args:
            in_index (list): list of index to be mapped
            drop_undecided (bool): if True, the undecided are dropped

        Returns:
            pd.DataFrame, columns=self.traincols + 'ismatch'
        '''
        newdata = pd.DataFrame()
        for ix in in_index:
            tablerow = self.build_training_table_from_row(ix, drop_undecided=drop_undecided)
            if newdata.shape[0] == 0:
                newdata = tablerow
            else:
                newdata = pd.concat([newdata, tablerow], axis=0, ignore_index=True)
        return newdata

_training_table_filename_ = 'training_table_prepared_20170911_79319rows.csv'


def standard_application(df,
                         warmstart=False,
                         training_filename=_training_table_filename_):
    '''
    create, clean and fit the model to the given database
    :param df: database to be deduplicated
    :param warmstart: if database needs to be clean
    :param training_filename: name of the training table file
    :param idcol: name of the column where new ids are issued
    :param queryidcol: name of the column where the original query is stored
    :return: Suricate model ready to launch calculations
    '''
    training_table = pd.read_csv(training_filename, encoding='utf-8', sep='|')
    sur = Suricate(df=df,
                   warmstart=warmstart, )
    sur.model_add(training_set=training_table)
    return sur


def load_model(training_filename=_training_table_filename_):
    # TO DO add here a routine to print out time fitting the model
    training_set = pd.read_csv(training_filename, sep='|', encoding='utf-8')
    from sklearn.ensemble import RandomForestClassifier
    mymodel = RandomForestClassifier(n_estimators=2000)
    # Define training set and target vector
    traincols = list(filter(lambda x: x != 'ismatch', training_set.columns))
    X_train = training_set[traincols].fillna(-1)  # fill na values
    y_train = training_set['ismatch']
    mymodel.fit(X_train, y_train)
    return mymodel, traincols


# %%
if __name__ == '__main__':
    pass

