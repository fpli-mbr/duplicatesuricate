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
import suricatefunctions as surfunc


class Suricate:
    """
    A class that uses a pandas.DataFrame to store the data (self.df) and special methods to eliminate doublons
    """

    def __init__(self, df, warmstart=True, idcol='possiblematches',
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

    def _parallel_filter_(self, query_index, row_index):
        '''
        check if the row could be a potential match or Not.
        Args:
            query_index: index of the query
            row_index: index of the row

        Returns:
        boolean True if potential match, False otherwise
        '''

        # if we compare two exact same indexes
        if row_index == query_index:
            return True
        else:
            # else if some ID match
            for c in ['dunsnumber', 'taxid', 'registerid']:
                if self.df.loc[query_index, c] == self.df.loc[row_index, c]:
                    return True
            # if they are not the same country we reject
            if self.df.loc[query_index, 'country'] != self.df.loc[row_index, 'country']:
                return False
            else:
                for c in ['streetaddress', 'companyname']:
                    filterscore = surfunc.compare_twostrings(
                        self.df.loc[query_index, c], self.df.loc[row_index, c])
                    if filterscore is not None:
                        if filterscore >= self._filterthreshold_:
                            return True
        return False

    def _parallel_calculate_comparison_score_(self, query_index, row_index):
        '''
        Return the score vector between the query index and the row index
        :param query_index: 
        :param row_index: 
        :return: comparison score
        '''

        score = pd.Series()
        # fill the table with the features calculated
        for c in self.featurecols:
            # fill the table with the features specific to each row
            score[c + '_feat_row'] = self.df.loc[row_index, c].copy()
            # fill the table with the features of the query
            score[c + '_feat_query'] = self.df.loc[query_index, c]

        # calculate the various distances
        for c in ['companyname', 'companyname_wostopwords', 'companyname_acronym',
                  'streetaddress', 'streetaddress_wostopwords', 'cityname', 'postalcode']:
            score[c + '_fuzzyscore'] = surfunc.compare_twostrings(self.df.loc[row_index, c],
                                                                  self.df.loc[query_index, c])

        # token scores
        for c in ['companyname_wostopwords', 'streetaddress_wostopwords']:
            score[c + '_tokenscore'] = surfunc.compare_tokenized_strings(
                self.df.loc[row_index, c], self.df.loc[query_index, c], tokenthreshold=0.5, countthreshold=0.5)

        # acronym score
        c = 'companyname'
        score['companyname_acronym_tokenscore'] = surfunc.compare_acronyme(self.df.loc[query_index, c],
                                                                           self.df.loc[row_index, c])

        # exactscore
        for c in ['country', 'state', 'dunsnumber', 'postalcode_1stdigit', 'postalcode_2digits', 'taxid',
                  'registerid']:
            score[c + '_exactscore'] = surfunc.exactmatch(self.df.loc[query_index, c], self.df.loc[row_index, c])

        score = score.fillna(-1)
        return score

    def _parallel_predict_(self, comparison_score,return_proba=False):
        '''
        returns boolean if the comparison_score should be a match or not
        Args:
            comparison_score: score vector
            return_proba(boolean): Default False: if True, returns the probability of being a match.
                If False, returns a boolean (True if proba >= decision threshold, False otherwise)

        Returns:
        boolean True if it is a match False otherwise
        '''

        # check column length are adequate
        if len(self.traincols) != len(comparison_score.index):
            additionalcolumns = list(filter(lambda x: x not in self.traincols, comparison_score.index))
            if len(additionalcolumns) > 0:
                print('unknown columns in traincols', additionalcolumns)
            missingcolumns = list(filter(lambda x: x not in comparison_score.index, self.traincols))
            if len(missingcolumns) > 0:
                print('columns not found in scoring vector', missingcolumns)

        proba = self.model.predict_proba(comparison_score.values.reshape(1, -1))[0][1]

        if return_proba is True:
            return proba

        else:
            if proba >= self._decisionthreshold_:
                return True
            else:
                return False

    def _parallel_computation(self, query_index, row_index,return_proba = False):
        '''
        for each row for a query, returns True (ismatch) or False (is not a match)
        Args:
            query_index: index of the query
            row_index: index of the possible match
            return_proba(boolean): If True, returns the probability score

        Returns:
        boolean True if it is a match False otherwise
        '''

        if self._parallel_filter_(query_index, row_index) is False:
            return False
        else:
            comparisonscore = self._parallel_calculate_comparison_score_(query_index, row_index)
            ismatch = self._parallel_predict_(comparison_score=comparisonscore,return_proba=return_proba)
            return ismatch

    def clean_db(self):
        """
        Clean the database
        Returns: None

        """
        companystopwords = companystopwords_list
        streetstopwords = streetstopwords_list
        endingwords = endingwords_list

        self.df['Index'] = self.df.index

        # check if columns is in the existing database, other create a null one
        for c in [self.idcol, self.queryidcol, 'latlng', 'state']:
            if c not in self.df.columns:
                self.df[c] = None

        # normalize the strings
        for c in ['companyname', 'streetaddress', 'cityname']:
            self.df[c] = self.df[c].apply(surfunc.normalizechars)

        # remove bad possible matches
        self.df.loc[self.df[self.idcol] == 0, self.idcol] = np.nan

        # convert all duns number as strings with 9 chars
        self.df['dunsnumber'] = self.df['dunsnumber'].apply(lambda r: surfunc.convert_int_to_str(r, 9))

        def cleanduns(s):
            # remove bad duns like DE0000000
            if pd.isnull(s):
                return None
            else:
                s = str(s).rstrip('00000')
                if len(s) <= 5:
                    return None
                else:
                    return s

        self.df['dunsnumber'] = self.df['dunsnumber'].apply(cleanduns)

        # convert all postal codes to strings
        self.df['postalcode'] = self.df['postalcode'].apply(lambda r: surfunc.convert_int_to_str(r))

        # convert all taxid and registerid to string
        for c in ['taxid', 'registerid']:
            if c in self.df.columns:
                self.df[c] = self.df[c].astype(str).replace(surfunc.nadict)
            else:
                self.df[c] = None

        # remove stopwords from company names
        self.df['companyname_wostopwords'] = self.df['companyname'].apply(
            lambda r: surfunc.rmv_stopwords(r, stopwords=companystopwords))

        # create acronyms of company names
        self.df['companyname_acronym'] = self.df['companyname'].apply(surfunc.acronym)

        # remove stopwords from street addresses
        self.df['streetaddress_wostopwords'] = self.df['streetaddress'].apply(
            lambda r: surfunc.rmv_stopwords(r, stopwords=streetstopwords, endingwords=endingwords))

        # Calculate word use frequency in company names
        self.df['companyname_wostopwords_wordfrequency'] = surfunc.calculate_token_frequency(
            self.df['companyname_wostopwords'])

        # Take the first digits and the first two digits of the postal code
        self.df['postalcode_1stdigit'] = self.df['postalcode'].apply(
            lambda r: np.nan if pd.isnull(r) else str(r)[:1]
        )
        self.df['postalcode_2digits'] = self.df['postalcode'].apply(
            lambda r: np.nan if pd.isnull(r) else str(r)[:2]
        )

        # Calculate length of strings
        for c in ['companyname', 'companyname_wostopwords', 'streetaddress']:
            mycol = c + '_len'
            self.df[mycol] = self.df[c].apply(lambda r: None if pd.isnull(r) else len(r))
            max_length = self.df[mycol].max()
            self.df.loc[self.df[mycol].isnull() == False, mycol] = self.df.loc[self.df[
                                                                                   mycol].isnull() == False, mycol] / max_length

        # Calculate number of tokens in string
        for c in ['companyname_wostopwords']:
            mycol = c + '_ntokens'
            self.df[mycol] = self.df[c].apply(lambda r: None if pd.isnull(r) else len(r.split(' ')))
            max_value = self.df[mycol].max()
            self.df.loc[self.df[mycol].isnull() == False, mycol] = self.df.loc[self.df[
                                                                                   mycol].isnull() == False, mycol] / max_value

        # Calculate frequency of city used
        self.df['cityfrequency'] = surfunc.calculate_cat_frequency(self.df['cityname'])

        # Define the list of big cities
        bigcities = ['munich',
                     'paris',
                     'madrid',
                     'hamburg',
                     'toulouse',
                     'berlin',
                     'bremen',
                     'london',
                     'ulm',
                     'stuttgart', 'blagnac']
        self.df['isbigcity'] = self.df['cityname'].isin(bigcities).astype(int)

        # Define the list of big companies
        for c in ['thales', 'zodiac', 'ge', 'safran']:
            self.df['has_' + c] = self.df['companyname_wostopwords'].apply(
                lambda r: 0 if pd.isnull(r) else c in r).astype(int)

        # Define the list of airbus names and equivalents
        airbus_names = ['airbus', 'casa', 'eads', 'cassidian', 'astrium', 'eurocopter']
        self.df['has_airbusequiv'] = self.df['companyname_wostopwords'].apply(
            lambda r: 0 if pd.isnull(r) else any(w in r for w in airbus_names)).astype(
            int)

        return None

    def fitmodel(self, training_set, n_estimators=2000, used_model=None):
        """
        this function initiate and fits the model on the specified training table
        Args:
            training_set (pd.DataFrame): supervised learning training table, has ismatch column
            n_estimators(int): number of estimators used for standard RandomForestModel
            used_model (scikit-learn Model): model used to do the predicition, default RandomForest model


        Returns:

        """

        # define the model
        if used_model is None:
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(n_estimators=2000)
        else:
            self.model = used_model

        start = pd.datetime.now()
        if self.verbose:
            print('shape of training table ', training_set.shape)
            print('number of positives in table', training_set['ismatch'].sum())

        # Define training set and target vector
        self.traincols = list(filter(lambda x: x != 'ismatch', training_set.columns))
        X_train = training_set[self.traincols].fillna(-1)  # fill na values
        y_train = training_set['ismatch']

        # fit the model
        self.model.fit(X_train, y_train)

        score = self.model.score(X_train, y_train)

        if self.verbose:
            end = pd.datetime.now()
            duration = (end - start).total_seconds()
            print('time ellapsed', duration, 'seconds')
            print('score on training data', score)

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
        possiblechoices = x.loc[(x['groupid'] == 0) | (x['groupid'].isnull())].index
        if possiblechoices.shape[0] == 0:
            del x
            return None
        else:
            a = np.random.choice(possiblechoices)
            del x, possiblechoices
            return a

    def _update_idcol_(self, goodmatches_index, query_index):
        """
        attribute a new matching id to the group of matches
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
                print('No maximum number given or index given')
                return None

        print('deduplication started at ', pd.datetime.now())
        for countdown in range(nmax):
            query_index = self._generate_query_index_(in_index)
            if query_index is None:
                print('no valid query available')
                break
            else:
                start = pd.datetime.now()
                goodmatches_index = self._return_goodmatches_(query_index=query_index)
                n_deduplicated = self._update_idcol_(goodmatches_index, query_index)
                end = pd.datetime.now()
                duration = (end - start).total_seconds()
                print('record',query_index, 'countdownm',countdown, 'of', nmax, 'n_deduplicated', n_deduplicated, 'duration', duration)
        print('deduplication finished at ', pd.datetime.now())

        return None

    def _calculate_scored_features_(self, query_index):
        """
        return a dataframe filled with numerical values corresponding to the possible matches
        Args:
            query_index: index of the query
    
        Returns:
            pd.DataFrame
        """
        ##PS: It would be nice to transform that into a function that applies directly to a row

        # get the list of possible matches
        database_index = self._filter_database_(query_index=query_index)
        tablescore = pd.DataFrame(index=database_index)
        query = self.df.loc[query_index]

        # fill the table with the features calculated
        for c in self.featurecols:
            # fill the table with the features specific to each row
            tablescore[c + '_feat_row'] = self.df.loc[database_index, c].copy()
            # fill the table with the features of the query
            tablescore[c + '_feat_query'] = query.loc[c]
            tablescore[c + '_feat_diff'] = tablescore[c + '_feat_row'] - query.loc[c]

        # calculate the various distances
        for c in ['companyname', 'companyname_wostopwords', 'companyname_acronym',
                  'streetaddress', 'streetaddress_wostopwords', 'cityname', 'postalcode']:
            b = query.loc[c]
            tablescore[c + '_fuzzyscore'] = self.df[c].apply(lambda a: surfunc.compare_twostrings(a, b))
            del b

        for c in ['companyname_wostopwords', 'streetaddress_wostopwords']:
            b = query.loc[c]
            tablescore[c + '_tokenscore'] = self.df[c].apply(
                lambda a: surfunc.compare_tokenized_strings(a, b, tokenthreshold=0.5, countthreshold=0.5))
            del b

        b = query.loc['companyname']
        tablescore['companyname_acronym_tokenscore'] = self.df['companyname'].apply(
            lambda a: surfunc.compare_acronyme(a, b))
        del b

        b = query.loc['latlng']
        tablescore['latlng_geoscore'] = self.df['latlng'].apply(lambda a: surfunc.geodistance(a, b))
        del b

        for c in ['country', 'state', 'dunsnumber', 'postalcode_1stdigit', 'postalcode_2digits', 'taxid', 'registerid']:
            b = query.loc[c]
            tablescore[c + '_exactscore'] = self.df[c].apply(lambda a: surfunc.exactmatch(a, b))
            del b

        tablecols = tablescore.columns.tolist()
        missingcolumns = list(filter(lambda x: x not in self.traincols, tablecols))
        if len(missingcolumns) > 0:
            raise NameError(
                'Missing columns not found in calculated score but present in training table' + str(missingcolumns))
        newcolumns = list(filter(lambda x: x not in tablecols, self.traincols))
        if len(newcolumns) > 0:
            raise NameError(
                'New columns present in calculated score but not found in training table' + str(missingcolumns))

            # rearrange the columns according to the tablescore order
        tablescore = tablescore[self.traincols]

        return tablescore

    def _return_goodmatches_(self, query_index):
        """
        return the index of the positive matches of the query
        Args:
            query_index: index of the query
    
        Returns:
            pandas.Series().index: positive matches
    
        """
        ismatch_boolean = self.df['Index'].apply(lambda ix: self._parallel_computation(query_index, ix))
        goodmatches_index = ismatch_boolean.loc[ismatch_boolean].index

        return goodmatches_index

    def _generate_labelled_scored_table_(self, query_index):
        """
        This functions uses existing group id to manually label the scoring table to prepare the supervised learning
        Args:
            query_index: index of the query
    
        Returns:
            pandas.DataFrame: labelled score table
    
        """

        # get the id of the group of the query
        groupid = self.df.loc[query_index, self.idcol]
        if groupid == 0:
            return None
        else:
            # return the calculated table
            tablescore = self._calculate_scored_features_(query_index)
            # groupresults is the correct labelling of the tables
            groupresults = self.df.loc[tablescore.index, self.idcol]
            verifiedresults = (groupresults == groupid)  # verified results is a boolean
            tablescore['ismatch'] = verifiedresults
            return tablescore

    def showgroup(self, groupid, cols=None):
        '''
        show the results from the deduplication
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
        show the results from the deduplication
        Args:
            query_index (index): index on which to check possible matches
        Returns:
            pd.DataFrame
        '''

        y_proba = self.df['Index'].apply(lambda x:self._parallel_computation(query_index,x,return_proba=True))
        y_proba=y_proba.loc[y_proba>0]
        x = self.df.loc[y_proba.index, self.displaycols].copy()
        x['score'] = y_proba
        x.sort_values(by='score', ascending=False, inplace=True)
        return x

    def build_training_table_from_grouplist(self, verified_groups_list):
        """
        build a scoring table to fit the model using the labels already classified
        Args:
            verified_groups_list: list, list of groups ids that were verified
        Returns:
            pandas.DataFrame: labelled score table for all verified matches
        """
        start = pd.datetime.now()
        alldata = pd.DataFrame()
        possibleindex = self.df.loc[(
                                        self.df[self.idcol].isnull() == False) & (
                                        self.df[self.idcol].isin(verified_groups_list))].index
        print('# of verified lines', len(possibleindex))
        for ix in possibleindex:
            scoredtable = self._generate_labelled_scored_table_(ix)
            if alldata.shape[0] == 0:
                alldata = scoredtable
            else:
                alldata = pd.concat([alldata, scoredtable], axis=0)
        alldata['ismatch'] = alldata['ismatch'].astype(int)
        if self.verbose:
            end = pd.datetime.now()
            duration = (end - start).total_seconds()
            print('building time', duration, 'seconds')
            print('shape of training table', alldata.shape[0])
            print('time per verified line', int(duration / len(possibleindex)))
            percent_true = alldata['ismatch'].sum() / alldata.shape[0]
            print('percent true', '{:.0%}'.format(percent_true))
            print('mean # of lines filtered for each verified line', int(alldata.shape[0] / len(possibleindex)))
        return alldata

    def building_training_table_from_queryids(self, verified_queries):
        """
        build a scoring table to fit the model using the labels already classified
        Args:
            verified_queries: list, list of queries (index) that were verified
        Returns:
            pandas.DataFrame: labelled score table for all verified matches
        """
        alldata = pd.DataFrame()
        start = pd.datetime.now()
        for qix in verified_queries:
            if pd.isnull(self.df.loc[qix, 'groupid']):
                pass
            else:
                scoredtable = self._generate_labelled_scored_table_(qix)
                if alldata.shape[0] == 0:
                    alldata = scoredtable
                else:
                    alldata = pd.concat([alldata, scoredtable], axis=0)
        alldata['ismatch'] = alldata['ismatch'].astype(int)
        if self.verbose:
            end = pd.datetime.now()
            duration = (end - start).total_seconds()
            print('building time', duration, 'seconds')
            print('shape of training table', alldata.shape[0])
            percent_true = alldata['ismatch'].sum() / alldata.shape[0]
            print('percent true', '{:.0%}'.format(percent_true))

        return alldata

    def export_results_to_excel(self, filename, *args, **kwargs):
        self.df.sort_values(by=[
            self.idcol, 'companyname', 'dunsnumber', 'taxid', 'registerid', 'streetaddress'], inplace=True,
            ascending=True)

        self.df[self.displaycols].to_excel(filename, *args, **kwargs)

companystopwords_list = ['aerospace',
                         'ag',
                         'and',
                         'co',
                         'company',
                         'consulting',
                         'corporation',
                         'de',
                         'deutschland',
                         'dr',
                         'electronics',
                         'engineering',
                         'europe',
                         'formation',
                         'france',
                         'gmbh',
                         'group',
                         'hotel',
                         'inc',
                         'ingenierie',
                         'international',
                         'kg',
                         'la',
                         'limited',
                         'llc',
                         'ltd',
                         'ltda',
                         'management',
                         'of',
                         'oy',
                         'partners',
                         'restaurant',
                         'sa',
                         'sarl',
                         'sas',
                         'service',
                         'services',
                         'sl',
                         'software',
                         'solutions',
                         'srl',
                         'systems',
                         'technologies',
                         'technology',
                         'the',
                         'uk',
                         'und']
streetstopwords_list = ['avenue', 'calle', 'road', 'rue', 'str', 'strasse', 'strae']
endingwords_list = ['strasse', 'str', 'strae']
_training_table_filename_ = 'training_table_prepared_201708_69584rows.csv'


def standard_model(df,
                   warmstart=False,
                   training_filename=_training_table_filename_,
                   idcol='groupid',
                   queryidcol='queryid'):
    '''
    create, clean and fit the model to the given database
    :param df: database to be deduplicated
    :param warmstart: if database needs to be clean
    :param training_filename: name of the training table file
    :param logfilename: name of the training file
    :param idcol: name of the column where new ids are issued
    :param queryidcol: name of the column where the original query is stored
    :return: Suricate model ready to launch calculations
    '''
    training_table = pd.read_csv(training_filename, encoding='utf-8', sep='|')
    idcol = idcol
    queryidcol = queryidcol
    sur = Suricate(df=df,
                   warmstart=warmstart,
                   idcol=idcol,
                   queryidcol=queryidcol)
    sur.fitmodel(training_set=training_table)
    return sur


# %%
if __name__ == '__main__':
    pass
