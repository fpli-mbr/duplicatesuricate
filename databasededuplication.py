# coding=utf-8
""" deduplication methods
# How to use it
# init a DeduplicationDatabase instance
# init a DeduplicationModel instance
# Build a training set to fit the deduplication model
# Fit the deduplicationModel
"""
import pandas as pd
import numpy as np
import stringcleaningtools as sct
import groupcompaniesfunc as group


class DeduplicationDatabase:
    """
    A wrap around Pandas.DataFrame class, with special methods to eliminate doublons
    How to wrap around the pandas.DataFrame ???
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

        self.displaycols = ['companyname', 'dunsnumber', 'cityname', 'country', 'streetaddress','possiblematches']
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
        #scoring threshold for fuzzyscore filtering
        self._filterthreshold_ = 0.8
        #scoring threshold using model predict_proba
        self._decisionthreshold_=0.6

    def clean_db(self):
        """
        Clean the database
        Returns: None

        """
        companystopwords = companystopwords_list
        streetstopwords = streetstopwords_list
        endingwords=endingwords_list

        # normalize the strings
        for c in ['companyname', 'streetaddress', 'cityname']:
            self.df[c] = self.df[c].apply(sct.normalizechars)

        # remove bad possible matches
        self.df.loc[self.df['possiblematches'] == 0, 'possiblematches'] = np.nan

        # convert all duns number as strings with 9 chars
        self.df['dunsnumber'] = self.df['dunsnumber'].apply(lambda r: sct.convert_int_to_str(r, 9))

        # remove stopwords from company names
        self.df['companyname_wostopwords'] = self.df['companyname'].apply(
            lambda r: sct.rmv_stopwords(r, stopwords=companystopwords))

        # create acronyms of company names
        self.df['companyname_acronym'] = self.df['companyname'].apply(sct.acronym)

        # remove stopwords from street addresses
        self.df['streetaddress_wostopwords'] = self.df['streetaddress'].apply(
            lambda r: sct.rmv_stopwords(r, stopwords=streetstopwords, endingwords=endingwords))

        # Calculate word use frequency in company names
        self.df['companyname_wostopwords_wordfrequency'] = sct.calculate_token_frequency(
            self.df['companyname_wostopwords'])

        # Take the first digits and the first two digits of the postal code
        self.df['postalcode_1stdigit'] = self.df['postalcode'].apply(
            lambda r:np.nan if pd.isnull(r) else str(r)[:1]
        )
        self.df['postalcode_2digits'] = self.df['postalcode'].apply(
            lambda r:np.nan if pd.isnull(r) else str(r)[:2]
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
        self.df['cityfrequency'] = sct.calculate_cat_frequency(self.df['cityname'])

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
            lambda r: any(w in r for w in airbus_names)).astype(
            int)

        return None

    def fitmodel(self, used_model, training_set):
        """
        this function initiate and fits the model on the specified training table
        Args:
            used_model (scikit-learn Model): model used to do the predicition
            training_set (pd.DataFrame): supervised learning training table, has ismatch column

        Returns:

        """

        # define the model
        self.model = used_model

        start = pd.datetime.now()
        if self.verbose:
            print('shape of training table ', training_set.shape)
            print('number of positives in table', training_set['ismatch'].sum())

        # Define training set and target vector
        traincols = list(filter(lambda x: x != 'ismatch', training_set.columns))
        X_train = training_set[traincols].fillna(-1)  # fill na values
        y_train = training_set['ismatch']

        # fit the model
        self.model.fit(X_train, y_train)

        if self.verbose:
            print('model fitted')
            end = pd.datetime.now()
            duration = (end - start).total_seconds()
            print('time ellapsed', duration, 'seconds')

        if self.verbose:
            start = pd.datetime.now()

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
        # to check why idcol has gid in it
        possiblechoices = self.df.loc[(self.df[self.idcol].astype(float) > 0) == False].loc[in_index]
        if possiblechoices.shape[0] == 0:
            return None
        else:
            return possiblechoices.sample().index[0]

    def _update_idcol_(self, goodmatches_index, query_index):
        """
        attribute a new matching id to the group of matches
        Args:
            goodmatches_index(list): index of the group of matches
            query_index: index to be used to save the original id of the matching record

        Returns:

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

            print('new match',goodmatches_matching_id)
            # take all lines who don't have a group id and give them the new id and save the index of the query
            nonallocatedmatches_index = self.df.loc[goodmatches_index].loc[
                self.df.loc[goodmatches_index, self.idcol].isnull()].index
            print('len of non allocated matches',len(nonallocatedmatches_index))
            self.df.loc[nonallocatedmatches_index, self.idcol] = goodmatches_matching_id
            self.df.loc[nonallocatedmatches_index, self.queryidcol] = query_index
        else:
            pass
        return None

    def _filter_database_(self, query_index):
        """
        return a reduced index of the database
        Args:
            query_index: index of the query

        Returns:
            list, index filtered
        """

        ### filter search on country
        required_index = []
        query = self.df.loc[query_index]
        for c in ['country']:
            if c in query.index:
                if pd.isnull(query[c]) is False:
                    b = query[c]
                    temp_index = self.df.loc[(self.df[c].apply(lambda a: group.exactmatch(a, b)) == 1)].index.tolist()
                    required_index = list(set(temp_index + required_index))
                    del temp_index

        # add fuzzy matching criterias based on the required index calculated above
        df2 = self.df.loc[required_index]
        filtercols=['companyname', 'streetaddress', 'cityname']
        for c in filtercols:
            if c in query.index:
                if pd.isnull(query[c]) is False:
                    b = query[c]
                    df2[c+'_score']=df2[c].apply(lambda a: sct.compare_twostrings(a, b))
                else:
                    df2[c+'_score']=0
        scorecols=[c+'_score' for c in filtercols]
        df2['filter_score']=df2[scorecols].fillna(0).max(axis=1)
        temp_index = df2.loc[df2['filter_score']>=self._filterthreshold_].index.tolist()
        all_index=list(set([query_index]+temp_index))

        # add dunsnumber / code information where exact match is asked
        exact_index = []
        for c in ['dunsnumber']:
            if c in query.index:
                if pd.isnull(query[c]) is False:
                    b = query[c]
                    temp_index = self.df.loc[(self.df[c].apply(lambda a: group.exactmatch(a, b)) == 1)].index.tolist()
                    exact_index = list(set(temp_index + exact_index))

        all_index = list(set(all_index + exact_index))
        return all_index

    def launch_calculation(self, nmax, in_index=None):
        """
        launches the deduplication process
        Args:
            nmax (int): maximum number of occurences
            in_index (list): deduplicate the following index

        Returns:

        """

        for countdown in range(nmax):
            query_index = self._generate_query_index_(in_index)
            if query_index is None:
                print('no valid query available')
                break
            else:
                goodmatches_index = self._return_goodmatches_(query_index=query_index)
                self._update_idcol_(goodmatches_index, query_index)
        return None

    def _calculate_scored_features_(self, query_index):
        """
        return a dataframe filled with numerical values corresponding to the possible matches
        Args:
            query_index: index of the query

        Returns:
            pd.DataFrame
        """

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
            tablescore[c+'_feat_diff']=tablescore[c + '_feat_row']-query.loc[c]

        # calculate the various distances
        for c in ['companyname', 'companyname_wostopwords', 'companyname_acronym',
                  'streetaddress', 'streetaddress_wostopwords', 'cityname', 'postalcode']:
            b = query.loc[c]
            tablescore[c + '_fuzzyscore'] = self.df[c].apply(lambda a: sct.compare_twostrings(a, b))
        for c in ['companyname_wostopwords', 'streetaddress_wostopwords']:
            b = query.loc[c]
            tablescore[c + '_tokenscore'] = self.df[c].apply(
                lambda a: sct.compare_tokenized_strings(a, b, tokenthreshold=0.5, countthreshold=0.5))

        b = query.loc['companyname']
        tablescore['companyname_acronym_tokenscore'] = self.df['companyname'].apply(lambda a: group.compare_acronyme(a, b))

        b = query.loc['latlng']
        tablescore['latlng_geoscore'] = self.df['latlng'].apply(lambda a: group.geodistance(a, b))

        for c in ['country', 'state', 'dunsnumber','postalcode_1stdigit','postalcode_2digits']:
            b = query.loc[c]
            tablescore[c + '_exactscore'] = self.df[c].apply(lambda a: group.exactmatch(a, b))

        return tablescore

    def _return_goodmatches_(self, query_index):
        """
        return the index of the positive matches of the query
        Args:
            query_index: index of the query

        Returns:
            pandas.Series().index: positive matches

        """
        # Create the scoring table
        tablescore = self._calculate_scored_features_(query_index)
        #fillna values
        tablescore=tablescore.fillna(-1)
        # Launche the model on it
        y_proba=pd.DataFrame(index=tablescore.index,data=self.model.predict_proba(tablescore))[1]
        y_pred = (y_proba>=self._decisionthreshold_)
        # Filter on positive matches
        goodmatches_index = y_pred.loc[y_pred].index

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
            # return the calculatedtable
            tablescore = self._calculate_scored_features_(query_index)
            # groupresults is the correct labelling of the tables
            groupresults = self.df.loc[tablescore.index, self.idcol]
            verifiedresults = (groupresults == groupid)  # verified results is a boolean
            tablescore['ismatch'] = verifiedresults
            return tablescore

    def build_traing_table_for_supervised_learning(self, verified_groups_list):
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
                                        self.df['possiblematches'].isnull() == False) & (
                                        self.df['possiblematches'].isin(verified_groups_list))].index
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
            percent_true=alldata['ismatch'].sum()/alldata.shape[0]
            print('percent true','{:.0%}'.format(percent_true))
            print('mean # of lines filtered for each verified line',int(alldata.shape[0]/len(possibleindex)))
        return alldata


companystopwords_list=['aerospace',
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
streetstopwords_list = ['avenue', 'calle', 'road', 'rue', 'str', 'strasse']
endingwords_list = ['strasse', 'str', 'strae']
_training_table_filename_='training_table_prepared_201707_69584rows.csv'
training_table = pd.read_csv(_training_table_filename_,index_col=0,encoding='utf-8',sep='|')

if __name__ == '__main__':
    pass
    filename='gid1000_verified2000_samples.csv'
    df = pd.read_csv(filename, index_col=0, sep='|', encoding='utf-8')
    db = DeduplicationDatabase(df, warmstart=False)
    verified_samples = np.arange(1, 10)
    #training_table = db.build_traing_table_for_supervised_learning(verified_groups_list=verified_samples)
    #or training table is already loaded
    from sklearn.ensemble import RandomForestClassifier
    mymodel = RandomForestClassifier(n_estimators=2000)
    db.fitmodel(mymodel, training_table)
    db.launch_calculation(nmax=10)

