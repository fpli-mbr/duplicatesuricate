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
import neatcleanstring as ncs
import suricatefunctions as surfunc


class Suricate:
    """
    A class that uses a pandas.DataFrame to store the data (self.df) and special methods to eliminate doublons
    """

    def __init__(self, df, warmstart=True, idcol='possiblematches',
                 queryidcol='queryid', verbose=True,log=None):
        """

        Args:
            df (pd.DataFrame): Input table for deduplication
            warmstart (bool): if True, does not apply the cleaning function
            idcol (str): name of the column where to store the deduplication results
            queryidcol (str): name of the column used to store the original match
            verbose (bool): Turns on or off prints
            log (pd.DataFrame): Log table used for deduplication
        """
        self.filename_log='log_table.csv'
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

        self.displaycols = [self.idcol,'companyname', 'dunsnumber', 'cityname', 'country', 'streetaddress']
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
        
        #create/load log table
        if log is None:
            self.logdf=pd.DataFrame(columns=['run_number','countdown',
            'timestamp_start','timestamp_end',
            'time_filtering_s','time_scoring_s','time_predicting_s','time_other_s',
            'query_index',
            'database_nrows','filtered_nrows','goodmatches_nrows',
            'deduplicated_nrows','matchid',
            'totalallocated_nrows','duration_s'])
            self.logrow=0
        else:
            self.logdf=log
            for c in ['timestamp_start','timestamp_end']:
                self.logdf[c]=pd.to_datetime(self.logdf[c])
            self.logrow=self.logdf.shape[0]+1
                                    
            

    def clean_db(self):
        """
        Clean the database
        Returns: None

        """
        companystopwords = companystopwords_list
        streetstopwords = streetstopwords_list
        endingwords=endingwords_list
        #check if columns is in the existing database, other create a null one
        for c in [self.idcol,self.queryidcol,'latlng','state']:
            if c not in self.df.columns:
                self.df[c]=None
          
        # normalize the strings
        for c in ['companyname', 'streetaddress', 'cityname']:
            self.df[c] = self.df[c].apply(ncs.normalizechars)

        # remove bad possible matches
        self.df.loc[self.df[self.idcol] == 0, self.idcol] = np.nan

        # convert all duns number as strings with 9 chars
        self.df['dunsnumber'] = self.df['dunsnumber'].apply(lambda r: ncs.convert_int_to_str(r, 9))
        def cleanduns(s):
            #remove bad duns like DE0000000
            if pd.isnull(s):
                return None
            else:
                s = str(s).rstrip('00000')
                if len(s)<=5:
                    return None
                else:
                    return s

        self.df['dunsnumber']=self.df['dunsnumber'].apply(cleanduns)
        
        # convert all postal codes to strings
        self.df['postalcode'] = self.df['postalcode'].apply(lambda r: ncs.convert_int_to_str(r))
        
        # remove stopwords from company names
        self.df['companyname_wostopwords'] = self.df['companyname'].apply(
            lambda r: ncs.rmv_stopwords(r, stopwords=companystopwords))

        # create acronyms of company names
        self.df['companyname_acronym'] = self.df['companyname'].apply(ncs.acronym)

        # remove stopwords from street addresses
        self.df['streetaddress_wostopwords'] = self.df['streetaddress'].apply(
            lambda r: ncs.rmv_stopwords(r, stopwords=streetstopwords, endingwords=endingwords))

        # Calculate word use frequency in company names
        self.df['companyname_wostopwords_wordfrequency'] = ncs.calculate_token_frequency(
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
        self.df['cityfrequency'] = ncs.calculate_cat_frequency(self.df['cityname'])

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

    def fitmodel(self, training_set,n_estimators=2000,used_model=None):
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

            #print('new match',goodmatches_matching_id)
            # take all lines who don't have a group id and give them the new id and save the index of the query
            nonallocatedmatches_index = self.df.loc[goodmatches_index].loc[
                self.df.loc[goodmatches_index, self.idcol].isnull()].index
            #print('len of non allocated matches',len(nonallocatedmatches_index))
            self.df.loc[nonallocatedmatches_index, self.idcol] = goodmatches_matching_id
            self.df.loc[nonallocatedmatches_index, self.queryidcol] = query_index
            
            #update log table information
            self.logdf.loc[self.logrow,'goodmatches_nrows']=len(goodmatches_index)
            self.logdf.loc[self.logrow,'deduplicated_nrows']=len(nonallocatedmatches_index)
            self.logdf.loc[self.logrow,'matchid']=goodmatches_matching_id
        else:
            #update log table information
            self.logdf.loc[self.logrow,'goodmatches_nrows']=0
            self.logdf.loc[self.logrow,'deduplicated_nrows']=None
            self.logdf.loc[self.logrow,'matchid']=None
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
        start=pd.datetime.now()
        ### filter search on country
        required_index = []
        query = self.df.loc[query_index]
        for c in ['country']:
            if c in query.index:
                if pd.isnull(query[c]) is False:
                    b = query[c]
                    temp_index = self.df.loc[(self.df[c].apply(lambda a: surfunc.exactmatch(a, b)) == 1)].index.tolist()
                    required_index = list(set(temp_index + required_index))
                    del temp_index

        # add fuzzy matching criterias based on the required index calculated above
        #I removed cityname to speed up the calculations
        df2 = self.df.loc[required_index]
        filtercols=['companyname', 'streetaddress']
        for c in filtercols:
            if c in query.index:
                if pd.isnull(query[c]) is False:
                    b = query[c]
                    df2[c+'_score']=df2[c].apply(lambda a: ncs.compare_twostrings(a, b))
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
                    temp_index = self.df.loc[(self.df[c].apply(lambda a: surfunc.exactmatch(a, b)) == 1)].index.tolist()
                    exact_index = list(set(temp_index + exact_index))

        all_index = list(set(all_index + exact_index))
        
        #update log table information
                #update the log file
        end=pd.datetime.now()
        self.logdf.loc[self.logrow,'time_filtering_s']=(end-start).total_seconds()
        self.logdf.loc[self.logrow,'filtered_nrows']=len(all_index)
        return all_index

    def launch_calculation(self, nmax, in_index=None):
        """
        launches the deduplication process
        Args:
            nmax (int): maximum number of occurences
            in_index (list): deduplicate the following index

        Returns:

        """

        print('deduplication started at ',pd.datetime.now())
        for countdown in range(nmax):
            query_index = self._generate_query_index_(in_index)
            if query_index is None:
                print('no valid query available')
                break
            else:
                #update log table information and display information
                self.logdf.loc[self.logrow,'countdown']=countdown
                self.logdf.loc[self.logrow,'query_index']=query_index
                start=pd.datetime.now()
                self.logdf.loc[self.logrow,'timestamp_start']=start
                self.logdf.loc[self.logrow,'database_nrows']=self.df.shape[0]
                goodmatches_index = self._return_goodmatches_(query_index=query_index)
                

                
                self._update_idcol_(goodmatches_index, query_index)
                
                #update log table information and display information
                end = pd.datetime.now()
                self.logdf.loc[self.logrow,'timestamp_end']=end
                self.logdf.loc[self.logrow,'totalallocated_nrows']=(self.df[self.idcol].isnull()==False).sum()
                duration = (end - start).total_seconds()
                self.logdf.loc[self.logrow,'duration_s']=duration
                self.logdf.loc[self.logrow,'time_other_s']=self.logdf.loc[self.logrow,'duration_s']-(
                        self.logdf.loc[self.logrow,'time_filtering_s']+self.logdf.loc[self.logrow,'time_scoring_s']+self.logdf.loc[self.logrow,'time_predicting_s'])        
                if self.verbose is True:
                    print('countdown',countdown+1,'of total ', nmax,' time ',duration,'n_deduplicated',self.logdf.loc[self.logrow,'deduplicated_nrows'])
                self.logrow+=1
                
        print('deduplication finished at ',pd.datetime.now())

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
            tablescore[c + '_fuzzyscore'] = self.df[c].apply(lambda a: ncs.compare_twostrings(a, b))
            del b
            
        for c in ['companyname_wostopwords', 'streetaddress_wostopwords']:
            b = query.loc[c]
            tablescore[c + '_tokenscore'] = self.df[c].apply(
                lambda a: ncs.compare_tokenized_strings(a, b, tokenthreshold=0.5, countthreshold=0.5))
            del b

        b = query.loc['companyname']
        tablescore['companyname_acronym_tokenscore'] = self.df['companyname'].apply(lambda a: surfunc.compare_acronyme(a, b))
        del b
        
        b = query.loc['latlng']
        tablescore['latlng_geoscore'] = self.df['latlng'].apply(lambda a: surfunc.geodistance(a, b))
        del b
        
        for c in ['country','state', 'dunsnumber','postalcode_1stdigit','postalcode_2digits']:
            b = query.loc[c]
            tablescore[c + '_exactscore'] = self.df[c].apply(lambda a: surfunc.exactmatch(a, b))
            del b
        tablecols=tablescore.columns.tolist()
        missingcolumns=list(filter(lambda x:x not in self.traincols,tablecols))
        if len(missingcolumns)>0:
            raise NameError('Missing columns not found in calculated score but present in training table'+str(missingcolumns))
        newcolumns=list(filter(lambda x:x not in tablecols,self.traincols))
        if len(newcolumns)>0:
            raise NameError('New columns present in calculated score but not found in training table'+str(missingcolumns))            
        
        # rearrange the columns according to the tablescore order
        tablescore=tablescore[self.traincols]
        
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
        start=pd.datetime.now()
        tablescore = self._calculate_scored_features_(query_index)
        #update the log file
        end=pd.datetime.now()
        self.logdf.loc[self.logrow,'time_scoring_s']=(end-start).total_seconds()
        #fillna values
        tablescore=tablescore.fillna(-1)
        
        start=pd.datetime.now()
        # Launche the model on it
        y_proba=pd.DataFrame(index=tablescore.index,data=self.model.predict_proba(tablescore))[1]
        y_pred = (y_proba>=self._decisionthreshold_)
        # Filter on positive matches
        goodmatches_index = y_pred.loc[y_pred].index
        #update the log file
        end=pd.datetime.now()
        self.logdf.loc[self.logrow,'time_predicting_s']=(end-start).total_seconds()
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
        
    def showgroup(self, groupid,cols=None):
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
        x = self.df.loc[self.df[self.idcol]==groupid,cols]
        return x
    
    def showpossiblematches(self, query_index):
        '''
        show the results from the deduplication
        Args:
            query_index (index): index on which to check possible matches
        Returns:
            pd.DataFrame
        '''
        tablescore = self._calculate_scored_features_(query_index)
        y_proba=pd.DataFrame(index=tablescore.index,data=self.model.predict_proba(tablescore))[1]
        x=self.df.loc[y_proba.index,self.displaycols].copy()
        x['score']=y_proba
        x.sort_values(by='score',inplace=True)
        return x
        
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
streetstopwords_list = ['avenue', 'calle', 'road', 'rue', 'str', 'strasse','strae']
endingwords_list = ['strasse', 'str', 'strae']
_training_table_filename_='training_table_prepared_201707_69584rows.csv'
#training_table = pd.read_csv(_training_table_filename_,index_col=0,encoding='utf-8',sep='|')

#%%
if __name__ == '__main__':
    pass
    filename_data='LFA1.csv'
    filename_training_table='training_table_prepared_201707_69584rows.csv'
    filename_out='LFA1_deduplicated.csv'
    
#    df=pd.read_csv(filename_out,index_col=0,sep='|',encoding='utf-8',error_bad_lines=False,nrows=10**4)
#        df.rename(columns={
#        'suppliername1':'companyname',
#        'street':'streetaddress',
#        'city':'cityname',
#    },
#             inplace=True)
    #df=df[[ 'dunsnumber', 'suppliername1',      'street', 'postalcode', 'city', 'country']]

#    idcol='groupid'
#    queryidcol='queryid'
#    df[idcol]=None
#    df[queryidcol]=None
#    training_table = pd.read_csv(filename_training_table,encoding='utf-8',sep='|',index_col=0,nrows=10**4)
#    trainingcols=training_table.columns.tolist()
#%%
if __name__ == '__main__':
    pass
#    sur = Suricate(df,warmstart=False,queryidcol=queryidcol,idcol=idcol)
#    sur.df.to_csv(filename_out,encoding='utf-8',sep='|')
#    sur.fitmodel(training_set=training_table)
#    sur.launch_calculation(nmax=40)
#    sur.df.sort_values(by=sur.idcol,inplace=True,ascending=True)
#    sur.df.to_csv(filename_out,encoding='utf-8',sep='|')

