__version__ = '0.4.2'

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
from fuzzywuzzy.fuzz import ratio, token_set_ratio

from pyspark.sql.functions import udf,lit
from pyspark.sql.types import IntegerType, FloatType, StructType, StructField, StringType, BooleanType
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier as SparkRF
from pyspark.ml import Pipeline


class Suricate:
    def __init__(self, input_records,
                 target_records,
                 classifier,
                 filterdict=None,
                 intermediate_thresholds=None,
                 cleanfunc=None,
                 idcol='gid', queryidcol='queryid', decision_threshold=0.5, verbose=True, spark=False):
        """
        Main class used for deduplication
        Args:
            input_records (pd.DataFrame): Input table for record linkage, records to link
            target_records (pd.DataFrame): Table of reference for record linkage
            classifier: evaluation classifier, has a .predict_proba and a .used_cols function
            filterdict (dict): define the all/any logic used detailed in filter_all_any {'all':['country_code'],'any':['duns']}
            intermediate_thresholds(dict): add an intermediary filter {'name_fuzzyscore':0.8}
            cleanfunc: cleaning function used for the databases
            idcol (str): name of the column where to store the deduplication results
            queryidcol (str): name of the column used to store the original match
            verbose (bool): Turns on or off prints
        """
        if cleanfunc is None:
            cleanfunc = lambda x: x

        self.input_records = cleanfunc(input_records)
        self.target_records = cleanfunc(target_records)
        self.input_records.index.name='ix_source'
        self.target_records.index.mame='ix_target'

        self.linker = RecordLinker(df=self.target_records,
                                   filterdict=filterdict,
                                   intermediate_thresholds=intermediate_thresholds,
                                   classifier=classifier, decision_threshold=decision_threshold
                                   )

        missingcols = list(filter(lambda x: x not in self.input_records.columns, self.linker.compared_cols))
        if len(missingcols) > 0:
            raise KeyError('RecordLinker does not have all necessary columns in input after cleaning', missingcols)

        missingcols = list(filter(lambda x: x not in self.target_records.columns, self.linker.compared_cols))
        if len(missingcols) > 0:
            raise KeyError('RecordLinker does not have all necessary columns in target after cleaning', missingcols)

        self.idcol = idcol
        self.queryidcol = queryidcol
        self.verbose = verbose

        if spark is True:
            self.linker._prepare_spark()

        self._results = {}
        pass

    def _generate_query_index_(self, in_index=None):
        """
        this function returns a random index from the input records with no group id to start the linkage process
        Args:
            in_index (pd.Index): index or list, default None the query should be in the selected index

        Returns:
            object: an index of the input records
        """

        if in_index is None:
            in_index = self.input_records.index

        x = self.input_records.loc[in_index]
        possiblechoices = x.loc[(x[self.idcol] == 0) | (x[self.idcol].isnull())].index
        if possiblechoices.shape[0] == 0:
            del x
            return None
        else:
            a = np.random.choice(possiblechoices)
            del x, possiblechoices
            return a

    def _find_matches_(self, query_index, n_matches_max=1):
        """
       search for records in the target records that match the query (input_records.loc[query_index])
        Args:
            query_index: index of the row to be deduplicated
            n_matches_max (int): max number of matches to be fetched. If None, all matches would be returned

        Returns:
            pd.Index (list of index in the target records)
        """

        # return the good matches as calculated by the evaluation

        goodmatches_index = self.linker.return_good_matches(query=self.input_records.loc[query_index])

        if goodmatches_index is None or len(goodmatches_index) == 0:
            return None
        elif n_matches_max is None:
            return goodmatches_index
        else:
            return goodmatches_index[:n_matches_max]

    def start_linkage(self, sample_size=10, on_inputs=None, n_matches_max=1, with_proba=False):
        """
        Takes as input an index of the input records, and returns a dict showing their corresponding matches
        on the target records
        Args:
            on_inputs (pd.Index): index of the records (from input_records) to be deduplicated
            sample_size (int): number of records to be deduplicated. If 'all' is provided, deduplicaate all
            n_matches_max (int): maximum number of possible matches to be returned.
                If none, all matches would be returned
            with_proba (bool): whether or not to return the probabilities

        Returns:
            pd.DataFrame : results in the form of {index_of_input_record:[list of index_of_target_records]} or
                    {index_of_input_record:{index_of_target_records:proba of target records}}
        """
        if on_inputs is None and n_matches_max is None and with_proba is True:
            print(
                'careful, huge number of results (cartesian product) will be returned. Limit to the best probables matches with n_matches_,ax or set with_proba = False')
        if on_inputs is None:
            on_inputs = self.input_records.index

        if sample_size == 'all' or sample_size is None:
            if on_inputs is not None:
                n_total = len(on_inputs)
            else:
                n_total = self.input_records.shape[0]
        else:
            n_total = sample_size

        on_inputs = on_inputs[:n_total]

        print('starting deduplication at {}'.format(pd.datetime.now()))
        self._results = {}
        if with_proba is False:
            for i, ix in enumerate(on_inputs):
                # timing
                time_start = pd.datetime.now()

                goodmatches_index = self._find_matches_(query_index=ix, n_matches_max=n_matches_max)

                if goodmatches_index is None:
                    self._results[ix] = None
                    n_deduplicated = 0
                else:
                    self._results[ix] = list(goodmatches_index)
                    n_deduplicated = len(self._results[ix])

                # timing
                time_end = pd.datetime.now()
                duration = (time_end - time_start).total_seconds()

                if self.verbose:
                    print(
                        '{} of {} inputs records deduplicated | found {} of {} max possible matches | time elapsed {} s'.format(
                            i + 1, n_total, n_deduplicated, n_matches_max, duration))

            print('finished work at {}'.format(pd.datetime.now()))
        else:
            # return proba
            for i, ix in enumerate(on_inputs):
                # timing
                time_start = pd.datetime.now()
                # get the probability vector
                y_proba = self.linker.predict_proba(query=self.input_records.loc[ix])
                # if none sve none
                if y_proba is None or len(y_proba) == 0:
                    self._results[ix] = None
                    n_deduplicated = 0
                else:
                    # take the top n_matches_max to save
                    if n_matches_max is not None:
                        y_proba = y_proba.iloc[:n_matches_max]

                    self._results[ix] = y_proba.to_dict()
                    n_deduplicated = len(self._results[ix])

                # timing
                time_end = pd.datetime.now()
                duration = (time_end - time_start).total_seconds()

                if self.verbose:
                    print(
                        '{} of {} inputs records deduplicated | found {} of {} max possible matches | time elapsed {} s'.format(
                            i + 1, n_total, n_deduplicated, n_matches_max, duration))

            print('finished work at {}'.format(pd.datetime.now()))

        # Melt the results dictionnary to have the form:
        # df.columns = ['ix_source','ix_target'] if with_proba is false, ['ix_source','ix_target','y_proba' otherwise]
        results = self.unpack_results(self._results, with_proba=with_proba)

        return results

    def build_visualcomparison_table(self, inputs, targets, display=None, fuzzy=None, exact=None, y_true=None,
                                     y_proba=None):
        """
        Create a comparison table for visual inspection of the results
        Both input index and target index are expected to have the same length
        Args:
            inputs (pd.Index): list of input index to be displayed
            targets (pd.Index): list of target index to be displayed
            display (list): list of columns to be displayed (optional,default self.compared_colds)
            fuzzy (list): list of columns on which to perform fuzzy score (optional, default None)
            exact (list): list of columns on which to calculate the number of exact_matching (optional, default None)
            y_true (pd.Series): labelled values (0 or 1) to say if it's a match or not
            y_proba (pd.Series): probability vector
        Returns:
            pd.DataFrame ['ix_source','ix_target'],['display_source','display_target','fuzzy_source','fuzzy_target']

        """

        if display is None:
            display = self.linker.compared_cols
        if fuzzy is None:
            fuzzy = []
        if exact is None:
            exact = []

        allcols = list(set(display + fuzzy + exact))

        # take all values from source records
        res = self.input_records.loc[inputs, allcols].copy()
        res.columns = [c + '_source' for c in allcols]
        res['ix_source'] = inputs

        # take all values from target records
        x = self.target_records.loc[targets, allcols].copy()
        x.columns = [c + '_target' for c in allcols]
        x.index.name='ix_target'
        res['ix_target'] = targets
        res.set_index('ix_target', inplace=True, drop=True)
        res.index.name='ix_target'
        res = pd.concat([res,x],axis=1)
        del x
        res.reset_index(inplace=True, drop=False)

        # add the true and the probability vector (optional)
        if y_true is not None:
            res['y_true'] = np.array(y_true)
        if y_proba is not None:
            res['y_proba'] = np.array(y_proba)

        # use multiIndex
        res.set_index(['ix_source', 'ix_target'], inplace=True, drop=True)

        # Launch scoring
        if len(fuzzy)>0:
            df_fuzzy = pd.DataFrame(index=res.index)
            for c in fuzzy:
                df_fuzzy[c + '_fuzzyscore'] = res.apply(
                    lambda r: _fuzzyscore(r[c + '_source'], r[c + '_target']), axis=1)
            # after the loop, take the sum of the exact score (n ids matchings)
            if len(fuzzy)>1:
                df_fuzzy['avg_fuzzyscore'] = df_fuzzy.fillna(0).mean(axis=1)
            res = res.join(df_fuzzy)

        if len(exact)>0:
            df_exact = pd.DataFrame(index=res.index)
            for c in exact:
                df_exact[c + '_exactscore'] = res.apply(
                    lambda r: _exactmatch(r[c + '_source'], r[c + '_target']), axis=1)
            # after the loop, take the sum of the exact score (n ids matchings)
            if len(exact)>1:
                df_exact['n_exactmatches'] = df_exact.fillna(0).sum(axis=1)
            res = res.join(df_exact)

        # Sort the columns by order
        ordered = []
        for c in allcols:
            ordered.append(c + '_source')
            ordered.append(c + '_target')
            if c in fuzzy:
                ordered.append(c + '_fuzzyscore')
            elif c in exact:
                ordered.append(c + '_exactscore')
        missing_cols = sorted(list(filter(lambda x: x not in ordered, res.columns)))
        ordered += missing_cols

        res = res.reindex(ordered, axis=1)

        return res

    def build_training_table(self, inputs, targets, y_true=None, with_proba=True, scoredict=None, fillna=0):
        """
        Create a scoring table, with a label (y_true), for supervised learning
        inputs, targets,y_true are expected to be of the same length
        Args:
            inputs (pd.Index): list of index of the input dataframe
            targets (pd.Index): list of index of the target dataframe
            y_true (pd.Series): labelled values (0 or 1) to say if it's a match or not, optional
            with_proba (bool): gives the probability score calculated by the tool, optional
            scoredict (dict): dictionnary of scores you want to calculate, default self.scoredict
            fillna (float): float value

        Returns:
            pd.DataFrame index=['ix_source','ix_target'],colums=[scores....,'y_true','y_proba']
        """

        training_table_complete = pd.DataFrame(columns=self.linker.score_cols)
        for t, u in zip(inputs, targets):
            similarity_vector = self.linker.scoringmodel.build_similarity_table(query=self.input_records.loc[t],
                                                                                on_index=pd.Index([u]),
                                                                                scoredict=scoredict)
            similarity_vector['ix_source'] = t
            similarity_vector['ix_target'] = u
            training_table_complete = pd.concat([training_table_complete, similarity_vector], ignore_index=True, axis=0)

        # fillna
        training_table_complete.fillna(fillna, inplace=True)

        # calculate the probability vector
        if with_proba:
            X_train = training_table_complete[self.linker.score_cols]
            y_proba = self.linker.classifier.predict_proba(X_train)
            training_table_complete['y_proba'] = y_proba
        if y_true is not None:
            training_table_complete['y_true'] = y_true

        # set index
        training_table_complete.set_index(['ix_source', 'ix_target'], inplace=True)

        return training_table_complete

    def unpack_results(self, res, with_proba=False):
        """
        Transform the dictionary_like output from start_linkage into a pd.DataFrame
        Format the results dictionnary to have the form:
        df.columns = ['ix_source','ix_target'] if with_proba is false, ['ix_source','ix_target','y_proba' otherwise]
        Will drop ix_source with no matches
        Args:
            res (dict): results {'ix_source_1':['ix_target_2','ix_target_3']} / {'ix_source':{'ix_target1':0.9,'ix_target2':0.5}}
            with_proba (bool): if the result dictionnary contains a probability vector

        Returns:
            pd.DataFrame

        """
        if with_proba is False:
            df = pd.DataFrame(columns=['ix_source', 'ix_target'])
            for ix_source in list(res.keys()):
                matches = res.get(ix_source)
                if matches is not None:
                    ixs_target = pd.Series(data=matches)
                    ixs_target.name = 'ix_target'
                    temp = pd.DataFrame(ixs_target).reset_index(drop=True)
                    temp['ix_source'] = ix_source
                    df = pd.concat([df, temp], axis=0, ignore_index=True)
            df.reset_index(inplace=True, drop=True)
        else:
            df = pd.DataFrame(columns=['ix_source', 'ix_target', 'y_proba'])
            for ix_source in list(res.keys()):
                probas = res.get(ix_source)
                if probas is not None:
                    ixs_target = pd.Series(probas)
                    ixs_target.index.name = 'ix_target'
                    ixs_target.name = 'y_proba'
                    temp = pd.DataFrame(ixs_target).reset_index(drop=False)
                    temp['ix_source'] = ix_source
                    df = pd.concat([df, temp], axis=0, ignore_index=True)
            df.reset_index(inplace=True, drop=True)
        return df

    def build_combined_table(self, inputs, targets, with_proba=False, y_true=None):
        """
        Combine a side-by-side visual comparison table (build_visualcomparison_table)
        And a scoring table created by build_training_table
        Args:
            inputs (pd.Index): list of index of the input dataframe
            targets (pd.Index): list of index of the target dataframe
            y_true (pd.Series): labelled values (0 or 1) to say if it's a match or not
            with_proba (bool): gives the probability score calculated by the tool

        Returns:
            pd.DataFrame
        """
        visual_table = self.build_visualcomparison_table(inputs=inputs,
                                                         targets=targets)
        scored_table = self.build_training_table(inputs=inputs,
                                                 targets=targets,
                                                 with_proba=with_proba,
                                                 y_true=y_true)
        combined_table = visual_table.join(scored_table, rsuffix='_fromscoretable', how='left')
        return combined_table


class RecordLinker:
    def __init__(self,
                 df, filterdict=None,
                 intermediate_thresholds=None,
                 fillna=0,
                 classifier=None,
                 decision_threshold=0.5,
                 verbose=True):
        """
        This class merges together the Scorer and the Classifier model
        it creates a similarity table
        evaluate the probability of being a match with the model
        and then can either:
        - return that probability with predict_proba
        - return a boolean: probability is higher than a decision threshold with predict
        - or return the index of the good matches with return_good_matches

        Args:
            df (pd.DataFrame): target records
            filterdict (dict): filtering dict with exact matches on an {'all':['country'],'any':[id1,id2]}
            intermediate_thresholds (dict): dictionary of minimum thresholds for intermediate scoring {'name_fuzzyscore':0.6}
            fillna (float): value with which to fill na values
            classifier : Model used to calculate a probability vector. Has .predict_proba function and .used_cols attribute ['name_tokenscore','street_fuzzyscore']
            decision_threshold (float), default 0.8
            verbose (bool): control print output
        """

        self.verbose = verbose

        self.df = df
        self.df.index.name='ix_target'

        # initiate query to empty
        self.query = pd.Series()

        self.decision_threshold = decision_threshold

        self.classifier = classifier

        # set all compared cols to empty
        self.compared_cols = []
        self.score_cols = []
        self.filterdict = {}
        self.intermediate_score = {}
        self.further_score = {}

        # calculate compared cols based on parameters given to calculate comparison operations to perform and optimize code
        # Set the following parameters:
        # .filterdict, .intermediate_score, .further_score
        # .compared_cols, .score_cols

        self._calculate_scoredict(filterdict=filterdict,
                                  intermediatethreshold=intermediate_thresholds,
                                  decision_cols=self.classifier.used_cols)

        self.scoredict = _transform_scorecols_scoredict(self.score_cols)

        # configure the intermediate decision function
        # If threshold: threshold_based, if no : let all pass)
        if intermediate_thresholds is not None:
            decision_int_func = lambda r: _threshold_based_decision(row=r, thresholds=intermediate_thresholds)
        else:
            decision_int_func = lambda r: 1

        # create associate scoring and filtering model
        self.scoringmodel = Scorer(df=df,
                                   filterdict=self.filterdict,
                                   score_intermediate=self.intermediate_score,
                                   decision_intermediate=decision_int_func,
                                   score_further=self.further_score,
                                   fillna=fillna
                                   )

        missingcols = list(filter(lambda x: x not in self.scoringmodel.score_cols, self.classifier.used_cols))
        if len(missingcols) > 0:
            raise KeyError('not all training columns are found in the output of the scorer:', missingcols)

        self.sparkdf = None
        pass

    def _calculate_scoredict(self, filterdict, intermediatethreshold, decision_cols):
        """
        Set the following parameters:

        Dictionnaries of comparison operations to perform, and in which order
            self.filterdict: {'all': ['country_code'], 'any': ['duns']}
            self.intermediate_score {'fuzzy': ['name']}
            self.furtherscore : {'acronym': None, 'exact': None, 'fuzzy': ['street','name_wostopwords'],'token': ['name']}

        List of columns:
            self.compared_cols=['country_code','duns','name','street','name_wostopwords']
            self.score_cols=['country_code_exactscore','duns_exactscore','name_fuzzyscore','name_tokenscore','street_fuzzyscore','name_wostopwords_fuzzyscore']

        Args:
            filterdict (dict): {'all':['country'],'any':[id1,id2]}
            intermediatethreshold (dict): {'name_fuzzyscore':0.6}
            decision_cols (list): ['name_tokenscore','street_fuzzyscore','name_wostopwords_fuzzyscore']

        Returns:
            None
        """
        self.compared_cols = []
        self.score_cols = []

        if filterdict is not None:
            for c in ['all', 'any']:
                if filterdict.get(c) is None:
                    filterdict[c] = None
            self.filterdict = filterdict

            incols, outcols = _transform_scoredict_scorecols(self.filterdict)
            self.compared_cols += incols
            self.score_cols += outcols
        else:
            self.filterdict = None

        if intermediatethreshold is not None and len(intermediatethreshold) > 0:
            score_intermediate = _transform_scorecols_scoredict(existing_cols=self.score_cols,
                                                                used_cols=list(intermediatethreshold.keys()))
        else:
            score_intermediate = None

        if score_intermediate is not None:
            self.intermediate_score = score_intermediate
            incols, outcols = _transform_scoredict_scorecols(self.intermediate_score)
            self.compared_cols += incols
            self.score_cols += outcols
        else:
            self.intermediate_score = None

        if decision_cols is not None and len(decision_cols) > 0:
            score_further = _transform_scorecols_scoredict(existing_cols=self.score_cols, used_cols=decision_cols)
        else:
            score_further = None

        if score_further is not None:
            self.further_score = score_further
            incols, outcols = _transform_scoredict_scorecols(self.further_score)
            self.compared_cols += incols
            self.score_cols += outcols
        else:
            self.further_score = None

        self.compared_cols = list(set(self.compared_cols))
        self.score_cols = list(set(self.score_cols))
        pass

    def return_good_matches(self, query, decision_threshold=None, on_index=None, n_matches_max=1):
        """
        Return the good matches
        - with the help of the scoring model, create a similarity table
        - with the help of the evaluation model, evaluate the probability of it being a match
        - using the decision threshold, decides if it is a match or not
        - return the index of the good matches
        Args:
            query (pd.Series): information available on our query
            decision_threshold (float): decision_threshold between 0 and 1
            on_index (pd.Index): index on which to return the records
            n_matches_max (int): number of matches to be returned

        Returns:
            pd.Index: the index of the target records identified as the same as the query by the algorithm

        """
        if decision_threshold is None:
            decision_threshold = self.decision_threshold

        y_bool = self.predict(query, decision_threshold=decision_threshold, on_index=on_index)

        if y_bool is None:
            return None
        else:
            goodmatches = y_bool.loc[y_bool].index
            if len(goodmatches) == 0:
                return None
            else:
                goodmatches = goodmatches[:max(n_matches_max, len(goodmatches))]
            return goodmatches

    def predict(self, query, decision_threshold=None, on_index=None):
        """
        Predict if it is a match or not.
        - with the help of the scoring model, create a similarity table
        - with the help of the evaluation model, evaluate the probability of it being a match
        - using the decision threshold, decides if it is a match or not

        Args:
            query (pd.Series): information available on our query
            decision_threshold (float): default None. number between 0 and 1. If not provided, take the default one from the model
            on_index (pd.Index): index on which to do the prediction

        Returns:
            pd.Series: a boolean vector: True if it is a match, false otherwise

        """
        if decision_threshold is None:
            decision_threshold = self.decision_threshold

        # calculate the probability of the records being the same as the query through the machine learning evaluation
        y_proba = self.predict_proba(query, on_index=on_index)
        if y_proba is None:
            return None
        else:
            assert isinstance(y_proba, pd.Series)
            # transform that probability in a boolean via a decision threshold
            # noinspection PyTypeChecker
            y_bool = (y_proba > decision_threshold)
            assert isinstance(y_bool, pd.Series)

            return y_bool

    def predict_proba(self, query, on_index=None, return_filtered=True):
        """
        Main method of this class:
        - with the help of the scoring model, create a similarity table
        - with the help of the evaluation model, evaluate the probability of it being a match
        - returns that probability

        Args:
            query (pd.Series): information available on our query
            on_index (pd.Index): index on which to do the prediction
            return_filtered (bool): whether or not to filter the table
        Returns:
            pd.Series : the probability vector of the target records being the same as the query

        """

        table_score_complete = self.scoringmodel.filter_compare(query=query, on_index=on_index,
                                                                return_filtered=return_filtered)

        if table_score_complete is None or table_score_complete.shape[0] == 0:
            return None
        else:
            # launch prediction using the predict_proba of the scikit-learn module

            y_proba = self.classifier.predict_proba(table_score_complete).copy()

            del table_score_complete

            # sort the results
            y_proba.sort_values(ascending=False, inplace=True)

            return y_proba

    def _showprobablematches(self, query, n_records=10, display=None, return_filtered=True):
        """
        Show the best matching recors after the filter_all_any method of the scorer
        Could be of interest to investigate the possible matches of a query
        Args:
            query (pd.Series):
            n_records (int): max number of records to be displayed
            display (list): list of columns to be displayed
            return_filtered (bool): whether or not to filter the table

        Returns:
            pd.DataFrame, incl. query

        """
        if n_records is None:
            n_records = self.df.shape[0]

        if display is None:
            display = self.compared_cols
        records = pd.DataFrame(columns=['proba'] + display)
        records.loc[0] = query[display].copy()
        records.rename(index={0: 'query'}, inplace=True)
        records.loc['query', 'proba'] = 'query'

        y_proba = self.predict_proba(query, return_filtered=return_filtered)
        if y_proba is not None and y_proba.shape[0] > 0:
            y_proba.sort_values(ascending=False, inplace=True)
            n_records = min(n_records, y_proba.shape[0])
            results = self.df.loc[y_proba.index[:n_records], display]
            results['proba'] = y_proba
            records = pd.concat([records, results], axis=0)
            return records
        else:
            return None

    def _showfilterstep(self, query, n_records=10, display=None, return_filtered=True):
        """
        Not used anymore
        Show the best matching recors after the filter_all_any method of the scorer
        Could be of interest to investigate the possible matches of a query
        Args:
            query (pd.Series):
            n_records (int): max number of records to be displayed
            display (list): list of columns to be displayed
            return_filtered (bool): whether or not to filter the table

        Returns:
            pd.DataFrame, incl query

        """
        if n_records is None:
            n_records = self.df.shape[0]

        if display is None:
            display = self.compared_cols
        records = pd.DataFrame(columns=['totalscore'] + display)
        records.loc[0] = query[display].copy()
        records.rename(index={0: 'query'}, inplace=True)
        records.loc['query', 'totalscore'] = 'query'

        table = self.scoringmodel.filter_all_any(query=query, return_filtered=return_filtered)
        if table is not None and table.shape[0] > 0:
            y_sum = table.sum(axis=1)
            y_sum.sort_values(ascending=False, inplace=True)
            n_records = min(n_records, y_sum.shape[0])
            results = self.df.loc[y_sum.index[:n_records], display]
            results['totalscore'] = y_sum
            records = pd.concat([records, results], axis=0)
            return records
        else:
            return None

    def _showscoringstep(self, query, n_records=10, display=None, return_filtered=True):
        """
        Not used anymore
        Show the total score of the scoring table after the filter_compare method of the scorer
        Could be of interest to investigate the possible matches of a query
        Args:
            query (pd.Series):
            n_records (int): max number of records to be displayed
            display (list): list of columns to be displayed
            return_filtered (bool): whether or not to filter the table

        Returns:
            pd.DataFrame, incl query as the first row

        """
        if n_records is None:
            n_records = self.df.shape[0]

        if display is None:
            display = self.compared_cols
        records = pd.DataFrame(columns=['totalscore'] + display)
        records.loc[0] = query[display].copy()
        records.rename(index={0: 'query'}, inplace=True)
        records.loc['query', 'totalscore'] = 'query'

        table = self.scoringmodel.filter_compare(query=query, return_filtered=return_filtered)

        if table is not None and table.shape[0] > 0:
            y_sum = table.sum(axis=1)
            y_sum.sort_values(ascending=False, inplace=True)
            n_records = min(n_records, y_sum.shape[0])
            results = self.df.loc[y_sum.index[:n_records], display]
            results['totalscore'] = y_sum
            records = pd.concat([records, results], axis=0)

            return records
        else:
            return None

    def _showsimilarityvector(self, query, target_index, scoredict=None):
        """
        returns the similarity vector, or the scoring vector, between a query and a target
        Args:
            query (pd.Series): name and attribute of the query
            target_index (obj): index of the target record
            scoredict (dict): dictionnary of the scores, default None --> default one used

        Returns:
            pd.Series similarity vector used for the decision function
        """

        scoring_vector = self.scoringmodel.build_similarity_table(query=query,
                                                                  on_index=pd.Index([target_index]),
                                                                  scoredict=scoredict)

        return scoring_vector

    def _prepare_spark(self,sqlContext=None):
        """
        Initialize a .sparkdf attribute
        Args:
            sqlContext (pyspark.sql.context.SQLContext):

        Returns:
            None
        """
        if sqlContext is None:
            sqlContext=self.classifier.sqlContext

        ds = _transform_pandas_spark(sqlContext=sqlContext,df=self.df[self.compared_cols],drop_index=False)

        for c in self.compared_cols:
            ds = ds.withColumnRenamed(existing=c,new=c+"_target")

        self.sparkdf = ds
        return None

    def _compare_spark(self, query):
        """
        Initialize a .sparkdf attribute
        Args:
            query (pd.Series):

        Returns:
            pyspar.sql.dataframe.DataFrame

        """
        mytype = _sparktypedict[type(query.name)]
        ds = self.sparkdf.withColumn('ix_source', lit(query.name).cast(mytype))

        for c in self.compared_cols:
            mytype = _sparktypedict[type(query[c])]
            ds = ds.withColumn(c + '_source', lit(query[c]).cast(mytype))

        fuzzycols = self.scoredict.get('fuzzy')
        if fuzzycols is not None:
            for c in fuzzycols:
                ds = ds.withColumn(c + '_fuzzyscore',
                                                       _fuzzy_udf(ds[c + '_source'],
                                                                  ds[c + '_target']))
        tokencols = self.scoredict.get('token')
        if tokencols is not None:
            for c in tokencols:
                ds = ds.withColumn(c + '_tokenscore',
                                                       _token_udf(ds[c + '_source'],
                                                                  ds[c + '_target']))
        exactcols = self.scoredict.get('exact')
        if exactcols is not None:
            for c in exactcols:
                ds = ds.withColumn(c + '_exactscore',
                                                       _exact_udf(ds[c + '_source'],
                                                                  ds[c + '_target']))
        acronymcols = self.scoredict.get('acronym')
        if acronymcols is not None:
            for c in acronymcols:
                ds = ds.withColumn(c + '_acronymscore',
                                                       _acronym_udf(ds[c + '_source'],
                                                                    ds[c + '_target']))

        ## TODO: add attributes
        # think how to initialize form pandas dataframe and then transform it as float
        # attributescols=self.scoredict.get('attributes')
        # if attributescols is not None:
        #     for c in attributescols:
        #         self.sparkdf = self.sparkdf.withColumn(c+'_query',lit(self.query[c].c))

        usecols=['ix_source','ix_target']+self.score_cols

        ds= ds.select(usecols)
        return ds

    def _predict_proba_spark(self,query):
        ds=self._compare_spark(query=query)
        y_proba = self.classifier.predict_proba(ds, index_col='ix_target')
        return y_proba


def _threshold_based_decision(row, thresholds):
    """
    if any  (or all values) values of the row are above the thresholds, return 1, else return 0
    Args:
        row (pd.Series): row to be decided
        thresholds (dict): threshold of values {'aggfunc':'all' or 'any','name_fuzzyscore':0.6,'street_tokenscore':0.8
                    'fillna':0}

    Returns:
        float

    """
    #
    navalue = thresholds.get('fillna')
    if navalue is None:
        navalue = 0
    elif navalue == 'dropna':
        row = row.dropna()
    row = row.fillna(navalue)

    aggfunc = thresholds.get('aggfunc')

    if aggfunc == 'all':
        f = all
    elif aggfunc == 'any':
        f = any
    else:
        f = any
    scorekeys = thresholds.keys()
    scorekeys = filter(lambda k: k.endswith('score'), scorekeys)
    result = map(lambda k: row[k] >= thresholds[k], list(scorekeys))
    result = f(result)

    return result


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


def _fuzzyscore(a, b):
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


_fuzzy_udf = udf(lambda a, b: _fuzzyscore(a, b), FloatType())


def _tokenscore(a, b):
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


_token_udf = udf(lambda a, b: _tokenscore(a, b), FloatType())


def _exactmatch(a, b):
    if pd.isnull(a) or pd.isnull(b):
        return 0.0
    else:
        if a == b:
            return 1.0
        else:
            return -1.0


_exact_udf = udf(lambda a, b: _exactmatch(a, b), FloatType())


def _acronym(s):
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


def _compare_acronym(a, b, minaccrolength=3):
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
        a_acronyme = _acronym(a)
        b_acronyme = _acronym(b)
        if min(len(a_acronyme), len(b_acronyme)) >= minaccrolength:
            a_score_acronyme = _tokenscore(a_acronyme, b)
            b_score_acronyme = _tokenscore(a, b_acronyme)
            if all(pd.isnull([a_score_acronyme, b_score_acronyme])):
                return 0.0
            else:
                max_score = np.max([a_score_acronyme, b_score_acronyme])
                return max_score
        else:
            return 0.0


_acronym_udf = udf(lambda a, b: _compare_acronym(a, b), FloatType())

_scorename = {'fuzzy': '_fuzzyscore',
             'token': '_tokenscore',
             'exact': '_exactscore',
             'acronym': '_acronymscore'}

_scorefuncs = {'fuzzy': _fuzzyscore,
              'token': _tokenscore,
              'exact': _exactmatch,
              'acronym': _compare_acronym}
_scoringkeys = list(_scorename.keys())


class Scorer:
    def __init__(self, df, filterdict=None, score_intermediate=None, decision_intermediate=None, score_further=None,
                 fillna=0):
        """
        This class is used to calculate similarity tables between a reference table and a possible query.
        It has three main steps in proceeding:
        - filter based on an all / any logic (filter_all_any)
        - calculate an intermediate score using  build_similarity_table
        - takes an intermediate decision function
        - if the decision function is positive, calculate further scores using  build_similarity_table
        - those three steps are meshed together in the filter_compare method
        Args:
            df (pd.DataFrame): reference records
            filterdict (dict): define the all/any logic used detailed in filter_all_any
            score_intermediate (dict): create the intermediate scoring table using a scoredict detailed in _unpackscoredict
            decision_intermediate (func): take a decision: function takes as input a row of the scoring table and returns a boolean
            score_further (dict): creates the additional scoring fields using a scoredict
            fillna (float): Value used to fill the na values
        Examples:
            filterdict : {'all':
        """

        self.df = df

        self.score_cols = []
        self.compared_cols = []

        ###
        if filterdict is not None:
            for c in ['all', 'any']:
                if filterdict.get(c) is None:
                    filterdict[c] = None
            self.filterdict = filterdict

            incols, outcols = _transform_scoredict_scorecols(self.filterdict)
            self.compared_cols += incols
            self.score_cols += outcols
        else:
            self.filterdict = None

        if score_intermediate is not None:
            self.intermediate_score = score_intermediate
            incols, outcols = _transform_scoredict_scorecols(self.intermediate_score)
            self.compared_cols += incols
            self.score_cols += outcols
        else:
            self.intermediate_score = None

        if score_further is not None:
            self.further_score = score_further
            incols, outcols = _transform_scoredict_scorecols(self.further_score)
            self.compared_cols += incols
            self.score_cols += outcols
        else:
            self.further_score = None

        ####

        self.total_scoredict = _transform_scorecols_scoredict(used_cols=self.score_cols)

        self.intermediate_func = decision_intermediate

        self.navalue = fillna

        self.input_records = pd.DataFrame()

        pass

    def filter_all_any(self, query, on_index=None, filterdict=None, return_filtered=True):
        """
        returns a pre-filtered table score calculated on the column names provided in the filterdict.
        in the values for 'any': an exact match on any of these columns ensure the row is kept for further analysis
        in the values for 'all': an exact match on all of these columns ensure the row is kept for further analysis
        if the row does not have any exact match for the 'any' columns, or if it has one bad match for the 'all' columns,
        it is filtered out
        MODIF: if return_filtered, this will not filter the table at all but just returns the scores
        Args:
            query (pd.Series): query
            on_index (pd.Index): index
            filterdict(dict): dictionnary two lists of values: 'any' and 'all' {'all':['country_code'],'any':['duns','taxid']}
            return_filtered (bool): whether or not to filter after calculation of the first scores (Not filtering used for deep inspections of the results)

        Returns:
            pd.DataFrame: a DataFrame with the exact score of the columns provided in the filterdict

        Examples:
            table = ['country_code_exactscore','duns_exactscore']
        """
        # create repository for the score
        table = pd.DataFrame(index=on_index)

        # Tackle the case where no index is given: use the whole index available
        if on_index is None:
            on_index = self.df.index

        # if no specific filterdict is given use the one from init
        if filterdict is None:
            filterdict = self.filterdict

        # if no filter dict is given returns an empty table with all of the rows selected: no filterdict has been applied!
        if filterdict is None:
            return table

        match_any_cols = filterdict.get('any')
        match_all_cols = filterdict.get('all')

        # same as comment above
        if match_all_cols is None and match_any_cols is None:
            return table

        df = self.df.loc[on_index]

        # perform the "any criterias match" logic
        if match_any_cols is not None:
            match_any_df = pd.DataFrame(index=on_index)
            for c in match_any_cols:
                match_any_df[c + '_exactscore'] = df[c].apply(
                    lambda r: _exactmatch(r, query[c]))
            y = (match_any_df == 1)
            assert isinstance(y, pd.DataFrame)

            anycriteriasmatch = y.any(axis=1)
            table = pd.concat([table, match_any_df], axis=1)
        else:
            anycriteriasmatch = pd.Series(index=on_index).fillna(False)

        # perform the "all criterias match" logic
        if match_all_cols is not None:
            match_all_df = pd.DataFrame(index=on_index)
            for c in match_all_cols:
                match_all_df[c + '_exactscore'] = df[c].apply(
                    lambda r: _exactmatch(r, query[c]))
            y = (match_all_df == 1)
            assert isinstance(y, pd.DataFrame)
            allcriteriasmatch = y.all(axis=1)

            table = pd.concat([table, match_all_df], axis=1)
        else:
            allcriteriasmatch = pd.Series(index=on_index).fillna(False)

        # perform the all criterias match OR at least one criteria match logic
        results = (allcriteriasmatch | anycriteriasmatch)

        assert isinstance(table, pd.DataFrame)

        if return_filtered is True:
            table = table.loc[results]

        return table

    def build_similarity_table(self,
                               query,
                               on_index=None,
                               scoredict=None):
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

        if on_index is None:
            on_index = self.df.index

        if scoredict is None:
            scoredict = self.total_scoredict

        table_score = pd.DataFrame(index=on_index)

        attributes_cols = scoredict.get('attributes')
        if attributes_cols is not None:
            for c in attributes_cols:
                table_score[c + '_source'] = query[c]
                table_score[c + '_target'] = self.df.loc[on_index, c]

        for c in _scoringkeys:
            table = self._compare(query, on_index=on_index, on_cols=scoredict.get(c), func=_scorefuncs[c],
                                  suffix=_scorename[c])
            table_score = pd.concat([table_score, table], axis=1)

        return table_score

    def filter_compare(self, query, on_index=None, return_filtered=True):
        """
        Simultaneously create a similarity table and filter the data.
        It works in three steps:
        - filter with a logic (exact match on any of these cols OR exact match on all of these columns)
        - intermediate score with dedicated comparison methods on selected columns
        - filter with an intermediate decision function
        - further score with dedicated comparison methods on selected columns
        - returns the final similarity table which is the concatenation of all of the scoring functions above on the rows
            that have been filtered
        MODIF : if return_filtered parameter is set to False, then it will not filter the data.

        Args:
            query (pd.Series): query
            on_index (pd.Index): index on which to filter and compare
            return_filtered (bool): whether or not to filter after calculation of the first scores

        Returns:
            pd.DataFrame similarity table
        """

        # pre filter the records for further scoring based on an all / any exact match
        if on_index is None:
            workingindex = self.df.index
        else:
            workingindex = on_index

        table_score_complete = self.filter_all_any(query=query,
                                                   on_index=workingindex,
                                                   filterdict=self.filterdict,
                                                   return_filtered=return_filtered
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
            y_intermediate = y_intermediate.astype(bool)

            assert isinstance(y_intermediate, pd.Series)
            assert (y_intermediate.dtype == bool)

            if return_filtered is True:
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
                duplicatecols = list(filter(lambda x: x in table_score_complete.columns, table_additional.columns))
                if len(duplicatecols) > 0:
                    table_additional.drop(duplicatecols, axis=1, inplace=True)

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
                table[colname] = self.df.loc[on_index, c].apply(lambda r: func(r, query[c]))
        return table

    def compare_nofilter(self, query, on_index):
        """
        Calculate the score, without filtering
        Only advised in order to calculate a training table, when on_index is a limited scope of the data
        Args:
            query (pd.Series):
            on_index (pd.Index):

        Returns:
            pd.DataFrame
        """
        table_score_complete = self.filter_compare(query=query,
                                                   on_index=on_index,
                                                   return_filtered=False)
        return table_score_complete


def _transform_scoredict_scorecols(scoredict):
    """
    Calculate, from the scoredict, two lists:
    - the list of the names of columns on which the scoring is performed compared_cols,used_cols
    - the list of the names of the scoring columns

    The names of the keys can be : 'all','any'
    - 'all','any': used only in the filter_all_any method
    - 'attributes':
    - 'fuzzy','token','exact','acronym': four kinds of comparison.
    Args:
        scoredict (dict): of the type {'fuzzy':['name','street'],'exact':['id'],'token':None}.\
        Should be of the form key:[list] or key:None.

    Returns:
        list,list : compared_cols, used_cols
    Examples:
        _unpack_scoredict({'fuzzy':['name','street'],'exact':['id'],'token':None,'attributes':['name_len'],
        all=['id','id2']}):
        returns ['name','street','id','name_len','id2'],['name_fuzzyscore','street_fuzzyscore','id_exactscore','id2_exactscore','name_len_query','name_len_row']
    """

    outputcols = []
    inputcols = []

    for d in ['all', 'any']:
        if scoredict.get(d) is not None:
            for c in scoredict[d]:
                inputcols.append(c)
                outputcols.append(c + '_exactscore')
    if scoredict.get('attributes') is not None:
        for c in scoredict['attributes']:
            inputcols.append(c)
            outputcols.append(c + '_source')
            outputcols.append(c + '_target')
    for k in _scorename.keys():
        if scoredict.get(k) is not None:
            for c in scoredict[k]:
                inputcols.append(c)
                outputcols.append(c + _scorename[k])
    return inputcols, outputcols


def _transform_scorecols_scoredict(used_cols, existing_cols=None):
    """
    From a set of existing comparison columns and columns needed for a decision function,
    calculate the scoring dict that is needed for the scorer to calculate all the needed columns.
    Args:
        existing_cols (list): list of existing columns that are already calculated.
        used_cols (list): list of columns needed for the decision function.

    Returns:
        dict: scoredict-type
    Examples:
        calculatescoredict(['name_fuzzyscore','id_exactscore'],existing_cols=['name_fuzzyscore'])
        returns {'exact':['id']}

        calculatescoredict(['name_fuzzyscore'])
        returns {'fuzzy':['name']}
    """
    if existing_cols is None:
        existing_cols = []

    x_col = list(filter(lambda x: x not in existing_cols, used_cols))
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
        return m_dic
    else:
        return None


class RuleBasedClassifier:
    """
    This evaluation model applies a hard-coded evaluation function to return a probability vector
    Examples:
        decisionfunc = lambda r:r[id_cols].mean()
        dm = FuncEvaluationModel(used_cols=id_cols,eval_func=decisionfunc)
        x_score = compare(query,target_records)
        y_proba = dm.predict_proba(x_score)
    """

    def __init__(self, used_cols, eval_func=None):
        """
        Create the model
        Args:
            used_cols (list): list of columns necessary for decision
            eval_func (func): evaluation function to be applied. must return a probability vector
        """
        self.used_cols = used_cols
        if eval_func is None:
            self.eval_func = lambda r: sum(r) / len(r)
        self.scoredict = _transform_scorecols_scoredict(self.used_cols)
        self.compared_cols = _transform_scoredict_scorecols(self.scoredict)[0]
        pass

    def fit(self):
        """
        pass
        Returns:
            None
        """
        pass

    @classmethod
    def from_dict(cls, scoredict, evalfunc=None):
        """
        Args:
            scoredict (dict): scoretype_dictionnary
            evalfunc (None): evaluation function, default sum

        Returns:
            RuleBasedClassifier

        Examples:
            scoredict={'attributes':['name_len'],
                        'fuzzy':['name','street']
                        'token':'name',
                        'exact':'id'
                        'acronym':'name'}
        """
        compared_cols, used_cols = _transform_scoredict_scorecols(scoredict)
        x = RuleBasedClassifier(used_cols=used_cols, eval_func=evalfunc)
        return x

    def predict_proba(self, x_score):
        """
        This is the evaluation function.
        It takes as input a DataFrame with each row being the similarity score between the query and the target records.
        It returns a series with the probability vector that the target records is the same as the query.
        The scoring tables column names must fit the columns used for the model
        If x_score is None or has no rows it returns None.
        Args:
            x_score (pd.DataFrame):the table containing the scoring records

        Returns:
            pd.Series : the probability vector of the target records being the same as the query
        """
        missing_cols = list(filter(lambda x: x not in x_score.columns, self.used_cols))
        if len(missing_cols) > 0:
            raise KeyError('not all training columns are found in the output of the scorer:', missing_cols)
        x_score = x_score[self.used_cols]

        y_proba = x_score.apply(lambda r: self.eval_func(r), axis=1)
        y_proba.name = 1

        return y_proba


class DummyClassifier:
    def __init__(self, scoredict):
        """
        Create a model used only for scoring (for example for creating training data)
        used_cols (list): list of columns necessary for decision
        eval_func (func): evaluation function to be applied. must return a probability vector
        Args:
            scoredict (dict): {'fuzzy':['name','street'],'token':['name_wostopwords'],'acronym':None}
        """
        self.scoredict = scoredict
        compared_cols, used_cols = _transform_scoredict_scorecols(scoredict)
        self.used_cols = used_cols
        self.compared_cols = compared_cols
        pass
    def fit(self,X,y):
        """
        Do nothing
        Args:
            X:
            y:

        Returns:

        """
        pass
    def predict_proba(self,X):
        """
        A dart-throwing chump generates a random probability vector for the sake of coherency with other classifier
        Args:
            X:

        Returns:
            pd.Series
        """
        y_proba = np.random.random(size=X.shape[0])
        y_proba=pd.Series(y_proba,index=X.shape[0])
        return y_proba

class ScikitLearnClassifier:
    """
    The evaluation model is based on machine learning, it is an implementation of the Random Forest algorithm.
    It requires to be fitted on a training table before making decision.

    Examples:
        dm = MLEvaluationModel()
        dm.fit(x_train,y_train)
        x_score = compare(query,target_records) where compare creates a similarity table
        y_proba = dm.predict_proba(x_score)
    """

    def __init__(self, verbose=True,
                 n_estimators=2000, model=None):
        """
        Create the model
        Args:
            verbose (bool): control print output
            n_estimators (int): number of estimators for the Random Forest Algorithm
            model: sklearn classifier model, default RandomForrest
        """
        self.verbose = verbose
        if model is None:
            self.model = RandomForestClassifier(n_estimators=n_estimators)
        else:
            self.model = model
        self.used_cols = []

        pass

    def fit(self, X, y):
        """
        fit the machine learning evaluation model on the provided data set.
        It takes as input a training table with numeric values calculated from previous examples.
        Args:
            X (pd.DataFrame): pandas DataFrame containing annotated data
            y (pd.Series):name of the target vector in the training_set

        Returns:
            None

        """

        self.used_cols = X.columns

        start = pd.datetime.now()

        if self.verbose:
            print('shape of training table ', X.shape)
            print('number of positives in table', y.sum())

        # fit the classifier
        self.model.fit(X, y)

        if self.verbose:
            # show precision and recall score of the classifier on training data
            y_pred = self.model.predict(X)
            precision = precision_score(y_true=y, y_pred=y_pred)
            recall = recall_score(y_true=y, y_pred=y_pred)
            print('precision score on training data:', precision)
            print('recall score on training data:', recall)

        if self.verbose:
            end = pd.datetime.now()
            duration = (end - start).total_seconds()
            print('time elapsed', duration, 'seconds')

        return None

    def predict_proba(self, x_score):
        """
        This is the evaluation function.
        It takes as input a DataFrame with each row being the similarity score between the query and the target records.
        It returns a series with the probability vector that the target records is the same as the query.
        The scoring table must not have na values.
        The scoring tables column names must fit the training table column names. (accessible via self.decisioncols).
        If x_score is None or has no rows it returns None.
        Args:
            x_score (pd.DataFrame): the table containing the scoring records

        Returns:
            pd.Series : the probability vector of the target records being the same as the query

        """
        if x_score is None or x_score.shape[0] == 0:
            return None
        else:
            missing_cols = list(filter(lambda x: x not in x_score.columns, self.used_cols))
            if len(missing_cols) > 0:
                raise KeyError('not all training columns are found in the output of the scorer:', missing_cols)

            # re-arrange the column order
            x_score = x_score[self.used_cols]

            # launch prediction using the predict_proba of the scikit-learn module
            y_proba = \
                pd.DataFrame(self.model.predict_proba(x_score), index=x_score.index)[1]
            assert isinstance(y_proba, pd.Series)
            return y_proba

class SparkClassifier():
    """
    The evaluation model is based on spark-powered machine learning, it is an implementation of the Random Forest algorithm.
    It requires to be fitted on a training table before making decision.

    Examples:
        dm = SparkMLEvaluationModel()
        dm.fit(Xytrain)
        y_proba = dm.predict_proba(x_score)
    """

    def __init__(self, sqlContext, verbose=True ):
        """
        Create the model
        Args:
            sqlContext (pyspark.sql.context.SQLContext):
            verbose (bool): control print output
        """
        self.verbose = verbose
        self.sqlContext = sqlContext
        self.used_cols = list()

        pass

    def fit(self, X, y):
        """
        fit the machine learning evaluation model on the provided data set.
        It takes as input a training table with numeric values calculated from previous examples.
        Args:
            X (pd.DataFrame): pandas DataFrame containing annotated data
            y (pd.Series):name of the target vector in the training_set

        Returns:
            None

        """
        start = pd.datetime.now()

        if self.verbose:
            print('shape of training table ', X.shape)
            print('number of positives in table', y.sum())


        self.used_cols = X.columns.tolist()


        # Format pandas DataFrame for use in spark, including types
        X=X.astype(float)
        assert isinstance(X,pd.DataFrame)
        X['y_train']=y
        X['y_train']=X['y_train'].astype(int)

        Xs = _transform_pandas_spark(self.sqlContext,df=X,drop_index=True)

        # Create the pipeline

        assembler = VectorAssembler(inputCols=list(self.used_cols),outputCol="features")
        labelIndexer = StringIndexer(inputCol="y_train", outputCol="label")
        rf_classifier = SparkRF(labelCol=labelIndexer.getOutputCol(), featuresCol=assembler.getOutputCol())
        pipeline=Pipeline(stages=[assembler,labelIndexer,rf_classifier])

        # fit the classifier
        self.pipeline_model = pipeline.fit(Xs)

        if self.verbose:
            # show precision and recall score of the classifier on training data
            y_pred = self.predict_proba(Xs,index_col=None)
            y_pred= (y_pred > 0.5)
            assert isinstance(y_pred,pd.Series)
            precision = precision_score(y_true=y, y_pred=y_pred)
            recall = recall_score(y_true=y, y_pred=y_pred)
            print('precision score on training data:', precision)
            print('recall score on training data:', recall)
        #
        if self.verbose:
            end = pd.datetime.now()
            duration = (end - start).total_seconds()
            print('time elapsed', duration, 'seconds')
        pass

    def _predict(self,X):
        """
        Args:
            X (pyspark.sql.dataframe.DataFrame):

        Returns:
            pyspark.sql.dataframe.DataFrame
        """

        x_pred=self.pipeline_model.transform(X)
        proba_udf = udf(lambda r: float(r[1]), FloatType())
        x_pred = x_pred.withColumn('y_proba', proba_udf(x_pred["probability"]))
        return x_pred


    def predict_proba(self, X,index_col=None):
        """
        This is the evaluation function.
        It takes as input a DataFrame with each row being the similarity score between the query and the target records.
        It returns a series with the probability vector that the target records is the same as the query.
        The scoring table must not have na values.
        The scoring tables column names must fit the training table column names. (accessible via self.decisioncols).
        If x_score is None or has no rows it returns None.
        Args:
            X (pyspark.sql.dataframe.DataFrame): the table containing the scoring records
            index_col (str): name, if any, of the column containing the index in the dataframe

        Returns:
            pd.Series : the probability vector of the target records being the same as the query

        """
        if type(X) == pd.DataFrame:
            if index_col is None:
                drop_index=False
                if X.index.name is None:
                    index_col='index'
                else:
                    index_col=X.index.name
            else:
                drop_index=True
            X=_transform_pandas_spark(sqlContext=self.sqlContext,df=X,drop_index=drop_index)

        x_pred = self._predict(X)
        if index_col in x_pred.schema.names:
            dp = x_pred.select([index_col,'y_proba']).toPandas()
            dp.set_index(index_col, inplace=True)
        else:
            dp = x_pred.select(['y_proba']).toPandas()
        return dp['y_proba']



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


_sparktypedict = {}
_sparktypedict[np.dtype('O')]= StringType()
_sparktypedict[np.dtype('int64')]= IntegerType()
_sparktypedict[np.dtype('float64')]= FloatType()
_sparktypedict[np.dtype('bool')]= BooleanType()

_sparktypedict[str]= StringType()
_sparktypedict[int]= IntegerType()
_sparktypedict[float]= FloatType()
_sparktypedict[bool]= BooleanType()

def _transform_pandas_spark(sqlContext,df,drop_index=False):
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
        x=df.reset_index(drop=False)
    else:
        x = df

    # create dataframe
    ds = sqlContext.createDataFrame(x, schema=schema)
    return ds




# Thank you
