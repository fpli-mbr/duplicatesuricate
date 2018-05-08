import pandas as pd
import numpy as np
from . import xarray
from . import utils
from . import linker
from . import retrain


class Suricate:
    def __init__(self, input_records, rlinker, verbose=True, display=None):
        """
        Main class used for deduplication
        Args:
            input_records (pd.DataFrame): Input table for record linkage, records to link
            rlinker (linker.RecordLinker)
            verbose (bool): Turns on or off prints
            display (list): list of columns
        """

        self.input_records = input_records

        self.input_records.index.name = 'ix_source'

        self.linker = rlinker
        self.verbose = verbose

        self.idcol = 'gid'

        self._results = {}

        if display is None:
            self.display = self.linker.connector.attributes
        else:
            self.display = set(display)
        pass


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
        query = self.input_records.loc[query_index]

        goodmatches_index = self.linker.match_index(query=query, n_matches_max=n_matches_max)

        return goodmatches_index

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
            # TODO: write func
            raise ModuleNotFoundError('function not defined yet')


        # Melt the results dictionnary to have the form:
        # df.columns = ['ix_source','ix_target'] if with_proba is false, ['ix_source','ix_target','y_proba' otherwise]
        results = self.unpack_results(self._results, with_proba=with_proba)
        results = retrain.unique_pairs(y_source=results['ix_source'],
                                       y_target=results['ix_targets'])
        return results

    def get_target(self, on_index, on_cols=None):
        results = self.linker.connector.fetch(on_index=on_index).toPandas()
        if on_cols is None:
            on_cols = self.linker.connector.attributes
        if type(on_cols) == set:
            cols = list(on_cols)
        elif type(on_cols) == list:
            cols = on_cols
        else:
            cols = list(on_cols)
        results = results[cols]
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
            display = list(self.linker.connector.attributes)
        if fuzzy is None:
            fuzzy = []
        if exact is None:
            exact = []

        inputs = np.array(inputs)
        targets = np.array(targets)

        allcols = list(set(display + fuzzy + exact))

        # take all values from source records
        res = self.input_records.loc[inputs, allcols].copy()
        res.columns = [c + '_source' for c in allcols]
        res['ix_source'] = inputs

        # take all values from target records
        x = self.get_target(on_index=targets, on_cols=allcols)
        x.columns = [c + '_target' for c in allcols]
        x.index.name = 'ix_target'
        res['ix_target'] = targets
        res.set_index('ix_target', inplace=True, drop=True)
        res.index.name = 'ix_target'
        res = pd.concat([res, x], axis=1)
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
        if len(fuzzy) > 0:
            df_fuzzy = pd.DataFrame(index=res.index)
            for c in fuzzy:
                df_fuzzy[c + '_fuzzyscore'] = res.apply(
                    lambda r: utils.fuzzyscore(r[c + '_source'], r[c + '_target']), axis=1)
            # after the loop, take the sum of the exact score (n ids matchings)
            if len(fuzzy) > 1:
                df_fuzzy['avg_fuzzyscore'] = df_fuzzy.fillna(0).mean(axis=1)
            res = res.join(df_fuzzy)

        if len(exact) > 0:
            df_exact = pd.DataFrame(index=res.index)
            for c in exact:
                df_exact[c + '_exactscore'] = res.apply(
                    lambda r: utils.exactmatch(r[c + '_source'], r[c + '_target']), axis=1)
            # after the loop, take the sum of the exact score (n ids matchings)
            if len(exact) > 1:
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
        missing_cols = sorted(list(filter(lambda m: m not in ordered, res.columns)))
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

        training_table_complete = pd.DataFrame(columns=list(self.linker.classifier.scores))
        if scoredict is None:
            scoredict = utils.ScoreDict.from_cols(self.linker.classifier.scores).to_dict()
        for t, u in zip(inputs, targets):
            query = self.input_records.loc[t]
            target = self.get_target(on_index=pd.Index([u]))
            similarity_vector = utils.build_similarity_table(query=query, targets=target, scoredict=scoredict)
            similarity_vector['ix_source'] = t
            similarity_vector['ix_target'] = u
            training_table_complete = pd.concat([training_table_complete, similarity_vector], ignore_index=True, axis=0)

        # fillna
        training_table_complete.fillna(fillna, inplace=True)

        # calculate the probability vector
        if with_proba:
            X_train = training_table_complete[list(self.linker.classifier.scores)]
            X_train = xarray.DepArray(X_train)
            y_proba = self.linker.classifier.predict_proba(X_train).toPandas()
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


class Clustricate(Suricate):
    @classmethod
    def from_files(cls, data, filterdict, scoredict, X_train, y_train, n_estimators=10, idcol='gid'):
        """

        Args:
            data (pd.DataFrame):
            filterdict:
            scoredict:
            X_train:
            y_train:
            n_estimators:
            idcol:

        Returns:

        """

        if idcol not in data.columns:
            data[idcol] = None

        lk = linker.create_pandas_linker(target=data,
                                         filterdict=filterdict,
                                         scoredict=scoredict,
                                         X_train=X_train, y_train=y_train, n_estimators=n_estimators)
        sur = Clustricate(input_records=data, rlinker=lk)
        sur.idcol = idcol
        return sur

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

    def find_duplicates(self, n_runs=5):
        self._find_duplicates(n_runs=n_runs)
        return None

    def _find_duplicates(self, n_runs=5):
        print('starting deduplication at {}'.format(pd.datetime.now()))
        for i in range(n_runs):
            time_start = pd.datetime.now()
            print('starting deduplication for {} of {}'.format(i+1, n_runs))
            new_query = self._generate_query_index_()
            if new_query is None:
                print('ending deduplication')
                break
            self._update_gid_query(query=new_query)
            # timing
            time_end = pd.datetime.now()
            duration = (time_end - time_start).total_seconds()

            if self.verbose:
                print(
                    '{} of {} inputs records deduplicated || time elapsed {} s'.format(
                        i + 1, n_runs, duration))
        print('done')
    def _update_gid_query(self, query):
        """
        update the self.idcol column with a new group id (gid)
        Args:
            query (object): part of the query_index

        Returns:
            None
        """
        goodmatches_index = self._find_matches_(query_index=query)
        if goodmatches_index is None:
            self.input_records.loc[query, self.idcol]=self.create_gid(query)
        else:
            gids = self.input_records.loc[goodmatches_index, self.idcol]
            assert isinstance(gids, pd.Series)
            gids = gids.dropna()
            if gids.shape[0] == 0:
                gid = self.create_gid(query)
            else:
                gid = gids.value_counts().index[0]
            self.input_records.loc[query, self.idcol] = gid
            y = self.input_records.loc[goodmatches_index, self.idcol]
            unmatched_index = y.loc[y.isnull()].index
            self.input_records.loc[unmatched_index, self.idcol] = gid
        return None

    def create_gid(self, query):
        from hashlib import sha1
        encoding = 'utf-8'
        s = str(query)
        hashedid = sha1(s.encode(encoding,'ignore')).hexdigest()
        hashedid='-'.join([hashedid[:5], hashedid[5:10], hashedid[10:15], hashedid[15:20]])
        # hashedid = hashedid.encode(encoding, 'ignore').decode(encoding)
        # hashedid = str(hashedid)
        return hashedid
    def show_group(self, gid=None, index=None):
        if gid is None and index is None:
            return None
        if index is not None:
            mygid = self.input_records.loc[index,self.idcol]
        else:
            mygid = gid
        res = self.input_records.loc[self.input_records[self.idcol]==mygid]
        return res

def create_pandas_suricate(source, target, filterdict, scoredict, X_train, y_train, n_estimators=500):
    lk = linker.create_pandas_linker(target=target,
                                     filterdict=filterdict,
                                     scoredict=scoredict,
                                     X_train=X_train, y_train=y_train, n_estimators=n_estimators)
    sur = Suricate(input_records=source, rlinker=lk)
    return sur
