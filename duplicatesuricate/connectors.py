import pandas as pd
from . import xarray
from . import utils


class _Connector:
    def __init__(self, target=None, **kwargs):
        '''

        Args:
            target:
            **kwargs:
        '''
        self.target = target
        self.attributes, self.relevance = self._config_init(**kwargs)
        assert isinstance(self.attributes, set)
        assert isinstance(self.attributes, set)
        self.output = self.attributes.union(self.relevance)
        assert isinstance(self.output, set)
        pass

    def search(self, query, on_index=None, **kwargs):
        '''

        Args:
            query (xarray.DepCol):
            on_index: if not None, this parameter specifies the output rows needed

        Returns:
            xarray.DepArray
        '''
        results = xarray.DepArray(self._search(query=query, on_index=on_index, **kwargs))
        assert set(results.columns) == self.output
        return results

    # noinspection PySetFunctionToLiteral,PySetFunctionToLiteral
    def _config_init(self, **kwargs):
        attributes = set(['info', 'info2'])
        relevance = set(['relevance'])
        return attributes, relevance

    def _config_search(self, **kwargs):
        pass

    def _search(self, query, on_index=None, **kwargs):
        results = xarray.DepArray(pd.DataFrame(columns=self.output))

        return results

    def fetch(self, on_index, on_cols=None):
        """

        Args:
            on_index (list):
            on_cols (list)

        Returns:
            xarray.DepArray
        """
        if on_cols is None:
            cols = list(self.attributes)
        else:
            cols = on_cols
        results = xarray.DepArray(self._fetch(on_index=on_index, on_cols=cols))
        assert set(results.columns) == self.attributes
        return results

    def _fetch(self, on_index, on_cols=None):
        """

        Args:
            on_index:
            on_cols (list):

        Returns:

        """
        results = xarray.DepArray(pd.DataFrame(columns=on_cols))
        return results


class PandasDF(_Connector):
    def _config_init(self, attributes, filterdict, scoredict, threshold=0.3):
        """

        Args:
            attributes:
            filterdict(dict): dictionnary two lists of values: 'any' and 'all' {'all':['country_code'],'any':['duns','taxid']}
            scoredict (dict):

        Returns:
            None
        """
        self.filterdict = utils.ScoreDict(filterdict)
        self.scoredict = utils.ScoreDict(scoredict)
        self.threshold = threshold
        relevance = self.filterdict.scores().union(self.scoredict.scores())
        return set(attributes), set(relevance)

    def _search(self, query, on_index=None, return_filtered=True):
        """
        Search engine that returns the records best matching the query
        Args:
            query (xarray.DepCol): attributes of the query
            on_index (pd.Index): optional, default None, if needed to compare the query to a specific set of records
            return_filtered (bool): default True, returns the records best matching the query and their relevance score.
                If set to False, returns all the records specified in on_index with their relevance score

        Returns:
            pd.DataFrame
        """
        q = query.toPandas()
        results1 = self.all_any(query=q, on_index=on_index, return_filtered=return_filtered)
        results2 = self.compare(query=q, on_index=results1.index, return_filtered=return_filtered)
        uniquecols = list(filter(lambda x: x not in results1.columns, results2.columns))
        results2 = results2[uniquecols]
        table = pd.concat([results1.loc[results2.index], results2], axis=1)
        return table

    def _fetch(self, on_index, on_cols=None):
        """
        This function returns the records corresponding to the index
        Args:
            on_index (pd.Index): index of the target records to be returned
            on_cols (list): optional, name of the columns to be returned. If None, all columns will be returned

        Returns:
            pd.DataFrame
        """
        if on_cols is None:
            on_cols = list(self.target.columns)
        res = self.target.loc[on_index, on_cols]
        return res

    def all_any(self, query, on_index=None, return_filtered=True):
        """
        returns a pre-filtered table score calculated on the column names provided in the filterdict.
        in the values for 'any': an exact match on any of these columns ensure the row is kept for further analysis
        in the values for 'all': an exact match on all of these columns ensure the row is kept for further analysis
        if the row does not have any exact match for the 'any' columns, or if it has one bad match for the 'all' columns,
        it is filtered out
        MODIF: if return_filtered is False, this will not filter the table at all but just returns the scores
        Args:
            query (pd.Series): query with its attributes
            on_index (pd.Index): optional, index of the target records compared. If None, all records will be targeted

        Returns:
            pd.DataFrame: a DataFrame with the exact score of the columns provided in the filterdict

        Examples:
            table = ['country_code_exactscore','duns_exactscore']
        """

        # Tackle the case where no index is given: use the whole index available
        if on_index is None:
            on_index = self.target.index

        table = pd.DataFrame(index=on_index)

        # if no filter dict is given returns an empty table with all of the rows selected:\
        #  no filterdict has been applied!
        if self.filterdict is None:
            return table

        match_any_cols = self.filterdict.get('any')
        match_all_cols = self.filterdict.get('all')

        # same as comment above
        if match_all_cols is None and match_any_cols is None:
            return table

        df = self.target.loc[on_index]

        # perform the "any criterias match" logic
        if match_any_cols is not None:
            match_any_df = pd.DataFrame(index=on_index)
            for c in match_any_cols:
                match_any_df[c + '_exactscore'] = df[c].apply(
                    lambda r: utils.exactmatch(r, query[c]))
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
                    lambda r: utils.exactmatch(r, query[c]))
            y = (match_all_df == 1)
            assert isinstance(y, pd.DataFrame)
            allcriteriasmatch = y.all(axis=1)

            table = pd.concat([table, match_all_df], axis=1)
        else:
            allcriteriasmatch = pd.Series(index=on_index).fillna(False)

        if return_filtered is True:
            # perform the all criterias match OR at least one criteria match logic
            results = (allcriteriasmatch | anycriteriasmatch)
            table = table.loc[results]

        out_index = table.index

        assert isinstance(table, pd.DataFrame)

        table = pd.concat([self.target.loc[out_index, self.attributes],
                           table], axis=1)
        return table

    def compare(self, query, on_index, return_filtered=True):
        """
        Compare a query against the dataframe and returns a similarity matrix
        Args:
            query (pd.Series): the record and its attributes
            on_index (pd.Index): the index of the records that are compared in the target records
            return_filtered (bool): default True, returns only the records qhere the similarity vector\
                is above a cetain threshold (any (row)> self.threshold)

        Returns:
            pd.DataFrame : a similarity matrix
        """
        targets = self.target.loc[on_index]
        table = utils.build_similarity_table(query=query, targets=targets, scoredict=self.scoredict)
        if return_filtered:
            results = table.apply(lambda r: any(r > self.threshold), axis=1)
            table = table.loc[results]
        return table


class SparkDF(_Connector):
    def _config_init(self, attributes, filterdict, scoredict, threshold = 0.3):
        self.filterdict = utils.ScoreDict(filterdict)
        self.scoredict = utils.ScoreDict(scoredict)
        self.threshold = threshold
        relevance = self.filterdict.scores().union(self.scoredict.scores())
        self.indexcol = 'index'
        return set(attributes), set(relevance)

    def _search(self, query, on_index=False, return_filtered=True):
        """
        Search engine that returns the records best matching the query
        Args:
            query (xarray.DepCol): attributes of the query
            on_index (SparkCol): optional, default None, if needed to compare the query to a specific set of records
            return_filtered (bool): default True, returns the records best matching the query and their relevance score.
                If set to False, returns all the records specified in on_index with their relevance score

        Returns:
            SparkDataFrame
        """
        q = query.toSpark()
        results1 = self.all_any(query=q, on_index=on_index, return_filtered=return_filtered)
        ix_1 = results1.select(self.indexcol)
        results2 = self.compare(query=q, on_index=ix_1, return_filtered=return_filtered)
        uniquecols = list(filter(lambda x: x not in results1.schema.names, results2.schema.names))
        results2 = results2.select(uniquecols)
        table = results1.join(results2, on=[self.indexcol], how = 'left_outer')
        return table

    def _fetch(self, on_index, on_cols=None):
        pass

    def compare(self, query, on_index=None, return_filtered=True):

        df = self.target
        for c in self.scoredict.get('fuzzy'):
            q_val = query.loc[c]
            df = df.withColumn('query', F.lit(q_val).cast(F.StringType()))
            df = df.withColumn('len', F.min(F.length(c), F.lit(len(q_val)).cast(T.IntegerType())))
            df = df.withColumn('levenshtein', F.levenshtein(c, 'query'))
            df = df.withColumn('score', F.col('levenshtein')/F.col('len'))


                c + '_fuzzyscore',
                F.levenshtein(
                    c,
                    F.lit(
                        query.loc[c]
                    )
                )/min(F.length(c))
            )
        pass

    def all_any(self, query, on_index=None, return_filtered=True):

        pass
