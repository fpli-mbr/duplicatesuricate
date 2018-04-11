import connectors
import classifiers
import comparators
import xarray
import pandas as pd
import functions

class RecordLinker:
    def __init__(self, connector, comparator, classifier):
        '''

        Args:
            connector (connectors._Connector):
            comparator (comparators._Comparator):
            classifier (classifiers._Classifier):
        '''
        self.connector = connector
        self.comparator = comparator
        self.classifier = classifier
        self._coherency()
        print('init ok')
        self.scores = self.classifier.scores
        self.compared = self.comparator.compared
        pass

    def _coherency(self):
        '''
        Checks if the different components can work with one another
        Returns:
            bool:
        '''
        assert self.comparator.compared.issubset(self.connector.attributes)
        assert len(self.connector.relevance.intersection(self.comparator.scored)) == 0
        assert self.classifier.scores.issubset(self.connector.relevance.union(self.comparator.scored))
        return True

    def predict_proba(self, query, on_index=None):
        """
        Main method of this class:
        - with the help of the scoring model, create a similarity table
        - with the help of the evaluation model, evaluate the probability of it being a match
        - returns that probability

        Args:
            query: information available on our query
            on_index (pd.Index): index on which to do the prediction
        Returns:
            xarray.DepCol: the probability vector of the target records being the same as the query

        """
        query = xarray.DepCol(query)
        output = self.connector.search(query, on_index=on_index)

        if output is None or output.count() == 0:
            return None
        else:

            relevance = output.select(self.connector.relevance)
            targets = output.select(self.connector.attributes)

            # create table of scores
            scores = self.comparator.compare(query, targets=targets)
            scores = scores.union(relevance)

            # launch prediction
            y_proba = self.classifier.predict_proba(scores)

            # sort the results
            y_proba = y_proba.sort(ascending=False)

            del scores

            return y_proba
    def predict(self, query, on_index=None):
        """

        Args:
            query:
            on_index (pd.Index):

        Returns:
            xarray.DepCol: the boolean vector of the target records being the same as the query
        """
        y_proba = self.predict_proba(query=query, on_index=on_index)
        if y_proba is None:
            return None
        y_proba = y_proba.toPandas()
        y_bool = (y_proba > self.classifier.threshold)
        assert isinstance(y_bool, pd.Series)
        y_bool = xarray.DepCol(y_bool)
        return y_bool

    def match_index(self, query, on_index=None, n_matches_max=None):
        """
        Args:
            query:
            on_index (pd.Index):

        Returns:
            list: a list of the matching indexes
        """
        if n_matches_max is None:
            n_matches_max = 1
        y_bool =self.predict(query=query, on_index=on_index)
        if y_bool is None:
            return None
        else:
            y_bool = y_bool.toPandas()
            goodmatches = y_bool.loc[y_bool].index
            if len(goodmatches) == 0:
                return None
            else:
                goodmatches = goodmatches[:max(n_matches_max, len(goodmatches))]
                goodmatches = list(goodmatches)
                return goodmatches

    def match_records(self, query, on_index=None):
        """

        Args:
            query:
            on_index (pd.Index):

        Returns:
            pd.DataFrame
        """
        goodmatches = self.match_index(query=query, on_index=on_index)
        results = self.connector.fetch(on_index=goodmatches)
        results = results.toPandas()
        return results


def  create_pandas_linker(source, filterdict, scoredict, scoredict2=None, scores=None, X_train=None, y_train=None):
        """

        Args:
            source:
            filterdict:
            scoredict:
            scoredict2:
            scores:

        Returns:
            RecordLinker
        """
        connector = connectors.PandasDF(source=source, attributes=source.columns, scoredict=scoredict, filterdict=filterdict)
        needed_scores = set(X_train.columns)
        needed_scores = needed_scores.difference(connector.relevance)
        needed_scores = functions.ScoreDict.from_cols(scorecols=needed_scores).to_dict()
        comparator = comparators.PandasComparator(scoredict=needed_scores)
        classifier = classifiers.ScikitLearnClassifier(n_estimators=100)
        classifier.fit(X_train, y_train)
        # classifier = classifiers.RuleBasedClassifier(scores)
        #
        lk = RecordLinker(connector=connector,
                                 comparator=comparator,
                                 classifier=classifier)
        return lk
