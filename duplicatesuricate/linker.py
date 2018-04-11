import connectors
import classifiers
import comparators
import xarray
import pandas as pd

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