from connectors import _Connector
from comparators import _Comparator
from classifiers import _Classifier
from xarray import _Array, _Col

class _RecordLinker:
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
        pass

    def _coherency(self):
        '''
        Checks if the different components can work with one another
        Returns:
            bool:
        '''
        assert self.comparator.compared.issubset(self.connector.attributes)
        assert self.classifier.used.issubset(self.connector.relevance.union(self.comparator.scored))
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
            _Col: the probability vector of the target records being the same as the query

        """
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
            y_proba = y_proba.sort_values(ascending=False)

            del scores

            return y_proba

es = _Connector()
fz = _Comparator()
ml = _Classifier()
rl = _RecordLinker(connector=es, comparator=fz, classifier=ml)