from connectors import _Connector
from comparators import _Comparator
from evaluators import _Evaluator

class _RecordLinker:
    def __init__(self, connector, comparator, evaluator):
        '''

        Args:
            connector (connectors._Connector):
            comparator (comparators._Comparator):
            evaluator (evaluators._Evaluator):
        '''
        self.connector = connector
        self.comparator = comparator
        self.evaluator = evaluator
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
        assert self.evaluator.used.issubset(self.connector.relevance.union(self.comparator.scored))
        return True

    def search(self, query):
        return self.connector.search(query)

    def compare(self,query, targets):
        return self.comparator.compare(query, targets)

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
        # TODO
        output = self.connector.search(query)

        relevance = output.select(self.connector.relevance)
        targets = output.select(self.connector.attributes)

        comparison =

        if table_score_complete is None or table_score_complete.shape[0] == 0:
            return None
        else:
            # launch prediction using the predict_proba of the scikit-learn module

            y_proba = self.classifier.predict_proba(table_score_complete).copy()

            del table_score_complete

            # sort the results
            y_proba.sort_values(ascending=False, inplace=True)

            return y_proba

es = _Connector()
fz = _Comparator()
ml = _Evaluator()
rl = _RecordLinker(connector=es, comparator=fz, evaluator=ml)