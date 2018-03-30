import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
from fuzzywuzzy.fuzz import ratio, token_set_ratio

# noinspection PyUnresolvedReferences
from pyspark.sql.functions import udf, lit
from pyspark.sql.types import IntegerType, FloatType, StructType, StructField, StringType, BooleanType
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier as SparkRF
from pyspark.ml import Pipeline


class _RecordLinker:
    def __init__(self, connector, comparator, evaluator):
        '''

        Args:
            connector (_Connector):
            comparator (_Comparator):
            evaluator (_Evaluator):
        '''
        self.connector = connector
        self.comparator = comparator
        self.evaluator = evaluator
        pass
    def _coherency(self):
        '''
        Checks if the different components can work with one another
        Returns:
            bool:
        '''


class _Connector:
    def __init__(self, source, **kwargs):
        '''

        Args:
            source:
            **kwargs:
        '''
        self.source = source
        self.attributes, self.relevance = self._config_init(**kwargs)
        assert isinstance(self.attributes, set)
        assert isinstance(self.attributes, set)
        self.output = self.attributes.union(self.relevance)
        assert isinstance(self.output, set)
        pass

    def search(self, query):
        '''

        Args:
            query:

        Returns:
            array
        '''
        results = pd.DataFrame(columns=self.output)

        assert set(results.columns) == self.output
        return results

    def _config_init(self, **kwargs):
        attributes = set(['info','info2'])
        relevance = set('relevance')
        return attributes, relevance

    def _config_search(self, **kwargs):
        pass


class _Comparator:
    def __init__(self, **kwargs):
        self.compared, self.scored = self._config_init(**kwargs)
        assert isinstance(self.compared, set)
        assert isinstance(self.scored, set)
        pass

    def _config_init(self, **kwargs):
        compared = set(['info'])
        scored = set(['info_score'])
        return compared, scored

    def compare(self, query, target):
        results = pd.DataFrame(columns=self.scored)

        assert set(results.columns) == self.scored
        return results

class _Evaluator:
    def __init__(self, **kwargs):
        """
        Create a model used only for scoring (for example for creating training data)
        used_cols (list): list of columns necessary for decision
        eval_func (func): evaluation function to be applied. must return a probability vector
        Args:
            scoredict (dict): {'fuzzy':['name','street'],'token':['name_wostopwords'],'acronym':None}
        """
        self.used = self._config_init(**kwargs)
        assert isinstance(self.used, set)
        pass

    def _config_init(self, **kwargs):
        used = set(['info_score','relevance'])
        return used

    def fit(self, X, y):
        """
        Do nothing
        Args:
            X:
            y:

        Returns:

        """
        pass

    def predict_proba(self, X):
        """
        A dart-throwing chump generates a random probability vector for the sake of coherency with other classifier
        Args:
            X:

        Returns:
            pd.Series
        """
        y_proba = np.random.random(size=X.shape[0])
        y_proba = pd.Series(y_proba, index=X.index)
        return y_proba