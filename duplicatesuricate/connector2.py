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
    def __init__(self):
        pass

class _Connector:
    def __init__(self, source, **kwargs):
        '''

        Args:
            source:
            **kwargs:
        '''
        self.source = source
        self.attributes, self.score = self._config_return(**kwargs)
        pass
    def search(self, query):
        return pd.DataFrame()
    def _config_return(self, **kwargs):
        attributes =['info']
        score = ['relevance']
        return attributes, score
    def _config_search(self, **kwargs):
        pass


