from xarray import _Array, _Col

import numpy as np
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier as SparkRF
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score

from deduplication import _transform_pandas_spark, _transform_scoredict_scorecols, _transform_scorecols_scoredict


class _Classifier:
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

    # noinspection PySetFunctionToLiteral
    def _config_init(self, *args,**kwargs):
        used = set(['info_score', 'relevance'])
        return used

    def fit(self, X, y):
        """
        Do nothing
        Args:
            X (_Array):
            y (_Col):

        Returns:
            None
        """
        pass

    def predict_proba(self, X):
        """
        A dart-throwing chump generates a random probability vector for the sake of coherency with other classifier
        Args:
            X (_Array):

        Returns:
            _Col
        """
        y_proba = np.random.random(size=X.count())
        y_proba = pd.Series(y_proba, index=X.index)
        return y_proba


class SparkClassifier:
    """
    The evaluation model is based on spark-powered machine learning, it is an implementation of the Random Forest algorithm.
    It requires to be fitted on a training table before making decision.

    Examples:
        dm = SparkMLEvaluationModel()
        dm.fit(Xytrain)
        y_proba = dm.predict_proba(x_score)
    """

    def __init__(self, sqlContext, verbose=True):
        """
        Create the model
        Args:
            sqlContext (pyspark.sql.context.SQLContext):
            verbose (bool): control print output
        """
        self.verbose = verbose
        self.sqlContext = sqlContext
        self.used_cols = list()
        self.pipeline_model = None

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
        X = X.astype(float)
        assert isinstance(X, pd.DataFrame)
        X['y_train'] = y
        X['y_train'] = X['y_train'].astype(int)

        Xs = _transform_pandas_spark(self.sqlContext, df=X, drop_index=True)

        # Create the pipeline

        assembler = VectorAssembler(inputCols=list(self.used_cols), outputCol="features")
        labelIndexer = StringIndexer(inputCol="y_train", outputCol="label")
        rf_classifier = SparkRF(labelCol=labelIndexer.getOutputCol(), featuresCol=assembler.getOutputCol())
        pipeline = Pipeline(stages=[assembler, labelIndexer, rf_classifier])

        # fit the classifier
        self.pipeline_model = pipeline.fit(Xs)

        if self.verbose:
            # show precision and recall score of the classifier on training data
            y_pred = self.predict_proba(Xs)
            assert isinstance(y_pred, pd.Series)
            # noinspection PyTypeChecker
            y_pred = (y_pred > 0.5)
            assert isinstance(y_pred, pd.Series)
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

    def _predict(self, X):
        """
        Args:
            X (pyspark.sql.dataframe.DataFrame):

        Returns:
            pyspark.sql.dataframe.DataFrame
        """

        x_pred = self.pipeline_model.transform(X)
        proba_udf = udf(lambda r: float(r[1]), FloatType())
        x_pred = x_pred.withColumn('y_proba', proba_udf(x_pred["probability"]))
        return x_pred

    def predict_proba(self, X, index_col=None):
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
                drop_index = False
                if X.index.name is None:
                    index_col = 'index'
                else:
                    index_col = X.index.name
            else:
                drop_index = True
            X = _transform_pandas_spark(sqlContext=self.sqlContext, df=X, drop_index=drop_index)

        x_pred = self._predict(X)
        if index_col in x_pred.schema.names:
            dp = x_pred.select([index_col, 'y_proba']).toPandas()
            dp.set_index(index_col, inplace=True)
        else:
            dp = x_pred.select(['y_proba']).toPandas()
        return dp['y_proba']


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
            print('proportion of positives in table: {0:.1%}'.format(y.sum() / X.shape[0]))

        # fit the classifier
        self.model.fit(X, y)

        if self.verbose:
            # show precision and recall score of the classifier on training data
            y_pred = self.model.predict(X)
            precision = precision_score(y_true=y, y_pred=y_pred)
            recall = recall_score(y_true=y, y_pred=y_pred)
            print('precision, recall score on training data: {0:.1%},{0:.1%}'.format(precision, recall))

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


class DummyClassifier:
    def __init__(self, scoredict=None):
        """
        Create a model used only for scoring (for example for creating training data)
        used_cols (list): list of columns necessary for decision
        eval_func (func): evaluation function to be applied. must return a probability vector
        Args:
            scoredict (dict): {'fuzzy':['name','street'],'token':['name_wostopwords'],'acronym':None}
        """
        if scoredict is None:
            self.scoredict = dict()
        else:
            self.scoredict = scoredict
        compared_cols, used_cols = _transform_scoredict_scorecols(scoredict)
        self.used_cols = used_cols
        self.compared_cols = compared_cols
        pass

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