# coding=utf-8
"""
Machine Learning evaluation model used for record linkage
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score

# Config RF Model
_training_path = '/Users/paulogier/Documents/8-PythonProjects/02-test_suricate/'
_training_name = 'training_table_prepared_20171107_79319rows.csv'
training_filename = _training_path + _training_name


def load_training_data(path, targetcol='ismatch', sep=',', encoding='utf-8', index_col=None):
    """
    Load the training data and return X_train, y_train
    Args:
        path (str): path of the training file
        targetcol (str):  name of the target vector in the training column
        sep (str): separator, default ','
        encoding (str): encoding, default 'utf-8'
        index_col (int): if the first column of the training table is the index, default False

    Returns:
        pd.DataFrame,pd.Series

    """
    extension = path.split('.')[-1]
    if extension == 'csv':
        df = pd.read_csv(filepath_or_buffer=path, sep=sep, encoding=encoding, index_col=index_col)
    elif extension in ['xlsx', 'xls']:
        df = pd.read_excel(path, index_col=index_col)
    else:
        raise TypeError('incorrect extension provided')
    assert isinstance(df, pd.DataFrame)

    if targetcol not in df.columns:
        raise KeyError(targetcol, ' not in table columns')

    y_train = df[targetcol]
    x_train = df.drop(targetcol, axis=1)
    assert isinstance(x_train, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    return x_train, y_train


class MLEvaluationModel:
    """
    The evaluation model is based on machine learning, it is an implementation of the Random Forest algorithm.
    It requires to be fitted on a training table before making decision.

    Examples:
        dm = MLEvaluationModel()
        x_train,y_train=dm.load_training_data('mytrainingdata.csv',targetcol='ismatch')
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

    def fit(self, x_train, y_train):
        """
        fit the machine learning evaluation model on the provided data set.
        It takes as input a training table with numeric values calculated from previous examples.
        Args:
            x_train (pd.DataFrame): pandas DataFrame containing annotated data
            y_train (pd.Series):name of the target vector in the training_set

        Returns:
            None

        """

        self.used_cols = x_train.columns

        start = pd.datetime.now()

        if self.verbose:
            print('shape of training table ', x_train.shape)
            print('number of positives in table', y_train.sum())

        # fit the evaluationmodel
        self.model.fit(x_train, y_train)

        if self.verbose:
            # show precision and recall score of the evaluationmodel on training data
            y_pred = self.model.predict(x_train)
            precision = precision_score(y_true=y_train, y_pred=y_pred)
            recall = recall_score(y_true=y_train, y_pred=y_pred)
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
                pd.DataFrame(self.model.predict_proba(x_score), index=x_score.index)[
                    1]
            assert isinstance(y_proba, pd.Series)
            return y_proba
