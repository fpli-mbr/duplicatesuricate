# coding=utf-8
"""
Machine Learning decisionmodel used for record linkage
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score

class RFModel:
    def __init__(self, verbose=True,
                 df=pd.DataFrame(),
                 n_estimators=2000):
        """
        The machine-learning par of the program
        Args:
            verbose (bool): control print output 
            df (pd.DataFrame): target records
            n_estimators (int): number of estimators for the Random Forest Algorithm
        """
        self.verbose = verbose
        self.model = RandomForestClassifier(n_estimators=n_estimators)

        pass

    def fit(self, warmstart=False, training_set=pd.DataFrame(), target_col='ismatch'):
        """
        fit the machine learning decisionmodel on the provided data set
        Args:
            warmstart (bool): if the decisionmodel is already trained
            training_set (pd.DataFrame): pandas DataFrame containing annotated data
            target_col (str):name of the target vector in the training_set

        Returns:
            None

        """

        if target_col not in training_set.columns:
            raise KeyError('target column ', target_col, ' not found in training set columns')
        # Define training set and target vector
        self.traincols = list(filter(lambda x: x != target_col, training_set.columns))

        if warmstart is False:
            start = pd.datetime.now()

            if self.verbose:
                print('shape of training table ', training_set.shape)
                print('number of positives in table', training_set[target_col].sum())

            # Define training set and target vector
            x_train = training_set[self.traincols].fillna(-1)  # fill na values
            y_train = training_set[target_col]

            # fit the decisionmodel
            self.model.fit(x_train, y_train)

            if self.verbose:
                # show precision and recall score of the decisionmodel on training data
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
        This is the heart of the program:
        Args:
            x_score (pd.DataFrame): the table containing the scoring records

        Returns:
            pd.Series : the probability vector of the target records being the same as the query
            
        """
        if x_score is None or x_score.shape[0] == 0:
            return None
        else:
            # re-arrange the column order
            x_score = x_score[self.traincols]

            # fill the na values
            x_score = x_score.fillna(-1)

            # launch prediction using the predict_proba of the scikit-learn module
            y_proba = \
                pd.DataFrame(self.model.predict_proba(x_score), index=x_score.index)[
                    1].copy()
            return y_proba