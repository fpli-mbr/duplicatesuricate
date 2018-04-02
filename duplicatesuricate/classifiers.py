import numpy as np
import pandas as pd

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
            X:
            y:

        Returns:
            None
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