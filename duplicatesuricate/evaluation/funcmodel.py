import pandas as pd


class FuncEvaluationModel:
    """
    This evaluation model applies a evaluation function to return a probability vector
    Examples:
        decisionfunc = lambda r:r[id_cols].mean()
        dm = FuncEvaluationModel(used_cols=id_cols,eval_func=decisionfunc)
        x_score = compare(query,target_records)
        y_proba = dm.predict_proba(x_score)
    """

    def __init__(self, used_cols, eval_func):
        """
        Create the model
        Args:
            used_cols (list): list of columns necessary for decision
            eval_func (func): evaluation function to be applied. must return a probability vector
        """
        self.used_cols = used_cols
        self.eval_func = eval_func
        pass

    def fit(self):
        """
        pass
        Returns:
            None
        """
        pass

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
