import pandas as pd
from duplicatesuricate.deduplication.scoring import Scorer
from duplicatesuricate.deduplication.scoring import _checkdict,_unpack_scoredict,_calculatescoredict,scorename,scoringkeys

class RecordLinker:
    def __init__(self,
                 df, filterdict=None,
                 intermediate_thresholds=None,
                 fillna=-1,
                 evaluator=None,
                 decision_threshold=0.8,
                 verbose=True):
        """
        This class merges together the Scorer and the Evaluation model.
        it creates a similarity table
        evaluate the probability of being a match with the model
        and then can either:
        - return that probability with predict_proba
        - return a boolean: probability is higher than a decision threshold with predict
        - or return the index of the good matches with return_good_matches

        Args:
            df (pd.DataFrame): target records
            filterdict (dict):
            intermediate_thresholds (dict):
            fillna (float):
            evaluator : Class used to calculate a probability vector. Has .predict_proba function
            decision_threshold (float), default 0.8
            verbose (bool): control print output
        """
        self.verbose = verbose

        self.df = df
        self.query = pd.Series()

        self.evaluationmodel = evaluator
        self.evaluation_cols=self.evaluationmodel.used_cols

        self.compared_cols = []
        self.scorecols=[]
        self.filterdict = None
        self.intermediate_score = None
        self.intermediate_function=None
        self.further_score = None

        self._calculate_scoredict(filterdict=filterdict,
                                  intermediatethreshold=intermediate_thresholds,
                                  used_cols=self.evaluation_cols)

        if intermediate_thresholds is not None:
            decision_int_func = lambda r:threshold_based_decision(r=r,thresholds=intermediate_thresholds)
        else:
            decision_int_func = None

        self.scoringmodel = Scorer(df=df,
                                   filterdict=self.filterdict,
                                   score_intermediate=self.intermediate_score,
                                   decision_intermediate=decision_int_func,
                                   score_further=self.further_score,
                                   fillna=fillna
                                   )

        self.decision_threshold = decision_threshold

        self.evaluation_cols=self.evaluationmodel.used_cols
        self.compared_cols=self.scoringmodel.compared_cols

        missingcols=list(filter(lambda x: x not in self.scoringmodel.scorecols, self.evaluationmodel.used_cols))
        if len(missingcols) > 0:
            raise KeyError('not all training columns are found in the output of the scorer:',missingcols)
        pass

    def _calculate_scoredict(self,filterdict,intermediatethreshold,used_cols):
        self.compared_cols = []
        self.scorecols = []

        if filterdict is not None:
            self.filterdict = _checkdict(filterdict, mandatorykeys=['all', 'any'], existinginput=self.scorecols)
            incols, outcols = _unpack_scoredict(self.filterdict)
            self.compared_cols += incols
            self.scorecols += outcols
        else:
            self.filterdict = None

        score_intermediate = _calculatescoredict(existing_cols=self.scorecols,used_cols=list(intermediatethreshold.keys()))
        if score_intermediate is not None:
            self.intermediate_score = _checkdict(score_intermediate, mandatorykeys=scoringkeys,
                                                 existinginput=self.scorecols)
            incols,outcols= _unpack_scoredict(self.intermediate_score)
            self.compared_cols+=incols
            self.scorecols+=outcols
        else:
            self.intermediate_score = None

        score_further=_calculatescoredict(existing_cols=self.scorecols,used_cols=used_cols)

        if score_further is not None:
            self.further_score = _checkdict(score_further, mandatorykeys=scoringkeys, existinginput=self.scorecols)
            incols,outcols= _unpack_scoredict(self.further_score)
            self.compared_cols+=incols
            self.scorecols+=outcols
        else:
            self.further_score = None
        pass

    def return_good_matches(self, query,decision_threshold=None):
        """
        Return the good matches
        - with the help of the scoring model, create a similarity table
        - with the help of the evaluation model, evaluate the probability of it being a match
        - using the decision threshold, decides if it is a match or not
        - return the index of the good matches
        Args:
            query (pd.Series): information available on our query

        Returns:
            pd.Index: the index of the target records identified as the same as the query by the algorithm

        """
        if decision_threshold is None:
            decision_threshold = self.decision_threshold

        y_bool = self.predict(query,decision_threshold=decision_threshold)

        if y_bool is None:
            return None
        else:
            goodmatches = y_bool.loc[y_bool].index
            return goodmatches

    def predict(self, query,decision_threshold=None):
        """
        Predict if it is a match or not.
        - with the help of the scoring model, create a similarity table
        - with the help of the evaluation model, evaluate the probability of it being a match
        - using the decision threshold, decides if it is a match or not

        Args:
            query (pd.Series): information available on our query
            decision_threshold (float): default None. number between 0 and 1. If not provided, take the default one from the model

        Returns:
            pd.Series: a boolean vector: True if it is a match, false otherwise

        """
        if decision_threshold is None:
            decision_threshold = self.decision_threshold

        # calculate the probability of the records being the same as the query through the machine learning evaluation
        y_proba = self.predict_proba(query)
        if y_proba is None:
            return None
        else:
            assert isinstance(y_proba,pd.Series)
            # transform that probability in a boolean via a decision threshold
            # noinspection PyTypeChecker
            y_bool = (y_proba > decision_threshold)
            assert isinstance(y_bool,pd.Series)

            return y_bool

    def predict_proba(self, query):
        """
        Main method of this class:
        - with the help of the scoring model, create a similarity table
        - with the help of the evaluation model, evaluate the probability of it being a match
        - returns that probability

        Args:
            query (pd.Series): information available on our query

        Returns:
            pd.Series : the probability vector of the target records being the same as the query

        """

        table_score_complete = self.scoringmodel.filter_compare(query=query)

        if table_score_complete is None or table_score_complete.shape[0] == 0:
            return None
        else:
            # launch prediction using the predict_proba of the scikit-learn module

            y_proba = self.evaluationmodel.predict_proba(table_score_complete)

            del table_score_complete

            # sort the results
            y_proba.sort_values(ascending=False, inplace=True)

            return y_proba


def threshold_based_decision(r, thresholds):
    """
    if all values of the row are above the thresholds, return 1, else return 0
    Args:
        r (pd.Series): row to be decided
        thresholds (dict): threshold of values

    Returns:
        float

    """
    r=r.fillna(0)
    for k in list(thresholds.keys()):
        if r[k] < thresholds[k]:
            return 0
    else:
        return 1