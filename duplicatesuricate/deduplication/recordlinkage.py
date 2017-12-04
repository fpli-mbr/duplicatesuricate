import pandas as pd
import numpy as np

from duplicatesuricate.deduplication.scoring import Scorer
from duplicatesuricate.deduplication.scoring import _checkdict, _unpack_scoredict, _calculatescoredict, scoringkeys


class RecordLinker:
    def __init__(self,
                 df, filterdict=None,
                 intermediate_thresholds=None,
                 fillna=-1,
                 evaluator=None,
                 decision_threshold=0.5,
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
            filterdict (dict): filtering dict with exact matches on an {'all':['country'],'any':[id1,id2]}
            intermediate_thresholds (dict): minimum threshold for intermediate scoring
            fillna (float): value with which to fill na values
            evaluator : Class used to calculate a probability vector. Has .predict_proba function
            decision_threshold (float), default 0.8
            verbose (bool): control print output
        """

        #TODO: update intermediate_threshold documentation

        self.verbose = verbose

        self.df = df
        self.query = pd.Series()

        self.evaluationmodel = evaluator
        self.evaluation_cols = self.evaluationmodel.used_cols

        self.compared_cols = []
        self.scorecols = []
        self.filterdict = {}
        self.intermediate_score = {}
        self.intermediate_function = None
        self.further_score = {}

        self._calculate_scoredict(filterdict=filterdict,
                                  intermediatethreshold=intermediate_thresholds,
                                  used_cols=self.evaluation_cols)

        if intermediate_thresholds is not None:
            decision_int_func = lambda r: threshold_based_decision(row=r, thresholds=intermediate_thresholds)
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

        self.evaluation_cols = self.evaluationmodel.used_cols
        self.compared_cols = self.scoringmodel.compared_cols

        missingcols = list(filter(lambda x: x not in self.scoringmodel.scorecols, self.evaluationmodel.used_cols))
        if len(missingcols) > 0:
            raise KeyError('not all training columns are found in the output of the scorer:', missingcols)
        pass

    def _calculate_scoredict(self, filterdict, intermediatethreshold, used_cols):
        self.compared_cols = []
        self.scorecols = []

        if filterdict is not None:
            self.filterdict = _checkdict(filterdict, mandatorykeys=['all', 'any'], existinginput=self.scorecols)
            incols, outcols = _unpack_scoredict(self.filterdict)
            self.compared_cols += incols
            self.scorecols += outcols
        else:
            self.filterdict = None


        score_intermediate = _calculatescoredict(existing_cols=self.scorecols,
                                            used_cols=list(intermediatethreshold.keys()))

        if score_intermediate is not None:
            self.intermediate_score = _checkdict(score_intermediate, mandatorykeys=scoringkeys,
                                                 existinginput=self.scorecols)
            incols, outcols = _unpack_scoredict(self.intermediate_score)
            self.compared_cols += incols
            self.scorecols += outcols
        else:
            self.intermediate_score = None

        score_further = _calculatescoredict(existing_cols=self.scorecols, used_cols=used_cols)

        if score_further is not None:
            self.further_score = _checkdict(score_further, mandatorykeys=scoringkeys, existinginput=self.scorecols)
            incols, outcols = _unpack_scoredict(self.further_score)
            self.compared_cols += incols
            self.scorecols += outcols
        else:
            self.further_score = None
        pass

    def return_good_matches(self, query, decision_threshold=None, on_index=None,n_matches_max=1):
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

        y_bool = self.predict(query, decision_threshold=decision_threshold, on_index=on_index)

        if y_bool is None:
            return None
        else:
            goodmatches = y_bool.loc[y_bool].index
            if len(goodmatches)==0:
                return None
            else:
                goodmatches=goodmatches[:max(n_matches_max,len(goodmatches))]
            return goodmatches

    def predict(self, query, decision_threshold=None, on_index=None):
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
        y_proba = self.predict_proba(query, on_index=on_index)
        if y_proba is None:
            return None
        else:
            assert isinstance(y_proba, pd.Series)
            # transform that probability in a boolean via a decision threshold
            # noinspection PyTypeChecker
            y_bool = (y_proba > decision_threshold)
            assert isinstance(y_bool, pd.Series)

            return y_bool

    def predict_proba(self, query, on_index=None):
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

        table_score_complete = self.scoringmodel.filter_compare(query=query,on_index=on_index)

        if table_score_complete is None or table_score_complete.shape[0] == 0:
            return None
        else:
            # launch prediction using the predict_proba of the scikit-learn module

            y_proba = self.evaluationmodel.predict_proba(table_score_complete)

            del table_score_complete

            # sort the results
            y_proba.sort_values(ascending=False, inplace=True)

            return y_proba

    def _showprobablematches(self, query, n_records=10, display=None):
        """
        Show the best matching recors after the filter_all_any method of the scorer
        Could be of interest to investigate the possible matches of a query
        Args:
            query (pd.Series):
            n_records (int): max number of records to be displayed
            display (list): list of columns to be displayed

        Returns:
            pd.DataFrame, incl. query

        """
        if display is None:
            display = self.compared_cols
        records = pd.DataFrame(columns=['proba']+display)
        records.loc[0]=query[display].copy()
        records.rename(index={0:'query'},inplace=True)
        records.loc['query','proba']='query'

        y_proba=self.predict_proba(query)
        if y_proba is not None and y_proba.shape[0]>0:
            print(y_proba.max())
            y_proba.sort_values(ascending=False,inplace=True)
            n_records = min(n_records, y_proba.shape[0])
            results=self.df.loc[y_proba.index[:n_records], display]
            results['proba']=y_proba
            records = pd.concat([records,results],axis=0)
            return records
        else:
            return None

    def _showfilterstep(self,query,n_records=10,display=None):
        """
        Not used anymore
        Show the best matching recors after the filter_all_any method of the scorer
        Could be of interest to investigate the possible matches of a query
        Args:
            query (pd.Series):
            n_records (int): max number of records to be displayed
            display (list): list of columns to be displayed

        Returns:
            pd.DataFrame, incl query

        """
        if display is None:
            display = self.compared_cols
        records = pd.DataFrame(columns=['totalscore']+display)
        records.loc[0]=query[display].copy()
        records.rename(index={0:'query'},inplace=True)
        records.loc['query', 'totalscore'] = 'query'

        table=self.scoringmodel.filter_all_any(query=query)
        if table is not None and table.shape[0]>0:
            y_sum = table.sum(axis=1)
            print(y_sum.max())
            print(y_sum.max())
            y_sum.sort_values(ascending=False,inplace=True)
            n_records = min(n_records, y_sum.shape[0])
            results=self.df.loc[y_sum.index[:n_records], display]
            results['totalscore']=y_sum
            records = pd.concat([records,results],axis=0)
            return records
        else:
            return None

    def _showscoringstep(self,query,n_records=10,display=None):
        """
        Not used anymore
        Show the total score of the scoring table after the filter_compare method of the scorer
        Could be of interest to investigate the possible matches of a query
        Args:
            query (pd.Series):
            n_records (int): max number of records to be displayed
            display (list): list of columns to be displayed

        Returns:
            pd.DataFrame, incl queryK

        """
        if display is None:
            display = self.compared_cols
        records = pd.DataFrame(columns=['totalscore']+display)
        records.loc[0]=query[display].copy()
        records.rename(index={0:'query'},inplace=True)
        records.loc['query', 'totalscore'] = 'query'

        table = self.scoringmodel.filter_compare(query=query)


        if table is not None and table.shape[0]>0:
            y_sum = table.sum(axis=1)
            print(y_sum.max())
            y_sum.sort_values(ascending=False,inplace=True)
            n_records = min(n_records, y_sum.shape[0])
            results=self.df.loc[y_sum.index[:n_records], display]
            results['totalscore']=y_sum
            records = pd.concat([records,results],axis=0)

            return records
        else:
            return None


def threshold_based_decision(row, thresholds):
    """
    if all values of the row are above the thresholds, return 1, else return 0
    Args:
        row (pd.Series): row to be decided
        thresholds (dict): threshold of values

    Returns:
        float

    """
    #TODO: Explain how this works
    navalue = thresholds.get('fillna')
    if navalue is None:
        navalue = 0
    elif navalue =='dropna':
        row = row.dropna()
    row = row.fillna(navalue)

    aggfunc = thresholds.get('aggfunc')

    if aggfunc == 'all':
        f=all
    elif aggfunc == 'any':
        f= any
    else:
        f=all
    keys=thresholds.keys()
    keys=filter(lambda k:k.endswith('score'),keys)
    result = map(lambda k:row[k]>=thresholds[k],list(keys))
    result = f(result)

    return result
