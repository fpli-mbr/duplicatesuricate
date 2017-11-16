import pandas as pd
from duplicatesuricate.scoring import scoringfunctions
from duplicatesuricate.configdir import configfile

def __init__(self, verbose=True,
             df=pd.DataFrame(),model=None):
    """
    The machine-learning par of the program
    Args:
        verbose (bool): control print output
        df (pd.DataFrame): target records
        id_cols: names of the columns which contain an id
        loc_col: name of the column of the location (often, country)
        fuzzy_filter_cols: fuzzy scoring cols for filtering
        feature_cols (list): list of columns with numerical values related to the query description (string length..)
        fuzzy_feature_cols (list): list of columns on which to calculate fuzzy string matching
        tokens_feature_cols (list): list of columns on which to calculate fuzzy token string matching
        exact_feature_cols (list): list of columns on which to calculate an exact matching (0 or 1)
        acronym_col (str): name of column on which to calculate the acronym score
        n_estimators (int): number of estimators for the Random Forest Algorithm
    """
    self.verbose = verbose

    self.df = df
    self.query = pd.Series()

    self.scorer = scoringfunctions.Scorer(df=self.df,
                                          filterdict=configfile.filter_dict,
                                          score1dict=configfile.score1_dict,
                                          score1func=configfile.score1decisionfunc,
                                          score2dict=configfile.score2_dict)
    self.model = model
    self.decision_threshold=configfile.decision_threshold

    if all(map(lambda x: x in self.scorer.existingcols, self.model.traincols)) is False:
        raise KeyError('not all training columns are found in the output of the scorer')
    pass

    def return_good_matches(self, target_records, query):
        """

        Args:
            target_records (pd.DataFrame): the table containing the target records
            query (pd.Series): information available on our query

        Returns:
            pd.Index: the index of the target records identified as the same as the query by the algorithm

        """
        y_bool = self.predict(target_records, query)
        if y_bool is None:
            return None
        else:
            goodmatches = y_bool.loc[y_bool].index
            return goodmatches


    def predict(self, target_records, query):
        """

        Args:
            target_records (pd.DataFrame): the table containing the target records
            query (pd.Series): information available on our query

        Returns:
            pd.Series: a boolean vector: True if it is a match, false otherwise

        """
        # calculate the probability of the records being the same as the query through the machine learning decisionmodel
        y_proba = self.predict_proba(target_records, query)
        if y_proba is None:
            return None
        else:
            # transform that probability in a boolean via a decision threshold
            y_bool = (y_proba > self.decision_threshold)
            return y_bool


    def predict_proba(self, target_records, query):
        """
        This is the heart of the program:
        - pre-filter all records
        - filter using fuzzy matching on selected columns, then select those above a certain threshold
        - do additional comparison on other columns
        - predict_proba on the table of comparison score using the ML decisionmodel
        - return probability of being a match
        Args:
            target_records (pd.DataFrame): the table containing the target records
            query (pd.Series): information available on our query

        Returns:
            pd.Series : the probability vector of the target records being the same as the query

        """
        self.df = target_records
        table_score_complete = self.scorer.filter_compare(query=query)

        if table_score_complete is None or table_score_complete.shape[0] == 0:
            return None
        else:
            # launch prediction using the predict_proba of the scikit-learn module
            y_proba = \
                pd.DataFrame(self.model.predict_proba(table_score_complete), index=table_score_complete.index)[
                    1].copy()

            del table_score_complete

            # sort the results
            y_proba.sort_values(ascending=False, inplace=True)

            return y_proba