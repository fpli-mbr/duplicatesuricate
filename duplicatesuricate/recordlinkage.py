# coding=utf-8
"""
Machine Learning model used for record linkage
"""
training_filename = 'filename.csv'
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
from duplicatesuricate import scoringfunctions


class RecordLinker:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.model = RandomForestClassifier(n_estimators=2000)

        self.df = pd.DataFrame()
        self.query = pd.Series()
        self.traincols = []
        # scoring threshold for fuzzy score filtering
        self.filter_threshold = 0.8
        # scoring threshold using model predict_proba
        self.decision_threshold = 0.6

        pass

    def train(self, warmstart=False, training_set=pd.DataFrame(), target_col='ismatch'):
        """
        train the machine learning model on the provided data set
        Args:
            warmstart (bool): if the model is already trained
            training_set (pd.DataFrame): pandas DataFrame containing annotated data
            target_col (str):name of the target vector in the training_set

        Returns:
            None

        """
        start = pd.datetime.now()

        if warmstart is True:
            if training_set.shape[0] == 0:
                training_set = pd.read_csv(training_filename, nrows=1)

            if target_col not in training_set.columns:
                raise KeyError('target column ', target_col, ' not found in training set columns')

            # Define training set and target vector
            self.traincols = list(filter(lambda x: x != target_col, training_set.columns))

        else:
            if training_set.shape[0] == 0:
                training_set = pd.read_csv(training_filename)

            if target_col not in training_set.columns:
                raise KeyError('target column ', target_col, ' not found in training set columns')

            if self.verbose:
                print('shape of training table ', training_set.shape)
                print('number of positives in table', training_set[target_col].sum())

            # Define training set and target vector
            self.traincols = list(filter(lambda x: x != target_col, training_set.columns))
            X_train = training_set[self.traincols].fillna(-1)  # fill na values
            y_train = training_set[target_col]

            # fit the model
            self.model.fit(X_train, y_train)

        if self.verbose:
            # show precision and recall score of the model on training data
            y_pred = self.model.predict(X_train)
            precision = precision_score(y_true=y_train, y_pred=y_pred)
            recall = recall_score(y_true=y_train, y_pred=y_pred)
            print('precision score on training data:', precision)
            print('recall score on training data:', recall)

        if self.verbose:
            end = pd.datetime.now()
            duration = (end - start).total_seconds()
            print('time elapsed', duration, 'seconds')

        return None

    def return_good_matches(self, target_records, query):
        """

        Args:
            target_records (pd.DataFrame): the table containing the target records
            query (pd.Series): information available on our query

        Returns:
            pd.Index: the index of the target records identified as the same as the query by the algorithm

        """
        y_bool = self.predict(target_records, query)
        return y_bool.loc[y_bool].index

    def predict(self, target_records, query):
        """
        
        Args:
            target_records (pd.DataFrame): the table containing the target records
            query (pd.Series): information available on our query

        Returns:
            pd.Series: a boolean vector: True if it is a match, false otherwise

        """
        # calculate the probability of the records being the same as the query through the machine learning model
        y_proba = self.predict_proba(target_records, query)

        # transform that probability in a boolean via a decision threshold
        y_bool = (y_proba > self.decision_threshold)

        return y_bool

    def predict_proba(self, target_records, query):
        """
        This
        Args:
            target_records (pd.DataFrame): the table containing the target records
            query (pd.Series): information available on our query

        Returns:
            pd.Series : the probability vector of the target records being the same as the query
            
        """
        self.df = target_records
        self.query = query.copy()

        filtered_index = self.pre_filter_records()

        table_score = self.create_similarity_features(filtered_index)

        predictions = self.step_three_predict(table_score)

        return predictions

    def pre_filter_records(self):
        """
        for each row for a query, returns True (ismatch) or False (is not a match)
        No arguments since the query has been saved in self.query
        Args:

        Returns:
            bool: The True if it is a match False otherwise
        """
        filtered_score = self.df['Index'].apply(lambda r: scoringfunctions.parallel_filter(self, r))

        return filtered_score.loc[filtered_score > self.filter_threshold].index

    def create_similarity_features(self, filtered_index):
        """
        Return a comparison table for all indexes of the filtered_index (as input)
        Args:
            filtered_index (pd.Index): index of rows on which to perform the deduplication

        Returns:
            pd.DataFrame, table of scores, where each column is a score (should be the same as self.traincols).
        """
        table_score = self.df.loc[filtered_index, 'Index'].apply(
            lambda r: scoringfunctions.parallel_calculate_comparison_score(self, r))
        return table_score

    def calculate_probability(self, table_score):
        """
        from a scoring table, apply the model and return the probability of being a match
        Args:
            table_score (pd.DataFrame): scoring table

        Returns:
            pd.Series of being a match
        
        Careful that column names, length and order is the same as the training table
        """
        # check column length are adequate
        if len(self.traincols) != len(table_score.columns):
            additional_columns = list(filter(lambda x: x not in self.traincols, table_score.columns))
            if len(additional_columns) > 0:
                print('unknown columns in traincols', additional_columns)
            missing_columns = list(filter(lambda x: x not in table_score.columns, self.traincols))
            if len(missingcolumns) > 0:
                print('columns not found in scoring vector', missing_columns)
        # check column order
        table_score = table_score[self.traincols]

        y_proba = pd.DataFrame(self.model.predict_proba(table_score), index=table_score.index)[1]

        return y_proba
