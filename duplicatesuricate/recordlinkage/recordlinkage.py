# coding=utf-8
"""
Machine Learning model used for record linkage
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score

from duplicatesuricate.recordlinkage import scoringfunctions
from duplicatesuricate import config


class RecordLinker:
    def __init__(self, verbose=True,
                 df=pd.DataFrame(),
                 id_cols=config.id_cols, loc_col=config.loc_col, fuzzy_filter_cols=config.fuzzy_filter_cols,
                 feature_cols=config.feature_cols,
                 fuzzy_feature_cols=config.fuzzy_feature_cols,
                 tokens_feature_cols=config.tokens_feature_cols,
                 exact_feature_cols=config.exact_feature_cols,
                 acronym_col=config.acronym_col,
                 n_estimators=2000):
        self.verbose = verbose
        self.model = RandomForestClassifier(n_estimators=n_estimators)

        self.df = df
        self.query = pd.Series()
        self.traincols = []

        # scoring threshold for fuzzy score filtering
        self.filter_threshold = 0.8
        # scoring threshold using model predict_proba
        self.decision_threshold = 0.6

        # name and list of column names to be used
        self.id_cols = id_cols
        self.loc_col = loc_col
        self.fuzzy_filter_cols = fuzzy_filter_cols
        self.feature_cols = feature_cols
        self.fuzzy_feature_cols = fuzzy_feature_cols
        self.tokens_feature_cols = tokens_feature_cols
        self.exact_feature_cols = exact_feature_cols
        self.acronym_col = acronym_col
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
        X_train = pd.DataFrame()
        y_train = pd.Series()

        if warmstart is True:
            if training_set.shape[0] == 0:
                training_set = pd.read_csv(config.training_filename, nrows=1, encoding='utf-8', sep=',')

            if target_col not in training_set.columns:
                raise KeyError('target column ', target_col, ' not found in training set columns')

            # Define training set and target vector
            self.traincols = list(filter(lambda x: x != target_col, training_set.columns))

        else:
            if training_set.shape[0] == 0:
                training_set = pd.read_csv(config.training_filename, encoding='utf-8', sep=',')

            if target_col not in training_set.columns:
                raise KeyError('target column ', target_col, ' not found in training set columns')

            if self.verbose:
                print('shape of training table ', training_set.shape)
                print('number of positives in table', training_set[target_col].sum())

            # Check that the training columns
            traincols = list(filter(lambda x: x != target_col, training_set.columns))

            if all(map(lambda x:x in traincols, config.similarity_cols)) is False:
                raise KeyError('output of scoring function and training columns do not match, check config file or Training file')

            self.traincols = config.similarity_cols

            # Define training set and target vector
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
        if y_bool is None:
            return None
        else:
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
        if y_proba is None:
            return None
        else:
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
        if len(filtered_index) == 0:
            return None
        else:
            table_score = self.create_similarity_features(filtered_index)

            # check column length are adequate
            if len(self.traincols) != len(table_score.columns):
                additional_columns = list(filter(lambda x: x not in self.traincols, table_score.columns))
                if len(additional_columns) > 0:
                    print('unknown columns in traincols', additional_columns)
                missing_columns = list(filter(lambda x: x not in table_score.columns, self.traincols))
                if len(missing_columns) > 0:
                    print('columns not found in scoring vector', missing_columns)
            # check column order
            table_score = table_score[self.traincols]

            y_proba = pd.DataFrame(self.model.predict_proba(table_score), index=table_score.index)[1]

            return y_proba

    def pre_filter_records(self):
        """
        for each row for a query, returns True (ismatch) or False (is not a match)
        No arguments since the query has been saved in self.query
        Args:

        Returns:
            pd.Index: the index of the potential matches in the target records table
        """
        filtered_score = scoringfunctions.filter_df(df=self.df,
                                                    query=self.query,
                                                    id_cols=self.id_cols,
                                                    loc_col=self.loc_col,
                                                    fuzzy_filter_cols=self.fuzzy_filter_cols)

        return filtered_score.loc[filtered_score > self.filter_threshold].index

    def create_similarity_features(self, filtered_index):
        """
        Return a comparison table for all indexes of the filtered_index (as input)
        Args:
            filtered_index (pd.Index): index of rows on which to perform the deduplication

        Returns:
            pd.DataFrame, table of scores, where each column is a score (should be the same as self.traincols).
        """
        table_score = scoringfunctions.build_similarity_table(df=self.df.loc[filtered_index],
                                                              query=self.query,
                                                              feature_cols=self.feature_cols,
                                                              fuzzy_feature_cols=self.fuzzy_feature_cols,
                                                              tokens_feature_cols=self.tokens_feature_cols,
                                                              exact_feature_cols=self.exact_feature_cols,
                                                              acronym_col=self.acronym_col,
                                                              traincols=self.traincols
                                                              )
        return table_score