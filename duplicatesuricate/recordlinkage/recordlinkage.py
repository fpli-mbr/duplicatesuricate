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

            if all(map(lambda x: x in traincols, config.similarity_cols)) is False:
                raise KeyError(
                    'output of scoring function and training columns do not match, check config file or Training file')

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
            print('len(goodmatches)', 0)
            return None
        else:
            goodmatches = y_bool.loc[y_bool].index
            print('len(goodmatches)', len(goodmatches))
            return goodmatches

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
        start = pd.datetime.now()

        # pre filter the records for further scoring: 0 if not possible choice, 0.5 if possible choice, 1 if sure choice
        y_pre_filter = self.pre_filter_records()
        y_pre_filter = y_pre_filter.loc[y_pre_filter > 0]
        ix_pre_filter = y_pre_filter.index

        end = pd.datetime.now()
        print('time pre-filtering', (end - start).total_seconds(), 'len(pre_filtering)', len(ix_pre_filter))
        start = end

        if len(ix_pre_filter) == 0:
            return None
        else:
            # do further scoring on the possible choices and the sure choices
            table_score_filter = self.create_similarity_features(filtered_index=ix_pre_filter, query=self.query,
                                                                 feature_cols=[],
                                                                 fuzzy_feature_cols=self.fuzzy_filter_cols)
            # add the y_pre_filter vector to make sure we select the sure choices (ids, ...) where the value is 1
            table_score_filter['y_pre_filter'] = y_pre_filter

            # this gives us a score: 1 if choice, max(fuzzy_score for filtered columns) for possible choices
            y_filter_score = table_score_filter.max(axis=1)

            # we then filter on that score with a pre-defined threshold
            ix_filter = y_filter_score.loc[y_filter_score > self.filter_threshold].index
            table_score_filter = table_score_filter.loc[ix_filter]

            # we remove the filter_score
            table_score_filter.drop(labels=['y_pre_filter'], axis=1, inplace=True)

            end = pd.datetime.now()
            print('time filtering', (end - start).total_seconds(), 'len(ix_filter)', len(ix_filter))
            start = end

            if table_score_filter.shape[0] == 0:
                return None
            else:
                # we perform further analysis on the filtered index:
                # we complete the fuzzy score with additional columns
                additional_fuzzy_cols = [c for c in self.fuzzy_feature_cols if not c in self.fuzzy_filter_cols]

                # we also include the exact matching, the features, the tokens matching
                table_score_additional = self.create_similarity_features(ix_filter, query=self.query,
                                                                         feature_cols=self.feature_cols,
                                                                         fuzzy_feature_cols=additional_fuzzy_cols,
                                                                         tokens_feature_cols=self.tokens_feature_cols,
                                                                         acronym_col=self.acronym_col,
                                                                         exact_feature_cols=self.exact_feature_cols)

                # we join the two tables to have a complete view of the score
                table_score_complete = table_score_filter.join(table_score_additional, how='left')

                end = pd.datetime.now()
                print('time additional scoring', (end - start).total_seconds())
                start = end

                # check column length are adequate
                traincols = pd.Index(self.traincols)
                traincols.name = 'training table'
                scorecols = table_score_complete.columns
                scorecols.name = 'scoring table'
                if check_column_same(traincols, scorecols) is False:
                    raise KeyError('training columns and scoring columns differ')
                del traincols, scorecols

                # re-arrange the column order
                table_score_complete = table_score_complete[self.traincols]

                # fill the na values
                table_score_complete = table_score_complete.fillna(-1)

                # launch prediction using the predict_proba of the scikit-learn module
                y_proba = \
                pd.DataFrame(self.model.predict_proba(table_score_complete), index=table_score_complete.index)[1].copy()

                end = pd.datetime.now()
                print('time predicting', (end - start).total_seconds())
                del table_score_complete

                # sort the results
                y_proba.sort_values(ascending=False, inplace=True)

                return y_proba

    def pre_filter_records(self):
        """
        for each row for a query, returns True (ismatch) or False (is not a match)
        No arguments since the query has been saved in self.query
        Args:

        Returns:
            pd.Series: the score of the potential matches in the target records table
        """
        filtered_float = scoringfunctions.filter_df(df=self.df,
                                                    query=self.query,
                                                    id_cols=self.id_cols,
                                                    loc_col=self.loc_col)
        return filtered_float

    def create_similarity_features(self, filtered_index, query, feature_cols=None, fuzzy_feature_cols=None,
                                   tokens_feature_cols=None,
                                   exact_feature_cols=None, acronym_col=None):
        """
        Return a comparison table for all indexes of the filtered_index (as input)
        Args:
            filtered_index (pd.Index): index of rows on which to perform the deduplication

        Returns:
            pd.DataFrame, table of scores, where each column is a score (should be the same as self.traincols).
        """
        table_score = scoringfunctions.build_similarity_table(df=self.df.loc[filtered_index],
                                                              query=query,
                                                              feature_cols=feature_cols,
                                                              fuzzy_feature_cols=fuzzy_feature_cols,
                                                              tokens_feature_cols=tokens_feature_cols,
                                                              exact_feature_cols=exact_feature_cols,
                                                              acronym_col=acronym_col
                                                              )

        return table_score


def check_column_same(a, b):
    if set(a) == set(b):
        return True
    else:
        missing_a_columns = list(filter(lambda x: x not in a, b))
        if len(missing_a_columns) > 0:
            print('unknown columns from', b.name, 'not in', a.name, ':', missing_a_columns)
        missing_b_columns = list(filter(lambda x: x not in b, a))
        if len(missing_b_columns) > 0:
            print('unknown columns from', a.name, 'not in', b.name, ':', missing_b_columns)
        return False
