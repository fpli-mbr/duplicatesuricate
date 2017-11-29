# backbone for doing the deduplication
import numpy as np
import pandas as pd
import neatmartinet as nm

from .recordlinkage import RecordLinker
from .scoring import Scorer

class Launcher:
    def __init__(self, input_records,
                 target_records,
                 linker,
                 cleanfunc=None,
                 idcol='groupid', queryidcol='queryid', verbose=True):
        """
        Args:
            input_records (pd.DataFrame): Input table for record linkage, records to link
            target_records (pd.DataFrame): Table of reference for record linkage
            linker (RecordLinker): record linked
            cleanfunc (func):
            idcol (str): name of the column where to store the deduplication results
            queryidcol (str): name of the column used to store the original match
            verbose (bool): Turns on or off prints
        Returns:
            None
        Examples:
            TODO: Examples
                scorer = Scorer(df=df_target_records,
                    filterdict=filter_dict,
                    score_intermediate=score1_inter_dict,
                    score_further=score2_more_dict,
                    decision_intermediate=intermediate_func)
                model = FuncEvaluationModel(used_cols=hard_cols,
                                            eval_func=hardcodedfunc)
                Lch=Launcher(input_records=df_input,
                            target_records=df_target,
                            cleanfunc=None,
                            scorer=scorer
                            evaluator=model)

        """
        #TODO: Complete docstring
        #TODO: add the possibility of having a group id
        #TODO: Re arrange with Record Linker

        self.linker =linker

        if cleanfunc is None:
            cleanfunc = lambda x:x

        self.input_records = cleanfunc(input_records)

        missingcols=list(filter(lambda x: x not in self.input_records.columns, self.linker.compared_cols))
        if len(missingcols) > 0:
            raise KeyError('RecordLinker does not have all necessary columns in input after cleaning',missingcols)

        self.target_records = cleanfunc(target_records)
        missingcols=list(filter(lambda x: x not in self.target_records.columns, self.linker.compared_cols))
        if len(missingcols) > 0:
            raise KeyError('RecordLinker does not have all necessary columns in target after cleaning',missingcols)

        self.idcol = idcol
        self.queryidcol = queryidcol
        self.verbose = verbose
        pass


    def _generate_query_index_(self, in_index=None):
        """
        this function returns a random index from the input records with no group id to start the linkage process
        Args:
            in_index (pd.Index): index or list, default None the query should be in the selected index

        Returns:
            object: an index of the input records
        """

        if in_index is None:
            in_index = self.input_records.index

        x = self.input_records.loc[in_index]
        possiblechoices = x.loc[(x[self.idcol] == 0) | (x[self.idcol].isnull())].index
        if possiblechoices.shape[0] == 0:
            del x
            return None
        else:
            a = np.random.choice(possiblechoices)
            del x, possiblechoices
            return a

    def _find_matches_(self, query_index, n_matches_max=1):
        """
       search for records in the target records that match the query (input_records.loc[query_index])
        Args:
            query_index: index of the row to be deduplicated
            n_matches_max (int): max number of matches to be fetched. If None, all matches would be returned

        Returns:
            pd.Index (list of index in the target records)
        """

        # return the good matches as calculated by the evaluation

        goodmatches_index = self.linker.return_good_matches(query=self.input_records.loc[query_index])

        if goodmatches_index is None or len(goodmatches_index) == 0:
            return None
        elif n_matches_max is None:
            return goodmatches_index
        else:
            return goodmatches_index[:n_matches_max]

    def start_linkage(self, in_index=None, n_matches_max=1):
        """
        Takes as input an index of the input records, and returns a dict showing their corresponding matches
        on the target records
        Args:
            in_index (pd.Index): index of the records (from input_records) to be deduplicated
            n_matches_max (int): maximum number of possible matches to be returned.
                If none, all matches would be returned

        Returns:
            dict : results in the form of {index_of_input_record:[list of index_of_target_records]}
        """
        if in_index is None:
            in_index = self.input_records.index

        n_total = len(in_index)

        #TODO: find why I wrote that...
        #if find_missing_keys_in_index(in_index, self.input_records.index) is True:
        #    raise KeyError('in_index called is not contained in input_records index')

        print('starting deduplication at {}'.format(pd.datetime.now()))
        results = {}
        for i, ix in enumerate(in_index):
            # timing
            time_start = pd.datetime.now()

            goodmatches_index = self._find_matches_(query_index=ix, n_matches_max=n_matches_max)

            if goodmatches_index is None:
                results[ix] = None
                n_deduplicated = 0
            else:
                results[ix] = list(goodmatches_index)
                n_deduplicated = len(results[ix])

            # timing
            time_end = pd.datetime.now()
            duration = (time_end - time_start).total_seconds()

            if self.verbose:
                print('{} of {} deduplicated | time elapsed {} s'.format(n_deduplicated, n_total, duration))

        print('finished work at {}'.format(pd.datetime.now()))
        if n_matches_max == 1:
            convertlisttovalue=lambda v:None if v is None else v[0]
            for k in results.keys():
                results[k]=convertlisttovalue(results[k])
        return results

    def format_results(self, res, display, fuzzy=None):
        """
        Return a formatted, side by side comparison of results
        Args:
            res (dict): results
            display (list): list of columns to be displayed
            fuzzy (list): list of columns on which to perform fuzzy score
        Returns:
            pd.DataFrame
        """
        # x = pd.Series(index=list(r.keys()),values=list(r.keys()))
        # df=x.apply(lambda r:pd.Series(r))
        # assert isinstance(df,pd.DataFrame)
        keys=list(res.keys())
        x=pd.Series(index=keys,data=keys, name='ix_source')
        x.index.name='ix_source'
        df=x.apply(lambda r:pd.Series(res[r]))
        assert isinstance(df,pd.DataFrame)
        if df.shape[1]>1:
            raise(ValueError('Too many columns to unpack, expected only one result per query'))
        df.rename(columns={0:'ix_target'},inplace=True)
        df.reset_index(inplace=True,drop=False)
        df.rename(columns={'index': 'ix_source'}, inplace=True)
        df.dropna(inplace=True)

        if fuzzy is None:
            allcols=display
        else:
            allcols=list(set(display+fuzzy))
        for c in allcols:
            df[c+'_source']=df['ix_source'].apply(lambda r:self.input_records.loc[r,c])
            df[c+'_target']=df['ix_target'].apply(lambda r:self.target_records.loc[r,c])
        if fuzzy is not None:
            for c in fuzzy:
                df[c+'_score']=df.apply(lambda r:nm.compare_twostrings(r[c+'_source'],r[c+'_target']),axis=1)
        return df