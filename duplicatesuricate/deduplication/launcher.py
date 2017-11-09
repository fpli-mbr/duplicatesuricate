# backbone for doing the deduplication
import pandas as pd
import numpy as np
from duplicatesuricate.preprocessing.companycleaning import clean_db
from duplicatesuricate.configdir import configfile
from duplicatesuricate.recordlinkage import RecordLinker


class Launcher:
    def __init__(self, input_records=pd.DataFrame(), target_records=pd.DataFrame(), idcol=configfile.idcol,
                 queryidcol=configfile.queryidcol, verbose=True):
        """
        Args:
            input_records (pd.DataFrame): Input table for record linkage, records to link
            target_records (pd.DataFrame): Table of reference for record linkage
            idcol (str): name of the column where to store the deduplication results
            queryidcol (str): name of the column used to store the original match
            verbose (bool): Turns on or off prints
        """
        self.input_records = input_records
        self.target_records = target_records
        self.idcol = idcol
        self.queryidcol = queryidcol
        self.verbose = verbose
        self.model = RecordLinker()
        pass

    def clean_records(self, source=True, target=True, cleanfunc=clean_db):
        """
        Launch the preprocessing (feature enrichment and calculations)
        Args:
            source (bool): whether to clean the input records
            target (bool): whether to clean the target records
            cleanfunc (function): function to be applied

        Returns:
            None
        """

        if source:
            self.input_records = cleanfunc(self.input_records)
        if target:
            self.target_records = cleanfunc(self.target_records)
        return None

    def add_model(self, model=RecordLinker()):
        """
        add trained record linkage model
        Args:
            model: RecordLinker

        Returns:
            None
        """
        self.model = model

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

        # return the good matches as calculated by the model

        goodmatches_index = self.model.return_good_matches(target_records=self.target_records,
                                                           query=self.input_records.loc[query_index])
        if goodmatches_index is None or len(goodmatches_index) == 0:
            return None
        elif n_matches_max is None:
            return goodmatches_index
        else:
            return goodmatches_index[:n_matches_max]

    def link_input_to_target(self, in_index=None, n_matches_max=1):
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

        if find_missing_keys_in_index(in_index, self.input_records.index) is True:
            raise KeyError('in_index called is not contained in input_records index')

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

        return results


def find_missing_keys_in_index(keys, ref_list, verbose=True):
    """
    Takes as input the a list of keys, and check if they are present in a reference list
    For example, make sure that all of the keys are in the index before launching a loop
    Args:
        keys (iterable): list of keys to be checked
        ref_list (iterable): list of reference keys
        verbose (bool): whether or not to print the statements

    Returns:
        bool: If True, then keys are missing
    """
    incorrect_keys = list(filter(lambda x: x not in ref_list, keys))
    if len(incorrect_keys) > 0:
        if verbose:
            print('those keys are missing in the index:', incorrect_keys)
        return True
    else:
        return False
