# backbone for doing the deduplication
import pandas as pd
import numpy as np
from duplicatesuricate.preprocessing.companycleaning import clean_db
import duplicatesuricate.config as config
from duplicatesuricate.recordlinkage import RecordLinker


class Launcher:
    def __init__(self, input_records=pd.DataFrame(), target_records=pd.DataFrame(),  idcol=config.idcol,
                 queryidcol=config.queryidcol, verbose=True):
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
        Deduplicate a row (search for duplicates in the target records and update the groupid col)
        Args:
            query_index: index of the row to be deduplicated
            n_matches_max (int): max number of matches to be fetched

        Returns:
            matches
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
        
        Args:
            in_index (pd.Index): 
            n_matches_max (int): 

        Returns:
            pd.DataFrame results
        """
        if in_index is None:
            in_index = self.input_records.index
        results = pd.DataFrame()
        for i, ix in enumerate(in_index):
            goodmatches_index = self._find_matches_(query_index=ix, n_matches_max=n_matches_max)
            if goodmatches_index is None:
                results.loc[ix] = None
            else:

                s = pd.Series(data=goodmatches_index, index=range(len(goodmatches_index)), name=ix)
                results=results.append(s)
        return results

