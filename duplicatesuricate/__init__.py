from .deduplication import Launcher,RecordLinker
from .evaluation import funcmodel

# from .configdir import configfile
# import pandas as pd
#
# def Basic_init(input_records, target_records, training_set, n_estimators=500, verbose=True):
#     """
#
#     Args:
#         input_records (pd.DataFrame):
#         target_records (pd.DataFrame):
#         training_set (pd.DataFrame):
#         n_estimators (int): number of estimators for the random forest algorithm
#         verbose (bool): verbose or not
#
#     Returns:
#         Launcher
#     """
#     suricate = Launcher(input_records=input_records, target_records=target_records,verbose=verbose)
#     suricate.clean_records()
#
#     irl = RFLinker(n_estimators=n_estimators, verbose=verbose)
#     irl.fit(training_set=training_set)
#     suricate.add_model(decisionmodel=irl)
#
#     return suricate