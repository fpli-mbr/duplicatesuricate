### Filtering function column names
loc_cols=['location','constructor']
id_cols=['serieid']
filter_dict = {'all': loc_cols, 'any': id_cols}

score1_dict = {'attributes': None,
               'fuzzy': ['productname'],
               'token': None,
               'exact': None,
               'acronym': None}
threshold_filter = 0.5

def score1decisionfunc(r):
    decisioncols=[c+'_fuzzyscore' for c in score1_dict['fuzzy']]
    data_row=r[decisioncols].fillna(0)
    min_row_value=data_row.min()
    result = (min_row_value > threshold_filter)
    return result

score2_dict = {'attributes': None,
               'fuzzy': None,
               'token': None,
               'exact': None,
               'acronym': None}

# #### SIMILITARITY TABLE COLUMN NAMES ####
# # COLUMN NAMES
# attributescols = ['companyname_len']
# fuzzyscorecols = ['companyname',
#                            'streetaddress',  'cityname', 'postalcode',
#                       'companyname_wostopwords', 'companyname_acronym','streetaddress_wostopwords']
# tokenscorecols = []
# exactscorecols = ['country', 'dunsnumber', 'taxid',
#                            'registerid','postalcode_1stdigit', 'postalcode_2digits']
#
# acronymscorecols = ['companyname']

# id_cols = ['dunsnumber', 'taxid', 'registerid1', 'registerid2', 'kapisid', 'cageid']
# loc_cols = ['country']

# # Similarity cols
# similarity_cols = [c + '_row' for c in attributescols] + [c + '_query' for c in attributescols]
# similarity_cols+=[c + '_fuzzyscore' for c in fuzzyscorecols]
# similarity_cols+=[c + '_tokenscore' for c in tokenscorecols]
# similarity_cols+=[c + '_exactscore' for c in exactscorecols]
# similarity_cols+=[acronymscorecols + '_acronym_fuzzyscore']
#
# # columns needed in the input and target file
# datacols=id_cols.copy()
# datacols+=[loc_col]
# datacols+=fuzzy_filter_cols.copy()
# datacols+=attributescols.copy()
# datacols+=fuzzyscorecols.copy()
# datacols+=tokenscorecols.copy()
# datacols+=exactscorecols.copy()
# datacols+=[acronymscorecols]

#
decision_threshold=0.8
_training_path = '/Users/paulogier/Documents/8-PythonProjects/02-test_suricate/'
_training_name= 'training_table_prepared_20171107_79319rows.csv'
training_filename = _training_path+_training_name