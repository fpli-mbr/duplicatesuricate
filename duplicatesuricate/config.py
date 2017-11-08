

# Name of the training table, formatted utf-8 with pipe | separator
_training_path = '/Users/paulogier/Documents/8-PythonProjects/02-test_suricate/'
_training_name= 'training_table_prepared_20171107_79319rows.csv'
training_filename = _training_path+_training_name

### Filtering function column names
id_cols = ['dunsnumber', 'taxid', 'registerid']
loc_col = 'country'
fuzzy_filter_cols = ['streetaddress', 'companyname']

#### SIMILITARITY TABLE COLUMN NAMES ####
# COLUMN NAMES
feature_cols = []
fuzzy_feature_cols = ['companyname',
                           'streetaddress',  'cityname', 'postalcode']
tokens_feature_cols = []
exact_feature_cols = ['country', 'dunsnumber', 'taxid',
                           'registerid']
acronym_col = 'companyname'
# Similarity cols
similarity_cols = [c + '_row' for c in feature_cols] + [c + '_query' for c in feature_cols]
similarity_cols+=[c + '_fuzzyscore' for c in fuzzy_feature_cols]
similarity_cols+=[c + '_tokenscore' for c in tokens_feature_cols]
similarity_cols+=[c + '_exactscore' for c in exact_feature_cols]
similarity_cols+=[acronym_col + '_acronym_fuzzyscore']
# columns needed in the input and target file
datacols=id_cols.copy()
datacols+=[loc_col]
datacols+=fuzzy_filter_cols.copy()
datacols+=feature_cols.copy()
datacols+=fuzzy_feature_cols.copy()
datacols+=tokens_feature_cols.copy()
datacols+=exact_feature_cols.copy()
datacols+=[acronym_col]

#
idcol = 'groupid'
queryidcol='original_query_name'

# List of stopwords
companystopwords_list = ['aerospace',
                         'ag',
                         'and',
                         'co',
                         'company',
                         'consulting',
                         'corporation',
                         'de',
                         'deutschland',
                         'dr',
                         'electronics',
                         'engineering',
                         'europe',
                         'formation',
                         'france',
                         'gmbh',
                         'group',
                         'hotel',
                         'inc',
                         'ingenierie',
                         'international',
                         'kg',
                         'la',
                         'limited',
                         'llc',
                         'ltd',
                         'ltda',
                         'management',
                         'of',
                         'oy',
                         'partners',
                         'restaurant',
                         'sa',
                         'sarl',
                         'sas',
                         'service',
                         'services',
                         'sl',
                         'software',
                         'solutions',
                         'srl',
                         'systems',
                         'technologies',
                         'technology',
                         'the',
                         'uk',
                         'und']
streetstopwords_list = ['avenue', 'calle', 'road', 'rue', 'str', 'strasse', 'strae']
endingwords_list = ['strasse', 'str', 'strae']
bigcities = ['munich',
             'paris',
             'madrid',
             'hamburg',
             'toulouse',
             'berlin',
             'bremen',
             'london',
             'ulm',
             'stuttgart', 'blagnac']
airbus_names = ['airbus', 'casa', 'eads', 'cassidian', 'astrium', 'eurocopter']

# Column names
# id_cols = ['dunsnumber', 'taxid', 'registerid']
# loc_col = ['country']
# fuzzy_filter_cols = ['streetaddress', 'companyname']
# feature_cols = ['companyname_wostopwords_wordfrequency',
#                      'companyname_len', 'companyname_wostopwords_len', 'streetaddress_len',
#                      'companyname_wostopwords_ntokens', 'cityfrequency', 'isbigcity', 'has_airbusequiv']
# fuzzy_feature_cols = ['companyname', 'companyname_wostopwords', 'companyname_acronym',
#                            'streetaddress', 'streetaddress_wostopwords', 'cityname', 'postalcode']
# tokens_feature_cols = ['companyname_wostopwords', 'streetaddress_wostopwords']
# exact_feature_cols = ['country', 'state', 'dunsnumber', 'postalcode_1stdigit', 'postalcode_2digits', 'taxid',
#                            'registerid']
# acronym_col = 'companyname'