### Filtering function column names
loc_cols=['location','constructor']
id_cols=['serieid']
filter_dict = {'all': loc_cols, 'any': id_cols}

score1_inter_dict = {'attributes': None,
               'fuzzy': ['productname'],
               'token': None,
               'exact': None,
               'acronym': None}
threshold_filter = 0.5

def intermediate_func(r):
    decisioncols=[c +'_fuzzyscore' for c in score1_inter_dict['fuzzy']]
    data_row=r[decisioncols].fillna(0)
    min_row_value=data_row.min()
    result = (min_row_value > threshold_filter)
    return result

score2_more_dict = {'attributes': None,
               'fuzzy': None,
               'token': ['productname'],
               'exact': None,
               'acronym': None}


#Config record linker
decision_threshold=0.8

#Config Hardcoded model
hard_threshold={'productname_fuzzyscore':0.5,
                'location_exactscore':1,
                'serieid_exactscore':1}

hard_cols=list(hard_threshold.keys())
def hardcodedfunc(r):
    r=r.fillna(0)
    for k in hard_cols:
        if r[k] < hard_threshold[k]:
            print(k,r[k])
            return 0
    else:
        return 1


#Config RF Model
_training_path = '/Users/paulogier/Documents/8-PythonProjects/02-test_suricate/'
_training_name= 'training_table_prepared_20171107_79319rows.csv'
training_filename = _training_path+_training_name