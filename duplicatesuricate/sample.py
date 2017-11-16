import pandas as pd
from duplicatesuricate.scoring.scoringfunctions import Scorer
from duplicatesuricate.configdir.configfile import score1decisionfunc,score2_dict,score1_dict,filter_dict
name_input_record = 'mytable.csv'
name_target_record = 'reference_table.csv'
name_training_table = 'training_table.csv'

if __name__ == '__main__':
    ### Load the reference records and the file to be checked as pd.DataFrame
    df_input_records = pd.read_excel('helicopters.xlsx',index_col=0,sheetname='source')
    df_target_records = pd.read_excel('helicopters.xlsx',index_col=0,sheetname='target')
    query=df_input_records.iloc[0]

    scorer = Scorer(df=df_target_records,
                    filterdict=filter_dict,
                    score1dict=score1_dict,
                    score2dict=score2_dict,
                    score1func=score1decisionfunc)
    results=scorer.filter_compare(query)
    print(results)

    ### results are in the form of a dict {index_of_input_record:[list of index_of_target_records]}
