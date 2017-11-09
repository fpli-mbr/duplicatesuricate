import pandas as pd
from duplicatesuricate import Basic_init

name_input_record = 'mytable.csv'
name_target_record = 'reference_table.csv'
name_training_table = 'training_table.csv'

if __name__ == '__main__':
    ### Load the reference records and the file to be checked as pd.DataFrame
    df_input_records = pd.read_csv(name_input_record,index_col=0)
    df_target_records = pd.read_csv(name_target_record,index_col=0)

    ### Load the training table
    df_training_table = pd.read_csv(name_training_table)

    ### Feed the records into the Launcher, train the model
    suricate = Basic_init(input_records=df_input_records,target_records=df_target_records,training_set=df_training_table)

    ### Launch the record linkage
    results=suricate.link_input_to_target()

    ### results are in the form of a dict {index_of_input_record:[list of index_of_target_records]}
