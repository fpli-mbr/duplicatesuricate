# Duplicate Suricate
This is an entity resolution algorithm (to deduplicate records of companies for example) using available information: Company name, street address, city name, Duns Number, Postal Code, Country, and Lat/Lng location derived from Google Maps.
It relies heavily on:
 - numpy and pandas to do the heavy indexing work (thus is not really parrelizable)
 - fuzzywuzzy (Levehstein distance) to do the string comparison
 - scikit-learn to do the predictions
 
 ##How does it work?
 ###Init
 ####Input table
 - it takes a pandas.DataFrame as input
 - the columns of interests for the deduplication can be names (company names), locations (cityname, street address, postal code), or IDs (taxid...)
 - you feed the table in the Suricate() class and it "cleans" the data
 #### Learning Model
 - uses as standard a Scikit-Learn Random Forest model
 - uses a training table to train the model
 
### Example

import pandas as pd
from duplicatesuricate import Basic_init

#### Load the reference records and the file to be checked as pd.DataFrame
df_input_records=pd.read_csv('mytable.csv',index_col=0)
df_target_records=pd.read_csv('reference_table.csv',index_col=0)

#### Load the training table
df_training_table = pd.read_csv('training_table.csv')

#### Feed the records into the Launcher, train the model
suricate = Basic_init(input_records=df_input_records,target_records=df_target_records,
                      training_set=df_training_table)

#### Launch the record linkage
results=suricate.link_input_to_target()

# results in the form of {index_of_input_record:[list of index_of_target_records]}


 

