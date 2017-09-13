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
 - uses a training table (attached) to train the model
 
 ### Process
 ####Per row
 ####Filter
 ####Score
 ####Predict
 ####Update group id
 
 

