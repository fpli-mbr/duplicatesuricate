# Duplicate Suricate
This is an entity resolution algorithm (to deduplicate records of companies for example) using available information: Company name, street address, city name, Duns Number, Postal Code, Country, and Lat/Lng location derived from Google Maps.
It relies heavily on pandas to do the heavy indexing work (thus is not really parrelizable), on Levehstein distance to do the string comparison. The prerequisite are my other package neat martinet, and the python fuzzywuzzy package.
