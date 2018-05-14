import pandas as pd
def _generate_words_sample():
    s1 = 'hello world'
    s2 = 'holy grail'
    s3 = 'hello big world'
    s4 = 'hello bigger world'
    s5 = 'holy molly'
    s6 = 'hello grail'
    mylist = [s1, s2, s3, s4, s5, s6]
    from itertools import combinations
    dp = pd.DataFrame(list(combinations(mylist, 2)))
    dp.columns = ['left', 'right']
    return dp
def _generate_two_dataframe():
    dp = _generate_words_sample()
    dp_left = dp[['left']]
    dp_right = dp[['right']]
    return dp_left, dp_right

def _generate_query():
    query = pd.Series({'left': 'hello world'})
    scoredict = {'fuzzy': ['left']}
    return query, scoredict

def test_zero():
    import findspark
    findspark.init()
    import pyspark
    from pyspark.sql import SQLContext
    try:
        sc = pyspark.SparkContext(appName="Fuzzy3")
        sqlContext = SQLContext(sc)
    except:
        sqlContext = SQLContext(sc)
    return sc, sqlContext

sc, sqlContext = test_zero()
dp = _generate_words_sample()
df_origin = sqlContext.createDataFrame(dp)

sc.stop()
del sqlContext
