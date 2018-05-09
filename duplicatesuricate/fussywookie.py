from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark import ml as ML

tk_left = 'tokens_left'
tk_right = 'tokens_right'
intersec_col = 'intersection'
diff_lr = 'diff_lr'
diff_rl = 'diff_rl'
sorted_sect = 'sorted_sect'
sorted_lr = 'sorted_lr'
sorted_rl = 'sorted_rl'
concat_sect = 'concat_sect'
concat_lr = 'concat_lr'
concat_rl = 'concat_rl'
combined_lr = 'combined_lr'
combined_rl = 'combined_rl'
ratio1 = 'ratio1'
ratio2 = 'ratio2'
ratio3 = 'ratio3'
tempcols = [tk_left, tk_right, intersec_col, diff_lr, diff_rl, sorted_sect, sorted_lr, sorted_rl, concat_sect,
           concat_lr, concat_rl, combined_lr, combined_rl, ratio1, ratio2, ratio3]


pattern="[-.;,| /]+"
transform_left_token = ML.feature.RegexTokenizer(inputCol=left, outputCol=tk_left, pattern=pattern)
transform_right_token = ML.feature.RegexTokenizer(inputCol=right, outputCol=tk_right, pattern=pattern)


def transform_set(df):
    differencer = F.udf(lambda x,y: list(set(x)-set(y)), T.ArrayType(T.StringType()))
    intersecter = F.udf(lambda x,y: list(set(x).intersection(set(y))), T.ArrayType(T.StringType()))

    df1 = df.withColumn(intersec_col, intersecter(tk_left, tk_right)).\
        withColumn(diff_lr, differencer(tk_left, tk_right)).\
        withColumn(diff_rl, differencer(tk_right, tk_left))
    return df1

transformer_set = ML.Transformer()
transformer_set.transform = transform_set


def transform_sort(df, inputCol, outputCol):
    df1 = df.withColumn(
        outputCol,
        F.sort_array(F.col(inputCol))
    )
    return df1

def transform_concat(df, inputCol, outputCol, pattern=' '):
    if type(inputCol) == list:
        in_col = inputCol
        df1 = df.withColumn(
        outputCol,
        F.trim(
            F.concat_ws(
                ' ',
                *in_col
            )
        )
    )
    else:
        in_col = inputCol
        df1 = df.withColumn(
        outputCol,
        F.trim(
            F.concat_ws(
                ' ',
                in_col
            )
        )
    )
    return df1
def sort_concat(df):
    df0 = transform_sort(df, inputCol=intersec_col, outputCol=sorted_sect)
    df1 = transform_sort(df0, inputCol=diff_lr, outputCol=sorted_lr)
    df2 = transform_sort(df1, inputCol=diff_rl, outputCol=sorted_rl)
    df3 = transform_concat(df2, inputCol=sorted_sect, outputCol=concat_sect)
    df4 = transform_concat(df3, inputCol=sorted_lr, outputCol=concat_lr)
    df5 = transform_concat(df4, inputCol=sorted_rl, outputCol=concat_rl)
    df6 = transform_concat(df5, inputCol=[concat_sect, concat_lr], outputCol=combined_lr)
    df7 = transform_concat(df6, inputCol=[concat_sect, concat_rl], outputCol=combined_rl)
    return df7
transformer_concat = ML.Transformer()
transformer_concat.transform = sort_concat

def fuzzy_score_spark(df, left, right, outputCol):
    df = df.withColumn(outputCol, 1- F.levenshtein(left, right) / F.greatest(F.length(left), F.length(right)))
    return df

LevenshteinComparator = ML.Transformer()
LevenshteinComparator.transform = fuzzy_score_spark

def transform_ratio(df):
    df = fuzzy_score_spark(df, left=combined_rl, right = combined_lr, outputCol = ratio1)
    df = fuzzy_score_spark(df, left=concat_sect, right = combined_lr, outputCol = ratio2)
    df = fuzzy_score_spark(df, left=concat_sect, right = combined_rl, outputCol = ratio3)
    df = df.withColumn('token_ratio', F.greatest('ratio1', 'ratio2', 'ratio3'))
    return df

transformer_score = ML.Transformer()
transformer_score.transform = transform_ratio

def transform_dropcols(df):
    for c in tempcols:
        df = df.drop(c)
    return df
transformer_drop = ML.Transformer()
transformer_drop.transform = transform_dropcols

pipeline = ML.Pipeline(stages=[
        transform_left_token,
        transform_right_token,
        transformer_set,
        transformer_concat,
        transformer_score,
        transformer_drop
    ])

def token_score_spark(df, left, right, outputCol='token_score'):
    df_end = pipeline.fit(df).transform(df)
    return df_end

def exact_score_spark(df, left, right, outputCol='exact_score'):
    df_end = df.withColumn(outputCol, (F.col(left) == F.col(right)).cast(T.IntegerType))
    return df_end