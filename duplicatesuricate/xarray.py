import pandas as pd

class DepArray:
    def __init__(self, X):
        if type(X) == pd.DataFrame:
            self.struct = 'pandas'
            self.columns = X.columns
        else:
            self.struct = None
            self.columns = X.schema.names
        self.df = X
    def toPandas(self):
        """

        Returns:
            pd.DataFrame
        """
        if self.struct == 'pandas':
            return self.df
        else:
            return self.df.toPandas()
    def toSpark(self,sqlContext):
        """

        Args:
            sqlContext:

        Returns:

        """
        #TODO: Docstring
        if self.struct == 'pandas':
            return sqlContext.createDataFrame(self.df)
        else:
            return self.df
    def fillna(self, na_value):
        self.df = self.df.fillna(na_value)
        return self
    def select(self, cols):
        '''

        Args:
            cols (list):

        Returns:
            _array
        '''
        if type(cols) == set:
            cols = list(cols)
        if self.struct == 'pandas':
            return DepArray(self.df.loc[:, cols])
        else:
            return DepArray(self.df.select(cols))

    def count(self):
        if self.struct == 'pandas':
            return self.df.shape[0]
        else:
            return self.df.count()
    def show(self):
        if self.struct == 'pandas':
            return self.df.head()
        else:
            return self.df.show()
    def union(self, X):
        """

        Args:
            X (DepArray):

        Returns:

        """
        if self.struct == 'pandas':
            if X.struct == 'pandas':
                newdf = pd.concat([self.df, X.df], axis=1)
            else:
                newdf = pd.concat([self.df, X.df.toPandas()], axis=1)
        else:
            newdf = self.df.union(X)
        return DepArray(newdf)
    def withColumn(self, colname, func, on_cols):
        if self.struct == 'pandas':
            x = self.df.copy()
            x[colname] = x.apply(lambda r: func(on_cols), axis = 1)
        else:
            x = self.df.withColumn(colname, func(on_cols))
        return DepArray(x)

class DepCol:
    def __init__(self,y):
        if y is None:
            y = pd.Series()
        if type(y) == pd.Series:
            self.struct = 'pandas'
            assert isinstance(y, pd.Series)
            self.y = y
        elif type(y) == DepCol:
            self.struct = y.struct
            self.y = y.y
        else:
            self.struct = None

        pass
    def sort(self, ascending = False):
        if self.struct == 'pandas':
            newy = self.y.sort_values(ascending=ascending)
        else:
            #TODO: complete with pyspark logic
            newy = self.y

        return DepCol(newy)
    def toPandas(self):
        """

        Returns:
            pd.Series
        """
        if self.struct == 'pandas':
            return self.y
        else:
            #TODO
            return pd.Series(self.y)
    def toSpark(self):
        #TODO
        return None