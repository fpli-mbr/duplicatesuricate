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
    def select(self, cols):
        '''

        Args:
            cols (list):

        Returns:
            _array
        '''
        if self.struct == 'pandas':
            return DepArray(self.df.loc[:, cols])
        else:
            return DepArray(self.df.select(cols))

    def count(self):
        if self.struct == 'pandas':
            return self.df.shape[0]
        else:
            return self.df.count()
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

class DepCol:
    def __init__(self,y):
        if type(y) == pd.Series:
            self.struct = 'pandas'
            assert isinstance(y, pd.Series)
            self.y = y
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