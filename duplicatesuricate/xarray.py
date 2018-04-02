import pandas as pd

class _Array:
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
            return _Array(self.df.loc[:, cols])
        else:
            return _Array(self.df.select(cols))

    def count(self):
        if self.struct == 'pandas':
            return self.df.shape[0]
        else:
            return self.df.count()
    def union(self, X):
        if self.struct == 'pandas':
            return pd.concat([self.df, X], axis=1)
        else:
            return self.df.union(X)

class _Col:
    def __init__(self,y):
        if type(y) == pd.Series:
            self.struct = 'pandas'
            assert isinstance(y, pd.Series)
            self.y = y
        else:
            self.struct = None

        pass
    def sort(self, ascending = True):
        if self.struct == 'pandas':
            return self.y.sort_values(ascending)
        else:
            #TODO: complete with pyspark logic
            return self.y.sort()