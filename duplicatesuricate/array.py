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

class _Col:
    def __init__(self,y):
        if type(X) == pd.Series:
            self.struct = 'pandas'
        else:
            self.struct = None
        self.y = y
        pass