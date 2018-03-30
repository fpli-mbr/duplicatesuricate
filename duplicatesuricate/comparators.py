import pandas as pd
from array import _Array, _Col

class _Comparator:
    def __init__(self, **kwargs):
        self.compared, self.scored = self._config_init(**kwargs)
        assert isinstance(self.compared, set)
        assert isinstance(self.scored, set)
        pass

    def _config_init(self, **kwargs):
        compared = set(['info'])
        scored = set(['info_score'])
        return compared, scored

    def compare(self, query, targets):
        '''

        Args:
            query (_Col):
            targets (_Array):

        Returns:
            _Array
        '''
        results = pd.DataFrame(columns=self.scored)

        assert set(results.columns) == self.scored
        return results