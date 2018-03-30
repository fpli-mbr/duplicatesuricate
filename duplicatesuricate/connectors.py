import pandas as pd
from array import _Array, _Col
class _Connector:
    def __init__(self, source=None, **kwargs):
        '''

        Args:
            source:
            **kwargs:
        '''
        self.source = source
        self.attributes, self.relevance = self._config_init(**kwargs)
        assert isinstance(self.attributes, set)
        assert isinstance(self.attributes, set)
        self.output = self.attributes.union(self.relevance)
        assert isinstance(self.output, set)
        pass

    def search(self, query):
        '''

        Args:
            query:

        Returns:
            _Array
        '''
        results = _Array(pd.DataFrame(columns=self.output))

        assert set(results.columns) == self.output
        return results

    def _config_init(self, **kwargs):
        attributes = set(['info', 'info2'])
        relevance = set(['relevance'])
        return attributes, relevance

    def _config_search(self, **kwargs):
        pass