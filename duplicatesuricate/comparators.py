import pandas as pd
from xarray import _Array, _Col
import functions

# noinspection PySetFunctionToLiteral,PySetFunctionToLiteral
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
        """

        Args:
            query (_Col):
            targets (_Array):

        Returns:
            _Array
        """
        results = _Array(self._compare(query=query, targets=targets))
        assert set(results.columns) == self.scored
        return results

    def _compare(self, query, targets):
        df = pd.DataFrame(columns=self.scored)
        return df


class PandasComparator(_Comparator):
    def _config_init(self, scoredict, **kwargs):
        self.scoredict = functions.ScoreDict(scoredict)
        compared = scoredict.compared()
        scores = scoredict.scores()
        return compared, scores
    def _compare(self, query, targets):
        table = functions.build_similarity_table(query=query,targets=targets,scoredict=self.scoredict)
        return table