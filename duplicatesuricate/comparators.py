import pandas as pd
import xarray
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
            query (xarray.DepCol):
            targets (xarray.DepArray):

        Returns:
            _Array
        """
        results = xarray.DepArray(self._compare(query=query, targets=targets))
        assert set(results.columns) == self.scored
        return results

    def _compare(self, query, targets):
        df = pd.DataFrame(columns=self.scored)
        return df


class PandasComparator(_Comparator):
    def _config_init(self, scoredict, **kwargs):
        """

        Args:
            scoredict (dict):
            **kwargs:

        Returns:

        """
        self.scoredict = functions.ScoreDict(scoredict)
        compared = self.scoredict.compared()
        scores = self.scoredict.scores()
        return compared, scores

    def _compare(self, query, targets):
        """

        Args:
            query (xarray.DepCol:
            targets (xarray.DepArray):

        Returns:
            pd.DataFrame
        """
        df = targets.toPandas()
        q = query.toPandas()
        table = functions.build_similarity_table(query=q,targets=df,scoredict=self.scoredict)
        return table