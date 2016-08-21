import numpy as np
import pytest

from sklearn import datasets
from sklearn.covariance import GraphLassoCV

from inverse_covariance.profiling import StatisticalPower
from inverse_covariance import (
    QuicGraphLassoCV,
    QuicGraphLassoEBIC,
)


class TestStatisticalPower(object):
    @pytest.mark.parametrize("params_in", [
        ({
            'model_selection_estimator': QuicGraphLassoCV(),
            'n_trials': 20,
            'n_features': 25,
        }),
        ({
            'model_selection_estimator': QuicGraphLassoEBIC(),
            'n_trials': 20,
            'n_features': 10,
            'n_jobs': 2,
        }),
        ({
            'model_selection_estimator': GraphLassoCV(),
            'n_trials': 20,
            'n_features': 20,
            'penalty_': 'alpha_',
        }),
    ])
    def test_integration_statistical_power(self, params_in):
        '''
        Just tests inputs/outputs (not validity of result).
        '''
        X = datasets.load_diabetes().data
        sp = StatisticalPower(**params_in)
        sp.fit(X)

        assert np.sum(sp.results_.flat) > 0
        assert sp.results_.shape == (5, sp.n_grid_points)
        assert len(sp.ks_) == 5
        assert len(sp.grid_) == sp.n_grid_points
