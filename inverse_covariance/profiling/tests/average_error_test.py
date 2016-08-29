import numpy as np
import pytest

from sklearn import datasets
from sklearn.covariance import GraphLassoCV

from inverse_covariance.profiling import AverageError
from inverse_covariance import (
    QuicGraphLassoCV,
    QuicGraphLassoEBIC,
)


class TestAverageError(object):
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
        ae = AverageError(**params_in)
        ae.fit(X)

        assert np.sum(ae.error_fro_.flat) > 0
        assert ae.error_fro_.shape == (5, ae.n_grid_points)
        assert np.sum(ae.error_supp_.flat) > 0
        assert ae.error_supp_.shape == (5, ae.n_grid_points)
        assert np.sum(ae.error_fp_.flat) > 0
        assert ae.error_fp_.shape == (5, ae.n_grid_points)
        assert np.sum(ae.error_fn_.flat) > 0
        assert ae.error_fn_.shape == (5, ae.n_grid_points)
        assert len(ae.ks_) == 5
        assert len(ae.grid_) == ae.n_grid_points
