import os
import numpy as np
import pytest

from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_allclose
from sklearn import datasets

from inverse_covariance import (
    QuicGraphLassoEBIC,
    AdaptiveGraphLasso,
    QuicGraphLassoCV,
)

class TestAdaptiveGraphLasso(object):
    @pytest.mark.parametrize("params_in", [
        ({
            'estimator': QuicGraphLassoCV(
                        cv=2, 
                        n_refinements=6,
                        init_method='cov',
                        score_metric='log_likelihood',
                    ),
            'method': 'binary',
        }),
        ({
            'estimator': QuicGraphLassoCV(
                        cv=2, 
                        n_refinements=6,
                        init_method='cov',
                        score_metric='log_likelihood',
                    ),
            'method': 'inverse',
        }),
        ({
            'estimator': QuicGraphLassoCV(
                        cv=2, 
                        n_refinements=6,
                        init_method='cov',
                        score_metric='log_likelihood',
                    ),
            'method': 'inverse_squared',
        }),
        ({
            'estimator': QuicGraphLassoEBIC(),
            'method': 'binary',
        }),
        ({
            'estimator': QuicGraphLassoEBIC(),
            'method': 'inverse',
        }),
        ({
            'estimator': QuicGraphLassoEBIC(),
            'method': 'inverse_squared',
        }),
    ])
    def test_integration_adaptive_graph_lasso(self, params_in):
        '''
        Just tests inputs/outputs (not validity of result).
        '''
        X = datasets.load_diabetes().data
        n_examples, n_features = X.shape
        
        model = AdaptiveGraphLasso(**params_in)
        model.fit(X)
        assert model.estimator_ is not None
        assert model.lam_ is not None

        assert np.sum(model.lam_[np.diag_indices(n_features)]) == 0

        if params_in['method'] == 'binary':
            uvals = set(model.lam_.flat)
            assert len(uvals) == 2
            assert 0 in uvals
            assert 1 in uvals
        elif params_in['method'] == 'inverse' or\
                params_in['method'] == 'inverse_squared':
            uvals = set(model.lam_.flat[model.lam_.flat != 0])
            assert len(uvals) > 0


