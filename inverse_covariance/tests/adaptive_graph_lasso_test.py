import numpy as np
import pytest

from inverse_covariance import (
    QuicGraphLassoEBIC,
    AdaptiveGraphLasso,
    QuicGraphLassoCV,
)
from inverse_covariance.profiling import (
    ClusterGraph,
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
        n_features = 20
        n_samples = 25
        cov, prec, adj = ClusterGraph(
            n_blocks=1,
            chain_blocks=False,
            seed=1,
        ).create(n_features, 0.8)
        prng = np.random.RandomState(2)
        X = prng.multivariate_normal(np.zeros(n_features), cov, size=n_samples)
        
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
