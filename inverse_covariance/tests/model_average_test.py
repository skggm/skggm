import numpy as np
import pytest

from sklearn import datasets
from sklearn.covariance import GraphLassoCV

from inverse_covariance import QuicGraphLassoCV, QuicGraphLasso, ModelAverage


class TestModelAverage(object):
    @pytest.mark.parametrize("params_in", [
        ({
            'estimator': QuicGraphLasso(),
            'n_trials': 10,
            'normalize': True,
            'subsample': 0.3,
            'penalization': 'random',
        }),
        ({
            'estimator': QuicGraphLasso(lam=0.5, mode='trace'),
            'n_trials': 15,
            'normalize': False,
            'subsample': 0.6,
            'penalization': 'fully-random',
        }),
        ({
            'estimator': QuicGraphLassoCV(),
            'n_trials': 10,
            'normalize': True,
            'subsample': 0.3,
            'penalization': 'random',
            'use_cache': True,
        }),
        ({
            'estimator': GraphLassoCV(),
            'n_trials': 10,
            'normalize': True,
            'subsample': 0.3,
            'penalization': 'subsampling',
            'use_cache': True,
            'penalty_name': 'alpha',
        }),
    ])
    def test_integration_quic_graph_lasso_cv(self, params_in):
        '''
        Just tests inputs/outputs (not validity of result).
        '''
        X = datasets.load_diabetes().data
        ma = ModelAverage(**params_in)
        ma.fit(X)

        n_examples, n_features = X.shape

        assert ma.proportion_.shape == (n_features, n_features)
        if ma.use_cache:
            assert len(ma.estimators_) == ma.n_trials
            assert len(ma.subsets_) == ma.n_trials
            if not ma.penalization == 'subsampling':
                assert len(ma.lams_) == ma.n_trials
            else:
                assert len(ma.lams_) == 0
        else:
            assert len(ma.estimators_) == 0
            assert len(ma.lams_) == 0
            assert len(ma.subsets_) == 0

        for eidx, e in enumerate(ma.estimators_):
            assert isinstance(e, params_in['estimator'].__class__)
            
            # sklearn doesnt have this but ours do
            if hasattr(e, 'is_fitted'):
                assert e.is_fitted == True

            # check that all lambdas used where different
            if not ma.penalization == 'subsampling' and eidx > 0:
                if hasattr(e, 'lam'):
                    prev_e = ma.estimators_[eidx - 1]
                    assert np.linalg.norm((prev_e.lam - e.lam).flat) > 0

        if ma.normalize == True:
            assert np.max(ma.proportion_) <= 1.0
        else:        
            assert np.max(ma.proportion_) <= ma.n_trials
                
        assert np.min(ma.proportion_) >= 0.0
        assert np.max(ma.proportion_) > 0.0


