import numpy as np
import pytest

from sklearn import datasets

from .. import QuicGraphLassoCV, ModelAverage

class TestQuicGraphLasso(object):
    @pytest.mark.parametrize("params_in, expected", [
        ({
            'estimator': QuicGraphLassoCV,
            'estimator_args': {},
            'num_trials': 10,
            'normalize': True,
            'subsample': 0.3,
        }, []),
    ])
    def test_integration_quic_graph_lasso_cv(self, params_in, expected):
        '''
        Just tests inputs/outputs (not validity of result).
        '''
        X = datasets.load_diabetes().data
        ma = ModelAverage(**params_in)
        ma.fit(X)

        n_examples, n_features = X.shape

        assert ma.proportion_.shape == (n_features, n_features)
        assert len(ma.estimators_) == ma.num_trials
        assert len(ma.lams_) == ma.num_trials

        for e in ma.estimators_:
            assert isinstance(e, params_in['estimator'])
            assert e.is_fitted == True

        if ma.normalize == True:
            assert np.max(ma.proportion_) <= 1.0
        else:        
            assert np.max(ma.proportion_) <= ma.num_trials
                
        assert np.min(ma.proportion_) >= 0.0


