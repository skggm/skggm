import os
import numpy as np
import pytest
from scipy.io import loadmat

from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_allclose
from sklearn import datasets

from inverse_covariance import (
    QuicGraphLasso,
    quic,
    QuicGraphLassoCV,
    QuicGraphLassoEBIC,
)


class TestQuicGraphLasso(object):
    @pytest.mark.parametrize("params_in, expected", [
        ({}, [3.2437533337151625, 3.4490549523890648, 9.7303201146340168, 3.673994442010553e-11]),
        ({
            'lam': 1.0,
            'max_iter': 100,
        }, [3.1622776601683795, 3.1622776601683795, 10.0, 0.0]),
        ({
            'lam': 0.5,
            'mode': 'trace',
        }, [3.2437533337151625, 3.4490549523890652, 32.290292419357321, 0.21836515326396364]),
        ({
            'lam': 0.5,
            'mode': 'path',
            'path': np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5]),
        }, [8.3256240637201717, 9.7862122341861983, 22.752074276274861, 1.6530965731149066e-08]),
        ({
            'lam': 1.0,
            'max_iter': 100,
            'init_method': 'cov',
        }, [0.0071706976421055616, 1394.564448134179, 50.890448754467911, 7.1054273576010019e-15]),
    ])
    def test_integration_quic_graph_lasso(self, params_in, expected):
        '''
        Just tests inputs/outputs (not validity of result).
        '''
        X = datasets.load_diabetes().data
        ic = QuicGraphLasso(**params_in)
        ic.fit(X)
        
        result_vec = [
            np.linalg.norm(ic.covariance_),
            np.linalg.norm(ic.precision_),
            np.linalg.norm(ic.opt_),
            np.linalg.norm(ic.duality_gap_),
        ]
        print result_vec
        assert_allclose(expected, result_vec)


    @pytest.mark.parametrize("params_in, expected", [
        ({'n_refinements': 1}, [4.69, 79.24, 2.67, 8.23e-05, 0.001]),
        ({'lam': 0.1 * np.ones((10,10)),
          'n_refinements': 1}, [4.69, 82.23, 2.64, 0.00069]),
    ])
    def test_integration_quic_graph_lasso_cv(self, params_in, expected):
        '''
        Just tests inputs/outputs (not validity of result).
        '''
        X = datasets.load_diabetes().data
        ic = QuicGraphLassoCV(**params_in)
        ic.fit(X)
        
        result_vec = [
            np.linalg.norm(ic.covariance_),
            np.linalg.norm(ic.precision_),
            np.linalg.norm(ic.opt_),
            np.linalg.norm(ic.duality_gap_),
        ]
        if isinstance(ic.lam_, float):
            result_vec.append(ic.lam_)
        elif isinstance(ic.lam_, np.ndarray):
            assert ic.lam_.shape == params_in['lam'].shape

        print result_vec
        assert_allclose(expected, result_vec, rtol=1e-1)

        assert len(ic.grid_scores) == len(ic.cv_lams_)


    @pytest.mark.parametrize("params_in, expected", [
        ({}, [3.1622776601683795, 3.1622776601683795, 0.91116275611548958]),
        ({'lam': 0.1 * np.ones((10, 10))}, [4.4495761722050329, 5.5097138516796109]),
    ])
    def test_integration_quic_graph_lasso_ebic(self, params_in, expected):
        '''
        Just tests inputs/outputs (not validity of result).
        '''
        X = datasets.load_diabetes().data
        ic = QuicGraphLassoEBIC(**params_in)
        ic.fit(X)
        
        result_vec = [
            np.linalg.norm(ic.covariance_),
            np.linalg.norm(ic.precision_),
        ]
        if isinstance(ic.lam_, float):
            result_vec.append(ic.lam_)
        elif isinstance(ic.lam_, np.ndarray):
            assert ic.lam_.shape == params_in['lam'].shape

        print result_vec
        assert_allclose(expected, result_vec, rtol=1e-1)

    def test_invalid_method(self):
        '''
        Test behavior of invalid inputs.
        '''
        X = datasets.load_diabetes().data
        ic = QuicGraphLasso(method='unknownmethod')
        assert_raises(NotImplementedError, ic.fit, X)

