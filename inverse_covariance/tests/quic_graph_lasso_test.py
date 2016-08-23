import os
import numpy as np
import pytest
from scipy.io import loadmat

from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_allclose
from sklearn import datasets

from .. import (
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
            'initialize_method': 'cov',
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


    @pytest.mark.parametrize("params_in, expected", [
        ({
            'lam': 0.5,
            'max_iter': 100,
        }, [593.747625799]),
        ({
            'lam': 1.0,
            'mode': 'path',
            'path': np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5]),
            'tol': 1e-16,
            'max_iter': 100,
        }, [692., 691.85406837, 689.44452037, 675.59092901, 643.30280267,
            593.7476258]),
        ({
            'lam': 0.5,
            'mode': 'trace',
            'tol': 1e-16,
            'max_iter': 11,
        }, [656.72143772, 638.21734896, 607.84863314, 595.38094248,
            594.01024824, 593.76739962,  593.74926575, 593.74777092,
            593.747634, 593.74762626, 593.74762583]),
    ])
    def test_ER_692(self, params_in, expected):
        '''
        Requires that inverse_covariance/tests/ER_692.mat exists. 
        It can be found in the MEX package archive from the [QUIC].
        http://www.cs.utexas.edu/~sustik/QUIC/
        
        Reproduces tests from pyquic: https://github.com/osdf/pyquic
        '''
        if not os.path.exists('inverse_covariance/tests/ER_692.mat'):
            print ('''Requires the file tests/ER_692.mat - this can be obtained in the MEX archive at http://www.cs.utexas.edu/~sustik/QUIC/''')
            assert False

        data = loadmat('inverse_covariance/tests/ER_692.mat')['S']
        X = np.zeros(data.shape)
        X[:] = data
        Theta, Sigma, opt, cputime, iters, dGap = quic(X, **params_in)
        print np.array(opt)
        assert_allclose(opt, expected, rtol=1e-2)

