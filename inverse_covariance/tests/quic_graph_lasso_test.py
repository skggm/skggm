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
        ({}, [4.7973002082865275, 2.1849691442858554, 13.938876329200646, 4.7809889380800996e-10]),
        ({
            'lam': 1.0,
            'max_iter': 100,
        }, [6.3245553203367599, 1.5811388300841893, 16.931471805599454, 1.7763568394002505e-15]),
        ({
            'lam': 0.5,
            'mode': 'trace',
        }, [4.7973002082865275, 2.1849691442858554, 47.669199462053712, 12.382778386518567]),
        ({
            'lam': 0.5,
            'mode': 'path',
            'path': np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5]),
        }, [11.085316489394547, 6.4455217296796823, 31.35533727044356, 1.7042279940093894e-08]),
        ({
            'lam': 1.0,
            'max_iter': 100,
            'initialize_method': 'cov',
        }, [0.014341395401919506, 697.28221834407839, 43.958976948867786, 8.8817841970012523e-16]),

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
        ({}, [4.695250607261749, 71.424414001397906, 2.8243718924865178, 0.00011952705621326443, 0.0015848931924611141]),
        ({'lam': np.eye(10)}, [4.7066725437645127, 80.453242746645287, 2.7220079143895006, 7.3715037470778455e-06]),
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
        ({}, [6.0436272886101987, 1.65463545689622, 0.91116275611548958]),
        ({'lam': np.eye(10)}, [4.8511097910208161, 14.753289369252375]),
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
        }, [923.1042]),
        ({
            'lam': 1.0,
            'mode': 'path',
            'path': np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5]),
            'tol': 1e-16,
            'max_iter': 100,
        }, [1171.6578, 1136.1222, 1097.9438, 1053.0555, 995.6587, 923.1042]),
        ({
            'lam': 0.5,
            'mode': 'trace',
            'tol': 1e-16,
            'max_iter': 11,
        }, [993.2862, 965.4918, 927.3593, 923.3665, 923.1369, 923.1083, 923.1045, 923.1042, 923.1042, 923.1042, 923.1042]),
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
        assert_allclose(opt, expected, rtol=1e-2)

