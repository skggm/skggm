import os
import numpy as np
import pytest
from scipy.io import loadmat

from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_allclose

from sklearn import datasets

from .. import InverseCovariance, quic


class TestInverseCovariance(object):
    @pytest.mark.parametrize("params_in, expected", [
        ({}, [65.22614889456429, 18.704554180634613, 534.23790896400169, 5.2531441951941815e-07]),
        ({
            'lam': 1.0,
            'max_iter': 100,
        }, [42.047592083257278, 10.511898020814318, 748.37105380749654, 0.0]),
        ({
            'lam': 0.5,
            'mode': 'trace',
        }, [65.22614889456429, 18.704554180634613, 1890.485640010804, 1414213562372882.5]),
        ({
            'lam': 0.5,
            'mode': 'path',
            'path': np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5]),
        }, [234.42034955185895, 66.11208447360967, 1020.7602391074518, 2.830908902050909e-06]),
        ({
            'lam': 1.0,
            'max_iter': 100,
            'initialize_method': 'cov',
        }, [0.014341395401919506, 697.28221834407839, 43.958976948867786, 8.8817841970012523e-16]),

    ])
    def test_fit(self, params_in, expected):
        '''
        Just tests inputs/outputs (not validity of result).
        '''
        X = datasets.load_diabetes().data
        ic = InverseCovariance(**params_in)
        ic.fit(X)
        
        result_vec = [
            np.linalg.norm(ic.covariance_),
            np.linalg.norm(ic.precision_),
            np.linalg.norm(ic.opt_),
            np.linalg.norm(ic.duality_gap_),
        ]
        print result_vec
        assert_allclose(expected, result_vec)


    def test_invalid_method(self):
        '''
        Test behavior of invalid inputs.
        '''
        X = datasets.load_diabetes().data
        ic = InverseCovariance(method='unknownmethod')
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

