import os
import numpy as np
import pytest
from scipy.io import loadmat

from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_allclose

from sklearn import datasets

from ..quic import QUIC


class TestQUIC(object):
    @pytest.mark.parametrize("params_in, expected", [
        ({}, [10.991008328453244, 40.263495336580313, 155.05405432239957, 5.6843418860808015e-14]),
        ({
            'lam': 1.0,
            'max_iter': 100,
        }, [21.501215410022848, 20.563604939391684, 451.8527039199829, 1.1368683772161603e-13]),
        ({
            'lam': 0.5,
            'mode': 'trace',
        }, [10.991008328453244, 40.263495336580313, 385.83756982219262, 13.677672436402311]),
        ({
            'lam': 0.5,
            'mode': 'path',
            'path': np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5]),
        }, [20.954440685849391, 139.75270122775521, 247.20884813107077, 5.2407042729174049e-13]),
    ])
    def test_fit(self, params_in, expected):
        '''
        Just tests inputs/outputs (not validity of result).
        '''
        X = datasets.load_diabetes().data
        X = np.dot(X, X.T)

        quic = QUIC(**params_in)
        quic.fit(X)
        
        result_vec = [
            np.linalg.norm(quic.covariance_),
            np.linalg.norm(quic.precision_),
            np.linalg.norm(quic.opt_),
            np.linalg.norm(quic.duality_gap_),
        ]
        print result_vec
        assert_array_almost_equal(expected, result_vec)


    def test_invalid_method(self):
        '''
        Test behavior of invalid inputs.
        '''
        X = datasets.load_diabetes().data
        X = np.dot(X, X.T)
        quic = QUIC(method='unknownmethod')
        assert_raises(NotImplementedError, quic.fit, X)


    def test_invalid_nonsquare(self):
        data = datasets.load_diabetes().data
        quic = QUIC()
        assert_raises(ValueError, quic.fit, data)


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
        Requires that tests/ER_692.mat exists. 
        It can be found in the MEX package archive from the [QUIC].
        http://www.cs.utexas.edu/~sustik/QUIC/
        
        Reproduces tests from pyquic: https://github.com/osdf/pyquic
        '''
        if not os.path.exists('tests/ER_692.mat'):
            print ('''Requires the file tests/ER_692.mat - this can be obtained in the MEX archive at http://www.cs.utexas.edu/~sustik/QUIC/''')
            assert False

        data = loadmat('tests/ER_692.mat')['S']
        X = np.zeros(data.shape)
        X[:] = data

        quic = QUIC(**params_in)
        quic.fit(X)

        assert_allclose(quic.opt_, expected, rtol=1e-2)

