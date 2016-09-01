import numpy as np
import pytest

from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_allclose
from sklearn import datasets

from inverse_covariance import (
    QuicGraphLasso,
    QuicGraphLassoCV,
    QuicGraphLassoEBIC,
    quic,
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
        lam = 0.5
        if 'lam' in params_in:
            lam = params_in['lam']
            del params_in['lam']

        S = np.corrcoef(X, rowvar=False)
        if 'init_method' in params_in:
            if params_in['init_method'] == 'cov':
                S = np.cov(X, rowvar=False)

            del params_in['init_method']        
        
        precision_, covariance_, opt_, cpu_time_, iters_, duality_gap_ =\
                quic(S, lam, **params_in)    
                    
        result_vec = [
            np.linalg.norm(covariance_),
            np.linalg.norm(precision_),
            np.linalg.norm(opt_),
            np.linalg.norm(duality_gap_),
        ]
        print result_vec
        assert_allclose(expected, result_vec)

    @pytest.mark.parametrize("params_in, expected", [
        ({'n_refinements': 1}, [4.6528, 32.335, 3.822, 1.5581289048993696e-06, 0.01]),
        ({'lam': 0.5 * np.ones((10,10)) - 0.5 * np.diag(np.ones((10,))),
          'n_refinements': 1}, [4.6765, 49.24459, 3.26151, 6.769744583801085e-07]),
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
        ({'lam': 0.5 * np.ones((10, 10))}, [4.797, 2.1849]),
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

