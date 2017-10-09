import numpy as np
from scipy import sparse
import pytest

from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_allclose
from sklearn import datasets

from clean import (
   twoway_standardize
)

def custom_init(n_rows, n_cols, with_mean=False):
    prng = np.random.RandomState(1)
    X = prng.normal(0, np.ones(shape=(n_rows,n_cols)))
    if with_mean:
        mu = np.ones(shape=(n_rows, 1)) * \
                prng.randint(1, 5, size=(1, n_cols))
    else:
        mu = np.zeros(shape=(n_rows,n_cols))
    var_rows = prng.lognormal(2, 1, size=(n_rows, 1))
    sqcov_rows = np.diag(np.sqrt(var_rows))
    var_cols = prng.lognormal(2, 1, size=(1, n_cols))
    sqcov_cols = np.diag(np.sqrt(var_cols))
    return mu + sqcov_rows * X * sqcov_cols

def test_invalid_argument():
    '''
    Test behavior of invalid sparse inputs.
    '''
    X = np.zeros(shape=(10,10))
    X_csc = sparse.csc_matrix(X)
    assert_raises(TypeError, twoway_standardize(X_csc))
    
    X_csr = sparse.csr_matrix(X)
    assert_raises(TypeError, twoway_standardize(X_csr))
