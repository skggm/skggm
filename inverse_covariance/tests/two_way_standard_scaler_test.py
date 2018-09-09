import pytest
import numpy as np
from scipy import sparse
from sklearn.utils.testing import assert_raises, assert_allclose
from sklearn.exceptions import NotFittedError

from inverse_covariance.two_way_standard_scaler import (
    two_way_standardize,
    TwoWayStandardScaler,
)


def custom_init(n_rows, n_cols, with_mean=False):
    prng = np.random.RandomState(1)
    X = prng.normal(0, np.ones(shape=(n_rows, n_cols)))
    if with_mean:
        mu = np.ones(shape=(n_rows, 1)) * prng.randint(1, 5, size=(1, n_cols))
    else:
        mu = np.zeros(shape=(n_rows, n_cols))

    var_rows = prng.lognormal(2, 1, size=(n_rows, 1))
    sqcov_rows = np.diag(np.sqrt(var_rows))
    var_cols = prng.lognormal(2, 1, size=(1, n_cols))
    sqcov_cols = np.diag(np.sqrt(var_cols))

    return mu + sqcov_rows * X * sqcov_cols


def test_fit_exception_on_sparse_input():
    """
    Test behavior of invalid sparse data matrix inputs.
    """
    X = np.zeros(shape=(10, 10))
    sparse_Xs = [sparse.csc_matrix(X), sparse.csr_matrix(X)]
    for sparse_X in sparse_Xs:
        scaler = TwoWayStandardScaler()
        assert_raises(TypeError, scaler.fit, sparse_X)


def test_transform_exception_not_fitted():
    """
    Test behavior of invalid sparse data matrix inputs.
    """
    scaler = TwoWayStandardScaler()
    assert_raises(NotFittedError, scaler.transform, np.zeros(shape=(10, 10)))


def test_transform_exception_on_sparse_input():
    """
    Test behavior of invalid sparse data matrix inputs.
    """
    X = np.zeros(shape=(10, 10))
    sparse_Xs = [sparse.csc_matrix(X), sparse.csr_matrix(X)]
    for sparse_X in sparse_Xs:
        scaler = TwoWayStandardScaler()
        scaler.fit(X)
        assert_raises(TypeError, scaler.transform, sparse_X)


@pytest.mark.parametrize(
    "data, with_std, expected",
    [
        (
            [[1, 0], [1, 0], [2, 1], [2, 1]],
            True,  # with_std=True
            [
                [0.5, 0.5, 1.5, 1.5],  # row_mean_
                [0.25, 0.25, 0.25, 0.25],  # row_var_
                [1.5, 0.5],  # col_mean_
                [0.25, 0.25],  # col_var_
                [0.5, 0.5, 0.5, 0.5],  # row_scale_
                [0.5, 0.5],  # col_scale_
                [4],  # [n_rows_seen_]
                [2],  # [n_cols_seen_]
            ],
        ),
        (
            [[1, 0], [1, 0], [2, 1], [2, 1]],
            False,  # with_std=False
            [
                [0.5, 0.5, 1.5, 1.5],  # row_mean_
                None,  # row_var_
                [1.5, 0.5],  # col_mean_
                None,  # col_var_
                None,  # row_scale_
                None,  # col_scale_
                [4],  # [n_rows_seen_]
                [2],  # [n_cols_seen_]
            ],
        ),
    ],
)
def test_fit(data, with_std, expected):
    scaler = TwoWayStandardScaler(with_std=with_std)
    scaler.fit(data)
    result = [
        scaler.row_mean_,
        scaler.row_var_,
        scaler.col_mean_,
        scaler.col_var_,
        scaler.row_scale_,
        scaler.col_scale_,
        [scaler.n_rows_seen_],
        [scaler.n_cols_seen_],
    ]
    assert_allclose(
        [i for e in expected if e is not None for i in e],
        [j for r in result if r is not None for j in r],
    )


@pytest.mark.parametrize(
    "data, with_std, expected",
    [
        (
            [
                [[1, 0], [1, 0], [2, 1], [2, 1]],
                [[1, 0], [1, 0], [2, 1], [2, 1]],
                [[1, 0], [1, 0], [2, 1], [2, 1]],
            ],  # multiple data examples for "online" estimation
            True,  # with_std=True
            [
                [0.5, 0.5, 1.5, 1.5],  # row_mean_
                [0.25, 0.25, 0.25, 0.25],  # row_var_
                [1.5, 0.5],  # col_mean_
                [0.25, 0.25],  # col_var_
                [0.5, 0.5, 0.5, 0.5],  # row_scale_
                [0.5, 0.5],  # col_scale_
                [12],  # [n_rows_seen_]
                [6],  # [n_cols_seen_]
            ],
        )
    ],
)
def test_partial_fit(data, with_std, expected):
    scaler = TwoWayStandardScaler(with_std=with_std)
    for d in data:
        scaler.partial_fit(d)

    result = [
        scaler.row_mean_,
        scaler.row_var_,
        scaler.col_mean_,
        scaler.col_var_,
        scaler.row_scale_,
        scaler.col_scale_,
        [scaler.n_rows_seen_],
        [scaler.n_cols_seen_],
    ]
    print(result)
    assert_allclose(
        [i for e in expected if e is not None for i in e],
        [j for r in result if r is not None for j in r],
    )


@pytest.mark.parametrize(
    "n_rows, n_cols, with_mean, with_std, expected",
    [
        (
            6,  # n_rows
            2,  # n_cols
            False,  # with_mean
            True,  # with_std
            [
                [1.24852525, -0.47021609],
                [-1.66629192, -3.38503326],
                [0.46966753, -1.24907381],
                [1.1966711, -0.52207024],
                [0.96470187, -0.75403946],
                [0.71346052, -1.00528082],
            ],
        ),
        (
            6,  # n_rows
            2,  # n_cols
            True,  # with_mean
            False,  # with_std
            [
                [1.85393809, -1.85393809],
                [-19.71891691, 19.71891691],
                [13.72700391, -13.72700391],
                [5.29676963, -5.29676963],
                [-19.41773454, 19.41773454],
                [18.25893982, -18.25893982],
            ],
        ),
        (
            6,  # n_rows
            2,  # n_cols
            True,  # with_mean
            True,  # with_std
            [
                [-1.41421356, 1.41421356],
                [-1.41421356, 1.41421356],
                [0.70710678, -0.70710678],
                [0.70710678, -0.70710678],
                [0.70710678, -0.70710678],
                [0.70710678, -0.70710678],
            ],
        ),
    ],
)
def test_two_way_standardize(n_rows, n_cols, with_mean, with_std, expected):
    X = custom_init(n_rows, n_cols, with_mean=with_mean)
    result = two_way_standardize(X, with_mean=with_mean, with_std=with_std)
    assert_allclose(result, expected)
