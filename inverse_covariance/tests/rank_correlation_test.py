import numpy as np

from sklearn.utils.testing import assert_array_almost_equal

from inverse_covariance.rank_correlation import (
    _compute_ranks,
    spearman_correlation,
    kendalltau_correlation,
)

Y = np.ones(shape=[10, 2])
X = .001 * np.random.randn(20, 1)
X = np.append(X, X, axis=1)


def test_compute_ranks():
    Y1 = 5.5 * Y / Y.shape[0]
    Y2 = _compute_ranks(Y)
    assert_array_almost_equal(Y1, Y2)


def test_spearman_correlation():
    A = np.ones(shape=[2, 2])
    A2 = spearman_correlation(X)
    assert_array_almost_equal(A, A2, decimal=3)


def test_kendalltau_correlation():
    A = np.ones(shape=[2, 2])

    A2 = kendalltau_correlation(X, weighted=False)
    assert_array_almost_equal(A, A2, decimal=3)

    A3 = kendalltau_correlation(X, weighted=True)
    assert_array_almost_equal(A, A3, decimal=3)
