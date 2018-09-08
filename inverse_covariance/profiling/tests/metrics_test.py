import numpy as np
import pytest

from inverse_covariance.profiling import metrics


class TestMetrics(object):
    @pytest.mark.parametrize(
        "m, m_hat, expected",
        [
            (
                np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]),
                np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]),
                (6, 6, 6),
            ),
            (
                np.array([[2, 1, 0], [1, 2, 3], [0, 5, 6]]),
                np.array([[1, 1, 0], [1, 2, 0], [0, 0, 3]]),
                (4, 2, 2),
            ),
            (
                np.array([[0, 1, 0], [1, 0, 3], [0, 5, 0]]),
                np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),
                (4, 2, 2),
            ),
        ],
    )
    def test__nonzero_intersection(self, m, m_hat, expected):
        result = metrics._nonzero_intersection(m, m_hat)
        print(result)
        assert result == expected

    @pytest.mark.parametrize(
        "m, m_hat, expected",
        [
            (
                np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]),
                np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]),
                0,
            ),
            (
                np.array([[2, 1, 0], [1, 2, 3], [0, 5, 6]]),
                np.array([[1, 1, 0], [1, 2, 0], [0, 0, 3]]),
                0,
            ),
            (
                np.array([[0, 1, 0], [1, 0, 3], [0, 5, 0]]),
                np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]]),
                1,
            ),
        ],
    )
    def test_support_false_positive_count(self, m, m_hat, expected):
        result = metrics.support_false_positive_count(m, m_hat)
        print(result)
        assert result == expected

    @pytest.mark.parametrize(
        "m, m_hat, expected",
        [
            (
                np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]),
                np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]),
                0,
            ),
            (
                np.array([[2, 1, 0], [1, 2, 3], [0, 5, 6]]),
                np.array([[1, 1, 0], [1, 2, 0], [0, 0, 3]]),
                1,
            ),
            (
                np.array([[0, 1, 0], [1, 0, 3], [0, 5, 0]]),
                np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]),
                0,
            ),
        ],
    )
    def test_support_false_negative_count(self, m, m_hat, expected):
        result = metrics.support_false_negative_count(m, m_hat)
        print(result)
        assert result == expected

    @pytest.mark.parametrize(
        "m, m_hat, expected",
        [
            (
                np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]),
                np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]),
                0,
            ),
            (
                np.array([[2, 1, 0], [1, 2, 3], [0, 5, 6]]),
                np.array([[1, 1, 0], [1, 2, 0], [0, 0, 3]]),
                1,
            ),
            (
                np.array([[0, 1, 0], [1, 0, 3], [0, 5, 0]]),
                np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]]),
                2,
            ),
        ],
    )
    def test_support_difference_count(self, m, m_hat, expected):
        result = metrics.support_difference_count(m, m_hat)
        print(result)
        assert result == expected

    @pytest.mark.parametrize(
        "m, m_hat, expected",
        [
            (
                np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]),
                np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]),
                1,
            ),
            (
                np.array([[2, 1, 0], [1, 2, 3], [0, 5, 6]]),
                np.array([[1, 1, 0], [1, 2, 0], [0, 0, 3]]),
                0,
            ),
            (
                np.array([[0, 1, 0], [1, 0, 3], [0, 5, 0]]),
                np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]]),
                0,
            ),
        ],
    )
    def test_has_exact_support(self, m, m_hat, expected):
        result = metrics.has_exact_support(m, m_hat)
        print(result)
        assert result == expected

    @pytest.mark.parametrize(
        "m, m_hat, expected",
        [
            (
                np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]),
                np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]),
                1,
            ),
            (
                np.array([[2, 1, 0], [1, 2, 3], [0, 5, 6]]),
                np.array([[1, 1, 0], [1, 2, 0], [0, 0, 3]]),
                1,
            ),
            (
                np.array([[0, 1, 0], [1, 0, 3], [0, 5, 0]]),
                np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]]),
                0,
            ),
        ],
    )
    def test_has_approx_support(self, m, m_hat, expected):
        result = metrics.has_approx_support(m, m_hat, 0.5)
        print(m, m_hat, result)
        assert result == expected

    @pytest.mark.parametrize(
        "m, m_hat, expected",
        [
            (
                np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]),
                np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]),
                0,
            ),
            (
                np.array([[2, 1, 0], [1, 2, 3], [0, 5, 6]]),
                np.array([[1, 1, 0], [1, 2, 0], [0, 0, 3]]),
                3.0,
            ),
            (
                np.array([[0, 1, 0], [1, 0, 3], [0, 5, 0]]),
                np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]]),
                3.16227766017,
            ),
        ],
    )
    def test_error_fro(self, m, m_hat, expected):
        result = metrics.error_fro(m, m_hat)
        print(m, m_hat, result)
        np.testing.assert_array_almost_equal(result, expected)
