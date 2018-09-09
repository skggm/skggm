import numpy as np
import pytest

from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_allclose
from sklearn import datasets

from inverse_covariance import (
    QuicGraphicalLasso,
    QuicGraphicalLassoCV,
    QuicGraphicalLassoEBIC,
    quic,
)


def custom_init(X):
    init_cov = np.cov(X, rowvar=False)
    return init_cov, np.max(np.abs(np.triu(init_cov)))


class TestQuicGraphicalLasso(object):
    @pytest.mark.parametrize(
        "params_in, expected",
        [
            (
                {},
                [
                    3.2437533337151625,
                    3.4490549523890648,
                    9.7303201146340168,
                    3.673994442010553e-11,
                ],
            ),  # NOQA
            (
                {"lam": 1.0, "max_iter": 100},
                [3.1622776601683795, 3.1622776601683795, 10.0, 0.0],
            ),
            (
                {"lam": 0.5, "mode": "trace"},
                [
                    3.2437533337151625,
                    3.4490549523890652,
                    32.290292419357321,
                    0.21836515326396364,
                ],
            ),  # NOQA
            (
                {
                    "lam": 0.5,
                    "mode": "path",
                    "path": np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5]),
                },
                [
                    8.3256240637201717,
                    9.7862122341861983,
                    22.752074276274861,
                    1.6530965731149066e-08,
                ],
            ),  # NOQA
            (
                {"lam": 1.0, "max_iter": 100, "init_method": "cov"},
                [
                    0.0071706976421055616,
                    1394.564448134179,
                    50.890448754467911,
                    7.1054273576010019e-15,
                ],
            ),  # NOQA
            (
                {"lam": 1.0, "max_iter": 100, "init_method": custom_init},
                [
                    0.0071706976421055616,
                    1394.564448134179,
                    50.890448754467911,
                    7.1054273576010019e-15,
                ],
            ),  # NOQA
            (
                {"lam": 1.0, "max_iter": 100, "init_method": "spearman"},
                [3.1622776601683795, 3.1622776601683795, 10.0, 1.7763568394002505e-15],
            ),  # NOQA
            (
                {"lam": 1.0, "max_iter": 100, "init_method": "kendalltau"},
                [3.1622776601683795, 3.1622776601683795, 10.0, 0.0],
            ),  # NOQA
        ],
    )
    def test_integration_quic_graphical_lasso(self, params_in, expected):
        """
        Just tests inputs/outputs (not validity of result).
        """
        X = datasets.load_diabetes().data
        ic = QuicGraphicalLasso(**params_in)
        ic.fit(X)

        result_vec = [
            np.linalg.norm(ic.covariance_),
            np.linalg.norm(ic.precision_),
            np.linalg.norm(ic.opt_),
            np.linalg.norm(ic.duality_gap_),
        ]
        print(result_vec)
        assert_allclose(expected, result_vec, atol=1e-1, rtol=1e-1)

    @pytest.mark.parametrize(
        "params_in, expected",
        [
            (
                {},
                [
                    3.2437533337151625,
                    3.4490549523890648,
                    9.7303201146340168,
                    3.673994442010553e-11,
                ],
            ),  # NOQA
            (
                {"lam": 1.0, "max_iter": 100},
                [3.1622776601683795, 3.1622776601683795, 10.0, 0.0],
            ),
            (
                {"lam": 0.5, "mode": "trace"},
                [
                    3.2437533337151625,
                    3.4490549523890652,
                    32.290292419357321,
                    0.21836515326396364,
                ],
            ),  # NOQA
            (
                {
                    "lam": 0.5,
                    "mode": "path",
                    "path": np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5]),
                },
                [
                    8.3256240637201717,
                    9.7862122341861983,
                    22.752074276274861,
                    1.6530965731149066e-08,
                ],
            ),  # NOQA
            (
                {"lam": 1.0, "max_iter": 100, "init_method": "cov"},
                [
                    0.0071706976421055616,
                    1394.564448134179,
                    50.890448754467911,
                    7.1054273576010019e-15,
                ],
            ),  # NOQA
            (
                {"lam": 1.0, "max_iter": 100, "init_method": "spearman"},
                [3.1622776601683795, 3.1622776601683795, 10.0, 0.0],
            ),  # NOQA
            (
                {"lam": 1.0, "max_iter": 100, "init_method": "kendalltau"},
                [3.1622776601683795, 3.1622776601683795, 10.0, 0.0],
            ),  # NOQA
        ],
    )
    def test_integration_quic_graphical_lasso_fun(self, params_in, expected):
        """
        Just tests inputs/outputs (not validity of result).
        """
        X = datasets.load_diabetes().data
        lam = 0.5
        if "lam" in params_in:
            lam = params_in["lam"]
            del params_in["lam"]

        S = np.corrcoef(X, rowvar=False)
        if "init_method" in params_in:
            if params_in["init_method"] == "cov":
                S = np.cov(X, rowvar=False)

            del params_in["init_method"]

        precision_, covariance_, opt_, cpu_time_, iters_, duality_gap_ = quic(
            S, lam, **params_in
        )

        result_vec = [
            np.linalg.norm(covariance_),
            np.linalg.norm(precision_),
            np.linalg.norm(opt_),
            np.linalg.norm(duality_gap_),
        ]
        print(result_vec)
        assert_allclose(expected, result_vec, atol=1e-1, rtol=1e-1)

    @pytest.mark.parametrize(
        "params_in, expected",
        [
            (
                {"n_refinements": 1},
                [4.6528, 32.335, 3.822, 1.5581289048993696e-06, 0.01],
            ),  # NOQA
            (
                {
                    "lam": 0.5 * np.ones((10, 10)) - 0.5 * np.diag(np.ones((10,))),
                    "n_refinements": 1,
                },
                [4.6765, 49.24459, 3.26151, 6.769744583801085e-07],
            ),  # NOQA
            (
                {
                    "lam": 0.5 * np.ones((10, 10)) - 0.5 * np.diag(np.ones((10,))),
                    "n_refinements": 1,
                    "init_method": "cov",
                },
                [0.0106, 21634.95296, 57.6289, 0.00039],
            ),
            (
                {
                    "lam": 0.5 * np.ones((10, 10)) - 0.5 * np.diag(np.ones((10,))),
                    "n_refinements": 1,
                    "init_method": custom_init,
                },
                [0.0106, 21634.95296, 57.6289, 0.00039],
            ),  # NOQA
            (
                {
                    "lam": 0.5 * np.ones((10, 10)) - 0.5 * np.diag(np.ones((10,))),
                    "n_refinements": 1,
                    "init_method": "spearman",
                },
                [
                    4.8315707207048622,
                    38.709631332689789,
                    2.8265068394116657,
                    1.5312382906085276e-07,
                ],
            ),  # NOQA
            (
                {
                    "lam": 0.5 * np.ones((10, 10)) - 0.5 * np.diag(np.ones((10,))),
                    "n_refinements": 1,
                    "init_method": "kendalltau",
                },
                [
                    4.9007318106601074,
                    85.081499460930743,
                    2.0463861650623159,
                    0.00012530384889419821,
                ],
            ),  # NOQA
        ],
    )
    def test_integration_quic_graphical_lasso_cv(self, params_in, expected):
        """
        Just tests inputs/outputs (not validity of result).
        """
        X = datasets.load_diabetes().data
        ic = QuicGraphicalLassoCV(**params_in)
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
            assert ic.lam_.shape == params_in["lam"].shape

        print(result_vec)
        assert_allclose(expected, result_vec, atol=1e-1, rtol=1e-1)

        assert len(ic.grid_scores_) == len(ic.cv_lams_)

    @pytest.mark.parametrize(
        "params_in, expected",
        [
            ({}, [3.1622776601683795, 3.1622776601683795, 0.91116275611548958]),
            ({"lam": 0.5 * np.ones((10, 10))}, [4.797, 2.1849]),
            (
                {"lam": 0.5 * np.ones((10, 10)), "init_method": custom_init},
                [0.0106, 35056.88460],
            ),  # NOQA
        ],
    )
    def test_integration_quic_graphical_lasso_ebic(self, params_in, expected):
        """
        Just tests inputs/outputs (not validity of result).
        """
        X = datasets.load_diabetes().data
        ic = QuicGraphicalLassoEBIC(**params_in)
        ic.fit(X)

        result_vec = [np.linalg.norm(ic.covariance_), np.linalg.norm(ic.precision_)]
        if isinstance(ic.lam_, float):
            result_vec.append(ic.lam_)
        elif isinstance(ic.lam_, np.ndarray):
            assert ic.lam_.shape == params_in["lam"].shape

        print(result_vec)
        assert_allclose(expected, result_vec, atol=1e-1, rtol=1e-1)

    def test_invalid_method(self):
        """
        Test behavior of invalid inputs.
        """
        X = datasets.load_diabetes().data
        ic = QuicGraphicalLasso(method="unknownmethod")
        assert_raises(NotImplementedError, ic.fit, X)
