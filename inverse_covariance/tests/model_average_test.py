import numpy as np
import pytest

from sklearn.covariance import GraphLassoCV

from inverse_covariance import QuicGraphicalLassoCV, QuicGraphicalLasso, ModelAverage
from inverse_covariance.profiling import ClusterGraph


class TestModelAverage(object):
    @pytest.mark.parametrize(
        "params_in",
        [
            (
                {
                    "estimator": QuicGraphicalLasso(),
                    "n_trials": 10,
                    "normalize": True,
                    "subsample": 0.3,
                    "penalization": "random",
                }
            ),
            (
                {
                    "estimator": QuicGraphicalLasso(lam=0.5, mode="trace"),
                    "n_trials": 10,
                    "normalize": False,
                    "subsample": 0.6,
                    "penalization": "fully-random",
                }
            ),
            (
                {
                    "estimator": QuicGraphicalLassoCV(cv=(2, 1)),
                    "n_trials": 2,
                    "normalize": True,
                    "subsample": 0.9,
                    "lam": 0.1,
                    "lam_perturb": 0.1,
                    "penalization": "random",
                }
            ),
            (
                {
                    "estimator": GraphLassoCV(cv=2),
                    "n_trials": 2,
                    "normalize": True,
                    "subsample": 0.9,
                    "penalization": "subsampling",
                    "penalty_name": "alpha",
                }
            ),
            (
                {
                    "estimator": QuicGraphicalLasso(),
                    "n_trials": 10,
                    "normalize": True,
                    "subsample": 0.3,
                    "lam": 0.1,
                    "lam_perturb": 0.1,
                    "penalization": "random",
                    "n_jobs": 2,
                }
            ),
        ],
    )
    def test_integration_quic_graph_lasso_cv(self, params_in):
        """
        Just tests inputs/outputs (not validity of result).
        """
        n_features = 30
        n_samples = 30
        cov, prec, adj = ClusterGraph(n_blocks=1, chain_blocks=False, seed=1).create(
            n_features, 0.8
        )
        prng = np.random.RandomState(2)
        X = prng.multivariate_normal(np.zeros(n_features), cov, size=n_samples)

        ma = ModelAverage(**params_in)
        ma.fit(X)

        n_examples, n_features = X.shape

        assert ma.proportion_.shape == (n_features, n_features)
        assert len(ma.estimators_) == ma.n_trials
        assert len(ma.subsets_) == ma.n_trials
        if not ma.penalization == "subsampling":
            assert len(ma.lams_) == ma.n_trials
        else:
            assert len(ma.lams_) == ma.n_trials
            assert ma.lams_[0] is None

        for eidx, e in enumerate(ma.estimators_):
            assert isinstance(e, params_in["estimator"].__class__)

            # sklearn doesnt have this but ours do
            if hasattr(e, "is_fitted"):
                assert e.is_fitted is True

            # check that all lambdas used where different
            if not ma.penalization == "subsampling" and eidx > 0:
                if hasattr(e, "lam"):
                    prev_e = ma.estimators_[eidx - 1]
                    assert np.linalg.norm((prev_e.lam - e.lam).flat) > 0

        if ma.normalize is True:
            assert np.max(ma.proportion_) <= 1.0
        else:
            assert np.max(ma.proportion_) <= ma.n_trials

        assert np.min(ma.proportion_) >= 0.0
        assert np.max(ma.proportion_) > 0.0
