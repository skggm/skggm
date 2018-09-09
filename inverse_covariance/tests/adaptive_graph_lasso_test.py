import numpy as np
import pytest

from inverse_covariance import (
    QuicGraphicalLassoEBIC,
    AdaptiveGraphicalLasso,
    QuicGraphicalLassoCV,
)
from inverse_covariance.profiling import ClusterGraph


class TestAdaptiveGraphicalLasso(object):
    @pytest.mark.parametrize(
        "params_in",
        [
            (
                {
                    "estimator": QuicGraphicalLassoCV(
                        cv=2,
                        n_refinements=6,
                        init_method="cov",
                        score_metric="log_likelihood",
                    ),
                    "method": "binary",
                }
            ),
            (
                {
                    "estimator": QuicGraphicalLassoCV(
                        cv=2,
                        n_refinements=6,
                        init_method="cov",
                        score_metric="log_likelihood",
                    ),
                    "method": "inverse",
                }
            ),
            (
                {
                    "estimator": QuicGraphicalLassoCV(
                        cv=2,
                        n_refinements=6,
                        init_method="cov",
                        score_metric="log_likelihood",
                    ),
                    "method": "inverse_squared",
                }
            ),
            ({"estimator": QuicGraphicalLassoEBIC(), "method": "binary"}),
            ({"estimator": QuicGraphicalLassoEBIC(), "method": "inverse"}),
            ({"estimator": QuicGraphicalLassoEBIC(), "method": "inverse_squared"}),
        ],
    )
    def test_integration_adaptive_graphical_lasso(self, params_in):
        """
        Just tests inputs/outputs (not validity of result).
        """
        n_features = 20
        n_samples = 25
        cov, prec, adj = ClusterGraph(n_blocks=1, chain_blocks=False, seed=1).create(
            n_features, 0.8
        )
        prng = np.random.RandomState(2)
        X = prng.multivariate_normal(np.zeros(n_features), cov, size=n_samples)

        model = AdaptiveGraphicalLasso(**params_in)
        model.fit(X)
        assert model.estimator_ is not None
        assert model.lam_ is not None

        assert np.sum(model.lam_[np.diag_indices(n_features)]) == 0

        if params_in["method"] == "binary":
            uvals = set(model.lam_.flat)
            assert len(uvals) == 2
            assert 0 in uvals
            assert 1 in uvals
        elif (
            params_in["method"] == "inverse" or params_in["method"] == "inverse_squared"
        ):
            uvals = set(model.lam_.flat[model.lam_.flat != 0])
            assert len(uvals) > 0
