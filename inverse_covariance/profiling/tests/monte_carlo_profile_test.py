import numpy as np
import pytest

from inverse_covariance.profiling import MonteCarloProfile


class FakeGraph(object):
    def create(self, n_features, alpha):
        identity = np.eye(n_features)
        return identity, identity, identity


def fake_metric(a, b):
    return 0.5


metrics = {"m0": fake_metric, "m1": fake_metric, "m2": fake_metric}


class TestMonteCarloProfile(object):
    @pytest.mark.parametrize(
        "params_in",
        [
            (
                {
                    "n_trials": 20,
                    "n_features": 10,
                    "graph": FakeGraph(),
                    "metrics": metrics,
                    "verbose": True,
                }
            ),
            (
                {
                    "n_trials": 20,
                    "n_features": 10,
                    "graph": FakeGraph(),
                    "metrics": metrics,
                    "n_samples_grid": [2, 3],
                    "alpha_grid": [2, 3],
                }
            ),
            (
                {
                    "n_trials": 20,
                    "n_features": 10,
                    "graph": FakeGraph(),
                    "n_jobs": 2,
                    "metrics": metrics,
                }
            ),
        ],
    )
    def test_integration_monte_carlo_profile(self, params_in):
        mc = MonteCarloProfile(**params_in)
        mc.fit()

        assert len(mc.results_) == len(params_in["metrics"])
        for key in params_in["metrics"]:
            assert key in mc.results_
            assert np.sum(mc.results_[key].flat) > 0
            assert mc.results_[key].shape == (len(mc.alphas_), len(mc.grid_))

        assert len(mc.precision_nnz_) == len(mc.alphas_) * len(mc.grid_)
        assert mc.precision_nnz_[0] == params_in["n_features"]  # for eye

        if isinstance(mc.n_samples_grid, int):
            assert len(mc.grid_) == mc.n_samples_grid
        else:
            assert mc.grid_ == mc.n_samples_grid

        if isinstance(mc.alpha_grid, int):
            assert len(mc.alphas_) == mc.alpha_grid
        else:
            assert mc.alphas_ == mc.alpha_grid

    @pytest.mark.parametrize("params_in", [({"n_trials": 20, "n_features": 10})])
    def test_integration_monte_carlo_profile_default(self, params_in):
        """Use default graph and metrics. """
        mc = MonteCarloProfile(**params_in)
        mc.fit()

        assert len(mc.results_) > 0

        for key in mc.results_:
            assert key in mc.results_
            assert np.sum(mc.results_[key].flat) > 0
            assert mc.results_[key].shape == (len(mc.alphas_), len(mc.grid_))

        assert len(mc.precision_nnz_) == len(mc.alphas_) * len(mc.grid_)

        if isinstance(mc.n_samples_grid, int):
            assert len(mc.grid_) == mc.n_samples_grid
        else:
            assert mc.grid_ == mc.n_samples_grid

        if isinstance(mc.alpha_grid, int):
            assert len(mc.alphas_) == mc.alpha_grid
        else:
            assert mc.alphas_ == mc.alpha_grid
