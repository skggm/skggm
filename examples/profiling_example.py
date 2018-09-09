"""
Example of Montecarlo Benchmarking
=============================

Run a MonteCarlo simulation on synthetic network structure and compare
estimation and model selection errors

"""


import sys
import matplotlib.pyplot as plt

sys.path.append("..")
from inverse_covariance.profiling import (
    MonteCarloProfile,
    support_false_positive_count,
    support_false_negative_count,
    support_difference_count,
    has_exact_support,
    has_approx_support,
    error_fro,
    LatticeGraph,
)

plt.ion()


def r_input(val):
    if sys.version_info[0] >= 3:
        return eval(input(val))

    return raw_input(val)


###############################################################################
# Setup metrics

metrics = {
    "fp_rate": support_false_positive_count,
    "fn_rate": support_false_negative_count,
    "support_error": support_difference_count,
    "prob_exact_support": has_exact_support,
    "prob_approx_support": has_approx_support,
    "frobenius": error_fro,
}


###############################################################################
# Run MC trials

mc = MonteCarloProfile(
    n_features=50,
    n_trials=10,
    graph=LatticeGraph(),
    n_samples_grid=10,
    alpha_grid=5,
    metrics=metrics,
    verbose=True,
    n_jobs=4,
)
mc.fit()

###############################################################################
# Plot results for each metric

for key in metrics:
    plt.figure()
    plt.plot(mc.grid_, mc.results_[key].T, linewidth=2)
    plt.title("metric = {}".format(key))
    legend_items = ["alpha={}".format(a) for a in mc.alphas_]
    plt.legend(legend_items)
    plt.show()


r_input("Any key to exit.")
