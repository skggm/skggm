"""
Visualize Regularization Path
=============================

Plot the edge level coefficients (inverse covariance entries)
as a function of the regularization parameter.

"""

import sys
import numpy as np
from sklearn.datasets import make_sparse_spd_matrix

sys.path.append("..")
from inverse_covariance import QuicGraphicalLasso
from inverse_covariance.plot_util import trace_plot
from inverse_covariance.profiling import LatticeGraph


def make_data(n_samples, n_features):
    prng = np.random.RandomState(1)
    prec = make_sparse_spd_matrix(
        n_features, alpha=.98, smallest_coef=.4, largest_coef=.7, random_state=prng
    )
    cov = np.linalg.inv(prec)
    d = np.sqrt(np.diag(cov))
    cov /= d
    cov /= d[:, np.newaxis]
    prec *= d
    prec *= d[:, np.newaxis]
    X = prng.multivariate_normal(np.zeros(n_features), cov, size=n_samples)
    X -= X.mean(axis=0)
    X /= X.std(axis=0)
    return X, cov, prec


def make_data_banded(n_samples, n_features):
    alpha = 0.1
    cov, prec, adj = LatticeGraph(
        n_blocks=2, random_sign=True, chain_blocks=True, seed=1
    ).create(n_features, alpha)
    prng = np.random.RandomState(2)
    X = prng.multivariate_normal(np.zeros(n_features), cov, size=n_samples)
    return X, cov, prec


def show_quic_coefficient_trace(X):
    path = np.logspace(np.log10(0.01), np.log10(1.0), num=50, endpoint=True)[::-1]
    estimator = QuicGraphicalLasso(lam=1.0, path=path, mode="path")
    estimator.fit(X)
    trace_plot(estimator.precision_, estimator.path_, n_edges=20)


def show_quic_coefficient_trace_truth(X, truth):
    path = np.logspace(np.log10(0.01), np.log10(1.0), num=50, endpoint=True)[::-1]
    estimator = QuicGraphicalLasso(lam=1.0, path=path, mode="path")
    estimator.fit(X)
    trace_plot(estimator.precision_, estimator.path_, n_edges=6, ground_truth=truth)


if __name__ == "__main__":
    # example 1
    n_samples = 10
    n_features = 5
    X, cov, prec = make_data(n_samples, n_features)

    print("Showing basic Erdos-Renyi example with ")
    print("   n_samples=10")
    print("   n_features=5")
    print("   n_edges=20")
    show_quic_coefficient_trace(X)

    # use ground truth for display
    print("Showing basic Erdos-Renyi example with ")
    print("   n_samples=100")
    print("   n_features=5")
    print("   n_edges=6")
    print("   ground_truth (shows only false pos and negatives)")
    show_quic_coefficient_trace_truth(X, prec)

    # example 2
    n_samples = 110
    n_features = 100
    X, cov, prec = make_data_banded(n_samples, n_features)

    print("Showing basic Lattice example with ")
    print("   n_samples=110")
    print("   n_features=100")
    print("   n_blocks=2")
    print("   random_sign=True")
    print("   n_edges=20")
    show_quic_coefficient_trace(X)

    # use ground truth for display
    print("Showing basic Lattice example with ")
    print("   n_samples=110")
    print("   n_features=100")
    print("   n_blocks=2")
    print("   random_sign=True")
    print("   n_edges=6")
    print("   ground_truth (shows only false pos and negatives)")
    show_quic_coefficient_trace_truth(X, prec)
