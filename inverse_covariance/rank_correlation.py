"""Nonparametric rank correlation estimators as alternative to
 linear correlation estimators."""
import numpy as np
from scipy.stats import (
    rankdata,
    kendalltau,
    weightedtau
)


def _compute_ranks(X):
    """
    Transform each column into ranked data. Tied ranks are averaged.

    Parameters
    ----------
    X : array-like, shape = (n_samples, n_features)
        The data matrix where each column is a feature.
         Row observations for each column will be replaced
         by correponding rank

    Returns
    -------
    Xrank
    """
    _, n_features = X.shape
    Xrank = np.zeros(shape=X.shape)
    for col in np.arange(n_features):
        Xrank[:, col] = rankdata(X[:, col], method='average')

    return Xrank


def spearman_correlation(X, rowvar=False):
    """
    Computes the spearman correlation estimate.
    This is effectively a bias corrected pearson correlation
    between rank transformed columns of X.

    Parameters
    ----------
    X: array-like, shape = [n_samples, n_features]
        Data matrix using which we compute the empirical
        correlation

    Returns
    -------
    rank_correlation

    References
    ----------

    Xue, Lingzhou; Zou, Hui.
    "Regularized rank-based estimation of high-dimensional
    nonparanormal graphical models."
    Ann. Statist. 40 (2012), no. 5, 2541--2571. doi:10.1214/12-AOS1041.

    Liu, Han, Fang; Yuan, Ming; Lafferty, John; Wasserman, Larry.
    "High-dimensional semiparametric Gaussian copula graphical models."
    Ann. Statist. 40.4 (2012): 2293-2326. doi:10.1214/12-AOS1037
    """

    Xrank = _compute_ranks(X)
    rank_correlation = np.corrcoef(Xrank, rowvar=rowvar)

    return 2 * np.sin(rank_correlation * np.pi / 6)


def kendalltau_correlation(X, rowvar=False, weighted=False):
    """
    Computes kendall's tau correlation estimate.
    The option to use scipy.stats.weightedtau is not recommended
    as the implementation does not appear to handle ties correctly.

    Parameters
    ----------
    X: array-like, shape = [n_samples, n_features]
        Data matrix using which we compute the empirical
        correlation

    Returns
    -------
    rank_correlation

    References
    ----------

    Liu, Han, Fang; Yuan, Ming; Lafferty, John; Wasserman, Larry.
    "High-dimensional semiparametric Gaussian copula graphical models."
    Ann. Statist. 40.4 (2012): 2293-2326. doi:10.1214/12-AOS1037

    Barber, Rina Foygel; Kolar, Mladen.
    "ROCKET: Robust Confidence Intervals via Kendall's Tau
    for Transelliptical Graphical Models."
     arXiv:1502.07641
    """

    if rowvar:
        X = X.T

    _, n_features = X.shape
    rank_correlation = np.eye(n_features)
    for row in np.arange(n_features):
        for col in np.arange(1+row, n_features):
            if weighted:
                rank_correlation[row, col], _ = weightedtau(
                    X[:, row], X[:, col], rank=False
                )
            else:
                rank_correlation[row, col], _ = kendalltau(
                    X[:, row], X[:, col]
                )
    rank_correlation = np.triu(rank_correlation, 1) + rank_correlation.T

    return np.sin(rank_correlation * np.pi / 2)


def trimmean_correlation(X, rowvar=False, weighted=False):
    """
    Computes a winsorized estimate of pairwise correlations

    Parameters
    ----------
    X: array-like, shape = [n_samples, n_features]
        Data matrix using which we compute the empirical
        correlation

    Returns
    -------
    rank_correlation

    References
    ----------

    Liu, Han, John Lafferty, and Larry Wasserman.
    "The nonparanormal: Semiparametric estimation of high dimensional
    undirected graphs."
    Journal of Machine Learning Research 10.Oct (2009): 2295-2328.
    """

    pass
