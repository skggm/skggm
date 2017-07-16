"""Nonparametric rank correlation estimators as alternative to
 linear correlation estimators."""
from __future__ import absolute_import

import numpy as np
from scipy.stats import (
    rankdata,
    kendalltau,
    weightedtau
)


def _compute_ranks(X, winsorize=False, truncation=None, verbose=True):
    """
    Transform each column into ranked data. Tied ranks are averaged.
    Ranks can optionally be winsorized as described in Liu 2009 otherwise
    this returns Tsukahara's scaled rank based Z-estimator.

    Parameters
    ----------
    X : array-like, shape = (n_samples, n_features)
        The data matrix where each column is a feature.
         Row observations for each column will be replaced
         by correponding rank.

    winsorize: bool
        Choose whether ranks should be winsorized (trimmed) or not. If True,
        then ranks will be winsorized using the truncation parameter.

    truncation: (float)
        The default value is given by 1/(4 n^(1/4) * sqrt(pi log n)), where
        n is the number of samples.

    Returns
    -------
    Xrank

    References
    ----------

    Liu, Han, John Lafferty, and Larry Wasserman.
    "The nonparanormal: Semiparametric estimation of high dimensional
    undirected graphs."
    Journal of Machine Learning Research 10.Oct (2009): 2295-2328.
    """
    n_samples, n_features = X.shape
    Xrank = np.zeros(shape=X.shape)

    if winsorize:
        if truncation is None:
            truncation = (
                1 / (
                    4 * np.power(n_samples, 0.25) *
                    np.sqrt(np.pi * np.log(n_samples))
                )
            )

        elif (truncation > 1):
            truncation = np.min(1.0, truncation)

    for col in np.arange(n_features):
        Xrank[:, col] = rankdata(X[:, col], method='average')
        Xrank[:, col] /= n_samples
        if winsorize:
            if n_samples > 100*n_features:
                Xrank[:, col] = n_samples * Xrank[:, col] / (n_samples + 1)
            else:
                lower_truncate = Xrank[:, col] <= truncation
                upper_truncate = Xrank[:, col] > 1-truncation
                Xrank[lower_truncate, col] = truncation
                Xrank[upper_truncate, col] = 1-truncation

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


def winsorized_rank_correlation(X, rowvar=False, weighted=False):
    """
    Computes rank correlations using a winsorized ranks.
    Relevant to high dimensional settings where the n_samples < n_features
    resulting too large variance in the ranks.

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
