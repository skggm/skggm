"""Nonparametric rank correlation estimators as alternative to
 linear correlation estimators."""
import numpy as np
from scipy.stats import rankdata


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


def spearman_correlation(X):
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
    """
    Xrank = _compute_ranks(X)
    rank_correlation = np.corrcoef(Xrank, rowvar=False)

    return 2 * np.sin(rank_correlation * np.pi / 6)


def kendalltau_correlation(X):
    """
    Computes kendall's tau correlation estimate.

    Parameters
    ----------
    X: array-like, shape = [n_samples, n_features]
        Data matrix using which we compute the empirical
        correlation

    Returns
    -------
    rank_correlation
    """

    pass
