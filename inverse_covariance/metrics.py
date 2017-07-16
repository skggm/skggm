"""Metrics for cross validation with Gaussian graphical models."""
import numpy as np
from sklearn.utils.extmath import fast_logdet


def log_likelihood(covariance, precision):
    """Computes the log-likelihood between the covariance and precision
    estimate.

    Parameters
    ----------
    covariance : 2D ndarray (n_features, n_features)
        Maximum Likelihood Estimator of covariance

    precision : 2D ndarray (n_features, n_features)
        The precision matrix of the covariance model to be tested

    Returns
    -------
    log-likelihood
    """
    assert covariance.shape == precision.shape
    dim, _ = precision.shape
    log_likelihood_ = -np.sum(covariance * precision) +\
        fast_logdet(precision) - dim * np.log(2 * np.pi)
    log_likelihood_ /= 2.
    return log_likelihood_


def kl_loss(covariance, precision):
    """Computes the KL divergence between precision estimate and
    reference covariance.

    The loss is computed as:

        Trace(Theta_1 * Sigma_0) - log(Theta_0 * Sigma_1) - dim(Sigma)

    Parameters
    ----------
    covariance : 2D ndarray (n_features, n_features)
        Maximum Likelihood Estimator of covariance

    precision : 2D ndarray (n_features, n_features)
        The precision matrix of the covariance model to be tested

    Returns
    -------
    KL-divergence
    """
    assert covariance.shape == precision.shape
    dim, _ = precision.shape
    logdet_p_dot_c = fast_logdet(np.dot(precision, covariance))
    return 0.5 * (np.sum(precision * covariance) - logdet_p_dot_c - dim)


def quadratic_loss(covariance, precision):
    """Computes ...

    Parameters
    ----------
    covariance : 2D ndarray (n_features, n_features)
        Maximum Likelihood Estimator of covariance

    precision : 2D ndarray (n_features, n_features)
        The precision matrix of the model to be tested

    Returns
    -------
    Quadratic loss
    """
    assert covariance.shape == precision.shape
    dim, _ = precision.shape
    return np.trace((np.dot(covariance, precision) - np.eye(dim))**2)


def ebic(covariance, precision, n_samples, n_features, gamma=0):
    '''
    Extended Bayesian Information Criteria for model selection.

    When using path mode, use this as an alternative to cross-validation for
    finding lambda.

    See:
        "Extended Bayesian Information Criteria for Gaussian Graphical Models"
        R. Foygel and M. Drton, NIPS 2010

    Parameters
    ----------
    covariance : 2D ndarray (n_features, n_features)
        Maximum Likelihood Estimator of covariance (sample covariance)

    precision : 2D ndarray (n_features, n_features)
        The precision matrix of the model to be tested

    n_samples :  int
        Number of examples.

    n_features : int
        Dimension of an example.

    lam: (float)
        Threshold value for precision matrix. This should be lambda scaling
        used to obtain this estimate.

    gamma : (float) \in (0, 1)
        Choice of gamma=0 leads to classical BIC
        Positive gamma leads to stronger penalization of large graphs.

    Returns
    -------
    ebic score (float).  Caller should minimized this score.
    '''
    l_theta = -np.sum(covariance * precision) + fast_logdet(precision)
    l_theta *= n_features / 2.

    # is something goes wrong with fast_logdet, return large value
    if np.isinf(l_theta) or np.isnan(l_theta):
        return 1e10

    mask = np.abs(precision.flat) > np.finfo(precision.dtype).eps
    precision_nnz = (np.sum(mask) - n_features) / 2.0  # lower off diagonal tri

    return (
        -2.0 * l_theta +
        precision_nnz * np.log(n_samples) +
        4.0 * precision_nnz * np.log(n_features) * gamma
    )
