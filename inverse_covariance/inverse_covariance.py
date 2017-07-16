from __future__ import absolute_import

import numpy as np
from sklearn.base import BaseEstimator

from . import metrics
from .rank_correlation import (
    spearman_correlation,
    kendalltau_correlation,
)


def _init_coefs(X, method='corrcoef'):
    if method == 'corrcoef':
        return np.corrcoef(X, rowvar=False), 1.0
    elif method == 'cov':
        init_cov = np.cov(X, rowvar=False)
        return init_cov, np.max(np.abs(np.triu(init_cov)))
    elif method == 'spearman':
        return spearman_correlation(X, rowvar=False), 1.0
    elif method == 'kendalltau':
        return kendalltau_correlation(X, rowvar=False), 1.0
    elif callable(method):
        return method(X)
    else:
        raise ValueError(
            ("initialize_method must be 'corrcoef' or 'cov', "
             "passed \'{}\' .".format(method))
        )


def _compute_error(comp_cov, covariance_, precision_,
                   score_metric='frobenius'):
    """Computes the covariance error vs. comp_cov.

    Parameters
    ----------
    comp_cov : array-like, shape = (n_features, n_features)
        The precision to compare with.
        This should normally be the test sample covariance/precision.

    scaling : bool
        If True, the squared error norm is divided by n_features.
        If False (default), the squared error norm is not rescaled.

    score_metric : str
        The type of norm used to compute the error between the estimated
        self.precision, self.covariance and the reference `comp_cov`.
        Available error types:

        - 'frobenius' (default): sqrt(tr(A^t.A))
        - 'spectral': sqrt(max(eigenvalues(A^t.A))
        - 'kl': kl-divergence
        - 'quadratic': quadratic loss
        - 'log_likelihood': negative log likelihood

    squared : bool
        Whether to compute the squared error norm or the error norm.
        If True (default), the squared error norm is returned.
        If False, the error norm is returned.
    """
    if score_metric == "frobenius":
        return np.linalg.norm(np.triu(comp_cov - covariance_, 1), ord='fro')
    elif score_metric == "spectral":
        error = comp_cov - covariance_
        return np.amax(np.linalg.svdvals(np.dot(error.T, error)))
    elif score_metric == "kl":
        return metrics.kl_loss(comp_cov, precision_)
    elif score_metric == "quadratic":
        return metrics.quadratic_loss(comp_cov, precision_)
    elif score_metric == "log_likelihood":
        return -metrics.log_likelihood(comp_cov, precision_)
    else:
        raise NotImplementedError(("Must be frobenius, spectral, kl, "
                                   "quadratic, or log_likelihood"))


def _validate_path(path):
    """Sorts path values from largest to smallest.

    Will warn if path parameter was not already sorted.
    """
    if path is None:
        return None

    new_path = np.array(sorted(set(path), reverse=True))
    if new_path[0] != path[0]:
        print('Warning: Path must be sorted largest to smallest.')

    return new_path


class InverseCovarianceEstimator(BaseEstimator):
    """
    Base class for inverse covariance estimators.

    Provides initialization method, metrics, scoring function,
    and ebic model selection.

    Parameters
    -----------
    score_metric : one of 'log_likelihood' (default), 'frobenius', 'spectral',
                  'kl', or 'quadratic'
        Used for computing self.score().

    init_method : one of 'corrcoef', 'cov', 'spearman', 'kendalltau',
        or a custom function.
        Computes initial covariance and scales lambda appropriately.
        Using the custom function extends graphical model estimation to
        distributions beyond the multivariate Gaussian.
        The `spearman` or `kendalltau` options extend inverse covariance
        estimation to nonparanormal and transelliptic graphical models.
        Custom function must return ((n_features, n_features) ndarray, float)
        where the scalar parameter will be used to scale the penalty lam.

    auto_scale : bool
        If True, will compute self.lam_scale_ = max off-diagonal value when
        init_method='cov'.
        If false, then self.lam_scale_ = 1.
        lam_scale_ is used to scale user-supplied self.lam during fit.

    Attributes
    ----------
    covariance_ : 2D ndarray, shape (n_features, n_features)
        Estimated covariance matrix

        This can also be a len(path) list of
        2D ndarray, shape (n_features, n_features)
        (e.g., see mode='path' in QuicGraphLasso)

    precision_ : 2D ndarray, shape (n_features, n_features)
        Estimated pseudo-inverse matrix.

        This can also be a len(path) list of
        2D ndarray, shape (n_features, n_features)
        (e.g., see mode='path' in QuicGraphLasso)

    sample_covariance_ : 2D ndarray, shape (n_features, n_features)
        Estimated sample covariance matrix

    lam_scale_ : (float)
        Additional scaling factor on lambda (due to magnitude of
        sample_covariance_ values).
    """
    def __init__(self, score_metric='log_likelihood', init_method='cov',
                 auto_scale=True):
        self.score_metric = score_metric
        self.init_method = init_method
        self.auto_scale = auto_scale

        self.covariance_ = None  # assumes a matrix of a list of matrices
        self.precision_ = None  # assumes a matrix of a list of matrices

        # these must be updated upon self.fit()
        # the first 4 will be set if self.init_coefs is used.
        #   self.sample_covariance_
        #   self.lam_scale_
        #   self.n_samples
        #   self.n_features
        self.is_fitted = False

        super(InverseCovarianceEstimator, self).__init__()

    def init_coefs(self, X):
        """Computes ...

        Initialize the following values:
            self.n_samples
            self.n_features
            self.sample_covariance_
            self.lam_scale_
        """
        self.n_samples, self.n_features = X.shape
        self.sample_covariance_, self.lam_scale_ = _init_coefs(
                X,
                method=self.init_method)

        if not self.auto_scale:
            self.lam_scale_ = 1.0

    def score(self, X_test, y=None):
        """Computes the score between cov/prec of sample covariance of X_test
        and X via 'score_metric'.

        Note: We want to maximize score so we return the negative error.

        Parameters
        ----------
        X_test : array-like, shape = [n_samples, n_features]
            Test data of which we compute the likelihood, where n_samples is
            the number of samples and n_features is the number of features.
            X_test is assumed to be drawn from the same distribution than
            the data used in fit (including centering).

        y : not used.

        Returns
        -------
        result : float or list of floats
            The negative of the min error between `self.covariance_` and
            the sample covariance of X_test.
        """
        if isinstance(self.precision_, list):
            print('Warning: returning a list of scores.')

        S_test, lam_scale_test = _init_coefs(
                X_test,
                method=self.init_method)
        error = self.cov_error(S_test, score_metric=self.score_metric)

        # maximize score with -error
        return -error

    def cov_error(self, comp_cov, score_metric='frobenius'):
        """Computes the covariance error vs. comp_cov.

        May require self.path_

        Parameters
        ----------
        comp_cov : array-like, shape = (n_features, n_features)
            The precision to compare with.
            This should normally be the test sample covariance/precision.

        scaling : bool
            If True, the squared error norm is divided by n_features.
            If False (default), the squared error norm is not rescaled.

        score_metric : str
            The type of norm used to compute the error between the estimated
            self.precision, self.covariance and the reference `comp_cov`.
            Available error types:

            - 'frobenius' (default): sqrt(tr(A^t.A))
            - 'spectral': sqrt(max(eigenvalues(A^t.A))
            - 'kl': kl-divergence
            - 'quadratic': quadratic loss
            - 'log_likelihood': negative log likelihood

        squared : bool
            Whether to compute the squared error norm or the error norm.
            If True (default), the squared error norm is returned.
            If False, the error norm is returned.

        Returns
        -------
        The min error between `self.covariance_` and `comp_cov`.

        If self.precision_ is a list, returns errors for each matrix, otherwise
        returns a scalar.
        """
        if not isinstance(self.precision_, list):
            return _compute_error(
                comp_cov,
                self.covariance_,
                self.precision_,
                score_metric
            )

        path_errors = []
        for lidx, lam in enumerate(self.path_):
            path_errors.append(_compute_error(
                comp_cov,
                self.covariance_[lidx],
                self.precision_[lidx],
                score_metric)
            )

        return np.array(path_errors)

    def ebic(self, gamma=0):
        """Compute EBIC scores for each model. If model is not "path" then
        returns a scalar score value.

        May require self.path_

        See:
        Extended Bayesian Information Criteria for Gaussian Graphical Models
        R. Foygel and M. Drton
        NIPS 2010

        Parameters
        ----------
        gamma : (float) \in (0, 1)
            Choice of gamma=0 leads to classical BIC
            Positive gamma leads to stronger penalization of large graphs.

        Returns
        -------
        Scalar ebic score or list of ebic scores.
        """
        if not self.is_fitted:
            return

        if not isinstance(self.precision_, list):
            return metrics.ebic(self.sample_covariance_,
                                self.precision_,
                                self.n_samples,
                                self.n_features,
                                gamma=gamma)

        ebic_scores = []
        for lidx, lam in enumerate(self.path_):
            ebic_scores.append(metrics.ebic(
                    self.sample_covariance_,
                    self.precision_[lidx],
                    self.n_samples,
                    self.n_features,
                    gamma=gamma))

        return np.array(ebic_scores)

    def ebic_select(self, gamma=0):
        """Uses Extended Bayesian Information Criteria for model selection.

        Can only be used in path mode (doesn't really make sense otherwise).

        See:
        Extended Bayesian Information Criteria for Gaussian Graphical Models
        R. Foygel and M. Drton
        NIPS 2010

        Parameters
        ----------
        gamma : (float) \in (0, 1)
            Choice of gamma=0 leads to classical BIC
            Positive gamma leads to stronger penalization of large graphs.

        Returns
        -------
        Lambda index with best ebic score.  When multiple ebic scores are the
        same, returns the smallest lambda (largest index) with minimum score.
        """
        if not isinstance(self.precision_, list):
            raise ValueError(
                "EBIC requires multiple models to select from."
            )
            return

        if not self.is_fitted:
            return

        ebic_scores = self.ebic(gamma=gamma)
        min_indices = np.where(np.abs(ebic_scores - ebic_scores.min()) < 1e-10)
        return np.max(min_indices)
