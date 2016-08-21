import numpy as np
import pytest

from sklearn import datasets
from sklearn.covariance import GraphLassoCV

from .. import StatisticalPower


class MockGraphLasso(object):
    """Will return result with statistical power performance linear in the size
    of n_examples/n_features.
    """
    def __init__(self, cov_true, prec_true):
        self.cov_true = cov_true
        self.prec_true = prec_true

        self.precision_ = None
        self.covariance_ = self.cov_true # ignoring covariance for now

    def fit(self, X, y=None, **fit_params):
        n_examples, n_features = X.shape
        ratio = 1. * n_examples / n_features

        prec_est = np.copy(self.prec_true)
        prec_est[np.nonzero(prec_est)]


        return self


class TestStatisticalPower(object):
