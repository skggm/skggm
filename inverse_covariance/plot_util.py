"""Various utilities for Gaussian graphical models.
"""
import numpy as np
from sklearn.utils.testing import assert_array_equal
from matplotlib import pyplot as plt
import seaborn

plt.ion()


def _check_path(in_path):
    path = sorted(set(in_path), reverse=True)
    assert_array_equal(path, in_path)


def trace_plot(results, path):
    """Plot the change in precision (or covariance) coefficients as a function 
    of changing lambda and l1-norm.

    Parameters
    -----------
    results: array of len(path) of 2D ndarray, shape (n_features, n_features)
        This is either precision_ or covariance_ from an InverseCovariance
        estimator in path mode, or a list of results for individual runs of 
        the GraphLasso.

    path:  array of floats (descending)
        This is path of lambdas explored. 
    """
    _check_path(path)
    assert len(path) == len(results)
    assert len(results) > 0

    path = np.array(path)
    dim, _ = results[0].shape

    # reshape data a bit:  
    # flatten each matrix into a column (so that coeffs are examples)
    # compute l1-norm of each column
    l1_norms = []
    coeffs = np.zeros((dim**2, len(results)))
    for ridx, result in enumerate(results):
        coeffs[:, ridx] = result.flat 
        l1_norms.append(np.linalg.norm(coeffs[:, ridx]))

    plt.figure()

    # show coefficients as a function of lambda
    plt.subplot(1, 2, 1)
    for result in results:
        plt.plot(l1_norms, coeffs.T, lw=1)

    plt.xlim([np.min(l1_norms), np.max(l1_norms)])
    plt.ylabel('Coefficients')
    plt.xlabel('l1 Norm')

    # show coefficients as a function of lambda
    log_path = np.log(path)
    plt.subplot(1, 2, 2)
    for result in results:
        plt.plot(log_path, coeffs.T, lw=1)

    plt.xlim([np.min(log_path), np.max(log_path)])
    plt.ylabel('Coefficients')
    plt.xlabel('log-Lambda')

    plt.show()
    raw_input('Press any key to continue.')

    