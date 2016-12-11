"""Various utilities for Gaussian graphical models."""
import numpy as np
from sklearn.utils.testing import assert_array_equal
from matplotlib import pyplot as plt
import seaborn

plt.ion()


def _check_path(in_path):
    path = sorted(set(in_path), reverse=True)
    assert_array_equal(path, in_path)


def trace_plot(precisions, path, n_edges=20):
    """Plot the change in precision (or covariance) coefficients as a function 
    of changing lambda and l1-norm.  Always ignores diagonals.

    Parameters
    -----------
    precisions : array of len(path) of 2D ndarray, shape (n_features, n_features)
        This is either precision_ or covariance_ from an InverseCovariance
        estimator in path mode, or a list of results for individual runs of 
        the GraphLasso.

    path :  array of floats (descending)
        This is path of lambdas explored. 

    n_edges :  int (default=20)
        Max number of edges to plot for each precision matrix along the path.
        Only plots the maximum magnitude values (evaluating the last precision 
        matrix).
    """
    _check_path(path)
    assert len(path) == len(precisions)
    assert len(precisions) > 0

    path = np.array(path)
    dim, _ = precisions[0].shape

    # determine which indices to track
    base_precision = np.copy(precisions[-1])
    base_precision[np.triu_indices(base_precision.shape[0])] = 0
    sidx = np.argsort(np.abs(base_precision.flat))[::-1][: n_edges]

    # reshape data a bit:  
    # flatten each matrix into a column (so that coeffs are examples)
    # compute l1-norm of each column
    l1_norms = []
    coeffs = np.zeros((dim**2, len(precisions)))
    for ridx, result in enumerate(precisions):
        coeffs[sidx, ridx] = result.flat[sidx]
        l1_norms.append(np.linalg.norm(coeffs[:, ridx]))

    # remove any zero rows
    coeffs = coeffs[np.linalg.norm(coeffs, axis=1) > 1e-10, :]

    plt.figure()

    # show coefficients as a function of lambda
    plt.subplot(1, 2, 1)
    for result in precisions:
        plt.plot(l1_norms, coeffs.T, lw=1)

    plt.xlim([np.min(l1_norms), np.max(l1_norms)])
    plt.ylabel('Coefficients')
    plt.xlabel('l1 Norm')

    # show coefficients as a function of lambda
    log_path = np.log(path)
    plt.subplot(1, 2, 2)
    for result in precisions:
        plt.plot(log_path, coeffs.T, lw=1)

    plt.xlim([np.min(log_path), np.max(log_path)])
    plt.ylabel('Coefficients')
    plt.xlabel('log-Lambda')

    plt.show()
    raw_input('Press any key to continue.')

    