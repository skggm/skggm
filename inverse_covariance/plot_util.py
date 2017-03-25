"""Various utilities for Gaussian graphical models."""
import sys
import numpy as np
from sklearn.utils.testing import assert_array_equal
from matplotlib import pyplot as plt
import seaborn  # NOQA

plt.ion()


def r_input(val):
    if sys.version_info[0] >= 3:
        return eval(input(val))

    return raw_input(val)  # NOQA


def _check_path(in_path):
    path = sorted(set(in_path), reverse=True)
    assert_array_equal(path, in_path)


def trace_plot(precisions, path, n_edges=20, ground_truth=None, edges=[]):
    """Plot the change in precision (or covariance) coefficients as a function
    of changing lambda and l1-norm.  Always ignores diagonals.

    Parameters
    -----------
    precisions : array of len(path) 2D ndarray, shape (n_features, n_features)
        This is either precision_ or covariance_ from an InverseCovariance
        estimator in path mode, or a list of results for individual runs of
        the GraphLasso.

    path :  array of floats (descending)
        This is path of lambdas explored.

    n_edges :  int (default=20)
        Max number of edges to plot for each precision matrix along the path.
        Only plots the maximum magnitude values (evaluating the last precision
        matrix).

    ground_truth : 2D ndarray, shape (n_features, n_features) (default=None)
        If not None, plot the top n_edges/2 false positive and top n_edges/2
        false negative indices when compared to ground_truth.

    edges : list (default=[])
        If not empty, use edges to determine which indicies of each precision
        matrix to track.  Should be arranged to index precisions[0].flat.

        If non-empty, n_edges and ground_truth will be ignored.
    """
    _check_path(path)
    assert len(path) == len(precisions)
    assert len(precisions) > 0

    path = np.array(path)
    dim, _ = precisions[0].shape

    # determine which indices to track
    if not edges:
        base_precision = np.copy(precisions[-1])
        base_precision[np.triu_indices(base_precision.shape[0])] = 0

        if ground_truth is None:
            # top n_edges strongest coefficients
            edges = np.argsort(np.abs(base_precision.flat))[::-1][: n_edges]
        else:
            # top n_edges/2 false positives and negatives compared to truth
            assert ground_truth.shape == precisions[0].shape
            masked_gt = np.copy(ground_truth)
            masked_gt[np.triu_indices(ground_truth.shape[0])] = 0

            intersection = np.intersect1d(
                np.nonzero(base_precision.flat)[0],
                np.nonzero(masked_gt.flat)[0]
            )

            # false positives
            fp_precision = np.copy(base_precision)
            fp_precision.flat[intersection] = 0
            fp_edges = np.argsort(
                np.abs(fp_precision.flat)
            )[::-1][: n_edges/2]

            # false negatives
            fn_precision = np.copy(masked_gt)
            fn_precision.flat[intersection] = 0
            fn_edges = np.argsort(
                np.abs(fn_precision.flat)
            )[::-1][: n_edges/2]

            edges = list(fp_edges) + list(fn_edges)

    assert len(edges) < len(precisions[0].flat)
    assert np.max(edges) < len(precisions[0].flat)
    assert np.min(edges) >= 0

    # reshape data a bit:
    # flatten each matrix into a column (so that coeffs are examples)
    # compute l1-norm of each column
    l1_norms = []
    coeffs = np.zeros((dim**2, len(precisions)))
    for ridx, result in enumerate(precisions):
        coeffs[edges, ridx] = result.flat[edges]
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
    r_input('Press any key to continue.')
