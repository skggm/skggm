import warnings
import numpy as np
from scipy import sparse

from sklearn.preprocessing.data import scale, _handle_zeros_in_scale
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.utils import check_array
from sklearn.externals.six import string_types
from sklearn.utils.extmath import _incremental_mean_and_var
from sklearn.utils.validation import check_is_fitted, FLOAT_DTYPES


def two_way_standardize(
    X,
    axis=0,
    with_mean=True,
    with_std=True,
    copy=True,
    max_iter=50,
    tol=1e-6,
    verbose=False,
):
    """Standardize a two-dimensional data matrix along both axes.
    Center to the mean and component wise scale to unit variance.
    Read more in the :ref:`User Guide <preprocessing_scaler>`.

    Parameters
    ----------
    X : {array-like, sparse matrix}
        The data to center and scale.

    axis : int (0 by default)
        axis used to compute the means and standard deviations along. If 0,
        independently standardize each feature, otherwise (if 1) standardize
        each sample.

    with_mean : boolean, True by default
        Is always true for two-way standardize

    with_std : boolean, True by default
        If True, scale the data to unit variance (or equivalently,
        unit standard deviation).

    copy : boolean, optional, default True
        set to False to perform inplace row normalization and avoid a
        copy (if the input is already a numpy array or a scipy.sparse
        CSC matrix and if axis is 1).

    max_iter : int, optional (50 by default)
        Set the maximum number of iterations of successive normalization algorithm

    tol : float, optional (1e-6 by default)
        Set the convergence threshold for successive normalization

    Notes
    -----
    This function invokes sklearn's scale function, thus the same restrictions
    for scale, apply here as well.

    The caller should pass a CSC matrix. The caller is expected to either set
    explicitly `with_mean=False`(in that case, only variance scaling will be
    performed on the features of the CSC matrix) or to call `X.toarray()` if
    the array fits in memory.

    For a comparison of the different scalers, transformers, and normalizers,
    see sklearn documentation `examples/preprocessing/plot_all_scaling.py

    See also
    --------
    StandardScaler: Performs scaling to unit variance using the``Transformer`` API,
    e.g. as part of a preprocessing :class:`sklearn.pipeline.Pipeline`.
    """
    n_rows, n_cols = np.shape(X)

    row_polish = np.copy(X.T)
    rows_X = np.copy(X.T)
    cols_X = np.copy(X)

    err_norm = np.inf
    n_iter = 0

    while n_iter <= max_iter and err_norm > tol:
        col_polish = scale(row_polish.T, axis=1, with_mean=with_mean, with_std=with_std)
        row_polish = scale(col_polish.T, axis=1, with_mean=with_mean, with_std=with_std)

        err_row = np.linalg.norm(rows_X - row_polish, "fro")
        err_col = np.linalg.norm(cols_X - col_polish, "fro")
        err_norm = .5 * err_row / (n_rows * n_cols) + .5 * err_col / (n_rows * n_cols)

        n_iter += 1
        if verbose:
            print("Iteration: {}, Convergence Err: {}".format(n_iter, err_norm))

        rows_X = np.copy(row_polish)
        cols_X = np.copy(col_polish)

    return row_polish.T


class TwoWayStandardScaler(BaseEstimator, TransformerMixin):
    """Standardize features by removing the mean and scaling to unit variance
    in both row and column dimensions.

    This class is modeled after StandardScaler in scikit-learn.
    Read more in the :ref:`User Guide <preprocessing_scaler>`.

    Parameters
    ----------
    copy : boolean, optional, default True
        If False, try to avoid a copy and do inplace scaling instead.
        This is not guaranteed to always work inplace; e.g. if the data is
        not a NumPy array or scipy.sparse CSR matrix, a copy may still be
        returned.

    with_mean : boolean, True by default
        If True, center the data before scaling.
        This does not work (and will raise an exception) when attempted on
        sparse matrices, because centering them entails building a dense
        matrix which in common use cases is likely to be too large to fit in
        memory.

    with_std : boolean, True by default
        If True, scale the data to unit variance (or equivalently,
        unit standard deviation).

    Attributes
    ----------
    row_scale_ : ndarray, shape (n_examples,)
        Per feature relative scaling of the data.

    col_scale_ : ndarray, shape (n_features,)
        Per feature relative scaling of the data.

    row_mean_ : array of floats with shape [n_examples]
        The mean value for each feature in the training set.

    col_mean_ : array of floats with shape [n_features]
        The mean value for each feature in the training set.

    row_var_ : array of floats with shape [n_examples]
        The variance for each feature in the training set. Used to compute
        `scale_`

    col_var_ : array of floats with shape [n_features]
        The variance for each feature in the training set. Used to compute
        `scale_`

    n_samples_seen_ : int
        The number of samples processed by the estimator. Will be reset on
        new calls to fit, but increments across ``fit`` calls.

    See also
    --------
    :class:`sklearn.preprocessing.StandardScaler`
    :class:`sklearn.decomposition.PCA`

    Notes
    -----
    See the implications of one-way vs. two-way standardization in here. TBD
    """

    def __init__(self, copy=True, with_mean=True, with_std=True):
        """Unlike StandardScaler, with_mean is always set to True, to ensure
        that two-way standardization is always performed with centering. The
        argument `with_mean` is retained for the sake of sklearn
        API compatibility.
        """
        self.with_mean = with_mean
        self.with_std = with_std
        self.copy = copy

    def _reset(self):
        fit_attrs = [
            "n_rows_",
            "n_cols_",
            "col_var_",
            "col_mean_",
            "row_var_",
            "row_mean_",
            "row_scale_",
            "col_scale_",
            "n_rows_seen_",
            "n_cols_seen_",
        ]
        for attr in fit_attrs:
            if hasattr(self, attr):
                delattr(self, attr)

    def _initial_state(self):
        return not hasattr(self, "n_rows_seen_")

    def fit(self, X, y=None):
        """Compute the mean and std for both row and column dimensions.

        Parameters
        ----------
        X : {array-like}, shape [n_rows, n_cols]
            The data used to compute the mean and standard deviation
            along both row and column axes

        y : Passthrough for ``Pipeline`` compatibility. Input is ignored.
        """
        self._reset()
        return self.partial_fit(X, y)

    def partial_fit(self, X, y=None):
        """Online computation of mean and std on X for later scaling."""
        X = check_array(
            X,
            accept_sparse=False,
            copy=self.copy,
            warn_on_dtype=True,
            dtype=FLOAT_DTYPES,
            estimator=self,
            ensure_min_features=2,
        )
        if sparse.issparse(X):
            raise NotImplemented(
                "Input is sparse: Algorithm for sparse matrices currently not supported."
            )

        self.n_rows_, self.n_cols_ = np.shape(X)

        # Q: This doesnt seem to actually get used in the transform, only in
        #    the inverse transform which it sounds like we should not support.

        # initialize variables on first pass
        if self._initial_state():
            self.col_mean_ = 0.
            self.col_var_ = 0. if self.with_std else None
            self.n_cols_seen_ = 0
            self.row_mean_ = 0.
            self.row_var_ = 0. if self.with_std else None
            self.n_rows_seen_ = 0

        self.col_mean_, self.col_var_, self.n_rows_seen_ = _incremental_mean_and_var(
            X, self.col_mean_, self.col_var_, self.n_rows_seen_
        )
        self.row_mean_, self.row_var_, self.n_cols_seen_ = _incremental_mean_and_var(
            X.T, self.row_mean_, self.row_var_, self.n_cols_seen_
        )

        self.row_scale_ = None
        self.col_scale_ = None
        if self.with_std:
            self.row_scale_ = _handle_zeros_in_scale(np.sqrt(self.row_var_))
            self.col_scale_ = _handle_zeros_in_scale(np.sqrt(self.col_var_))

        return self

    def transform(self, X, copy=False):
        """Perform standardization by centering and scaling

        Parameters
        ----------
        X : array-like, shape [n_rows, n_cols]
            The data used to scale along the features axis.

        copy : bool, optional (default: False)
            Copy the input X or not.
        """
        check_is_fitted(self, "row_scale_")
        X = check_array(
            X,
            accept_sparse=False,
            copy=copy,
            warn_on_dtype=True,
            dtype=FLOAT_DTYPES,
            estimator=self,
        )
        n_rows, n_cols = np.shape(X)
        if self.n_cols_ != n_cols:
            raise ValueError("Number of features must be same as for fit().")

        if sparse.issparse(X):
            raise NotImplementedError(
                "Input is sparse: Algorithm for sparse matrices currently not supported."
            )

        return two_way_standardize(
            X, with_mean=self.with_mean, with_std=self.with_std, copy=self.copy
        )

    def inverse_transform(self, X, copy=False):
        """Scale back the data to the original representation

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to scale along the features axis.

        copy : bool, optional (default: False)
            Copy the input X or not.

        Returns
        -------
        X_tr : array-like, shape [n_samples, n_features]
            Transformed array.
        """
        if sparse.issparse(X):
            raise NotImplementedError(
                "Input is sparse: Algorithm for sparse matrices currently not supported."
            )

        raise NotImplementedError("Reversing two way transformation is not accurate.")
