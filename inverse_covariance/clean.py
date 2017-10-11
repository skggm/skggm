import numpy as np
from scipy import sparse
from scipy import stats

from sklearn.preprocessing.data import (
    scale,
    _handle_zeros_in_scale
    )
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.utils import check_array
from sklearn.externals import six
from sklearn.externals.six import string_types
from sklearn.utils.extmath import row_norms
from sklearn.utils.extmath import _incremental_mean_and_var
from sklearn.utils.sparsefuncs_fast import (inplace_csr_row_normalize_l1,
                                      inplace_csr_row_normalize_l2)
from sklearn.utils.sparsefuncs import (inplace_column_scale,
                                 mean_variance_axis, incr_mean_variance_axis,
                                 min_max_axis)
from sklearn.utils.validation import (check_is_fitted, check_random_state,
                                FLOAT_DTYPES)


def twoway_standardize(X, axis=0, with_mean=True, with_std=True, copy=True, 
                       max_iter=50, tol=1e-6, verbose=False):
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
    This function invokes sklearn's scale function. Thus, the same restrictions
    for scale, apply here as well.
    This implementation will refuse to center scipy.sparse matrices
    since it would make them non-sparse and would potentially crash the
    program with memory exhaustion problems.
    Instead the caller is expected to either set explicitly
    `with_mean=False` (in that case, only variance scaling will be
    performed on the features of the CSC matrix) or to call `X.toarray()`
    if he/she expects the materialized dense array to fit in memory.
    To avoid memory copy the caller should pass a CSC matrix.
    For a comparison of the different scalers, transformers, and normalizers,
    see sklearn documentation `examples/preprocessing/plot_all_scaling.py
    See also
    --------
    StandardScaler: Performs scaling to unit variance using the``Transformer`` API
        (e.g. as part of a preprocessing :class:`sklearn.pipeline.Pipeline`).
    """  # noqa

    X = check_array(X, accept_sparse=None, copy=copy, warn_on_dtype=True,
                        dtype=FLOAT_DTYPES)
    Xrow_polish = np.copy(X.T)
    Xcol_polish = np.copy(X)
    [n_rows,n_cols] = np.shape(X)

    if sparse.issparse(X):
        print('Input is sparse')
        raise NotImplemented(
                "Algorithm for sparse matrices currently not supported.")

    else:
        n_iter = 0
        err_norm = np.inf
        oldXrow = np.copy(Xrow_polish)
        oldXcol = np.copy(Xcol_polish)
        while n_iter <= max_iter and err_norm > tol :
            Xcol_polish = scale(Xrow_polish.T, axis=1,
                                    with_mean=True,
                                    with_std=with_std
                                   )
            Xrow_polish = scale(Xcol_polish.T, axis=1,
                                    with_mean=True,
                                    with_std=with_std
                                   )
            n_iter += 1
            err_norm_row = np.linalg.norm(oldXrow-Xrow_polish,'fro')
            err_norm_col = np.linalg.norm(oldXcol-Xcol_polish,'fro')
            err_norm = .5 * err_norm_row/(n_rows*n_cols) + .5 * err_norm_col/(n_rows*n_cols)
            if verbose:
                print('Iteration: {}, Convergence Err: {}'.format(n_iter,err_norm))
            oldXrow = np.copy(Xrow_polish)
            oldXcol = np.copy(Xcol_polish)

        X = Xrow_polish
    return X


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
    scale_ : ndarray, shape (n_features,)
        Per feature relative scaling of the data.
        .. versionadded:: 0.17
           *scale_*
    mean_ : array of floats with shape [n_features]
        The mean value for each feature in the training set.
    var_ : array of floats with shape [n_features]
        The variance for each feature in the training set. Used to compute
        `scale_`
    n_samples_seen_ : int
        The number of samples processed by the estimator. Will be reset on
        new calls to fit, but increments across ``partial_fit`` calls.
    Examples
    --------
    >>> from inverse_covariance.clean import TwoWayStandardScaler
    >>>
    >>> data = [[1, 0], [1, 0], [2, 1], [2, 1]]
    >>> scaler = StandardScaler()
    >>> print(scaler.fit(data))
    StandardScaler(copy=True, with_mean=True, with_std=True)
    >>> print(scaler.mean_)
    [ 3.0  0.5]
    >>> print(scaler.transform(data))
    [[-1. -1.]
     [-1. -1.]
     [ 1.  1.]
     [ 1.  1.]]
    See also
    --------
    twoway_standardize: Equivalent function without the estimator API.
    :class:`sklearn.preprocessing.StandardScaler`
    :class:`sklearn.decomposition.PCA`
        Further removes the linear correlation across features with 'whiten=True'.
    Notes
    -----
    See the implications of one-way vs. two-way standardization in here. TBD
    
    """  # noqa

    def __init__(self, copy=True, with_mean=True, with_std=True):
        """Unlike StandardScaler, with_mean is always set to True, to ensure
        that two-way standardization is always performed with centering. The
        argument `with_mean` is retained for the sake of model API compatibility.
        """
        self.with_mean = True
        self.with_std = with_std
        self.copy = copy

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.
        __init__ parameters are not touched.
        """

        # Checking one attribute is enough, becase they are all set together
        # in partial_fit
        if hasattr(self, 'col_scale_'):
            del self.row_scale_
            del self.row_mean_
            del self.row_var_
            del self.col_scale_
            del self.col_mean_
            del self.col_var_
            del self.n_rows_seen_
            del self.n_cols_seen_

    def fit(self, X, y=None):
        """Compute the mean and std for both row and column dimensions.
        Parameters
        ----------
        X : {array-like}, shape [n_rows, n_cols]
            The data used to compute the mean and standard deviation
            along both row and column axes
        y : Passthrough for ``Pipeline`` compatibility. Input is ignored.
        """

        return self.partial_fit(X, y)

    def partial_fit(self, X, y=None):
        """Compute the mean and std for both row and column dimensions.
        Equivalent to fit. Online algorithm not supported at this time.
        Parameters
        ----------
        X : {array-like}, shape [n_rows, n_cols]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        y : Passthrough for ``Pipeline`` compatibility.
        """
        X = check_array(X, accept_sparse=None, copy=self.copy,
                        warn_on_dtype=True, dtype=FLOAT_DTYPES)

        if sparse.issparse(X):
            print('Input is sparse')
            raise NotImplemented(
                "Algorithm for sparse matrices currently not supported.")
        else:
            # First pass
            if not hasattr(self, 'n_rows_seen_'):
                self.col_mean_ = .0
                self.n_rows_seen_ = 0
                if self.with_std:
                    self.col_var_ = .0
                else:
                    self.col_var_ = None

            self.col_mean_, self.col_var_, self.n_rows_seen_ = \
                            _incremental_mean_and_var(X, self.col_mean_, self.col_var_,
                                                      self.n_rows_seen_)
            if not hasattr(self, 'n_cols_seen_'):
                self.row_mean_ = .0
                self.n_cols_seen_ = 0
                if self.with_std:
                    self.row_var_ = .0
                else:
                    self.row_var_ = None
            self.row_mean_, self.row_var_, self.n_cols_seen_ = \
                                        _incremental_mean_and_var(X.T, self.row_mean_, self.row_var_,
                                                                  self.n_cols_seen_)

        if self.with_std:
            self.row_scale_ = _handle_zeros_in_scale(np.sqrt(self.row_var_))
            self.col_scale_ = _handle_zeros_in_scale(np.sqrt(self.col_var_))
        else:
            self.row_scale_ = None
            self.col_scale_ = None

        return self

    def transform(self, X, y='deprecated', copy=None):
        """Perform standardization by centering and scaling
        Parameters
        ----------
        X : array-like, shape [n_rows, n_cols]
            The data used to scale along the features axis.
        y : (ignored)
            .. deprecated:: 0.19
               This parameter will be removed in 0.21.
        copy : bool, optional (default: None)
            Copy the input X or not.
        """
        if not isinstance(y, string_types) or y != 'deprecated':
            warnings.warn("The parameter y on transform() is "
                          "deprecated since 0.19 and will be removed in 0.21",
                          DeprecationWarning)

        check_is_fitted(self, 'row_scale_')

        copy = copy if copy is not None else self.copy
        X = check_array(X, accept_sparse=None, copy=copy, warn_on_dtype=True,
                        estimator=self, dtype=FLOAT_DTYPES)

        if sparse.issparse(X):
            print('Input is sparse')
            raise NotImplemented(
                "Algorithm for sparse matrices currently not supported.")
        else:
            X = twoway_standardize(X)
        return X

    def inverse_transform(self, X, copy=None):
        """Scale back the data to the original representation
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to scale along the features axis.
        copy : bool, optional (default: None)
            Copy the input X or not.
        Returns
        -------
        X_tr : array-like, shape [n_samples, n_features]
            Transformed array.
        """
        check_is_fitted(self, 'scale_')

        copy = copy if copy is not None else self.copy
        if sparse.issparse(X):
            if self.with_mean:
                raise ValueError(
                    "Cannot uncenter sparse matrices: pass `with_mean=False` "
                    "instead See docstring for motivation and alternatives.")
            if not sparse.isspmatrix_csr(X):
                X = X.tocsr()
                copy = False
            if copy:
                X = X.copy()
            if self.scale_ is not None:
                inplace_column_scale(X, self.scale_)
        else:
            X = np.asarray(X)
            if copy:
                X = X.copy()
            if self.with_std:
                X *= self.scale_
            if self.with_mean:
                X += self.mean_
        return X