"""Compatibility helpers for the supported range of scikit-learn versions."""
from sklearn.utils import as_float_array as _sklearn_as_float_array


def as_float_array(X, copy=False, ensure_all_finite=False):
    """``sklearn.utils.as_float_array`` with a version-stable finiteness kwarg.

    scikit-learn renamed the ``force_all_finite`` argument to
    ``ensure_all_finite`` in 1.6 and removed the old spelling in 1.9.  Prefer
    the new name and fall back to the old one so skggm works across the
    supported range (>=1.5) without emitting deprecation warnings on newer
    releases or breaking on older ones.
    """
    try:
        return _sklearn_as_float_array(X, copy=copy, ensure_all_finite=ensure_all_finite)
    except TypeError:
        return _sklearn_as_float_array(X, copy=copy, force_all_finite=ensure_all_finite)
