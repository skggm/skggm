from sklearn.utils.estimator_checks import check_estimator
from inverse_covariance import (
    QuicGraphicalLasso,
    QuicGraphicalLassoCV,
    QuicGraphicalLassoEBIC,
    AdaptiveGraphicalLasso,
    ModelAverage,
)


def test_quic_graphical_lasso():
    return check_estimator(QuicGraphicalLasso)


def test_quic_graphical_lasso_cv():
    return check_estimator(QuicGraphicalLassoCV)


def test_quic_graphical_lasso_ebic():
    return check_estimator(QuicGraphicalLassoEBIC)


def test_adaptive_graphical_lasso():
    return check_estimator(AdaptiveGraphicalLasso)


def test_model_average():
    return check_estimator(ModelAverage)
