from sklearn.utils.estimator_checks import check_estimator
from inverse_covariance import (
    QuicGraphLasso,
    QuicGraphLassoCV,
    QuicGraphLassoEBIC,
    AdaptiveGraphLasso,
    ModelAverage,
)


def test_quic_graph_lasso():
    return check_estimator(QuicGraphLasso)


def test_quic_graph_lasso_cv():
    return check_estimator(QuicGraphLassoCV)


def test_quic_graph_lasso_ebic():
    return check_estimator(QuicGraphLassoEBIC)


def test_adaptive_graph_lasso():
    return check_estimator(AdaptiveGraphLasso)


def test_model_average():
    return check_estimator(ModelAverage)
