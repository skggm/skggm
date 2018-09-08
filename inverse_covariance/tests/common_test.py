from sklearn.utils.estimator_checks import check_estimator

from inverse_covariance import QuicGraphLasso, QuicGraphLassoCV, QuicGraphLassoEBIC


def test_quic_graph_lasso():
    return check_estimator(QuicGraphLasso)


def test_quic_graph_lasso_cv():
    return check_estimator(QuicGraphLassoCV)


def test_quic_graph_lasso_ebic():
    return check_estimator(QuicGraphLassoEBIC)
