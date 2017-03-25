import numpy as np


def _nonzero_intersection(m, m_hat):
    '''Count the number of nonzeros in and between m and m_hat.

    Returns
    ----------
    m_nnz :  number of nonzeros in m (w/o diagonal)

    m_hat_nnz : number of nonzeros in m_hat (w/o diagonal)

    intersection_nnz : number of nonzeros in intersection of m/m_hat
                      (w/o diagonal)
    '''
    n_features, _ = m.shape

    m_no_diag = m.copy()
    m_no_diag[np.diag_indices(n_features)] = 0
    m_hat_no_diag = m_hat.copy()
    m_hat_no_diag[np.diag_indices(n_features)] = 0

    m_hat_nnz = len(np.nonzero(m_hat_no_diag.flat)[0])
    m_nnz = len(np.nonzero(m_no_diag.flat)[0])

    intersection_nnz = len(np.intersect1d(
        np.nonzero(m_no_diag.flat)[0],
        np.nonzero(m_hat_no_diag.flat)[0]
    ))

    return m_nnz, m_hat_nnz, intersection_nnz


def support_false_positive_count(m, m_hat):
    '''Count the number of false positive support elements in
    m_hat in one triangle, not including the diagonal.
    '''
    m_nnz, m_hat_nnz, intersection_nnz = _nonzero_intersection(m, m_hat)
    return int((m_hat_nnz - intersection_nnz) / 2.0)


def support_false_negative_count(m, m_hat):
    '''Count the number of false negative support elements in
    m_hat in one triangle, not including the diagonal.
    '''
    m_nnz, m_hat_nnz, intersection_nnz = _nonzero_intersection(m, m_hat)
    return int((m_nnz - intersection_nnz) / 2.0)


def support_difference_count(m, m_hat):
    '''Count the number of different elements in the support in one triangle,
    not including the diagonal.
    '''
    m_nnz, m_hat_nnz, intersection_nnz = _nonzero_intersection(m, m_hat)
    return int((m_nnz + m_hat_nnz - (2 * intersection_nnz)) / 2.0)


def has_exact_support(m, m_hat):
    '''Returns 1 if support_difference_count is zero, 0 else.
    '''
    m_nnz, m_hat_nnz, intersection_nnz = _nonzero_intersection(m, m_hat)
    return int((m_nnz + m_hat_nnz - (2 * intersection_nnz)) == 0)


def has_approx_support(m, m_hat, prob=0.01):
    """Returns 1 if model selection error is less than or equal to prob rate,
    0 else.

    NOTE: why does np.nonzero/np.flatnonzero create so much problems?
    """
    m_nz = np.flatnonzero(np.triu(m, 1))
    m_hat_nz = np.flatnonzero(np.triu(m_hat, 1))

    upper_diagonal_mask = np.flatnonzero(np.triu(np.ones(m.shape), 1))
    not_m_nz = np.setdiff1d(upper_diagonal_mask, m_nz)

    intersection = np.in1d(m_hat_nz, m_nz)  # true positives
    not_intersection = np.in1d(m_hat_nz, not_m_nz)  # false positives

    true_positive_rate = 0.0
    if len(m_nz):
        true_positive_rate = 1. * np.sum(intersection) / len(m_nz)
        true_negative_rate = 1. - true_positive_rate

    false_positive_rate = 0.0
    if len(not_m_nz):
        false_positive_rate = 1. * np.sum(not_intersection) / len(not_m_nz)

    return int(np.less_equal(true_negative_rate + false_positive_rate, prob))


def error_fro(m, m_hat):
    return np.linalg.norm(np.triu(m - m_hat, 1), ord='fro')
