def exact_support(prec, prec_hat):
    # Q: why do we need something like this?, and why must eps be so big?
    # Q: can we automatically determine what this threshold should be?    
    eps = .2
    #eps = np.finfo(prec_hat.dtype).eps # too small
    prec_hat[np.abs(prec_hat) <= eps] = 0.0
    
    not_empty = np.count_nonzero(np.triu(prec,1))>0
    
    return not_empty & np.array_equal(
            np.nonzero(np.triu(prec,1).flat)[0],
            np.nonzero(np.triu(prec_hat,1).flat)[0])

'''
def count_support_diff(m, m_hat):
    
    n_features, _ = m.shape

    m_no_diag = m.copy()
    m_no_diag[np.diag_indices(n_features)] = 0
    m_hat_no_diag = m_hat.copy()
    m_hat_no_diag[np.diag_indices(n_features)] = 0

    m_nnz = len(np.nonzero(m_no_diag.flat)[0])
    m_hat_nnz = len(np.nonzero(m_hat_no_diag.flat)[0])

    nnz_intersect = len(np.intersect1d(np.nonzero(m_no_diag.flat)[0],
                                       np.nonzero(m_hat_no_diag.flat)[0]))
    return m_nnz + m_hat_nnz - (2 * nnz_intersect)
'''

def support_diff(m, m_hat):
    '''Count the number of different elements in the support in one triangle,
    not including the diagonal. 
    '''
    n_features, _ = m.shape

    m_no_diag = m.copy()
    m_no_diag[np.diag_indices(n_features)] = 0
    m_hat_no_diag = m_hat.copy()
    m_hat_no_diag[np.diag_indices(n_features)] = 0

    m_nnz = len(np.nonzero(m_no_diag.flat)[0])
    m_hat_nnz = len(np.nonzero(m_hat_no_diag.flat)[0])

    nnz_intersect = len(np.intersect1d(np.nonzero(m_no_diag.flat)[0],
                                       np.nonzero(m_hat_no_diag.flat)[0]))
    return (m_nnz + m_hat_nnz - (2 * nnz_intersect)) / 2.0


def approx_support(prec, prec_hat, prob=.01):
    """
    Returns True if model selection error is less than or equal to prob%
    """        
    #eps = .2
    #prec_hat[np.abs(prec_hat) <= eps] = 0.0

    # Why does np.nonzero/np.flatnonzero create so much problems? 
    A = np.flatnonzero(np.triu(prec,1))
    B = np.flatnonzero(np.triu(prec_hat,1))
    ud = np.flatnonzero(np.triu(np.ones(prec.shape),1))
    notA = np.setdiff1d(ud,A)

    B_in_A_bool = np.in1d(B,A) # true positives
    B_notin_A_bool = np.in1d(B,notA) # false positives
    #print np.sum(B_in_A_bool), np.shape(A)[0]
    #print np.sum(B_notin_A_bool), np.shape(notA)[0]
    
    if np.shape(A)[0]:
        tpr = float(np.sum(B_in_A_bool))/len(A)
        tnr = 1.0-tpr
    else:
        tpr = 0.0
        tnr = 0.0        
    if np.shape(notA)[0]:
        fpr = float(np.sum(B_notin_A_bool))/len(notA)
    else:
        fpr = 0.0
        
    #print tnr,fpr
    
    return np.less_equal(tnr+fpr,prob), tpr, fpr


def _false_support(m, m_hat):
    '''Count the number of false positive and false negatives supports in 
    m_hat in one triangle, not including the diagonal.
    '''
    n_features, _ = m.shape

    m_no_diag = m.copy()
    m_no_diag[np.diag_indices(n_features)] = 0
    m_hat_no_diag = m_hat.copy()
    m_hat_no_diag[np.diag_indices(n_features)] = 0

    m_nnz = len(np.nonzero(m_no_diag.flat)[0])
    m_hat_nnz = len(np.nonzero(m_hat_no_diag.flat)[0])

    nnz_intersect = len(np.intersect1d(np.nonzero(m_no_diag.flat)[0],
                                       np.nonzero(m_hat_no_diag.flat)[0]))

    false_positives = (m_hat_nnz - nnz_intersect) / 2.0
    false_negatives = (m_nnz - nnz_intersect) / 2.0
    return false_positives, false_negatives