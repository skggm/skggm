import numpy as np 

from sklearn.base import clone
from sklearn.datasets import make_sparse_spd_matrix
from sklearn.externals.joblib import Parallel, delayed
from matplotlib import pyplot as plt
#import seaborn

from .. import QuicGraphLasso


plt.ion()
prng = np.random.RandomState(1)



def _new_graph(n_features, alpha):
    global prng
    prec = make_sparse_spd_matrix(n_features,
                                  alpha=alpha, # prob that a coeff is zero
                                  smallest_coef=0.7,
                                  largest_coef=0.7,
                                  random_state=prng)
    cov = np.linalg.inv(prec)
    d = np.sqrt(np.diag(cov))
    cov /= d
    cov /= d[:, np.newaxis]
    prec *= d
    prec *= d[:, np.newaxis]
    return cov, prec


def _new_sample(n_samples, n_features, cov):
    X = prng.multivariate_normal(np.zeros(n_features), cov, size=n_samples)
    X -= X.mean(axis=0)
    X /= X.std(axis=0)
    return X


def _plot_avg_error(results, grid, ks):
    plt.figure()
    plt.plot(grid, results.T, lw=2)
    plt.xlabel('n/p (n_samples / n_features)')
    plt.ylabel('Average error')
    legend_text = []
    for ks in ks:
        legend_text.append('sparsity={}'.format(ks))
    plt.legend(legend_text)
    plt.show()


def _support_diff(m, m_hat):
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


def _ae_trial(trial_estimator, n_samples, n_features, cov, prec):
    X = _new_sample(n_samples, n_features, cov)
    new_estimator = clone(trial_estimator)
    new_estimator.fit(X)

    error_fro = np.linalg.norm(prec - new_estimator.precision_, ord='fro')
    error_supp = _support_diff(prec, new_estimator.precision_)
    error_fp, error_fn = _false_support(prec, new_estimator.precision_)

    return error_fro, error_supp, error_fp, error_fn


class AverageError(object):
    """Compute the average error of a model selector for
    different values of alpha over grid of n_samples / n_features.

    Use this to compare model selection methods.

    For each choice of alpha, we select a fixed test graph.
    For each choice of n_samples / n_features, we learn the model selection
    penalty just once and apply this learned value to each subsequent random
    trial (new instances with the fixed covariance).

    Parameters
    -----------        
    model_selection_estimator : An inverse covariance estimator instance 
        This estimator must be able to select a penalization parameter. 
        Use .penalty_ to obtain selected penalty.

    n_features : int (default=50)
        Fixed number of features to test.

    n_trials : int (default=100)
        Number of examples to draw to measure P(recovery).

    penalty_ : string (default='lam_')
        Name of the selected best penalty in estimator
        e.g., 'lam_' for QuicGraphLassoCV, QuicGraphLassoEBIC,
              'alpha_' for GraphLassoCV

    n_grid_points : int (default=10)
        Number of grid points for sampling n_samples / n_features between (0,1)

    verbose : bool (default=False)
        Print out progress information.

    n_jobs: int, optional
        number of jobs to run in parallel (default 1).

    Methods
    ----------
    show() : Plot the results.

    Attributes
    ----------
    grid_ : array of size (n_grid_points, )
        Array of n_samples / n_features ratios.

    alphas_ : array of size (n_alpha_grid_points, )
        Array of alphas used to generate test graphs 
        (see .statistical_power._new_graph)

    ks_ : array of size (n_alpha_grid_points, )
        The sparsity of each test graph.

    error_fro_ : matrix of size (n_alpha_grid_points, n_grid_points)
        The average Frobenius error for each alpha and 
        n_samples / n_features grid point.

    error_supp_ : matrix of size (n_alpha_grid_points, n_grid_points)
        The average support difference for each alpha and 
        n_samples / n_features grid point.

    error_fp_ : matrix of size (n_alpha_grid_points, n_grid_points)
        The average false positive difference for each alpha and 
        n_samples / n_features grid point.

    error_fn_ : matrix of size (n_alpha_grid_points, n_grid_points)
        The average false negative difference for each alpha and 
        n_samples / n_features grid point.
    """
    def __init__(self, model_selection_estimator=None, n_features=50, 
                n_trials=100, n_grid_points=10, verbose=False, penalty_='lam_',
                n_jobs=1):
        self.model_selection_estimator = model_selection_estimator  
        self.n_features = n_features
        self.n_grid_points = n_grid_points
        self.n_trials = n_trials
        self.verbose = verbose
        self.penalty_ = penalty_ # class name for model selected penalty
        self.n_jobs = n_jobs

        self.is_fitted = False
        self.error_fro_ = None
        self.error_supp_ = None
        self.error_fp_ = None
        self.error_fn_ = None
        self.alphas_ = None
        self.ks_ = None
        self.grid_ = None
 
    def fit(self, X=None, y=None):
        n_alpha_grid_points = 3

        self.error_fro_ = np.zeros((n_alpha_grid_points, self.n_grid_points))
        self.error_supp_ = np.zeros((n_alpha_grid_points, self.n_grid_points))
        self.error_fp_ = np.zeros((n_alpha_grid_points, self.n_grid_points))
        self.error_fn_ = np.zeros((n_alpha_grid_points, self.n_grid_points))

        self.grid_ = np.linspace(1, 4, self.n_grid_points)
        self.alphas_ = np.linspace(0.95, 0.99, n_alpha_grid_points)[::-1]
        self.ks_ = []

        for aidx, alpha in enumerate(self.alphas_):
            if self.verbose:
                print 'at alpha {} ({}/{})'.format(
                    alpha,
                    aidx,
                    n_alpha_grid_points,
                )

            # draw a new fixed graph for alpha
            cov, prec = _new_graph(self.n_features, alpha)
            n_nonzero_prec = np.count_nonzero(prec.flat)
            self.ks_.append(n_nonzero_prec)
            if self.verbose:
                print '   Graph has {} nonzero entries'.format(n_nonzero_prec)

            for sidx, sample_grid in enumerate(self.grid_):
                n_samples = int(sample_grid * self.n_features)
                
                # model selection (once)
                X = _new_sample(n_samples, self.n_features, cov)
                ms_estimator = clone(self.model_selection_estimator)
                ms_estimator.fit(X)
                lam = getattr(ms_estimator, self.penalty_)
                
                if self.verbose:
                    display_lam = lam
                    if isinstance(lam, np.ndarray):
                        display_lam = np.linalg.norm(lam)
                    print '   ({}/{}), n_samples = {}, selected lambda = {}'.format(
                            sidx,
                            self.n_grid_points,
                            n_samples,
                            display_lam)

                # setup default trial estimator
                trial_estimator = QuicGraphLasso(lam=lam,
                                                 mode='default',
                                                 init_method='corrcoef')

                # estimate statistical power
                errors = Parallel(
                    n_jobs=self.n_jobs,
                    verbose=False,
                    backend='threading',
                    #max_nbytes=None,
                    #batch_size=1,
                )(
                    delayed(_ae_trial)(
                        trial_estimator, n_samples, self.n_features, cov, prec
                    )
                    for nn in range(self.n_trials))

                error_fro, error_supp, error_fp, error_fn = zip(*errors)
                self.error_fro_[aidx, sidx] = np.mean(error_fro)
                self.error_supp_[aidx, sidx] = np.mean(error_supp)
                self.error_fp_[aidx, sidx] = np.mean(error_fp)
                self.error_fn_[aidx, sidx] = np.mean(error_fn)

            if self.verbose:
                print 'Results at this row:'
                print '   fro = {}'.format(self.error_fro_[aidx, :])
                print '   supp = {}'.format(self.error_supp_[aidx, :])
                print '   fp = {}'.format(self.error_fp_[aidx, :])
                print '   fn = {}'.format(self.error_fn_[aidx, :])

        self.is_fitted = True
        return self

    def show(self):
        if not self.is_fitted:
            print 'Not fitted.'
            return

        errors_to_plot = [
            ('Frobenius', self.error_fro_),
            ('Support', self.error_supp_),
            ('False Positive', self.error_fp_),
            ('False Negative', self.error_fn_),
        ]
        for name, result in errors_to_plot:
            _plot_avg_error(result, self.grid_, self.ks_)
            plt.title(name)

