import numpy as np 

from sklearn.base import clone
from sklearn.datasets import make_sparse_spd_matrix
from sklearn.externals.joblib import Parallel, delayed
from matplotlib import pyplot as plt
import seaborn

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


def _plot_spower(results, grid, ks):
    plt.figure()
    plt.plot(grid, results.T, lw=2)
    plt.xlabel('n/p (n_samples / n_features)')
    plt.ylabel('P(exact recovery)')
    legend_text = []
    for ks in ks:
        legend_text.append('sparsity={}'.format(ks))
    plt.legend(legend_text)
    plt.show()


def _exact_support(prec, prec_hat):
    # Q: why do we need something like this?, and why must eps be so big?
    # Q: can we automatically determine what this threshold should be?
    eps = 0.2
    #eps = np.finfo(prec_hat.dtype).eps # too small
    prec_hat[np.abs(prec_hat) <= eps] = 0.0
    
    return np.array_equal(
            np.nonzero(prec.flat)[0],
            np.nonzero(prec_hat.flat)[0])


def _sp_trial(trial_estimator, n_samples, n_features, cov, prec):
    X = _new_sample(n_samples, n_features, cov)
    new_estimator = clone(trial_estimator)
    new_estimator.fit(X)
    return _exact_support(prec, new_estimator.precision_)


class StatisticalPower(object):
    """Compute the statistical power P(exact support) of a model selector for
    different values of alpha over grid of n_samples / n_features.

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

    trial_estimator : An inverse covariance estimator instance (default=None)
        Estimator to use on each instance after selecting a penalty lambda.
        If None, this will use QuicGraphLasso with lambda obtained with 
        model_selection_estimator.
        Use .penalty to set selected penalty.

    penalty_ : string (default='lam_')
        Name of the selected best penalty in estimator
        e.g., 'lam_' for QuicGraphLassoCV, QuicGraphLassoEBIC,
              'alpha_' for GraphLassoCV

    penalty : string (default='lam')
        Name of the penalty kwarg in the estimator.  
        e.g., 'lam' for QuicGraphLasso, 'alpha' for GraphLasso

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

    results_ : matrix of size (n_alpha_grid_points, n_grid_points)
        The statisical power, P(exact support recovery) for each alpha and 
        n_samples / n_features grid point.
    """
    def __init__(self, model_selection_estimator=None, n_features=50, 
                trial_estimator=None, n_trials=100, n_grid_points=10,
                verbose=False, penalty_='lam_', penalty='lam', n_jobs=1):
        self.model_selection_estimator = model_selection_estimator  
        self.trial_estimator = trial_estimator
        self.n_features = n_features
        self.n_grid_points = n_grid_points
        self.n_trials = n_trials
        self.verbose = verbose
        self.penalty_ = penalty_ # class name for model selected penalty
        self.penalty = penalty # class name for setting penalty
        self.n_jobs = n_jobs

        self.is_fitted = False
        self.results_ = None
        self.alphas_ = None
        self.ks_ = None
        self.grid_ = None
 
    def fit(self, X=None, y=None):
        n_alpha_grid_points = 5

        self.results_ = np.zeros((n_alpha_grid_points, self.n_grid_points))
        self.grid_ = np.linspace(0.25, 4, self.n_grid_points)
        self.alphas_ = np.linspace(0.99, 0.999, n_alpha_grid_points)[::-1]
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
                if self.trial_estimator is None:
                    trial_estimator = QuicGraphLasso(lam=lam,
                                                     mode='default',
                                                     init_method='corrcoef')
                else:
                    trial_estimator = self.trial_estimator

                # patch trial estimator with this lambda
                trial_estimator.set_params(**{
                    self.penalty: lam, 
                })

                # estimate statistical power
                exact_support_counts = Parallel(
                    n_jobs=self.n_jobs,
                    verbose=False,
                    backend='threading',
                    #max_nbytes=None,
                    #batch_size=1,
                )(
                    delayed(_sp_trial)(
                        trial_estimator, n_samples, self.n_features, cov, prec
                    )
                    for nn in range(self.n_trials))

                self.results_[aidx, sidx] = 1. * np.sum(exact_support_counts) / self.n_trials

            if self.verbose:
                print 'Results at this row: {}'.format(self.results_[aidx, :])

        self.is_fitted = True
        return self

    def show(self):
        if not self.is_fitted:
            print 'Not fitted.'
            return

        _plot_spower(self.results_, self.grid_, self.ks_)

