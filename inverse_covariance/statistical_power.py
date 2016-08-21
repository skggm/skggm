import numpy as np 

from sklearn.datasets import make_sparse_spd_matrix
from matplotlib import pyplot as plt
#import seaborn

from . import QuicGraphLasso


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
    plt.xlabel('n/p (n_samples/n_features)')
    plt.ylabel('P(exact recovery)')
    legend_text = []
    for ks in ks:
        legend_text.append('sparsity={}'.format(ks))
    plt.legend(legend_text)
    plt.show()


class GraphLassoSP(object):
    """Compute the statistical power P(exact support) of a model selector for
    different values of alpha over grid of n_samples/n_features.

    Once the model is chosen, we will run QuicGraphLasso with
    lam = self.penalty for multiple instances of a graph.
    You can override the choice of the naive estimator (such as using the adaptive
    method with )

    Expects that estimator.precision_ is a matrix, already model selected.

    Note:  We want to run model selection once at 

    Note:  Look into what sklearn's clone feature does

    Note:  There is no lambda for model_average, so we need to think about how 
           we would do that in this framework.
    """
    def __init__(self, model_selection_estimator=None,
                model_selection_estimator_args={}, n_features=10, 
                n_trials=10, n_grid_points=10, verbose=False, penalty='lam_'):
        self.model_selection_estimator = model_selection_estimator  
        self.model_selection_estimator_args = model_selection_estimator_args
        self.n_features = n_features
        self.n_grid_points = n_grid_points
        self.n_trials = n_trials
        self.verbose = verbose
        self.penalty = penalty
        # new variable trial_estimator (e.g., QuicGraphLasso)

        self.is_fitted = False
        self.results = None
        self.alphas = None
        self.ks = None
        self.grid = None

    def exact_support(self, prec, prec_hat):
        # Q: why do we need something like this?, and why must eps be so big?
        # Q: can we automatically determine what this threshold should be?
        eps = 0.2
        prec_hat[np.abs(prec_hat) < eps] = 0.0
        
        return np.array_equal(
                np.nonzero(prec.flat)[0],
                np.nonzero(prec_hat.flat)[0])
 
    def fit(self, X=None, y=None):
        n_alpha_grid_points = self.n_grid_points / 2

        self.results = np.zeros((n_alpha_grid_points, self.n_grid_points))
        self.grid = np.linspace(0.25, 4, self.n_grid_points)
        self.alphas = np.linspace(0.99, 0.999, n_alpha_grid_points)[::-1]
        self.ks = []

        for aidx, alpha in enumerate(self.alphas):
            if self.verbose:
                print 'at alpha {} ({}/{})'.format(
                    alpha,
                    aidx,
                    n_alpha_grid_points,
                )

            # draw a new fixed graph for alpha
            cov, prec = _new_graph(self.n_features, alpha)
            n_nonzero_prec = np.count_nonzero(prec.flat)
            self.ks.append(n_nonzero_prec)
            print '   graph has {} nonzero entries'.format(n_nonzero_prec)

            for sidx, sample_grid in enumerate(self.grid):
                n_samples = int(sample_grid * self.n_features)
                
                # do model selection (once)
                X = _new_sample(n_samples, self.n_features, cov)
                ms_estimator = self.model_selection_estimator(
                        **self.model_selection_estimator_args)
                ms_estimator.fit(X)
                lam = getattr(ms_estimator, self.penalty)
                print ms_estimator.lam_scale_
                if self.verbose:
                    print '   ({}/{}), n_samples = {}, selected lambda = {}'.format(
                            sidx,
                            self.n_grid_points,
                            n_samples,
                            lam)

                for nn in range(self.n_trials):                    
                    # estimate example with lam=ms_estimator.penalty
                    X = _new_sample(n_samples, self.n_features, cov)
                    new_estimator = QuicGraphLasso(
                            lam=lam,
                            mode='default',
                            initialize_method='corrcoef') # maybe try corrcoef since 'cov' will modify lamba
                    new_estimator.fit(X)
                    
                    self.results[aidx, sidx] += self.exact_support(
                            prec,
                            new_estimator.precision_)

                    del new_estimator

                self.results[aidx, sidx] /= self.n_trials

            if self.verbose:
                print 'Results at this row: {}'.format(self.results[aidx, :])

        self.is_fitted = True
        return self

    def show(self):
        if not self.is_fitted:
            print 'Not fitted.'
            return

        _plot_spower(self.results, self.grid, self.ks)

