import numpy as np 

from sklearn.datasets import make_sparse_spd_matrix
from matplotlib import pyplot as plt
import seaborn

from . import QuicGraphLasso


plt.ion()
prng = np.random.RandomState(1)


def _new_graph(n_features, alpha):
    global prng
    prec = make_sparse_spd_matrix(n_features,
                                  alpha=alpha, # prob that a coeff is nonzero
                                  smallest_coef=0.4,
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


def _plot_spower(results, alphas):
    plt.figure()
    plt.plot(results, lw=2)
    plt.xlabel('n/p (n_samples/n_features)')
    plt.ylabel('P(exact recovery)')
    legend_text = []
    for alpha in alphas:
        legend_text.append('alpha={}'.format(alpha))
    plt.legend(legend_text)
    plt.show()


class GraphLassoSP(object):
    """Compute the statistical power P(exact support) of a model selector for
    different values of alpha over grid of n_samples/n_features.

    Once the model is chosen, we will run QuicGraphLasso with
    lam = self.penalty for multiple instances of a graph.

    Expects that estimator.precision_ is a matrix, already model selected.

    Note:  We want to run model selection once at 
    """
    def __init__(self, estimator=None, estimator_args={}, n_features=10, 
                n_trials=10, n_grid_points=10, verbose=False, penalty='lam_'):
        self.estimator = estimator 
        self.estimator_args = estimator_args
        self.n_features = n_features
        self.n_grid_points = n_grid_points
        self.n_trials = n_trials
        self.verbose = verbose
        self.penalty = penalty

        self.is_fitted = False
        self.results = None
        self.alphas = None

    def exact_support(self, prec, prec_hat):
        return np.array_equal(np.nonzero(prec.flat)[0], np.nonzero(prec_hat.flat)[0])
 
    def fit(self, X=None, y=None):
        # we'll have each row correspond to a different alpha
        self.results = np.zeros((self.n_grid_points, self.n_grid_points))

        grid = np.linspace(1, 10, self.n_grid_points)
        self.alphas = np.linspace(0.1, 1, self.n_grid_points)
        for aidx, alpha in enumerate(self.alphas):
            if self.verbose:
                print 'At alpha {} ({}/{})'.format(
                    alpha,
                    aidx,
                    self.n_grid_points,
                )

            # draw a new fixed graph for alpha
            cov, prec = _new_graph(self.n_features, alpha)

            for sidx, sample_grid in enumerate(grid):
                n_samples = int(sample_grid * self.n_features)
                
                if self.verbose:
                    print ' | ({}/{})'.format(sidx, self.n_grid_points)

                # do model selection (once)
                X = _new_sample(n_samples, self.n_features, cov)
                ms_estimator = self.estimator(**self.estimator_args)
                ms_estimator.fit(X)
                lam = getattr(ms_estimator, self.penalty)
                if self.verbose:
                    print 'Selected lambda = {}'.format(lam)

                for nn in range(self.n_trials):                    
                    # estimate example with lam=ms_estimator.penalty
                    X = _new_sample(n_samples, self.n_features, cov)
                    new_estimator = QuicGraphLasso(
                            lam=lam,
                            mode='default',
                            initialize_method='cov')
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

        _plot_spower(self.results, self.alphas)
        raw_input()

