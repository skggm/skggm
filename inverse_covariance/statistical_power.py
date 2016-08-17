import numpy as np 

from sklearn.datasets import make_sparse_spd_matrix
from matplotlib import pyplot as plt
import seaborn

plt.ion()
prng = np.random.RandomState(1)

def _new_sample(n_samples, n_features, alpha):
    global prng
    prec = make_sparse_spd_matrix(n_features,
                                  alpha=alpha, # prob that a coeff is nonzero
                                  smallest_coef=0.1,
                                  largest_coef=0.9)
    cov = np.linalg.inv(prec)
    d = np.sqrt(np.diag(cov))
    cov /= d
    cov /= d[:, np.newaxis]
    prec *= d
    prec *= d[:, np.newaxis]
    X = prng.multivariate_normal(np.zeros(n_features), cov, size=n_samples)
    X -= X.mean(axis=0)
    X /= X.std(axis=0)

    return X, cov, prec


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


class StatisticalPower(object):
    """

    Expects that estimator.precision_ is a matrix, already model selected.
    """
    def __init__(self, estimator=None, estimator_args={}, n_features=30, 
                n_trials=10, n_grid_points=10, metric='support', verbose=False):
        self.estimator = estimator 
        self.estimator_args = estimator_args
        self.n_features = n_features
        self.n_grid_points = n_grid_points
        self.metric = metric
        self.n_trials = n_trials
        self.verbose = verbose

        self.is_fitted = False
        self.results = None
        self.alphas = None

    def exact_support(self, prec, prec_hat):
        return np.nonzero(prec) == np.nonzero(prec_hat)

    def fit(self, X=None, y=None):
        # we'll have each row correspond to a different alpha
        self.results = np.zeros((self.n_grid_points, self.n_grid_points))

        grid = np.linspace(0.1, 1, self.n_grid_points)
        self.alphas = np.linspace(0, 1, self.n_grid_points)
        for aidx, alpha in enumerate(self.alphas):
            if self.verbose:
                print 'At alpha {} ({}/{})'.format(
                    alpha,
                    aidx,
                    self.n_grid_points,
                )
            for sidx, sample_grid in enumerate(grid):
                n_samples = int(sample_grid * self.n_features)
                
                if self.verbose:
                    print ' | ({}/{})'.format(sidx, self.n_grid_points)

                for nn in range(self.n_trials):
                    X, cov, prec = _new_sample(n_samples, self.n_features, alpha)
                    new_estimator = self.estimator(**self.estimator_args)
                    new_estimator.fit(X)

                    if self.metric == 'support':
                        self.results[aidx, sidx] += self.exact_support(
                                new_estimator.precision_,
                                prec)
                    else:
                        raise NotImplementedError(
                            "Only metric='support' has been implemented.") 

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

