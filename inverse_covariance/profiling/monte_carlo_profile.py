from __future__ import absolute_import

import numpy as np 
from sklearn.base import clone
from sklearn.datasets import make_sparse_spd_matrix
from sklearn.externals.joblib import Parallel, delayed

from .metrics import error_fro
from .erdos_renyi_graph import ErdosRenyiGraph
from .. import QuicGraphLasso, QuicGraphLassoCV


def _sample_mvn(n_samples, cov, prng):
    '''Draw a multivariate normal sample from the graph defined by cov.
    
    Parameters
    ----------- 
    n_samples : int

    cov : matrix of shape (n_features, n_features) 
        Covariance matrix of the graph.

    prng : np.random.RandomState instance.
    '''
    n_features, _ = cov.shape
    return prng.multivariate_normal(np.zeros(n_features), cov, size=n_samples)


def _mc_trial(estimator, metrics, n_samples, cov, prec, prng):
    '''Draw a new multivariate normal sample via cov and prng and use cloned
    estimator to estimate inverse covariance.  Compute estimate performance.
    Returns dict of performance.
    
    Used as a helper function for Parallel/delayed.
    '''
    X = _sample_mvn(n_samples, cov, prng)
    new_estimator = clone(estimator)
    new_estimator.fit(X)
    return {k: f(prec, new_estimator.precision_) for k,f in metrics.items()}


class MonteCarloProfile(object):
    """Compute performance metrics over multiple random trials (multivariate 
    normal sample instances for a given graph). 

    The graph is sampled once to obtain a penalty (via the ms_estimator) and 
    then sampled and fit for n_trials for each (alpha, n_samples) parameter 
    pair.

    Parameters
    -----------        
    n_features : int (default=50)
        Fixed number of features to test.

    n_trials : int (default=100)
        Number of examples to draw to measure P(recovery).

    ms_estimator : "Model selection estimator" (default=None)
        An inverse covariance estimator instance. This estimator must be able to
        select a penalization parameter that can be accessed via the instance 
        variable .lam_.  

    mc_estimator : "Monte Carlo trial estimator" (default=None)
        An inverse covariance estimator instance. Estimator to use on each 
        instance after selecting a penalty lambda. The penalty parameter 'lam' 
        will be overriden by with penalty selected by ms_estimator.
        If None, this will use QuicGraphLasso.

    graph :  An instance of a class with the method `.create(n_features, alpha)`
        that returns (cov, prec, adj).
        graph.create() will be used to draw a new graph instance in each trial.
        default: ErdosRenyiGraph()

    n_samples_grid : int (default=10) or array of floats
        Grid points for choosing number of samples.
        If integer, defines a linear grid (5, 200) 
        Else uses array as grid.

    alpha_grid : int (default=5) or array of floats
        Grid points used in making new graphs via graph.create().
        If integer, defines a logarithmic grid (0.14, 0.4)
        Else, uses array as grid.

    metrics : dict of functions: scalar = func(prec, prec_estimate)
        The key for each function will be used for reporting results.
        default: {'frobenius': error_fro}

    verbose : bool (default=False)
        Print out progress information.

    n_jobs: int, optional
        number of jobs to run in parallel (default 1).

    seed : np.random.RandomState seed. (default=2)


    Attributes
    ----------
    grid_ : array of size (n_samples_grid, ) or n_samples_grid
        Array of n_samples / n_features ratios.

    alphas_ : array of size (alpha_grid, ) or alpha_grid
        Array of alphas used to generate graphs.

    precision_nnz_ : array of size (len(alphas_), )
        The sparsity of each test graph precision.

    results_ : dict of matrices of size (len(alphas_), len(grid_))
        Each key corresponds to a function from metrics.
    """
    def __init__(self, n_features=50, n_trials=100, ms_estimator=None,
                 mc_estimator=None, graph=None, n_samples_grid=10, alpha_grid=5,
                 metrics={'frobenius': error_fro}, verbose=False,  n_jobs=1, 
                 seed=2):
        self.n_features = n_features
        self.n_trials = n_trials
        self.ms_estimator = ms_estimator  
        self.mc_estimator = mc_estimator
        self.graph = graph
        self.n_samples_grid = n_samples_grid
        self.alpha_grid = alpha_grid
        self.metrics = metrics
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.prng = np.random.RandomState(seed)

        if self.graph is None:
            self.graph = ErdosRenyiGraph()

        if self.ms_estimator is None:
            self.ms_estimator = QuicGraphLassoCV()

        if self.mc_estimator is None:
            self.mc_estimator = QuicGraphLasso(lam=0.5,
                                               mode='default',
                                               init_method='corrcoef')

        if isinstance(self.n_samples_grid, int):
            self.grid_ = np.linspace(5, 200, self.n_samples_grid)
        else:
            self.grid_ = self.n_samples_grid
        
        if isinstance(self.alpha_grid, int):
            self.alphas_ = np.logspace(np.log(0.15),np.log10(0.4), self.alpha_grid)
        else:
            self.alphas_ = self.alpha_grid

        self.is_fitted = False
        self.results_ = None
        self.precision_nnz_ = None
 
    def fit(self, X=None, y=None):
        n_alphas = len(self.alphas_)
        n_grids = len(self.grid_)

        self.precision_nnz_ = []
        self.results_ = {k: np.zeros((n_alphas, n_grids)) for k in self.metrics}

        for aidx, alpha in enumerate(self.alphas_):
            if self.verbose:
                print 'alpha {} ({}/{})'.format(
                    alpha,
                    aidx,
                    n_alphas,
                )

            # draw a new fixed graph for alpha
            cov, prec, adj = self.graph.create(self.n_features, alpha)

            # track nnz of graph precision
            precision_nnz = np.count_nonzero(prec.flat)
            self.precision_nnz_.append(precision_nnz)
            if self.verbose:
                print '   Graph has {} nonzero entries'.format(precision_nnz)

            for sidx, grid_point in enumerate(self.grid_):
                n_samples = int(grid_point * self.n_features)
                
                # model selection (once per n_samples grid point)
                X = _sample_mvn(n_samples, cov, self.prng)
                ms_estimator = clone(self.ms_estimator)
                ms_estimator.fit(X)
                
                if self.verbose:
                    display_lam = ms_estimator.lam_
                    if isinstance(display_lam, np.ndarray):
                        display_lam = np.linalg.norm(display_lam)
                    print '   ({}/{}), n_samples = {}, selected lambda = {}'.format(
                            sidx,
                            n_grids,
                            n_samples,
                            display_lam)

                # patch trial estimator with this lambda
                self.mc_estimator.set_params(**{'lam': ms_estimator.lam_})

                # run estimator on n_trials instances
                trials = Parallel(
                    n_jobs=self.n_jobs,
                    verbose=False,
                    backend='threading',
                    #max_nbytes=None,
                    #batch_size=1,
                )(
                    delayed(_mc_trial)(
                        self.mc_estimator, self.metrics, n_samples, cov, prec,
                        self.prng
                    )
                    for nn in range(self.n_trials))

                for key in self.metrics:
                    results_by_key = np.array([t[key] for t in trials])
                    self.results_[key][aidx, sidx] =\
                            1. * np.sum(results_by_key) / self.n_trials

            if self.verbose:
                for key in self.metrics:
                    print 'Results for {}: {}'.format(key, self.results_[key][aidx, :])

        self.is_fitted = True
        return self