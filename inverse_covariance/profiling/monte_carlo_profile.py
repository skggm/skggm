from __future__ import absolute_import

import numpy as np
from functools import partial
from sklearn.base import clone
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


def _ms_fit(indexed_params, estimator, n_features, graph, prng):
    # unpack params
    index, (alpha, grid_point) = indexed_params

    # draw a new fixed graph for alpha
    cov, prec, adj = graph.create(n_features, alpha)

    # model selection (once per n_samples grid point)
    n_samples = int(grid_point * n_features)
    X = _sample_mvn(n_samples, cov, prng)
    ms_estimator = clone(estimator)
    ms_estimator.fit(X)

    return index, ((cov, prec, adj), ms_estimator.lam_, n_samples)


def _mc_fit(indexed_params, estimator, metrics, prng):
    # unpack params
    index, (nn, (cov, prec, adj), lam, n_samples) = indexed_params

    # compute mc trial
    X = _sample_mvn(n_samples, cov, prng)
    mc_estimator = clone(estimator)
    mc_estimator.set_params(lam=lam)
    mc_estimator.fit(X)
    results = {k: f(prec, mc_estimator.precision_) for k, f in metrics.items()}

    return index, results


def _cpu_map(fun, param_grid, n_jobs, verbose):
    return Parallel(
        n_jobs=n_jobs,
        verbose=verbose,
        backend='threading',  # any sklearn backend should work here
    )(
        delayed(fun)(
            params
        )
        for params in param_grid)


def _spark_map(fun, indexed_param_grid, sc, seed):
    '''We cannot pass a RandomState instance to each spark worker since it will
    behave identically across partitions.  Instead, we explictly handle the
    partitions with a newly seeded instance.

    The seed for each partition will be the "seed" (MonteCarloProfile.seed) +
    "split_index" which is the partition index.

    Following this trick:
        https://wegetsignal.wordpress.com/2015/05/08/
                generating-random-numbers-for-rdd-in-spark/
    '''
    def _wrap_random_state(split_index, partition):
        prng = np.random.RandomState(seed + split_index)
        yield map(partial(fun, prng=prng), partition)

    par_param_grid = sc.parallelize(indexed_param_grid)
    indexed_results = par_param_grid.mapPartitionsWithIndex(
        _wrap_random_state).collect()
    return [item for sublist in indexed_results for item in sublist]


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
        An inverse covariance estimator instance. This estimator must be able
        to select a penalization parameter that can be accessed via the
        instance variable .lam_.

    mc_estimator : "Monte Carlo trial estimator" (default=None)
        An inverse covariance estimator instance. Estimator to use on each
        instance after selecting a penalty lambda. The penalty parameter 'lam'
        will be overriden by with penalty selected by ms_estimator.
        If None, this will use QuicGraphLasso.

    graph :  An instance of a class with the method .create(n_features, alpha)
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

    n_jobs: int (optional)
        number of jobs to run in parallel (default 1).

    sc: sparkContext (optional)
        If a sparkContext object is provided, n_jobs will be ignore and the
        work will be parallelized via spark.

    seed : np.random.RandomState starting seed. (default=2)


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
                 mc_estimator=None, graph=None, n_samples_grid=10,
                 alpha_grid=5, metrics={'frobenius': error_fro}, verbose=False,
                 n_jobs=1, sc=None, seed=2):
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
        self.sc = sc
        self.seed = seed
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
            self.alphas_ = np.logspace(
                np.log(0.15), np.log10(0.4), self.alpha_grid
            )
        else:
            self.alphas_ = self.alpha_grid

        self.is_fitted = False
        self.results_ = None
        self.precision_nnz_ = None

    def fit(self, X=None, y=None):
        n_alphas = len(self.alphas_)
        n_grids = len(self.grid_)

        self.precision_nnz_ = []
        self.results_ = {
            k: np.zeros((n_alphas, n_grids)) for k in self.metrics
        }

        # build an indexed set (or generator) of grid points
        param_grid = [(a, g) for a in self.alphas_ for g in self.grid_]
        indexed_param_grid = list(zip(range(len(param_grid)), param_grid))

        ms_fit = partial(_ms_fit,
                         estimator=self.ms_estimator,
                         n_features=self.n_features,
                         graph=self.graph,
                         prng=self.prng)

        if self.verbose:
            print('Getting parameters via model selection...')

        if self.sc is not None:
            ms_results = _spark_map(
                ms_fit, indexed_param_grid, self.sc, self.seed
            )
        else:
            ms_results = _cpu_map(
                ms_fit, indexed_param_grid, self.n_jobs, self.verbose
            )

        # ensure results are ordered
        ms_results = sorted(ms_results, key=lambda r: r[0])

        # track nnz of graph precision
        self.precision_nnz_ = [
            np.count_nonzero(graph[1].flat) for _, (graph, _, _) in ms_results
        ]

        # build param grid for mc trials
        # following results in an grid where nn indexes each trial of each
        # param grid:
        #  (0, graph_0, lam_0, n_samples_), (1, graph_0, lam_0, n_samples_0),..
        trial_param_grid = [
            (nn, graph, lam, n_samples)
            for _, (graph, lam, n_samples) in ms_results
            for nn in range(self.n_trials)
        ]
        indexed_trial_param_grid = list(
            zip(range(len(trial_param_grid)), trial_param_grid)
        )

        mc_fit = partial(
            _mc_fit,
            estimator=self.mc_estimator,
            metrics=self.metrics
        )

        if self.verbose:
            print('Fitting MC trials...')

        if self.sc is not None:
            mc_results = _spark_map(
                mc_fit,
                indexed_trial_param_grid,
                self.sc,
                self.seed + len(param_grid)
            )
        else:
            mc_results = _cpu_map(
                partial(mc_fit, prng=self.prng),
                indexed_trial_param_grid,
                self.n_jobs,
                self.verbose
            )

        # ensure results are ordered correctly
        mc_results = sorted(mc_results, key=lambda r: r[0])

        # strip mc_results
        mc_results = [r for _, r in mc_results]

        # reduce
        param_grid_matrix_index = [
            (a, g)
            for a in range(len(self.alphas_))
            for g in range(len(self.grid_))
        ]
        for param_index in range(len(param_grid)):
            trial_start = param_index * self.n_trials
            trials = mc_results[trial_start: trial_start + self.n_trials]
            aidx, gidx = param_grid_matrix_index[param_index]
            for key in self.metrics:
                results_by_key = np.array([t[key] for t in trials])
                self.results_[key][aidx, gidx] =\
                    1. * np.sum(results_by_key) / self.n_trials

        if self.verbose:
                for key in self.metrics:
                    print('Results for {}: {}'.format(
                        key, self.results_[key][aidx, :]
                    ))

        self.is_fitted = True
        return self
