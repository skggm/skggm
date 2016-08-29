# skggm : Gaussian graphical models in scikit-learn.


# Included in this package
- **QuicGraphLasso** [[doc]](https://github.com/jasonlaska/skggm/blob/develop/inverse_covariance/quic_graph_lasso.py#L138-L216)

    This `InverseCovarianceEstimator` wraps the [QUIC](http://jmlr.org/papers/volume15/hsieh14a/hsieh14a.pdf) algorithm as a scikit-learn compatible estimator. We expect this to be used interchangeably with the built-in `GraphLasso` (and changing some param names, e.g., `alpha` to `lam`).  

    The primary output parameters of interest are: `covariance_`, `precision_`, and `lam_`.

    Notable advantages of this implementation over sklearn's built-in implementation are support for a matrix penalization term and speed.

    - **QuicGraphLassoCV** [[doc]](https://github.com/jasonlaska/skggm/blob/develop/inverse_covariance/quic_graph_lasso.py#L365-L439)
        
        Provides an optimized implementation for cross-validation model selection in similar fashion to sklearn's [GraphLassoCV](http://scikit-learn.org/stable/modules/generated/sklearn.covariance.GraphLassoCV.html).  While `QuicGraphLasso` can be used with [GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html), this implementation yields similar results in less time.

    - **QuicGraphLassoEBIC** [[doc]](https://github.com/jasonlaska/skggm/blob/develop/inverse_covariance/quic_graph_lasso.py#L616-L681)

        Provided as a convenience class to use the extended Bayesian information criteria for model selection.  This criteria can also be applied directly to `QuicGraphLasso` after being run in `path` mode.

    - **quic**

        Python function to run QUIC algorithm (independent of sklearn estimator).

- **AdaptiveGraphLasso** [[doc]](https://github.com/jasonlaska/skggm/blob/develop/inverse_covariance/adaptive_graph_lasso.py#L13-L48)

    This `InverseCovarianceEstimator` performs a two step estimation procedure.  It obtains an initial sparse estimate (QuicGraphLassoCV by default), derives a new penalization matrix from the result, and refits.  This technique works well to refine the non-zero precision values once a reasonable support is estimated.

- **ModelAverage** [[doc]](https://github.com/jasonlaska/skggm/blob/develop/inverse_covariance/model_average.py#L66-L162)
    
    This ensemble estimator computes several fits with random penalties and random subsamples (similar to sklearn's [RandomizedLasso](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RandomizedLasso.html).  The result is a `proportion_` matrix indicating the probability of a non-zero at each index. This can be used in conjunction with the `AdaptiveGraphLasso` for a final estimate.


- **InverseCovarianceEstimator** [[doc]](https://github.com/jasonlaska/skggm/blob/develop/inverse_covariance/inverse_covariance.py#L80-L123)
    
    Base class with common metrics and EBIC model selection criteria.

- **trace_plot**

    Utility to plot `lam_` paths.

- **profiling**

    Submodule that includes `AverageError`, `StatisticalPower`, etc. to compare performance between methods.


### Installation

- Note about pip

- Note about setup

If you would like to work with a forked branch directly, you will need to run  `python build_deps.py (from /scikitquic)` to compile pyquic.


### Tests
To run the tests, execute the following lines.  

    python -m pytest inverse_covariance/tests/
    python -m pytest inverse_covariance/profiling/tests

# Examples

## Usage 
In `examples/estimator_suite.py` we reproduce the [plot_sparse_cov](http://scikit-learn.org/stable/auto_examples/covariance/plot_sparse_cov.html) example from the scikit-learn documentation for each method provided (however, the variations chosen are not exhaustive).

An example run for `n_examples=100` and `n_features=20` yielded the following results. 

<img src="images/estimator_suite_scorecard_100x20.png" alt="(n_examples, n_features) = (100, 20)" width="650">

<img src="images/estimator_suite_plots_page0_100x20.png" alt="(n_examples, n_features) = (100, 20)" width="600">

<img src="images/estimator_suite_plots_page1_100x20.png" alt="(n_examples, n_features) = (100, 20)" width="600">

For slightly higher dimensions of `n_examples=600` and `n_features=120` we obtained:

<img src="images/estimator_suite_scorecard_600x120.png" alt="(n_examples, n_features) = (600, 120)" width="650">

# References

### BIC / EBIC Model Selection

* ["Extended Bayesian Information Criteria for Gaussian Graphical Models"](https://papers.nips.cc/paper/4087-extended-bayesian-information-criteria-for-gaussian-graphical-models) R. Foygel and M. Drton NIPS 2010

### QuicGraphLasso / QuicGraphLassoCV

* ["QUIC: Quadratic Approximation for sparse inverse covariance estimation"](http://jmlr.org/papers/volume15/hsieh14a/hsieh14a.pdf) by C. Hsieh, M. A. Sustik, I. S. Dhillon, P. Ravikumar, Journal of Machine Learning Research (JMLR), October 2014.

* QUIC implementation found [here](http://www.cs.utexas.edu/~sustik/QUIC/) and [here](http://bigdata.ices.utexas.edu/software/1035/) with cython bindings forked from [pyquic](https://github.com/osdf/pyquic)

### Adaptive refitting (two-step methods)

* ["High dimensional covariance estimation based on Gaussian graphical models"](http://www.jmlr.org/papers/volume12/zhou11a/zhou11a.pdf) S. Zhou, P. R{\"u}htimann, M. Xu, and P. B{\"u}hlmann

* ["Relaxed Lasso"](http://stat.ethz.ch/~nicolai/relaxo.pdf) N. Meinshausen, December 2006.

### Randomized model averaging 

* ["Stability Selection"](https://arxiv.org/pdf/0809.2932v2.pdf) N. Meinhausen and P. Buehlmann, May 2009

* ["Random Lasso"](https://arxiv.org/abs/1104.3398) S. Wang, B. Nan, S. Rosset, and J. Zhu, Apr 2011
            
* ["Mixed effects models for resampled network statistics improves statistical power to find differences in multi-subject functional connectivity"](http://biorxiv.org/content/early/2016/03/14/027516) M. Narayan and G. Allen, March 2016

