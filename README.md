# skggm : Gaussian graphical models in scikit-learn.


# Included in this package
- **QuicGraphLasso** [[doc]](https://github.com/jasonlaska/skggm/blob/develop/inverse_covariance/quic_graph_lasso.py#L138-L216)

    This `InverseCovarianceEstimator` wraps the [QUIC](http://jmlr.org/papers/volume15/hsieh14a/hsieh14a.pdf) algorithm as a scikit-learn compatible estimator. In general, this can be used interchangeably with the built-in `GraphLasso` by swaping `alpha` for `lam`.  

    Notable advantages of this implementation over sklearn's built-in implementation are support for a matrix penalization term and speed.

    - **QuicGraphLassoCV** [[doc]](https://github.com/jasonlaska/skggm/blob/develop/inverse_covariance/quic_graph_lasso.py#L360-L427)
        
        Provides an optimized implementation for cross-validation model selection in similar fashion to sklearn's [GraphLassoCV](http://scikit-learn.org/stable/modules/generated/sklearn.covariance.GraphLassoCV.html).  While `QuicGraphLasso` can be used with [GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html), this implementation yields similar results in less time.

    - **QuicGraphLassoEBIC** [[doc]](https://github.com/jasonlaska/skggm/blob/develop/inverse_covariance/quic_graph_lasso.py#L605-L664)

        Provided as a convenience class to use the extended Bayesian information criteria for model selection.  This criteria can also be applied directly to `QuicGraphLasso` after being run in `path` mode.

    - **quic**

        Python function to run QUIC algorithm (independent of sklearn estimator).

- AdaptiveInverseCovariance (two stage adaptive meta estimator) [TODO: better name]
- Ensemble meta estimator
- Numerous usage examples
- InverseCovarianceEstimator with common model selection metrics (such as EBIC and metrics for cross validation)


^ These are notes, will clean up later.


### Installation

- Note about pip

- Note about setup

If you would like to work with a forked branch directly, you will need to run  `python build_deps.py (from /scikitquic)` to compile pyquic.


### Runing tests:
Download the test data file `ER_692.mat` from `http://www.cs.utexas.edu/~sustik/QUIC/`.  The file is contained in the MEX archive.  Move this file to `/scikitquic/inverse_covariance/tests`.

    python -m pytest inverse_covariance/tests/
    python -m pytest inverse_covariance/profiling/tests

# Examples

## Estimator Suite
In `examples/estimator_suite.py` we reproduce the [plot_sparse_cov](http://scikit-learn.org/stable/auto_examples/covariance/plot_sparse_cov.html) example from the scikit-learn documentation for each method provided.

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

