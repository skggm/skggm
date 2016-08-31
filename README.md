# skggm : Gaussian graphical models in scikit-learn.

Given **n**, **p**-dimensional independently drawn Gaussian random samples **X \in R^{n, p}**, the maximum likelihood estimate of the inverse covariance matrix **S** can be computed via the _graphical lasso_, i.e., the program

<img src="images/graphlasso_program.png" alt="\ell_1 penalized inverse covariance estimation" width="500">

where  **Lambda \in R^{p, p}** is a symmetric non-negative weight matrix and

<img src="images/weighted_ell_1.png" alt="\ell_1 penalized inverse covariance estimation" width="200"> 

is a regularization term that promotes sparsity \[[Hsieh et al.](http://jmlr.org/papers/volume15/hsieh14a/hsieh14a.pdf)\]. This is a generalization of the scalar-**lambda** formulation found in \[[Friedman et al.](http://statweb.stanford.edu/~tibs/ftp/glasso-bio.pdf)\] and implemented [here](http://scikit-learn.org/stable/modules/generated/sklearn.covariance.GraphLassoCV.html) .

The graphical lasso finds diverse applications in **TODO X, Y, Z**.

In this package we provide a [scikit-learn](http://scikit-learn.org)-compatible implementation of the program above and a collection of modern best practices for working with the graphical lasso.  To get started, test out 

    
    from inverse_covariance import QuicGraphLassoCV
    
    model = QuicGraphLassoCV()
    model.fit(X)  # X is data matrix of shape (n_samples, n_features) 
    
    # see: model.covariance_, model.precision_, model.lam_
    

and then head over to `examples/estimator_suite.py` for other example usage.

---

This is an ongoing effort. We'd love your feedback on which algorithms we should provide bindings for next and how you're using the package. We also welcome contributions. 

[@jasonlaska](https://github.com/jasonlaska) and [@mnarayn](https://github.com/mnarayan)

---

## Included in `inverse_covariance` 
- **QuicGraphLasso** [[doc]](https://github.com/jasonlaska/skggm/blob/develop/inverse_covariance/quic_graph_lasso.py#L138-L216)

    _QuicGraphLasso_ is an implementation of [QUIC](http://jmlr.org/papers/volume15/hsieh14a/hsieh14a.pdf) wrapped as a scikit-learn compatible estimator \[[Hsieh et al.](http://jmlr.org/papers/volume15/hsieh14a/hsieh14a.pdf)\] . The estimator can be run in `default` mode for a fixed penalty or in `path` mode to explore a sequence of penalties efficiently.  The penalty `lam` can be a scalar or matrix.

    The primary outputs of interest are: `covariance_`, `precision_`, and `lam_`.  _QuicGraphLasso_ also includes the `score(X_test)`, `ebic(gamma=0)`, and `ebic_select(gamma=0)` class methods.

    The interface largely mirrors the built-in _[GraphLasso](http://scikit-learn.org/stable/modules/generated/sklearn.covariance.GraphLasso.html)_ although some param names have been changed (e.g., `alpha` to `lam`). Some notable advantages of this implementation over _GraphLasso_ are:

    - support for a matrix penalization term

    - speed

- **QuicGraphLassoCV** [[doc]](https://github.com/jasonlaska/skggm/blob/develop/inverse_covariance/quic_graph_lasso.py#L365-L439)
    
    _QuicGraphLassoCV_ is an optimized cross-validation model selection implementation in similar fashion to scikit-learn's _[GraphLassoCV](http://scikit-learn.org/stable/modules/generated/sklearn.covariance.GraphLassoCV.html)_. While _QuicGraphLasso_ can be used with _[GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html)_, this estimator yields similar results in less time.

    As with _QuicGraphLasso_, this implementation also supports matrix penalization.

- **QuicGraphLassoEBIC** [[doc]](https://github.com/jasonlaska/skggm/blob/develop/inverse_covariance/quic_graph_lasso.py#L616-L681)

    _QuicGraphLassoEBIC_ is provided as a convenience class to use the _Extended Bayesian Information Criteria_ (EBIC) for model selection \[[Foygel et al.](https://papers.nips.cc/paper/4087-extended-bayesian-information-criteria-for-gaussian-graphical-models)\].  

    As noted earlier, for class methods are provided with _QuicGraphLasso_ to compute EBIC scores and select the best penalty when used in `path` mode. This may be a faster, more flexible approach when experimenting with several model selection methods.

- **quic**

    Python function to run QUIC algorithm (independent of sklearn estimator).

- **AdaptiveGraphLasso** [[doc]](https://github.com/jasonlaska/skggm/blob/develop/inverse_covariance/adaptive_graph_lasso.py#L13-L48)

    _AdaptiveGraphLasso_ performs a two step estimation procedure.  
    
    1. Obtain an initial sparse estimate.  By default QuicGraphLassoCV be used for the initial estimate but you can pass in your own estimator instance via the parameter `estimator`. 

    2. Derive a new penalization matrix from the original estimate.  We currently provide three methods for this: `binary`, `1/|coeffs|`, and `1/|coeffs|^2`.  The `binary` method only requires the initial estimate's support (and this can be be used with _ModelAverage_ below).

    This technique works well to refine the non-zero precision values given a reasonable initial support estimate.

- **ModelAverage** [[doc]](https://github.com/jasonlaska/skggm/blob/develop/inverse_covariance/model_average.py#L66-L162)
    
    This ensemble estimator computes several fits with random penalties and random subsamples (similar to sklearn's [RandomizedLasso](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RandomizedLasso.html).  The result is a `proportion_` matrix indicating the probability of a non-zero at each index. This can be used in conjunction with the `AdaptiveGraphLasso` for a final estimate.


- **InverseCovarianceEstimator** [[doc]](https://github.com/jasonlaska/skggm/blob/develop/inverse_covariance/inverse_covariance.py#L80-L123)
    
    Base class with common scoring metrics (`log_likelihood`, `frobenius`, `kl-loss`) and EBIC model selection criteria.

- **trace_plot**

    Utility to plot `lam_` paths.

- **.profiling**

    Submodule that includes `profiling.AverageError`, `profiling.StatisticalPower`, etc. to compare performance between methods.


## Installation

- Note about pip

- Note about setup

If you would like to work with a forked branch directly, you will need to run  `python build_deps.py (from /scikitquic)` to compile pyquic.


## Tests
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

## Plotting the regularization path
We've provided a utility function `inverse_covariance.trace_plot` that can be used to display the coefficients as a function of `lam_`.  This can be used with any estimator that returns a path.  The example in `examples/trace_plot_example.py` yields:

<img src="images/trace_plot.png" alt="Trace plot" width="400">

## Profiling utilities
We've provided some utilities in `inverse_covariance.profiling` to compare performance across the estimators. 

For example, below is the comparison of the average support error between `QuicGraphLassoCV` and its randomized model average equivalent (the example found in `examples/compare_model_selection.py`).  The support error of `QuicGraphLassoCV` is dominated by the false-positive rate which grows substantially as the number of samples grows.

<img src="images/model_avg_support.png" alt="" width="300">
<img src="images/quicgraphlassocv_support.png" alt="" width="300">


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

### Convergence test

* ["The graphical lasso: New Insights and alternatives"](https://web.stanford.edu/~hastie/Papers/glassoinsights.pdf) Mazumder and Hastie, 2012.

