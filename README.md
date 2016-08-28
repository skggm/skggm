# skggm
Gaussian graphical models in scikit-learn.

# Included in this package
- InverseCovarianceEstimator with common model selection metrics (such as EBIC and metrics for cross validation)
- QuicGraphLasso & QuicGraphLassoCV estimators
- AdaptiveInverseCovariance (two stage adaptive meta estimator) [TODO: better name]
- Ensemble meta estimator
- Numerous usage examples

^ These are notes, will clean up later.


### Installation

- Note about pip

- Note about setup

If you would like to work with a forked branch directly, you will need to run  `python build_deps.py (from /scikitquic)` to compile pyquic.


### Runing tests:
Download the test data file `ER_692.mat` from `http://www.cs.utexas.edu/~sustik/QUIC/`.  The file is contained in the MEX archive.  Move this file to `/scikitquic/inverse_covariance/tests`.

    python -m pytest inverse_covariance/tests/
    python -m pytest inverse_covariance/profiling/tests

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

