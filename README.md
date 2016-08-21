# skggm
Gaussian graphical models in scikit-learn.

# Included in this package
- InverseCovarianceEstimator with common model selection metrics (such as EBIC and metrics for cross validation)
- QuicGraphLasso & QuicGraphLassoCV estimators
- AdaptiveInverseCovariance (two stage adaptive meta estimator) [TODO: better name]
- Ensemble meta estimator
- Numerous usage examples

^ These are notes, will clean up later.

# Setup pyquic submodule

This project currently depends on the pyquic module for python QUIC bindings.  We include this project as a submodule which needs to be checked out independently.  

### When first checking out this repo run:
    git submodule init
    git submodule update --checkout --remote

### Updating pyquic in your working branch
    git submodule update --checkout --remote -f

### Setting up pyquic

1. Run `python setup.py (from /scikitquic)`.  This will compile pyquic and copy the module into `/scikitquic/inverse_covariance/`.

2. Download the test data file `ER_692.mat` from `http://www.cs.utexas.edu/~sustik/QUIC/`.  The file is contained in the MEX archive.  Move this file to `/scikitquic/inverse_covariance/tests`.

NOTE:  During cleanup, we should rename setup.py since that file will be used for setting up the entire python package.

### Runing tests:
    py.test inverse_covariance/tests/
    python -m pytest inverse_covariance/tests/

# TODO/Notes:

1. The `py_quic` module isn't really set up in a way to be used as an included submodule.  To deal with this I've added a script that moves the compiled module directory to `/scikitquic` (this is not preferable). Make this easier to use, we might just do our own python bindings in the future.

2. I've also submitted a PR to the original repo/author to fix the -faltivec issue https://github.com/osdf/pyquic/pull/1.  

3. README to be cleaned up after notes collected here

4. Directory structure of project tbd, right now main file is at top level.

# References

### BIC / EBIC Model Selection

* ["Extended Bayesian Information Criteria for Gaussian Graphical Models"](https://papers.nips.cc/paper/4087-extended-bayesian-information-criteria-for-gaussian-graphical-models) R. Foygel and M. Drton NIPS 2010

### QuicGraphLasso / QuicGraphLassoCV

* ["QUIC: Quadratic Approximation for sparse inverse covariance estimation"](http://jmlr.org/papers/volume15/hsieh14a/hsieh14a.pdf) by C. Hsieh, M. A. Sustik, I. S. Dhillon, P. Ravikumar, Journal of Machine Learning Research (JMLR), October 2014.

* QUIC implementation found [here](http://www.cs.utexas.edu/~sustik/QUIC/) and [here](http://bigdata.ices.utexas.edu/software/1035/) with cython bindings forked from [pyquic](https://github.com/osdf/pyquic)

### Randomized model averaging 

* ["Stability Selection"](https://arxiv.org/pdf/0809.2932v2.pdf) N. Meinhausen and P. Buehlmann, May 2009

* ["Mixed effects models for resampled network statistics improves statistical power to find differences in multi-subject functional connectivity"](http://biorxiv.org/content/early/2016/03/14/027516) M. Narayan and G. Allen, March 2016

