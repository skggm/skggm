# scikitquic
scikit-learn wrappers for various implementations QUIC


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

### Runing tests:
    py.test inverse_covariance/tests/

# TODO/Notes:

1. The `py_quic` module isn't really set up in a way to be used as an included submodule.  To deal with this I've added a script that moves the compiled module directory to `/scikitquic` (this is not preferable). Make this easier to use, we might just do our own python bindings in the future.

2. I've also submitted a PR to the original repo/author to fix the -faltivec issue https://github.com/osdf/pyquic/pull/1.  

3. README to be cleaned up after notes collected here

4. Directory structure of project tbd, right now main file is at top level.

# Acknowledgements

* [QUIC](http://www.cs.utexas.edu/~sustik/QUIC/) algorithm is explained in *Sparse Inverse Covariance Matrix Estimation Using Quadratic Approximation* by Cho-Jui Hsieh, Mátyás A. Sustik, Inderjit S. Dhillon, Pradeep Ravikumar and forked from [http://www.cs.utexas.edu/~sustik/QUIC/](http://www.cs.utexas.edu/~sustik/QUIC/).

* Cython bindings for QUIC forked from [https://github.com/osdf/pyquic](https://github.com/osdf/pyquic)
