# scikitquic
scikit-learn wrappers for various implementations QUIC


# Setup pyquic submodule

This project currently depends on the pyquic module for python QUIC bindings.  We include this project as a submodule which needs to be checked out independently.  

### When first checking out this repo run:
    git submodule init
    git submodule update --checkout --remote

### Updating pyquic in your working branch
    git submodule update --checkout --remote

### Run setup script to compile the submodule
    run: python setup.py (in /scikitquic)
This will also move the py_quic module into the top `/scikitquic` directory.

### To test submodule, see readme at:
    https://github.com/osdf/pyquic

# TODO/Notes:

1. The `py_quic` module isn't really set up in a way to be used as an included submodule.  To deal with this I've added a script that moves the compiled module directory to `/scikitquic` (this is not preferable). Make this easier to use, we might just do our own python bindings in the future.

2. I've also submitted a PR to the original repo/author to fix the -faltivec issue https://github.com/osdf/pyquic/pull/1.  

