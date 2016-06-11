# scikit-quic
scikit-learn wrappers for various implementations QUIC


# Setup pyquic submodule

This project currently depends on the pyquic module for python QUIC bindings.  We include this project as a submodule which needs to be checked out independently.  

### When first checking out this repo run:
    git submodule init
    git submodule update --checkout --remote

### Updating pyquic in your working branch
    git submodule update --checkout --remote

### See readme at:
    https://github.com/osdf/pyquic

### After compiling pyquic
    run: python setup.py (in /scikit-quic)

# TODO/Notes:

The `py_quic` module isn't really set up in a way to be used as an included submodule.  To deal with this I've added a script that flattens the compiled module directory to `/pyquic` (this is not preferable). I've also submitted a PR to the original repo/author to fix the -faltivec issue https://github.com/osdf/pyquic/pull/1.  To deal with this, we might just do our own python bindings or add a script