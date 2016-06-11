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