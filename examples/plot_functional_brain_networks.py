"""
Estimate Functional Connectivity using an estimator for Sparse Inverse Covariances
==================================================================================
This example constructs a functional connectome using the sparse penalized MLE
estimator implemented using QUIC.

This function extracts time-series from the ABIDE dataset, with nodes defined
using regions of interest from the
  Power-264 atlas (Power, 2011).
  Power, Jonathan D., et al. "Functional network organization of the human
  brain."
  Neuron 72.4 (2011): 665-678.

Then we estimate separate inverse covariance matrices for one subject
"""
##############################################################################

import numpy as np
import matplotlib.pyplot as plt
from nilearn import datasets, plotting, input_data

import sys

sys.path.append("..")
sys.path.append("../inverse_covariance")
from inverse_covariance import (
    QuicGraphicalLasso,
    QuicGraphicalLassoCV,
    QuicGraphicalLassoEBIC,
    AdaptiveGraphicalLasso,
)

plt.ion()

# Fetch the coordinates of power atlas
power = datasets.fetch_coords_power_2011()
coords = np.vstack((power.rois["x"], power.rois["y"], power.rois["z"])).T

# Loading the functional datasets
abide = datasets.fetch_abide_pcp(n_subjects=1)
abide.func = abide.func_preproc

# print basic information on the dataset
# 4D data
print("First subject functional nifti images (4D) are at: %s" % abide.func[0])

###############################################################################
# Masking: taking the signal in a sphere of radius 5mm around Power coords

masker = input_data.NiftiSpheresMasker(
    seeds=coords,
    smoothing_fwhm=4,
    radius=5.,
    standardize=True,
    detrend=True,
    low_pass=0.1,
    high_pass=0.01,
    t_r=2.5,
)

timeseries = masker.fit_transform(abide.func[0])


###############################################################################
# Extract and plot sparse inverse covariance

estimator_type = "QuicGraphicalLasso"

if estimator_type == "QuicGraphicalLasso":
    # Compute the sparse inverse covariance via QuicGraphicalLasso
    estimator = QuicGraphicalLasso(
        init_method="cov", lam=0.5, mode="default", verbose=1
    )
    estimator.fit(timeseries)

elif estimator_type == "QuicGraphicalLassoCV":
    # Compute the sparse inverse covariance via QuicGraphicalLassoCV
    estimator = QuicGraphicalLassoCV(init_method="cov", verbose=1)
    estimator.fit(timeseries)

elif estimator_type == "QuicGraphicalLassoEBIC":
    # Compute the sparse inverse covariance via QuicGraphicalLassoEBIC
    estimator = QuicGraphicalLassoEBIC(init_method="cov", verbose=1)
    estimator.fit(timeseries)

elif estimator_type == "AdaptiveQuicGraphicalLasso":
    # Compute the sparse inverse covariance via
    # AdaptiveGraphicalLasso + QuicGraphicalLassoEBIC + method='binary'
    model = AdaptiveGraphicalLasso(
        estimator=QuicGraphicalLassoEBIC(init_method="cov"), method="binary"
    )
    model.fit(timeseries)
    estimator = model.estimator_

# Display the sparse inverse covariance
plt.figure(figsize=(7.5, 7.5))
plt.imshow(
    np.triu(-estimator.precision_, 1), interpolation="nearest", cmap=plt.cm.RdBu_r
)
plt.title("Precision (Sparse Inverse Covariance) matrix")
plt.colorbar()

# And now display the corresponding graph
plotting.plot_connectome(
    -estimator.precision_,
    coords,
    title="Functional Connectivity using Precision Matrix",
    edge_threshold="99.2%",
    node_size=20,
)
plotting.show()

eval(input("Press any key to exit.."))
