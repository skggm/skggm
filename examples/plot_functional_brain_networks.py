"""
Estimate Functional Connectivity using an estimator for Sparse Inverse Covariances
==================================================================================
This example constructs a functional connectome using the sparse penalized MLE estimator implemented using QUIC.

This function extracts time-series from the ABIDE dataset, with nodes defined using regions of interest from the 
Power-264 atlas (Power, 2011).
Power, Jonathan D., et al. "Functional network organization of the
human brain." Neuron 72.4 (2011): 665-678.

Then we estimate separate inverse covariance matrices for one subject

"""


##############################################################################

import numpy as np
import matplotlib.pyplot as plt
from nilearn import datasets, connectome, plotting, input_data

# Retrieve the atlas and the data
from nilearn import datasets

# Fetch the coordinates of power atlas
power = datasets.fetch_coords_power_2011()
coords = np.vstack((
    power.rois['x'],
    power.rois['y'],
    power.rois['z'],
)).T


atlas_filename, labels = power.maps, power.labels


# Loading the functional datasets
abide = datasets.fetch_abide_pcp(n_subjects=1)
abide.func = abide.func_preproc

# print basic information on the dataset
print('First subject functional nifti images (4D) are at: %s' %
      abide.func[0])  # 4D data
      
      
      
###############################################################################
# Masking: taking the signal in a sphere of radius 5mm around Power coords

masker = input_data.NiftiSpheresMasker(seeds=coords,
                                       smoothing_fwhm=4,
                                       radius=5.,
                                       standardize=True,
                                       detrend=True,
                                       low_pass=0.1,
                                       high_pass=0.01,
                                       t_r=2.5)

timeseries = masker.fit_transform(abide.func[0])



###############################################################################
# Extract and plot covariance and sparse covariance

# Compute the sparse inverse covariance
from quic.quic import QUIC

Shat =  np.cov(timeseries[:,:],rowvar=0)
estimator = QUIC(lam = float(.5*np.max(np.triu(np.abs(Shat),1))), mode='default',verbose=1)
estimator.fit(Shat)
#estimator.fit(timeseries)

# Display the sparse inverse covariance
plt.figure(figsize=(7.5, 7.5))
plt.imshow(np.triu(-estimator.precision_,1), interpolation="nearest", vmax=1, vmin=-1, cmap=plt.cm.PRGn)

# Display the labels, if available. Not available for Power's Coordinates
# x_ticks = plt.xticks(range(len(labels)), labels, rotation=90)
# y_ticks = plt.yticks(range(len(labels)), labels)
plt.title('Precision (Sparse Inverse Covariance) matrix')
plt.colorbar()

# And now display the corresponding graph
plotting.plot_connectome(estimator.precision_, coords,
                         title='Functional Connectivity using Precision Matrix',
                         edge_threshold="99.2%",
                         node_size=20)
plotting.show()