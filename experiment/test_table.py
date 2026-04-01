import numpy as np
import nibabel as nib
import dask.array as da
import os
from bspline import B_spline_bases
from util import kronecker_vector_product, preprocess_Z, compute_mu_mean
from dask.diagnostics import ProgressBar

spacing = 5
polynomial_order = 1
simulated_dset = False

smooth_lesion_mask_filename = "/well/nichols/users/pra123/brain_lesion_project/simulation_experiment/data/UKB/smooth_lesion_mask_RealDataset_approximate_model_Poisson_log_link_func.nii.gz"
brain_mask = nib.load(smooth_lesion_mask_filename)
brain_mask_data = brain_mask.get_fdata()
mask_indices = np.where(brain_mask_data > 0)

data = np.load(f"data/UKB/masked_data_RealDataset_approximate_model_Poisson_log_link_func_spacing_{spacing}.npz")
Y, Z, B = data["Y"], data["Z"], data["X_spatial"]

age_mean, age_std = np.mean(Z[:, 2]), np.std(Z[:, 2])
cvr_mean, cvr_std = np.mean(Z[:, 4]), np.std(Z[:, 4])
Z = preprocess_Z(simulated_dset, Z, polynomial_order)
Z = Z * 50 / Z.shape[0]
B = B * 50 / B.shape[0]
Z = np.concatenate([Z, np.ones((Z.shape[0], 1))], axis=1)
B = np.concatenate([B, np.ones((B.shape[0], 1))], axis=1)
_M, _R = Z.shape # (13677, 6)
_N, _P = B.shape # (14807, 890)
Z_mean = np.mean(Z, axis=0).reshape(1, -1) # shape: (1, 6)
Z_age_10_mean = Z_mean.copy()
Z_age_10_mean[0, 2] += 10.0 / age_std * 50 / Z.shape[0] # shape: (1, 6)
Z_cvr_1_mean = Z_mean.copy()
Z_cvr_1_mean[0, 4] += 1.0 / cvr_std * 50 / Z.shape[0] # shape: (1, 6)
results_Poisson_5_linear = np.load("results/UKB/Probability_comparison_RealDataset_approximate_model_dask_approximate_Poisson_log_link_func_spacing_5_linear.npz")
beta = results_Poisson_5_linear["beta"] # shape:(5340, 1)

eta_mean = kronecker_vector_product(Z_mean, B, beta, use_dask=True, block_size=10000)
mu_mean = da.exp(eta_mean) # shape: (14807, 1)
with ProgressBar():
    mu_mean = mu_mean.compute()

mask_indices = np.where(brain_mask_data > 0)
mu_nifti_data = np.zeros(brain_mask_data.shape, dtype=np.float32)
# Assign p-vals/z-statistics to the masked voxels
mu_nifti_data[mask_indices] = mu_mean.ravel()

P_nifti_data = mu_nifti_data * np.exp(-mu_nifti_data)

eta_age_10_mean = kronecker_vector_product(Z_age_10_mean, B, beta, use_dask=True, block_size=10000)
mu_age_10_mean = da.exp(eta_age_10_mean) # shape: (14807, 1)
with ProgressBar():
    mu_age_10_mean = mu_age_10_mean.compute()
mu_age_10_nifti_data = np.zeros(brain_mask_data.shape, dtype=np.float32)
# Assign p-vals/z-statistics to the masked voxels
mu_age_10_nifti_data[mask_indices] = mu_age_10_mean.ravel()

P_age_10_nifti_data = mu_age_10_nifti_data * np.exp(-mu_age_10_nifti_data)

RR_age_10 = P_age_10_nifti_data / P_nifti_data
RD_age_10 = P_age_10_nifti_data - P_nifti_data

print(np.nanmean(RR_age_10))
exit()

# eta_cvr_1_mean = kronecker_vector_product(Z_cvr_1_mean, B, beta, use_dask=True, block_size=10000)
# mu_cvr_1_mean = da.exp(eta_cvr_1_mean) # shape: (14807, 1)
# with ProgressBar():
#     mu_cvr_1_mean = mu_cvr_1_mean.compute()
# mu_cvr_1_nifti_data = np.zeros(brain_mask_data.shape, dtype=np.float32)
# # Assign p-vals/z-statistics to the masked voxels
# mu_cvr_1_nifti_data[mask_indices] = mu_cvr_1_mean.ravel()

# P_cvr_1_nifti_data = mu_cvr_1_nifti_data * np.exp(-mu_cvr_1_nifti_data)

# RR_cvr_1 = P_cvr_1_nifti_data / P_nifti_data
# RD_cvr_1 = P_cvr_1_nifti_data - P_nifti_data


print(RR_cvr_1[53,74,42])
print(RR_cvr_1[36,75,41])
print(RR_cvr_1[30,39,45])
print(RR_cvr_1[34,25,39])
print(RR_cvr_1[59,38,45])
print(RR_cvr_1[55,24,38])

# print(3.0976*RD_cvr_1[53,74,42])
# print(3.0976*RD_cvr_1[36,75,41])
# print(3.0976*RD_cvr_1[30,39,45])
# print(3.0976*RD_cvr_1[34,25,39])
# print(3.0976*RD_cvr_1[59,38,45])
# print(3.0976*RD_cvr_1[55,24,38])