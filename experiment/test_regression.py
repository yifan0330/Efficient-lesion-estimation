import numpy as np
from util import compute_mu_mean, compute_P_mean, preprocess_Z, compute_mu
from plot import plot_brain
import nibabel as nib
import dask.array as da
import os

# simulated_dset = True
# homogeneous = True
# polynomial_order = 3

# gradient_mode = "dask"
# preconditioner_mode = "approximate"

# filename_1 = "_Homogeneous" if homogeneous else "_BumpSignals"

# data_path = os.getcwd()+f"/data/3D/3D_data_Simulation{filename_1}_2_group_[2000, 4000]_Poisson_log_link_func.npz"
# data = np.load(data_path)
# B, Z = data["X_spatial"], data["Z"]
# B = np.concatenate([data["X_spatial"], np.ones((data["X_spatial"].shape[0], 1))], axis=1)
# Z = np.concatenate([Z, np.ones((Z.shape[0], 1))], axis=1)
# _M, _R = Z.shape
# _N, _P = B.shape
# Z = preprocess_Z(simulated_dset, Z, polynomial_order)

# # re-scale
# if homogeneous:
#     Z_rescaled = Z * 10 / _M
#     B_rescaled = B * 10 / _N
# else:
#     Z_rescaled = Z * 50 / _M
#     B_rescaled = B * 50 / _N

# baseline_results = np.load(f"results/3D/3D_Probability_comparison_Simulation{filename_1}_approximate_model_dask_dask_2_group_[2000, 4000]_Poisson_log_link_func.npz")
# baseline_beta = baseline_results["beta"]
# baseline_MU = compute_mu(Z_rescaled, B_rescaled, baseline_beta, mode="dask", block_size=10000)
# print(np.min(baseline_MU), np.max(baseline_MU), np.mean(baseline_MU))

# results_path = os.getcwd()+f"/results/3D/3D_Probability_comparison_Simulation{filename_1}_approximate_model_{gradient_mode}_{preconditioner_mode}_2_group_[2000, 4000]_Poisson_log_link_func.npz"
# results = np.load(results_path)
# beta = results["beta"]
# MU_hat = compute_mu(Z, B, beta, mode="dask", block_size=10000)

# relative_bias = np.abs(MU_hat - baseline_MU) / baseline_MU
# relative_std = np.std(MU_hat - baseline_MU) / np.mean(baseline_MU)
# relative_mse = np.mean((MU_hat - baseline_MU) ** 2) / np.mean(baseline_MU ** 2)
# print(np.mean(relative_bias), np.mean(relative_std), np.mean(relative_mse))





# real_data = np.load("data/UKB/masked_data_RealDataset_approximate_model_Poisson_log_link_func_spacing_5.npz")
# B = real_data["X_spatial"]
# Z = real_data["Z"]
# ZTZ = Z.T @ Z  # shape: (11, 11)
# BTB = B.T @ B  # shape: (1086, 1086)
# XTX = np.kron(ZTZ, BTB) # shape: (11946, 11946)
# XTX_inv = np.linalg.inv(XTX)
# print(XTX_inv.shape)
# print(np.min(np.diag(XTX_inv)), np.max(np.diag(XTX_inv)), np.mean(np.diag(XTX_inv)))

# prod = XTX @ XTX_inv
# print(prod)
# print(np.diag(prod))
# print(np.min(np.diag(prod)), np.max(np.diag(prod)), np.mean(np.diag(prod)))



polynomial_order = 3
filename = "linear" if polynomial_order == 1 else "cubic" if polynomial_order == 3 else None
simulated_dset = False
spacing = 5

smooth_lesion_mask = nib.load("data/UKB/smooth_lesion_mask_RealDataset_approximate_model_Poisson_log_link_func.nii.gz")

data = np.load(f"data/UKB/masked_data_RealDataset_approximate_model_Poisson_log_link_func_spacing_{spacing}.npz")
Y = data["Y"]
Y_mean = np.mean(Y, axis=0)

results = np.load(f"results/UKB/Probability_comparison_RealDataset_approximate_model_dask_approximate_Poisson_log_link_func_spacing_{spacing}_{filename}.npz")
beta, MU_mean, P_mean = results["beta"], results["MU_mean"], results["P_mean"]
print(beta.shape)

print(np.min(MU_mean), np.max(MU_mean), np.mean(MU_mean))
P_mean = MU_mean * np.exp(-MU_mean)
print(P_mean.shape)
print(np.min(P_mean), np.max(P_mean), np.mean(P_mean))

# plot_brain(p=1.76*np.sqrt(P_mean), brain_mask=smooth_lesion_mask, threshold=0)
plot_brain(p=np.sqrt(Y_mean), brain_mask=smooth_lesion_mask, threshold=0)
