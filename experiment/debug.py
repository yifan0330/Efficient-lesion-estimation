import numpy as np
from util import compute_gradient, compute_preconditioner, kronecker_vector_product
from util import eigen_clip, compute_log_poisson_nll
import logging
import dask.array as da
from dask.diagnostics import ProgressBar

logger = logging.getLogger()
logger.setLevel(logging.INFO)

data_path = "data/3D/3D_data_Simulation_Homogeneous_2_group_[2000, 4000]_Poisson_log_link_func.npz"
data = np.load(data_path)
Y = data["Y"]
B = np.concatenate([data["X_spatial"], np.ones((data["X_spatial"].shape[0], 1))], axis=1)
Z = np.concatenate([data["Z"], np.ones((data["Z"].shape[0], 1))], axis=1)
# Dimensions
_M, _R = Z.shape
_N, _P = B.shape
# re-scale
Z = Z * 50 / _M
B = B * 50 / _N

checkpoint_path = "checkpoint/homogeneous"
results = np.load(checkpoint_path+"/iter_75.npy.npz")
beta, G, C = results["beta"], results["G"], results["C"]

max_iter = 100
gradient_mode = "dask"
preconditioner_mode = "dask"
block_size = 10000
alpha = 0.5
mu_Z, mu_B = None, None
compute_nll = True
nll_mode = "dask"
tol = 1e-10

logging.info(f"Starting from iteration 76")
for iteration in range(76, max_iter):
    G = compute_gradient(Z, B, beta, Y, mode=gradient_mode, block_size=block_size)
    C = compute_preconditioner(Z, B, beta, mu_Z=mu_Z, mu_X=mu_B, 
                                mode=preconditioner_mode, block_size=block_size, damping_factor=1e-1)
    C = eigen_clip(C, min_val=-1e4, max_val=1e4)
    beta_new = beta - alpha * C @ G
    np.savez(f"{checkpoint_path}/iter_{iteration}.npy", beta=beta_new, G=G, C=C)
    delta_beta = np.linalg.norm(beta_new - beta)
    beta = beta_new
    if compute_nll:
        nll = compute_log_poisson_nll(Z, B, beta, Y, mode=nll_mode, block_size=block_size)
        logging.info(f"--> Iteration: {iteration}, delta beta: {delta_beta}, NLL: {nll}")
    else:
        logging.info(f"--> Iteration: {iteration}, delta beta: {delta_beta}")
    if delta_beta < tol:
        logging.info(f"Converged in {iteration + 1} iterations.")
        break



# path = "results/UKB/XTWX_RealDataset_approximate_model_dask_approximate_Poisson_log_link_func_spacing_10.npz"
# XTWX = np.load(path)['XTWX']

# data = np.load("data/UKB/masked_data_RealDataset_approximate_model_Poisson_log_link_func_spacing_10.npz")
# B = np.concatenate([data["X_spatial"], np.ones((data["X_spatial"].shape[0], 1))], axis=1)
# B = B * 100 / B.shape[0]

# C = np.array([1,0,0,0,0,0,0,0])
# CB = np.einsum('i,jk->jik', C.ravel(), B)  # shape: (n_voxel, n_covariates, n_bases)
# CB_flat = CB.reshape(B.shape[0], -1)  # shape: (n_voxel, n_covariates*n_bases)

# XTWX = XTWX + 1e-6 * np.eye(XTWX.shape[0])

# def get_quantile(arr, pcts=[0, 10, 25, 50, 75, 90, 100]):
#     r = np.percentile(arr, pcts)
#     print([f"{pct}: {r[i]}" for i, pct in enumerate(pcts)])

# def robust_inverse(XTWX, eps=1e-8):
#     XTWX = (XTWX + XTWX.T) / 2
#     U, S, VT = np.linalg.svd(XTWX, full_matrices=False)
#     M = (S > eps)
#     S_inv = (S ** -1) * M
#     XTWX_inv = VT.T @ np.diag(S_inv) @ U.T
#     return XTWX_inv

# def robust_inverse_generalised(XTWX, Q, eps=1e-16):
#     if Q.shape[1] != XTWX.shape[0]:
#         raise ValueError("Mismatch in dimensions")
#     XTWX = (XTWX + XTWX.T) / 2
#     U, S, VT = np.linalg.svd(XTWX, full_matrices=False)
#     M = (S > eps)
#     S_inv = S ** -1 # (S ** -1) * M + np.min(S)**(-1) * (1-M)
#     U = ((U + VT.T) / 2) * M[None, :]
#     QU = Q @ U
#     diag_cov = np.sum(QU ** 2 * S_inv, axis=1)
#     return diag_cov

# R = robust_inverse_generalised(XTWX, CB_flat)
# get_quantile(R)

# quit()
# R2 = np.diagonal(np.linalg.pinv(XTWX))

# print(R1.max(), R1.min())
# print(R2.max(), R2.min())
# exit()

# contrast_vector = np.array([1,0,0,0,0,0,0,0])
# B = np.load("data/UKB/masked_data_RealDataset_approximate_model_Poisson_log_link_func_spacing_10.npz")["X_spatial"]
# B = np.concatenate([data["X_spatial"], np.ones((data["X_spatial"].shape[0], 1))], axis=1)
# print(B.shape)
# exit()
# CB = np.einsum('i,jk->jik', contrast_vector, B)  # shape: (n_voxel, n_covariates, n_bases)
# CB_flat = CB.reshape(self._N, -1)  # shape: (n_voxel, n_covariates*n_bases)
# contrast_var_eta = np.sum(CB_flat @ cov_beta_full * CB_flat, axis=1) # shape: (n_voxel,)


# quit()

# data = np.load("/tmp/zby.npz")
# Z, B, Y = [data[i] for i in data.files]
# _M, _R = Z.shape
# _N, _P = B.shape
# block_size = 10000

# Z = Z * 0.01



# beta = 0.01 * np.random.rand(_R * _P, 1)

# eta = kronecker_vector_product(Z/_M, B/_N, beta, use_dask=True, block_size=block_size)
# mu = da.exp(eta)
# with ProgressBar():
#     mu = mu.compute()
# get_quantile(mu)
# XTmu = kronecker_vector_product(Z.T/_M, B.T/_N, mu, use_dask=True, block_size=block_size)
# with ProgressBar():
#     XTmu = XTmu.compute()
# get_quantile(XTmu)
# quit()


# G = compute_gradient(Z, B, beta, Y, mode="dask", block_size=10000)
# # C = compute_preconditioner(Z, B, beta, mode="dask", block_size=10000, damping_factor=1e-2)
# print(G)