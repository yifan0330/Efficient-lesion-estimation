import os
import numpy as np
import scipy 
from util import * 
from absl import logging
import dask

dask.config.set({"distributed.worker.nthreads": 1})  # Threads per worker
dask.config.set({"distributed.workers": os.cpu_count()})  # Number of workers

logging.set_verbosity(logging.INFO)

# data = np.load("results/1D_Probability_comparison_Simulation_Homogeneous_2_group_Poisson_log_link_func.npz", allow_pickle=True)
# MU = data['MU']#[:1200]
# Y = data['Y']#[:1200]
# Y = Y.astype(bool)
# Z = data['Z']#[:1200]
# B = data['X_spatial']

Y = np.random.randint(0, 2, size=(15000, 220000), dtype=bool)
Z = np.random.randn(15000, 5)
B = np.random.randn(220000, 100)

Z = np.concatenate([Z, np.ones((Z.shape[0], 1))], axis=1)
B = np.concatenate([B, np.ones((B.shape[0], 1))], axis=1)

_M, _R = Z.shape
_N, _P = B.shape
logging.info("-" * 50)
logging.info(f"Y Shape (Sparse): {Y.shape}")
logging.info(f"Z Shape: {Z.shape}")
logging.info(f"B Shape: {B.shape}")
logging.info("-" * 50)
############################################



# beta_exact = fit_multiplicative_log_glm(Z, B, Y, max_iter=20, 
#                                   gradient_mode="offload",
#                                   preconditioner_mode="exact",
#                                   compute_nll=True)
# mu_bar_exact = compute_mu_mean(Z, B, beta_exact, mode="dask")

# beta_dask = fit_multiplicative_log_glm(Z, B, Y, max_iter=20, 
#                                   gradient_mode="dask",
#                                   preconditioner_mode="dask",
#                                   block_size=10000,
#                                   compute_nll=True)
# mu_bar_dask = compute_mu_mean(Z, B, beta_dask, mode="dask")

beta_approx_pre = fit_multiplicative_log_glm(Z, B, Y, max_iter=100, 
                                  gradient_mode="dask",
                                  preconditioner_mode="approximate",
                                  block_size=10000,
                                  compute_nll=True)
mu_bar_approximate_preconditioner = compute_mu_mean(Z, B, beta_approx_pre, mode="dask")

beta_approx_full = fit_multiplicative_log_glm(Z, B, Y, max_iter=100, 
                                  alpha=0.1, # use a smaller step size for stability
                                  gradient_mode="approximate",
                                  preconditioner_mode="approximate",
                                  compute_nll=True)
mu_bar_approximate = compute_mu_mean(Z, B, beta_approx_full, mode="dask")

# print(np.linalg.norm(mu_bar_dask - mu_bar_exact) / np.linalg.norm(mu_bar_exact))
# print(np.linalg.norm(mu_bar_approximate_preconditioner - mu_bar_exact) / np.linalg.norm(mu_bar_exact))
print(np.linalg.norm(mu_bar_approximate - mu_bar_exact) / np.linalg.norm(mu_bar_exact))

quit()

Y = da.from_array(Y, chunks=(1000, 1000))
mu = da.exp(kronecker_vector_product(Z, B, beta_dask, use_dask=True, block_size=1000))
d = (Y.reshape(-1, 1) - mu)
M = efficient_kronT_diag_kron(Z, B, d, use_dask=True, block_size=1000)
with ProgressBar():
    M = M.compute()
print(M.shape)



