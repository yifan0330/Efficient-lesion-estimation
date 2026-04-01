import numpy as np
from time import time
from tqdm import tqdm 
import gc

begin = time()

_M, _N = 1000, 2000
_R, _P = 1, 103

Z = np.random.normal(0.0, 1.0, size=(_M, _R))
X = np.random.normal(0.0, 1.0, size=(_N, _P))
beta_w = np.random.normal(0.0, 1.0, size=(_P, _R))
P = np.random.normal(0.0, 1.0, size=(_M, _N))
cov_beta_w = np.random.normal(0.0, 1.0, size=(_P * _R, _P * _R))

def compute_covariance(Z, X, beta_w, P, cov_beta_w):
    unstacked_cov_beta_w  = np.stack(np.split(cov_beta_w, _P, axis=-1))
    unstacked_cov_beta_w = np.stack(np.split(unstacked_cov_beta_w, _P, axis=-2)) # [_P, _P, _R, _R]

    A = Z @ beta_w.T # [_M, _P]
    eta = A @ X.T # [_M, _N]
    P = np.exp(eta) # [_M, _N]

    cov_A = unstacked_cov_beta_w @ Z.T[None, None, :, :] # [_P, _P, _R, _M]
    cov_A = np.sum(cov_A * Z.T[None, None, :, :], axis=-2) # [_P, _P, _M]
    cov_A = np.moveaxis(cov_A, -1, 0) # [_M, _P, _P]
    cov_eta = X[None, :, :] @ cov_A @ X.T[None, :, :] # [_M, _N, _N]
    cov_P = cov_eta * P[:, :, None] * P[:, None, :] # [_M, _N, _N]
    del unstacked_cov_beta_w, A, P, eta, cov_A, cov_eta
    gc.collect()
    return cov_P

def batch_compute_covariance(Z, X, beta_w, P, cov_beta_w, batch_size=20):
    _M = Z.shape[0]
    _N = X.shape[0]
    split_indices = np.arange(0, _M, batch_size)
    cov_P = np.memmap("cov_P.dat", mode="w+", shape=(_M, _N, _N))
    for left_index in tqdm(split_indices, total=len(split_indices)):
        right_index = min(left_index + batch_size, _M)
        Z_i = Z[left_index:right_index]
        P_i = P[left_index:right_index]
        cov_P_i = compute_covariance(Z_i, X, beta_w, P_i, cov_beta_w)
        cov_P[left_index:right_index] = cov_P_i
        del Z_i, P_i, cov_P_i
        gc.collect()
    return cov_P
  
cov_P = batch_compute_covariance(Z, X, beta_w, P, cov_beta_w, batch_size=20)
cov_P.flush()
print(cov_P.shape)

print(time() - begin)