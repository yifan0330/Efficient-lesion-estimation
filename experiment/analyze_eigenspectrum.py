import numpy as np
import matplotlib.pyplot as plt
import gc
import time

N = 1400 # number of voxels
M = 1300 # number of subjects
P = 80 # number of spatial features
R = 4 # number of covariates

rng = np.random.default_rng(42)

Z = np.random.rand(M, R)
B = np.random.rand(N, P)
beta = np.random.uniform(-1, 0, size=R * P)
Y = np.random.poisson(0.01, size=(M, N))

X = np.kron(Z, B).reshape(M, N, R*P) # shape (M*N, R*P)
X_reshaped = X.reshape(M*N, R*P)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def robust_inverse(XTWX, eps=1e-8):
    XTWX = (XTWX + XTWX.T) / 2
    U, S, VT = np.linalg.svd(XTWX, full_matrices=False)
    # choose the threshold that at least 50% eigenvalues are kept
    eps = min(np.median(S), eps)
    M = (S > eps)
    # S_inv = (S ** -1) * M
    # XTWX_inv = VT.T @ np.diag(S_inv) @ U.T
    S_inv = S ** -1
    U = ((U + VT.T) / 2) * M[None, :]
    XTWX_inv = U @ np.diag(S_inv) @ U.T
    return XTWX_inv

mu = np.exp(X @ beta) # shape (M*N,)

# Meat term
W = Y - mu
BW = W.dot(B)    # shape (M, P)
T = (Z[:, :, None] * BW[:, None, :]).reshape(M, P * R)  # shape (M, R*P)
Meat = T.T.dot(T)   # shape (R*P, R*P)
del W, BW, T
gc.collect()
# eigendecomposition: Q^T * diag(Lambda) * Q = Meat
Lambda, Q = np.linalg.eigh(Meat) # dimensions: (R*P,), (R*P, R*P)

# Bread term
S = X * mu[..., np.newaxis]
Bread = (S.transpose(0, 2, 1) @ S).sum(axis=0)  # sum over subjects -> (R*P, R*P)
Bread_inv = np.linalg.pinv(Bread)

# Sandiwch estimator: Cov(beta) = B^-1 M B^-1
# Eigendecomposition of the meat term: Q^T * diag(Lambda) * Q = Meat
# Cov(beta) = (Q^T diag(Lambda)^-1/2) (diag(Lambda)^(1/2) Q B^-1 Q^T diag(Lambda)^(-1/2))^-1 (diag(Lambda)^-1/2 Q B Q^T diag(Lambda)^(-1/2))^-1 (diag(Lambda)^-1/2 Q)

d = 1.0 / np.sqrt(Lambda) 
QBQT = Q @ Bread @ Q.T
C = d[:, None] * QBQT * d[None, :] # shape (R*P, R*P)
C_inv = np.linalg.pinv(C) # shape (R*P, R*P)

def analyze_eigenspectrum(H, save_to=None):
    eigenvalues = np.linalg.eigvalsh(H)
    eigenvalues = np.sort(np.real(eigenvalues))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    indices = np.arange(eigenvalues.size)
    axes[0].plot(
        indices,
        eigenvalues,
        color='tab:blue',
        linewidth=1.6,
        marker='o',
        markersize=3,
        alpha=0.85,
    )
    axes[0].set_title('Eigenspectrum (sorted, ascending)')
    axes[0].set_xlabel('Index')
    axes[0].set_ylabel('Eigenvalue')
    axes[0].grid(True, linestyle='--', alpha=0.4)

    abs_eigs = np.maximum(np.abs(eigenvalues), 1e-16)
    axes[1].semilogy(indices, abs_eigs, color='tab:purple', linewidth=1.6)
    axes[1].set_title('Magnitude Spectrum (log scale)')
    axes[1].set_xlabel('Index')
    axes[1].set_ylabel('|Eigenvalue|')
    axes[1].grid(True, which='both', linestyle='--', alpha=0.4)

    fig.suptitle('Bread Matrix Eigenvalue Analysis', fontsize=13)
    fig.tight_layout()

    if save_to:
        fig.savefig(save_to, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
        
analyze_eigenspectrum(Bread, save_to="eigenspectrum_bread.png")
analyze_eigenspectrum(C, save_to="eigenspectrum_C.png")