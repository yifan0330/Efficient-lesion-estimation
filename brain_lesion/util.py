import numpy as np
import scipy
from tqdm import tqdm
from absl import logging
from scipy.optimize import minimize, line_search
import dask.array as da
from dask.diagnostics import ProgressBar

def kronecker_vector_product(Z, B, beta, use_dask=False, block_size=1000):
    """ Efficient implementation of Kron(Z, B)beta.

    1. Reshape X to [_R, _P]
    2. Compute quadratic form of Z @ beta @ B.T

    Args:
        Z: Matrix of shape [_M, _R]
        B: Matrix of shape [_N, _P]
        beta: Matrix of shape [_R * _P, 1]
    Returns:
        Matrix of shape [_M * _N, 1]
    """
    _M, _R =  Z.shape
    _N, _P =  B.shape
    beta = np.reshape(beta, (_R, _P))
    if use_dask:
        if not isinstance(Z, da.Array):
            Z = da.from_array(Z, chunks=(block_size, _R))
        if not isinstance(B, da.Array):
            B = da.from_array(B, chunks=(block_size, _P))
        if not isinstance(beta, da.Array):
            beta = da.from_array(beta, chunks=(_R, _P))
        ret = (Z @ beta @ B.T).reshape((_M*_N, 1))
        return ret
    return np.einsum('...mr,...rp,...np->...mn', Z, beta, B).reshape((_M*_N, 1))

def compute_gradient(Z, 
                     B, 
                     beta, 
                     Y, 
                     mode="approximate",
                     block_size=1000):
    """
    Z: [_M, _R]
    B: [_N, _P]
    beta: [_R * _P,]
    Y: [_M, _N] binary matrix
    """
    _M, _R = Z.shape
    _N, _P = B.shape
    if scipy.sparse.issparse(Y):
        Y = Y.tocsr()
    G = Y @ B # [_M, _P]
    G = Z.T @ G # [_R, _P]
    XTY = G.reshape((_R * _P, 1)) # [_R * _P, 1]
    if mode == "approximate":
        Z_bar = Z - Z.mean(axis=1, keepdims=True)
        eta_bar = compute_eta_mean(Z, B, beta) # [_N, 1]
        exp_eta_bar = np.exp(eta_bar)
        XTmu = np.kron(Z.T.sum(axis=1, keepdims=True), B.T @ exp_eta_bar) # [_R*_P, 1]
        B_bar = B * exp_eta_bar # [_N, _P]
        XTmu += kronecker_vector_product(Z.T @ Z_bar, B.T @ B_bar, beta) # [_R*_P, 1]
    elif mode == "offload":
        X = np.memmap("/tmp/X.dat", dtype=np.float64, mode="w+", shape=(_M, _N))
        for j in tqdm(range(0, _N, block_size)):
            for i in range(0, _M, block_size):
                i_end = min(i + block_size, _M)
                j_end = min(j + block_size, _N)
                mu = np.exp(kronecker_vector_product(Z[i:i_end, :], B[j:j_end, :], beta))
                X[i:i_end, j:j_end] = mu.reshape((i_end-i, j_end-j))[:, :]
        X.flush()
        XTmu = np.zeros((_R, _P))
        for j in tqdm(range(0, _N, block_size)):
            for i in range(0, _M, block_size):
                i_end = min(i + block_size, _M)
                j_end = min(j + block_size, _N)
                XTmu += Z[i:i_end, :].T @ X[i:i_end, j:j_end] @ B[j:j_end, :] # [_R, _P]
        XTmu = XTmu.reshape((_R*_P, 1))
    elif mode == "exact":
        eta = kronecker_vector_product(Z, B, beta) # [_M*_N, 1]
        XTmu = kronecker_vector_product(Z.T, B.T, np.exp(eta)) # [_R*_P, 1]
    elif mode == "dask":
        mu = da.exp(kronecker_vector_product(Z, B, beta, use_dask=True, block_size=block_size))
        XTmu = kronecker_vector_product(Z.T, B.T, mu, use_dask=True, block_size=block_size)
        with ProgressBar():
            XTmu = XTmu.compute()
    else:
        raise ValueError("Unknown mode = {}".format(mode))
    return -(XTY - XTmu)

def compute_preconditioner(Z, 
                           B, 
                           beta=None,
                           mu_Z=None, 
                           mu_X=None, 
                           mode="approximate",
                           block_size=1000):
    _M, _R = Z.shape
    _N, _P = B.shape
    if mode == "approximate":
        assert mu_Z is not None and mu_X is not None
        ZTWZ = Z.T @ (Z * mu_Z)
        BTWB = B.T @ (B * mu_X)
        ZTWZ_inv = np.linalg.pinv(ZTWZ + 1e-4 * np.eye(ZTWZ.shape[0])) # [_R, _R]
        BTWB_inv = np.linalg.pinv(BTWB + 1e-4 * np.eye(BTWB.shape[0])) # [_P, _P]
        XTWX_inv = np.kron(ZTWZ_inv, BTWB_inv)  # [_R*_P, _R*_P]
        return XTWX_inv
    elif mode == "dask":
        assert beta is not None
        mu = da.exp(kronecker_vector_product(Z, B, beta, use_dask=True, block_size=block_size))
        XTmuX = efficient_kronT_diag_kron(Z, B, mu, use_dask=True, block_size=block_size)
        with ProgressBar():
            XTmuX = XTmuX.compute()
        return np.linalg.pinv(XTmuX + 1e-8 * np.eye(XTmuX.shape[0]))
    elif mode == "exact":
        assert beta is not None
        mu = np.exp(kronecker_vector_product(Z, B, beta))
        ZB = np.kron(Z, B) * (mu**0.5) # [_M*_N, _R*_P]
        XTmuX = ZB.T @ ZB # [_R*_P, _R*_P]
        return np.linalg.pinv(XTmuX + 1e-8 * np.eye(XTmuX.shape[0]))
    else:
      raise ValueError("Unknown mode = {}".format(mode))

def compute_eta_mean(Z, 
                     B, 
                     beta):
    """
    Args:
      Z: [_M, _R]
      B: [_N, _P]
      beta: [_R * _P, 1]
    Returns:
      eta_bar: [_N, 1]
    """
    eta_bar = np.mean(Z, axis=0, keepdims=True)
    eta_bar = kronecker_vector_product(eta_bar, B, beta)
    return eta_bar

def compute_mu_mean(Z, 
                    B, 
                    beta,
                    mode="approximate",
                    block_size=100):
    _M, _R = Z.shape
    _N, _P = B.shape
    if mode == "exact":
        eta = kronecker_vector_product(Z, B, beta)
        mu = np.exp(eta)
        mu_mean = mu.reshape((_M, _N)).mean(axis=0)
        return mu_mean
    elif mode == "dask":
        mu = da.exp(kronecker_vector_product(Z, B, beta, use_dask=True, block_size=block_size))
        mu_mean = mu.reshape((_M, _N)).mean(axis=0)
        with ProgressBar():
            mu_mean = mu_mean.compute()
        return mu_mean
    elif mode == "approximate":
        Z_bar = Z - Z.mean(axis=1, keepdims=True)
        eta_bar = compute_eta_mean(Z, B, beta) # [_N, 1]
        exp_eta_bar = np.exp(eta_bar) # [_N, 1]
        B_bar = B * exp_eta_bar
        Z_bar_mean = Z_bar.mean(axis=0, keepdims=True)
        mu_bar = exp_eta_bar + kronecker_vector_product(Z_bar_mean, B_bar, beta)
        return mu_bar
    elif mode == "offload":
        mu = np.memmap("/tmp/mu.dat", dtype=np.float64, mode="w+", shape=(_M, _N))
        for j in range(0, _N, block_size):
            for i in range(0, _M, block_size):
                i_end = min(i + block_size, _M)
                j_end = min(j + block_size, _N)
                mu[i:i_end, j:j_end] = np.exp(kronecker_vector_product(Z[i:i_end, :], B[j:j_end, :], beta)).reshape((i_end-i, j_end-j))[:, :]
        mu.flush()
        mu_mean = mu.mean(axis=0)
        return mu_mean

def log_poisson_likelihood(lam, Y, use_dask=False, block_size=1000):
    """Compute the log Poisson likelihood.
    Args:
        lam: Array of shape [n_samples, n_features]
        Y: Array of shape [n_samples, n_features]
    Returns:
        NLL: Negative log likelihood
    """
    if use_dask:
        if not isinstance(lam, da.Array):
            lam = da.from_array(lam, chunks=(block_size,))
        if not isinstance(Y, da.Array):
            Y = da.from_array(Y, chunks=(block_size,))
        lam, Y = lam.ravel(), Y.ravel()
        return (da.multiply(Y, da.log(lam)) - lam).mean()
    return (Y.multiply(np.log(lam)) - lam).mean()

def compute_log_poisson_nll(Z, B, beta, Y, mode="exact", block_size=1000):
    _M, _R = Z.shape
    _N, _P = B.shape
    if mode == "exact":
        mu = np.exp(kronecker_vector_product(Z, B, beta))
        nll = -log_poisson_likelihood(mu, Y.reshape(-1, 1))
    elif mode == "dask":
        if not isinstance(Y, da.Array):
            Y = da.from_array(Y, chunks=(block_size, block_size))
        if not isinstance(Z, da.Array):
            Z = da.from_array(Z, chunks=(block_size, _R))
        if not isinstance(B, da.Array):
            B = da.from_array(B, chunks=(block_size, _P))
        mu = da.exp(kronecker_vector_product(Z, B, beta, use_dask=True, block_size=block_size))
        nll = -log_poisson_likelihood(mu, Y, use_dask=True, block_size=block_size)
        nll = nll.compute()
    return nll

def irls_log_glm(X, y, max_iter=50, tol=1e-10, compute_nll=False):
    """IRLS for Log Poisson GLM Regression.
    Args:
        X: Feature array array of shape [n_samples, n_features]
        y: Response array of shape [n_samples,]
        max_iter: Maximum number of iterations
        tol: Tolerance for convergence
    Returns:
        beta: Estimated coefficients
    """
    n_samples, n_features = X.shape
    logging.info("-" * 50)
    logging.info("IRLS for Log Poisson GLM")
    logging.info(f"n_samples: {n_samples}, n_features: {n_features}")
    logging.info("-" * 50)
    beta = np.zeros((n_features,))
    for iteration in range(max_iter):
        eta = X.dot(beta)
        mu = np.exp(eta)
        z = eta + (y - mu) / mu
        XTmuX = X.T.dot(mu[:, None] * X)
        XTmuX = XTmuX + 1e-8 * np.eye(n_features)
        XTmuz = X.T.dot(mu * z)
        beta_new = np.linalg.solve(XTmuX, XTmuz)
        delta_beta = np.linalg.norm(beta_new - beta)
        beta = beta_new
        if compute_nll:
            nll = -log_poisson_likelihood(np.exp(X.dot(beta)), y)
            logging.info(f"--> Iteration: {iteration}, delta beta: {delta_beta}, NLL: {nll}")
        else:
            logging.info(f"--> Iteration: {iteration}, delta beta: {delta_beta}")
        if delta_beta < tol:
            logging.info(f"Converged in {iteration + 1} iterations.")
            break
    return beta

def fit_additive_log_glm(Z, B, Y, mode="approximate"):
    _M, _R = Z.shape
    _N, _P = B.shape
    if scipy.sparse.issparse(Y):
        Y = Y.tocsr()
    if mode == "approximate":
        beta, gamma = np.zeros((_P,)), np.zeros((_R,))
        Y_Z = np.array(Y.mean(axis=1)).reshape(-1,)
        Y_B = np.array(Y.mean(axis=0)).reshape(-1,)
        def objective(params):
            beta = params[:_P]
            gamma = params[_P:]
            B_beta = B @ beta
            Z_gamma = Z @ gamma
            l = (Y_B * B_beta).mean() + (Y_Z * Z_gamma).mean() - np.exp(B_beta).mean() * np.exp(Z_gamma).mean()
            return -l
        def jac(params):
            beta = params[:_P]
            gamma = params[_P:]
            B_beta = B @ beta
            Z_gamma = Z @ gamma
            d_beta = B.T @ Y_B / float(_N) - B.T @ np.exp(B_beta) / float(_N) * np.exp(Z_gamma).mean()
            d_gamma = Z.T @ Y_Z / float(_M) - Z.T @ np.exp(Z_gamma) / float(_M) * np.exp(B_beta).mean()
            return -np.concatenate([d_beta, d_gamma])
        res = minimize(fun=objective, jac=jac, x0=np.concatenate([beta, gamma]), 
                    method="L-BFGS-B", options={"disp": True})
        beta = res.x[:_P]
        gamma = res.x[_P:]
        return beta, gamma
    elif mode == "exact":
        raise NotImplementedError

def eigen_clip(M, min_val=0.1, max_val=10.0):
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals_clipped = np.clip(eigvals, min_val, max_val)
    return eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T

def fit_multiplicative_log_glm(Z, B, Y, tol=1e-10, max_iter=100, 
                               alpha=1.0,
                               gradient_mode="approximate", 
                               preconditioner_mode="approximate",
                               nll_mode="dask",
                               block_size=1000,
                               compute_nll=False):
    """Fit Multiplicative Log GLM.
    Args:
        Z: Matrix of shape [_M, _R]
        B: Matrix of shape [_N, _P]
        Y: binary matrix of shape [_M, _N]
        tol: Tolerance for convergence
        max_iter: Maximum number of iterations
    Returns:
        beta: Estimated coefficients
    """
    _M, _R = Z.shape
    _N, _P = B.shape
    logging.info("-" * 50)
    logging.info("Multiplicative Log Poisson GLM")
    logging.info(f"n_subject: {_M}, n_voxel: {_N}, n_covariates: {_R}, n_basis: {_P}")
    logging.info("-" * 50)
    if preconditioner_mode == "approximate":
        beta_B, beta_Z = fit_additive_log_glm(Z, B, Y, mode=preconditioner_mode)
        mu_Z = np.exp(Z @ beta_Z)[:, None]
        mu_B = np.exp(B @ beta_B)[: , None]
    else:
        mu_Z, mu_B = None, None
    beta = np.zeros((_R * _P, 1))
    for iteration in range(max_iter):
        G = compute_gradient(Z, B, beta, Y, mode=gradient_mode, block_size=block_size)
        C = compute_preconditioner(Z, B, beta, mu_Z=mu_Z, mu_X=mu_B, mode=preconditioner_mode, block_size=block_size)
        beta_new = beta - alpha * C @ G
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
    return beta

def efficient_kronT_diag_kron(Z, B, d, use_dask=False, block_size=1000):
    """
    Efficiently computes kron(Z, B)^T @ diag(d) @ kron(Z, B)

    Args:
      Z : array of shape (_M, _R)
          The first matrix.
      B : array of shape (_N, _P)
          The second matrix.
      d : array of length (_M * _N)
          The diagonal entries of D, which will be reshaped to (_M, _N).
      use_dask : bool
          Whether to use dask for computation.
      block_size : int
          The block size for dask computation.

    Returns:
      result : array of shape (_R * _P, _R * _P)
    """
    _M, _R = Z.shape
    _N, _P = B.shape
    if use_dask:
        if not isinstance(Z, da.Array):
            Z = da.from_array(Z, chunks=(block_size, _R))
        if not isinstance(B, da.Array):
            B = da.from_array(B, chunks=(block_size, _P))
        d_reshaped = d.reshape((_M, _N))
        M = da.einsum('ij,jr,js->irs', d_reshaped, B, B)  
        result_blocks = da.einsum('ik,il,irs->klrs', Z, Z, M)
        result = result_blocks.transpose(0, 2, 1, 3).reshape(_R * _P, _R * _P)
        return result
    d_reshaped = d.reshape((_M, _N))
    M = np.einsum('ij,jr,js->irs', d_reshaped, B, B)  
    result_blocks = np.einsum('ik,il,irs->klrs', Z, Z, M)
    result = result_blocks.transpose(0, 2, 1, 3).reshape(_R * _P, _R * _P)
    return result


