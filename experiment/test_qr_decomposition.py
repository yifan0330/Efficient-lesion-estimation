import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular

def poisson_sandwich(
    X,                 # shape (I, n, p)
    y,                 # shape (I, n)
    mu,                # shape (I, n)
    *,
    meat="cluster",    # "cluster" or "iid"
    method="qr",       # "qr" or "svd"
    tol=None,
    ridge=0.0,
    return_diagnostics=False
    ):
    """
    Robust sandwich covariance for Poisson log-link GLM
    with leading dimension I.

    Shapes:
        X:  (I, n, p)
        y:  (I, n)
        mu: (I, n)

    Bread:
        A = sum_i X_i^T W_i X_i

    Meat options:
        "iid":     B = sum_i X_i^T diag((y_i - mu_i)^2) X_i
        "cluster": B = sum_i U_i U_i^T, U_i = X_i^T (y_i - mu_i)

    Returns
    -------
    cov : (p, p)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)

    I, n, p = X.shape

    if y.shape != (I, n) or mu.shape != (I, n):
        raise ValueError("y and mu must have shape (I, n) matching X.")

    if np.any(mu <= 0):
        raise ValueError("All mu must be > 0.")

    # Residuals
    r = y - mu

    # --------------------------------------------------
    # Bread: stack weighted designs across i
    # A = sum_i X_i^T W_i X_i
    # --------------------------------------------------

    # Xw_i = diag(sqrt(mu_i)) X_i
    Xw = X * np.sqrt(mu)[..., None]     # shape (I, n, p)

    # Stack along row dimension
    Xw_stack = Xw.reshape(I * n, p)     # shape (I*n, p)

    # Ridge stabilization
    if ridge > 0:
        Xw_stack = np.vstack([Xw_stack, np.sqrt(ridge) * np.eye(p)])

    # --------------------------------------------------
    # Meat factor C
    # --------------------------------------------------
    meat = meat.lower()

    if meat == "iid":
        # Xr_i = diag(|r_i|) X_i
        Xr = X * np.abs(r)[..., None]     # (I, n, p)
        Xr_stack = Xr.reshape(I * n, p)
        C = Xr_stack.T                   # (p, I*n)
        meat_kind = "iid"

    elif meat == "cluster":
        # U_i = X_i^T r_i
        # B = sum_i U_i U_i^T
        U = np.zeros((p, I))
        for i in range(I):
            U[:, i] = X[i].T @ r[i]
        C = U                             # (p, I)
        meat_kind = "cluster"

    else:
        raise ValueError("meat must be 'iid' or 'cluster'.")

    # --------------------------------------------------
    # Stable solve
    # --------------------------------------------------

    eps = np.finfo(float).eps
    if tol is None:
        tol = np.sqrt(eps)

    diag = {
        "method": method,
        "meat": meat_kind,
        "ridge": ridge,
        "tol": tol,
        "I": I,
        "n": n,
        "p": p
    }

    method = method.lower()

    if method == "qr":

        # Xw_stack = Q R
        Q, R = np.linalg.qr(Xw_stack, mode="reduced")

        if R.shape != (p, p):
            raise RuntimeError("QR failed: not enough total rows relative to p.")

        try:
            from scipy.linalg import solve_triangular
            Z = solve_triangular(R.T, C, lower=True, check_finite=False)
            Y = solve_triangular(R, Z, lower=False, check_finite=False)
            diag["solver"] = "scipy.linalg.solve_triangular"
        except Exception:
            Z = np.linalg.solve(R.T, C)
            Y = np.linalg.solve(R, Z)
            diag["solver"] = "numpy.linalg.solve"

        cov = Y @ Y.T
        cov = 0.5 * (cov + cov.T)

    elif method == "svd":

        U_svd, s, Vt = np.linalg.svd(Xw_stack, full_matrices=False)

        smax = s.max() if s.size else 0.0
        cutoff = tol * smax
        keep = s > cutoff

        inv_s2 = np.zeros_like(s)
        inv_s2[keep] = 1.0 / (s[keep] ** 2)

        temp = Vt @ C
        temp2 = inv_s2[:, None] * temp
        Y = Vt.T @ temp2

        cov = Y @ Y.T
        cov = 0.5 * (cov + cov.T)

        diag.update({
            "rank_kept": int(np.sum(keep)),
            "rank_total": int(len(s)),
            "cutoff": float(cutoff)
        })

    else:
        raise ValueError("method must be 'qr' or 'svd'.")

    if return_diagnostics:
        return cov, diag

    return cov


def poisson_sandwich_kron(
    Z,                 # shape (M, R) - subject covariates
    B,                 # shape (N, P) - spatial bases
    y,                 # shape (M, N)
    mu,                # shape (M, N)
    *,
    meat="cluster",
    ridge=0.0,
    return_diagnostics=False
    ):
    """
    Memory-efficient sandwich covariance for Poisson log-link GLM,
    exploiting  X[i,j,:] = kron(Z[i,:], B[j,:])  (never materialised).

    The full design X would be  (M, N, R*P)  which is ~24 GB for
    typical problem sizes.  This function avoids forming it.

    Bread  A = sum_i X_i^T diag(mu_i) X_i
         Block (k,k') of A = B^T diag(w_{kk'}) B
         where  w_{kk'}[j] = sum_i Z[i,k] Z[i,k'] mu[i,j]

    Cluster meat  C_i = kron(z_i, B^T r_i)
    iid     meat  block structure same as bread but with r^2
    """
    Z  = np.asarray(Z,  dtype=float)
    B  = np.asarray(B,  dtype=float)
    y  = np.asarray(y,  dtype=float)
    mu = np.asarray(mu, dtype=float)

    M, R = Z.shape
    N, P = B.shape
    p = R * P

    if y.shape != (M, N) or mu.shape != (M, N):
        raise ValueError("y and mu must have shape (M, N) matching Z and B.")
    if np.any(mu <= 0):
        raise ValueError("All mu must be > 0.")

    r = y - mu                                          # (M, N)

    # ------------------------------------------------------------------
    # Bread:  A  (p x p)
    # w[k,l,j] = sum_i Z[i,k]*Z[i,l]*mu[i,j]
    # ------------------------------------------------------------------
    w_bread = np.einsum('ik,il,ij->klj', Z, Z, mu)     # (R, R, N)

    A = np.zeros((p, p))
    for k in range(R):
        for k2 in range(k, R):
            block = B.T @ (B * w_bread[k, k2, :, None]) # (P, P)
            A[k*P:(k+1)*P, k2*P:(k2+1)*P] = block
            if k != k2:
                A[k2*P:(k2+1)*P, k*P:(k+1)*P] = block  # symmetric

    if ridge > 0:
        A += ridge * np.eye(p)

    # ------------------------------------------------------------------
    # Meat
    # ------------------------------------------------------------------
    meat_kind = meat.lower()

    if meat_kind == "cluster":
        # U_i = X_i^T r_i = kron(z_i, B^T r_i)
        Bt_r = B.T @ r.T                                # (P, M)
        U = np.zeros((p, M))
        for k in range(R):
            U[k*P:(k+1)*P, :] = Bt_r * Z[:, k][None, :] # (P, M)
        C = U                                            # (p, M)
        Bmeat = None

    elif meat_kind == "iid":
        # Same block structure as bread but weighted by r^2
        w_meat = np.einsum('ik,il,ij->klj', Z, Z, r**2)  # (R, R, N)
        Bmeat = np.zeros((p, p))
        for k in range(R):
            for k2 in range(k, R):
                block = B.T @ (B * w_meat[k, k2, :, None])
                Bmeat[k*P:(k+1)*P, k2*P:(k2+1)*P] = block
                if k != k2:
                    Bmeat[k2*P:(k2+1)*P, k*P:(k+1)*P] = block
        C = None
    else:
        raise ValueError("meat must be 'iid' or 'cluster'.")

    # ------------------------------------------------------------------
    # Solve:  cov = A^{-1} meat A^{-1}
    # ------------------------------------------------------------------
    try:
        L, low = cho_factor(A)
        if meat_kind == "cluster":
            Y = cho_solve((L, low), C)           # A^{-1} U,  (p, M)
            cov = Y @ Y.T
        else:
            D   = cho_solve((L, low), Bmeat)     # A^{-1} Bmeat
            cov = cho_solve((L, low), D.T).T     # (A^{-1} Bmeat) A^{-1}
    except np.linalg.LinAlgError:
        print("Cholesky failed — falling back to pseudo-inverse")
        Ainv = np.linalg.pinv(A)
        if meat_kind == "cluster":
            Y = Ainv @ C
            cov = Y @ Y.T
        else:
            cov = Ainv @ Bmeat @ Ainv

    cov = 0.5 * (cov + cov.T)

    if return_diagnostics:
        diag_info = {
            "method": "kron_cholesky",
            "meat": meat_kind,
            "ridge": ridge,
            "M": M, "N": N, "R": R, "P": P, "p": p,
        }
        return cov, diag_info
    return cov


# ---------------------------
# Example usage / correctness check
# ---------------------------
if __name__ == "__main__":
    data = np.load("debug_sandwich.npz")
    Z, B, Y, MU = data["Z"], data["B"], data["Y"], data["MU"]
    _M, _R = Z.shape
    _N, _P = B.shape

    Y  = Y.reshape(_M, _N)   # (M, N)
    MU = MU.reshape(_M, _N)  # (M, N)

    print(f"Z: {Z.shape}, B: {B.shape}, Y: {Y.shape}, MU: {MU.shape}")
    print(f"Full X would be ({_M}, {_N}, {_R*_P}) = "
          f"{_M * _N * _R * _P * 8 / 1e9:.1f} GB — not materialised")

    # Memory-efficient: never forms the (M, N, R*P) design matrix
    COV, diag = poisson_sandwich_kron(
        Z, B, Y, MU,
        meat="cluster", ridge=0.0001, return_diagnostics=True
    )
    print("Diagnostics:", diag)
    d = np.diag(COV)
    print(f"diag(COV) — min: {np.min(d):.6e}, mean: {np.mean(d):.6e}, max: {np.max(d):.6e}")