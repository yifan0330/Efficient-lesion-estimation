"""Inference module for brain lesion estimation models.

Provides classes for statistical inference (Wald tests, sandwich estimators,
Fisher information) on spatial GLMs and mass-univariate models.
"""

import gc
import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
from statsmodels.stats.multitest import fdrcorrection
from tqdm import tqdm

from model import MassUnivariateRegression, SpatialBrainLesionModel
from plot import plot_brain, save_nifti
from util import (
    compute_mu,
    efficient_kronT_diag_kron,
    eigenspectrum,
    robust_inverse,
    robust_inverse_generalised,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Helper: QQ-plot (shared by multiple inference classes)
# ---------------------------------------------------------------------------

def plot_qq(p_vals, filename, significance_level=0.05,
            title_list=("group_0 - group_1", "group_1 - group_0")):
    """QQ-plot of observed vs expected -log10 p-values with Beta CIs.

    Parameters
    ----------
    p_vals : np.ndarray, shape (M, N)
        P-values, one row per contrast direction.
    filename : str
        Output figure path.
    significance_level : float
        Significance level for CI bands and threshold.
    title_list : tuple of str
        Subplot titles.
    """
    M, N = p_vals.shape
    th_p = np.arange(1 / float(N), 1 + 1 / float(N), 1 / float(N))
    th_p_log = -np.log10(th_p)
    k_array = np.arange(1, N + 1)
    CI_lower = scipy.stats.beta.ppf(significance_level / 2, k_array, N - k_array + 1)
    CI_upper = scipy.stats.beta.ppf(1 - significance_level / 2, k_array, N - k_array + 1)

    fig, axes = plt.subplots(1, M, figsize=(12 * M, 11))
    if M == 1:
        axes = [axes]
    for i in range(M):
        sorted_p = np.sort(p_vals[i])
        pct_rejected = np.sum(sorted_p < 0.05) / N
        ax = axes[i]
        ax.fill_between(
            th_p_log, -np.log10(CI_lower), -np.log10(CI_upper),
            color="grey", alpha=0.5,
            label=f"{int((1 - significance_level) * 100)}% Beta CI",
        )
        ax.plot(th_p_log, np.full(N, -np.log10(0.05)), "y--",
                label="threshold at -log10(0.05)")
        ax.plot(th_p_log, -np.log10(th_p), color="orange", ls="--", label="y=x")
        ax.plot(th_p_log, -np.log10(significance_level * th_p), "r-",
                label="FDR(BH) control")
        ax.scatter(th_p_log, -np.log10(sorted_p), c="#1f77b4", s=4)
        lim = np.max(-np.log10(k_array / N))
        ax.set(xlim=[0, lim], ylim=[0, lim],
               xlabel="Expected -log10(P)", ylabel="Observed -log10(P)")
        ax.set_title(
            f"{title_list[i]}: {pct_rejected * 100:.2f}% voxels rejected",
            fontsize=20,
        )
        ax.legend()
    fig.savefig(filename)
    plt.close(fig)


# ---------------------------------------------------------------------------
#  Full-model inference
# ---------------------------------------------------------------------------

class BrainInference_full:
    """Inference for the full (non-approximate) spatial brain-lesion model."""

    def __init__(self, model, space_dim, marginal_dist, link_func,
                 regression_terms, random_seed, fewer_voxels=False,
                 dtype=torch.float64, device='cpu'):
        self.model = model
        self.space_dim = space_dim
        self.marginal_dist = marginal_dist
        self.link_func = link_func
        self.regression_terms = regression_terms
        self.random_seed = random_seed
        self.fewer_voxels = fewer_voxels
        self.dtype = dtype
        self.device = device
        self._kwargs = {"device": self.device, "dtype": self.dtype}
    
    def load_params(self, data, params):
        """Load spatial bases, covariates, outcomes, and fitted parameters."""
        self.group_names = list(data.keys())
        self.n_group = len(self.group_names)

        X_spatial = data["X_spatial"]
        self.X_spatial_array = np.concatenate(
            [X_spatial, np.ones((X_spatial.shape[0], 1))], axis=1,
        )
        self.X_spatial = torch.tensor(self.X_spatial_array, **self._kwargs)
        self.n_subject_per_group = data["Z"].shape[0]

        P = params["P"]
        self.P_mean = np.stack(
            [np.mean(P_group, axis=0) for P_group in P], axis=0,
        )  # (n_group, n_voxel)
        self.eta = np.log(self.P_mean)

        self.Y = torch.tensor(data["Y"], **self._kwargs)
        intercept_col = np.ones((data["Z"].shape[0], 1))
        self.Z = torch.tensor(
            np.concatenate([data["Z"], intercept_col], axis=1), **self._kwargs,
        )

        self.n_subject, self.n_covariates = self.Z.shape
        self.beta = torch.tensor(params["beta"], **self._kwargs)
        self.beta_array = params["beta"]
        # spatial coefficient dimension
        self.n_voxel, self.n_bases = self.X_spatial.shape

    def create_contrast(self, contrast_vector=None, contrast_name=None):
        """Build and normalise the contrast vector for hypothesis testing."""
        self.contrast_vector = contrast_vector
        self.contrast_name = contrast_name
        self.contrast_vector = (
            np.eye(self.n_covariates)
            if contrast_vector is None
            else np.array(contrast_vector).reshape(1, -1)
        )
        if self.contrast_vector.shape[1] != self.n_covariates:
            raise ValueError(
                f"Contrast vector shape {self.contrast_vector.shape} "
                f"doesn't match number of covariates ({self.n_covariates})."
            )
        self.contrast_vector = (
            self.contrast_vector
            / np.sum(np.abs(self.contrast_vector), axis=1, keepdims=True)
        )

    def run_inference(self, method="FI", inference_filename=None,
                      fig_filename=None, lesion_mask=None, alpha=0.05):
        """Run generalised linear hypothesis test and plot results."""
        z_threshold = scipy.stats.norm.ppf(1 - alpha)
        if not os.path.exists(inference_filename):
            p_vals, z_stats = self._glh_con_group(method)
            np.savez(inference_filename, p_vals=p_vals, z_stats=z_stats)
        else:
            p_vals = np.load(inference_filename)["p_vals"]
        logger.info("p_vals shape: %s", p_vals.shape)
        logger.info("Plotting inference results to %s", fig_filename)
        plot_brain(
            p=z_stats, brain_mask=lesion_mask,
            threshold=z_threshold, output_filename=fig_filename,
        )

    def __glh_con_group(self, method, batch_size=20):
        all_bar_Z = {}
        for group in self.group_names:
            all_bar_Z[group] = np.mean(self.Z[group], axis=0) # shape: (n_covariates,)
    
        n_subject_list = list(self.n_subject.values())
        group_ratio = n_subject_list / np.max(n_subject_list)
        # self.contrast_vector /= group_ratio
        contrast_eta = self.contrast_vector @ self.eta # shape: (1, n_voxel)
        # Estimate the variance of beta, from either FI or sandwich estimator
        if method == "FI":
            all_F_beta = self._Fisher_info() # shape: (n_covariates_expand, n_covariates_expand)
            all_cov_beta = {}
            for group in self.group_names:
                F_beta = all_F_beta[group]
                cov_beta = [np.linalg.inv(F_beta[:,i,:,i]+1e-6*np.eye(self.n_bases)) for i in range(self.n_covariates[group])]
                all_cov_beta[group] = cov_beta
                del F_beta
            del all_F_beta
        elif method == "sandwich":
            bread_term = self.bread_term(self.Z, self.X_spatial_array, self.P) # list: len = n_covariates
            meat_term = self.meat_term(self.Z, self.X_spatial_array, self.P, self.Y) # list: len = n_covariates
            # Sandwich estimator
            cov_beta = [B @ M @ B.T for B, M in zip(bread_term, meat_term)]
            del bread_term, meat_term
        logger.info("Variance of beta computed")

        all_var_bar_eta = {}
        for group in self.group_names:
            var_bar_eta = list()
            for s in range(self.n_covariates[group]):
                # for covariate s, at voxel j
                # bar_eta_sj = bar_Z_s * X_j^T @ beta_s -- dim: (1,)
                # COV(bar_eta_sj) = bar_Z_s * X_j^T @ COV(beta_s) @ X_j -- dim: (1,)
                # COV(bar_eta_s) = bar_P_s**2 * X @ COV(beta_s) @ X^T -- dim: (n_voxel, n_voxel)
                var_bar_eta_s = all_bar_Z[group][s] * np.einsum('ij,jk,ik->i', self.X_spatial_array, all_cov_beta[group][s], self.X_spatial_array)
                var_bar_eta.append(var_bar_eta_s)
            var_bar_eta = np.stack(var_bar_eta, axis=0) # shape: (n_covariate, n_voxel)
            all_var_bar_eta[group] = var_bar_eta
            del var_bar_eta
        del all_cov_beta
        # Compute the numerator of the Z test

        a = np.concatenate([np.sum(all_var_bar_eta[group], axis=0).reshape(1, -1) for group in self.group_names], axis=0) # shape: (n_group, n_voxel)
        logger.debug("Aggregated variance shape: %s", a.shape)

        contrast_var_bar_eta = self.contrast_vector**2 @ a # shape: (1, n_voxel)
        contrast_std_bar_eta = np.sqrt(contrast_var_bar_eta) # shape: (1, n_voxel)
        # Conduct Wald test (Z test)
        z_stats_eta = contrast_eta / contrast_std_bar_eta
        z_stats = np.concatenate([z_stats_eta, -z_stats_eta], axis=0) # shape: (2, n_voxel)
        p_vals = scipy.stats.norm.sf(z_stats)  # (2, n_voxel)
        logger.info(
            "p-values: min=%.4g, max=%.4g, significant=%d, shape=%s",
            np.min(p_vals), np.max(p_vals),
            np.count_nonzero(p_vals < 0.05), p_vals.shape,
        )
        return p_vals, z_stats

    def _Fisher_info(self):
        """Compute or load cached Fisher information matrix."""
        Fisher_info_filename = (
            f"{os.getcwd()}/results/{self.space_dim}/GRF_{self.n_subject}/"
            f"{self.model}_{self.marginal_dist}_{self.link_func}/Fisher_info_{self.random_seed}.npz"
        )
        os.makedirs(os.path.dirname(Fisher_info_filename), exist_ok=True)
        if os.path.exists(Fisher_info_filename):
            all_H = np.load(Fisher_info_filename, allow_pickle=True)
            # Convert NpzFile to dictionary
            all_H = {group: all_H[group] for group in all_H.files}
        else:
            # Load Y, Z
            start_time = time.time()
            # Compute the Fisher information matrix
            if self.model == "SpatialBrainLesion":
                nll = lambda beta: SpatialBrainLesionModel._neg_log_likelihood(
                                                                            self.marginal_dist,
                                                                            self.link_func,
                                                                            self.regression_terms,
                                                                            self.X_spatial,
                                                                            self.Y,
                                                                            self.Z,
                                                                            beta,
                                                                            self.device)
                params = (self.beta)
                # Hessian
                H = torch.autograd.functional.hessian(nll, params, create_graph=False)
            elif self.model == "MassUnivariateRegression":
                beta_age = self.beta[:, 2]
                # remove beta_age from beta
                beta_other = self.beta.clone()
                beta_other[:, 2] = 0.0
                # beta_age
                nll = lambda beta: MassUnivariateRegression._neg_log_likelihood(
                                                                            self.marginal_dist,
                                                                            self.link_func,
                                                                            self.regression_terms,
                                                                            self.X_spatial,
                                                                            self.Y,
                                                                            self.Z,
                                                                            beta,
                                                                            beta_other,
                                                                            self.device)
                params = (beta_age)
                # Hessian
                H = torch.autograd.functional.hessian(nll, params, create_graph=False)
            logger.info("Fisher information computed in %.1fs", time.time() - start_time)

        return H.detach().cpu().numpy()

    def bread_term(self, Z, X_spatial, P):
        """Compute bread term (inverse Hessian blocks) of sandwich estimator."""
        start_time = time.time()
        A = np.einsum('ia,ib->iab', Z, Z)  # (n_subject, n_cov, n_cov)
        B = np.einsum('ij,jk,jl->ikl', P, X_spatial, X_spatial)  # (n_subject, n_bases, n_bases)
        H = np.einsum('iab,icd->acbd', A, B)  # (n_cov, n_bases, n_cov, n_bases)
        bread_term = [
            np.linalg.inv(H[i, :, i, :] + 1e-6 * np.eye(self.n_bases))
            for i in range(self.n_covariates)
        ]
        del H, Z, A, B
        gc.collect()
        logger.info("Bread term computed in %.1fs", time.time() - start_time)
        return bread_term
    
    def meat_term(self, Z, X_spatial, P, Y):
        """Compute meat term of sandwich estimator."""
        start_time = time.time()
        R = Y - P  # (n_subject, n_voxel)
        L = np.dot(R, X_spatial)  # (n_subject, n_bases)
        V = [Z[:, i, None] * L for i in range(self.n_covariates)]
        meat_term = [Vi.T @ Vi for Vi in V]
        del R, L, V
        gc.collect()
        logger.info("Meat term computed in %.1fs", time.time() - start_time)
        return meat_term
    
    def batch_compute_covariance(self, var_P, Z, X, P, cov_beta_w, batch_size=20):
        n_subject = Z.shape[0]
        split_indices = np.arange(0, n_subject, batch_size)
        for left_index in tqdm(split_indices, total=len(split_indices)):
            right_index = min(left_index + batch_size, n_subject)
            Z_i = Z[left_index:right_index]
            P_i = P[left_index:right_index]
            var_P_i = self.compute_covariance(Z_i, X, P_i, cov_beta_w)
            var_P[left_index:right_index] = var_P_i[:]
            var_P.flush()
            del Z_i, P_i, var_P_i
            gc.collect()

    def compute_covariance(self, Z, X, P, cov_beta_w):
        """Compute variance of P from coefficient covariance."""
        unstacked = np.stack(np.split(cov_beta_w, self.n_bases, axis=-1))
        unstacked = np.stack(np.split(unstacked, self.n_bases, axis=-2))  # [_P, _P, _R, _R]
        cov_A = unstacked @ Z.T[None, None, :, :]
        cov_A = np.sum(cov_A * Z.T[None, None, :, :], axis=-2)
        cov_A = np.moveaxis(cov_A, -1, 0)  # (n_batch, n_bases, n_bases)
        var_eta = np.einsum('np,mpq,nq->mn', X, cov_A, X)  # (n_batch, n_voxel)
        var_P = P ** 2 * var_eta
        del unstacked, cov_A, var_eta
        gc.collect()
        return var_P
    
    def plot_1d(self, p_vals, filename, significance_level=0.05):
        """QQ-plot of p-values. Delegates to module-level ``plot_qq``."""
        plot_qq(p_vals, filename, significance_level)


# ---------------------------------------------------------------------------
#  Approximate-model inference
# ---------------------------------------------------------------------------

class BrainInference_Approximate:
    """Inference for the approximate (Kronecker-structured) spatial GLM."""

    def __init__(self, model, marginal_dist, link_func, regression_terms,
                dtype=torch.float64, device='cpu'):
        self.model = model
        self.marginal_dist = marginal_dist
        self.link_func = link_func
        self.regression_terms = regression_terms
        self.dtype = dtype
        self.device = device
    
    def load_params(self, data, params):
        """Load spatial bases, group info, covariates, and fitted parameters."""
        self.G = data["G"].item()
        self.group_names = list(self.G.keys())
        self.n_subject_per_group = [len(self.G[group]) for group in self.group_names]
        self.n_group = len(self.group_names)

        self.B = np.concatenate([data["X_spatial"], np.ones((data["X_spatial"].shape[0], 1))], axis=1)
        self.Z = np.concatenate([data["Z"], np.ones((data["Z"].shape[0], 1))], axis=1)
        self._M, self._R = self.Z.shape
        self._N, self._P = self.B.shape

        self.beta = params["beta"]
        self.MU = compute_mu(self.Z, self.B, self.beta, mode="dask", block_size=1000)
        self.Y = data["Y"]  # (n_subject, n_voxel)
        self.Y_reshape = self.Y.reshape(-1, 1)
    
    def create_contrast(self, contrast_vector=None, contrast_name=None):
        """Build and normalise the contrast vector."""
        self.contrast_vector = contrast_vector
        self.contrast_name = contrast_name
        self.contrast_vector = (
            np.eye(self.n_group)
            if contrast_vector is None
            else np.array(contrast_vector).reshape(1, -1)
        )
        if self.contrast_vector.shape[1] != self._R:
            raise ValueError(
                f"Contrast vector shape {self.contrast_vector.shape} "
                f"doesn't match number of covariates ({self._R})."
            )
        self.contrast_vector = (
            self.contrast_vector
            / np.sum(np.abs(self.contrast_vector), axis=1, keepdims=True)
        )
        
    def run_inference(self, method="FI", fig_filename=None):
        """Run hypothesis test and optionally produce a QQ-plot."""
        p_vals = self._glh_con_group(method)
        logger.info("Output figure: %s", fig_filename)
        if fig_filename is not None:
            self.plot_1d(p_vals, fig_filename, 0.05)

    def _glh_con_group(self, method, use_dask=True, batch_size=20):
        """Generalised linear hypothesis test (Wald / sandwich)."""
        bar_Z = np.mean(self.Z, axis=0)  # (n_covariates,)
        # Scale contrast by group size
        group_n_subjects = bar_Z.reshape(1, -1)[:, -self.n_group - 1:-1]
        group_ratio = group_n_subjects / np.max(group_n_subjects)
        self.contrast_vector[:, -self.n_group - 1:-1] /= group_ratio

        beta_reshape = self.beta.reshape(self._P, self._R, order="F")
        bar_eta_covariates = (bar_Z * beta_reshape).T @ self.B.T  # (n_cov, n_voxel)
        contrast_eta = self.contrast_vector @ bar_eta_covariates  # (1, n_voxel)
        del bar_eta_covariates

        # Covariance of beta
        start_time = time.time()
        if method == "FI":
            F = efficient_kronT_diag_kron(
                self.Z, self.B, self.MU, use_dask=use_dask, block_size=1e4,
            )
            cov_beta = [
                robust_inverse(F[i * self._P:(i + 1) * self._P,
                                 i * self._P:(i + 1) * self._P]
                               + 1e-6 * np.eye(self._P))
                for i in range(self._R)
            ]
            del F
            logger.info("Fisher information computed in %.1fs", time.time() - start_time)
        elif method == "sandwich":
            bread = self.bread_term(self.Z, self.B, self.MU)
            meat = self.meat_term(self.Z, self.B, self.MU, self.Y_reshape)
            cov_beta = [B @ M @ B for B, M in zip(bread, meat)]
            del bread, meat
            logger.info("Sandwich estimator computed in %.1fs", time.time() - start_time)
        logger.info("Variance of beta computed")

        # Variance of linear predictor per covariate
        var_bar_eta = np.stack([
            bar_Z[s] * np.einsum('ij,jk,ik->i', self.B, cov_beta[s], self.B)
            for s in range(self._R)
        ], axis=0)  # (n_cov, n_voxel)
        del cov_beta
        gc.collect()

        # Wald test
        contrast_var = self.contrast_vector ** 2 @ var_bar_eta  # (1, n_voxel)
        contrast_std = np.sqrt(contrast_var)
        z_stats_eta = contrast_eta / contrast_std
        z_stats = np.concatenate([z_stats_eta, -z_stats_eta], axis=0)  # (2, n_voxel)
        p_vals = scipy.stats.norm.sf(z_stats)
        logger.info(
            "p-values: min=%.4g, mean=%.4g, max=%.4g",
            np.min(p_vals), np.mean(p_vals), np.max(p_vals),
        )
        return p_vals
    
    def bread_term(self, Z, B, P, use_dask=True, block_size=1000):
        """Bread term (inverse Hessian blocks) via efficient Kronecker product."""
        XTWX = efficient_kronT_diag_kron(Z, B, P, use_dask=use_dask, block_size=block_size)
        XTWX = XTWX.reshape(self._P, self._R, self._P, self._R, order="F")
        bread = [
            robust_inverse(XTWX[:, i, :, i] + 1e-6 * np.eye(self._P))
            for i in range(self._R)
        ]
        del XTWX
        gc.collect()
        return bread

    def meat_term(self, Z, B, P, Y, use_dask=True, block_size=1000):
        """Meat term of sandwich estimator."""
        R = (Y - P).reshape(self._M, self._N)
        L = R @ B  # (n_subject, n_bases)
        V = [Z[:, i, None] * L for i in range(self._R)]
        meat = [Vi.T @ Vi for Vi in V]
        del R, L, V
        gc.collect()
        return meat

    def plot_1d(self, p_vals, filename, significance_level=0.05):
        """QQ-plot of p-values. Delegates to module-level ``plot_qq``."""
        plot_qq(p_vals, filename, significance_level)

# ---------------------------------------------------------------------------
#  UKB (UK Biobank) inference
# ---------------------------------------------------------------------------

class BrainInference_UKB:
    """Inference for brain lesion models fitted to UK Biobank data."""

    def __init__(self, model, marginal_dist, link_func, regression_terms,
                 dtype=torch.float64, device='cpu'):
        self.model = model
        self.marginal_dist = marginal_dist
        self.link_func = link_func
        self.regression_terms = regression_terms
        self.dtype = dtype
        self.device = device

    def load_params(self, data, params):
        """Load data and fitted parameters, then compute derived quantities."""
        B, Z = data["X_spatial"], data["Z"]
        B = B * 50 / B.shape[0]
        Z = Z * 50 / Z.shape[0]
        self.B = np.concatenate([B, np.ones((B.shape[0], 1))], axis=1)
        self.Y = data["Y"]
        self.Z = np.concatenate([Z, np.ones((Z.shape[0], 1))], axis=1)
        self._M, self._R = self.Z.shape
        self._N, self._P = self.B.shape
        # Load parameters and re-scale
        self.beta = params["beta"]
        # MU
        if self.model == "SpatialBrainLesion":
            self.MU = compute_mu(self.Z, self.B, self.beta, mode="dask", block_size=5000) # shape: (n_subject*n_voxel, 1)
            self.MU = self.MU.reshape(self._M, self._N) # shape: (n_subject, n_voxel)
            P = self.MU * np.exp(-self.MU) # shape: (n_subject, n_voxel)
            P_mean = np.mean(P, axis=0) # shape: (n_voxel,)

    def create_contrast(self, contrast_vector=None, contrast_name=None, polynomial_order=1):
        """Build and normalise the contrast matrix for hypothesis testing."""
        self.contrast_vector = contrast_vector
        self.contrast_name = contrast_name
        if contrast_name == "age":
            if polynomial_order == 1:
                self.contrast_vector = np.array([0, 1, 0, 0, 0]).reshape(-1, self._R)
            else:
                self.contrast_vector = np.array([
                                            [0, 0, 1, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 1, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 1, 0, 0, 0]
                                        ]).reshape(-1, self._R)
        else:
            self.contrast_vector = (
                np.eye(self._R)
                if contrast_vector is None
                else np.array(contrast_vector).reshape(-1, self._R)
            )
        self._S = self.contrast_vector.shape[0]
        # standardization (row sum 1)
        self.contrast_vector = self.contrast_vector / np.sum(np.abs(self.contrast_vector), axis=1).reshape((-1, 1))

    def run_inference(self, alpha=0.05, method="FI", lesion_mask=None, XTWX_filename=None, Fisher_info_filename=None,
                      meat_term_filename=None, bread_term_filename=None, p_vals_filename=None, 
                      z_vals_filename=None,fig_filename=None):
        """Run statistical inference: compute p-values, z-stats, and save results."""
        self.XTWX_filename = XTWX_filename
        self.Fisher_info_filename = Fisher_info_filename
        self.meat_term_filename = meat_term_filename
        self.bread_term_filename = bread_term_filename
        self.p_vals_filename = p_vals_filename
        self.z_vals_filename = z_vals_filename
        self.fig_dir = os.path.dirname(fig_filename)
        z_threshold = scipy.stats.norm.ppf(1-alpha)
        # Generalised linear hypothesis testing
        if os.path.exists(self.p_vals_filename) and os.path.exists(self.z_vals_filename):
            p_vals = np.load(self.p_vals_filename)["p_vals"]
            z_stats = np.load(self.z_vals_filename)["z_stats"]
            logger.info("Loaded p-values and z-stats from file.")
        else:
            if self.model == "SpatialBrainLesion":
                p_vals, z_stats = self.SpatialGLM_glh_con_group(method, lesion_mask, True, 1e4)
            elif self.model == "MassUnivariateRegression":
                p_vals, z_stats = self.MUM_glh_con_group(lesion_mask)
            else:
                raise ValueError(f"Model {self.model} not supported for inference.")
            np.savez(self.p_vals_filename, p_vals=p_vals)
            np.savez(self.z_vals_filename, z_stats=z_stats)
            logger.info("Saved p-values and z-stats to file.")
        # Plot the estimated P, standard error of P, and p-values
        self.histogram_z_stats(z_stats, fig_filename.replace(".png", "_z_stats_histogram.png"))
        save_nifti(p_vals.flatten(), lesion_mask, os.path.join(self.fig_dir, f"p_vals_{self.model}_{method}.nii.gz"))
        save_nifti(z_stats.flatten(), lesion_mask, os.path.join(self.fig_dir, f"z_stats_{self.model}_{method}.nii.gz"))
        plot_brain(p=z_stats, brain_mask=lesion_mask, threshold=z_threshold, output_filename=fig_filename)
    
    def SpatialGLM_glh_con_group(self, method, lesion_mask, use_dask=True, block_size=1e6):
        """Generalised linear hypothesis test for the Spatial GLM model.

        Returns
        -------
        p_vals, z_stats : ndarrays
        """
        # Estimate the variance of beta via FI or sandwich estimator
        if method == "FI":
            if not os.path.exists(self.XTWX_filename):
                XTWX = efficient_kronT_diag_kron(self.Z, self.B, self.MU, use_dask=use_dask, block_size=block_size) # shape: (n_covariates*n_bases, n_covariates*n_bases)
                np.savez(self.XTWX_filename, XTWX=XTWX)
            else:
                XTWX = np.load(self.XTWX_filename)["XTWX"]

        CB = np.einsum('ij,kl->ikjl', self.contrast_vector, self.B) # shape: (_S, _N, _R, _P)
        CB_flat = CB.reshape(self._S, self._N, -1) # shape: (_S, _N, _R*_P)
        # (C \otimes B) \beta
        CB_beta = CB_flat @ self.beta  # shape: (_S, _N, 1)
        CB_beta = CB_beta.squeeze(-1) # shape: (_S, _N)
        # get the path of self.fig_filename
        plot_brain(p=CB_beta.flatten(), brain_mask=lesion_mask, threshold=0, vmax=None, output_filename=os.path.join(self.fig_dir, "numerator_map_SGLM.png"))
        # shape: (_S, _N) 
        if method == "FI":
            cov_beta = np.linalg.pinv(XTWX) # shape: (_R*_P, _R*_P)
            tmp = np.einsum('snk,kl->snl', CB_flat, cov_beta)         # (S, N, K)
            contrast_var_eta = np.sum(tmp * CB_flat, axis=-1, keepdims=True)  # (S, N, 1)
            plot_brain(p=np.sqrt(contrast_var_eta).flatten(), brain_mask=lesion_mask, threshold=0, vmax=None, output_filename=os.path.join(self.fig_dir, "denominator_map_SGLM_FI.png"))
        elif method == "sandwich":
            cov_beta, diag = self.poisson_sandwich_kron(self.Z, self.B, self.Y, self.MU, meat="iid", ridge=0, return_diagnostics=True)
            logger.info(
                "cov_beta diag stats: min=%.4g, mean=%.4g, max=%.4g",
                np.min(np.diag(cov_beta)), np.mean(np.diag(cov_beta)), np.max(np.diag(cov_beta)),
            )
            tmp = np.einsum('snk,kl->snl', CB_flat, cov_beta)         # (S, N, K)
            contrast_var_eta = np.sum(tmp * CB_flat, axis=-1, keepdims=True)  # (S, N, 1)
            plot_brain(p=np.sqrt(contrast_var_eta).flatten(), brain_mask=lesion_mask, threshold=0, vmax=None, output_filename=os.path.join(self.fig_dir, "denominator_map_SGLM_sandwich.png"))
        if self._S == 1:
            contrast_std_eta = np.sqrt(contrast_var_eta) # shape: (_N, 1)
            # Conduct Wald test (Z test)
            z_stats = CB_beta.reshape(-1, 1) / contrast_std_eta.reshape(-1, 1) # shape: (_N, 1)
            logger.info("z stats range: min=%.4g, max=%.4g", np.min(z_stats), np.max(z_stats))
            # one-sided p-values
            p_vals = scipy.stats.norm.sf(z_stats) # shape: (_N, 1)
        else:
            chi_square_stats = np.empty(shape=(0,))
            for j in range(self._N):
                CB_j = CB_flat[:, j, :]  # shape: (_S, _R*_P)
                CB_beta_j = CB_beta[:, j].reshape(1, self._S) # shape: (1, _S)
                v_j = CB_j @ cov_beta @ CB_j.T # shape: (_S, _S)
                v_j_inv = np.linalg.pinv(v_j) # shape: (_S, _S)
                chi_square_j = CB_beta_j @ v_j_inv @ CB_beta_j.T
                chi_square_stats = np.concatenate((chi_square_stats, chi_square_j.reshape(1,)), axis=0)
            p_vals = 1 - scipy.stats.chi2.cdf(chi_square_stats, df=self._S)
            logger.info(
                "Chi-square test: %d voxels significant at p<0.05 (out of %d)",
                np.count_nonzero(p_vals < 0.05), len(p_vals),
            )
            p_vals = p_vals.reshape((1,-1))
            # convert p-values to z-stats (one-sided)
            z_stats = scipy.stats.norm.isf(p_vals / 2)
            # save to nifti file
    
        return p_vals, z_stats

    def MUM_glh_con_group(self, lesion_mask):
        """Generalised linear hypothesis test for the Mass Univariate model.

        Returns
        -------
        p_vals, z_stats : ndarrays
        """
        contrast_beta_covariates = self.contrast_vector @ self.beta  # (1, n_voxel)

        # Compute Fisher information for the single non-zero contrast
        if np.count_nonzero(self.contrast_vector) != 1:
            raise NotImplementedError(
                "FI method only implemented for single non-zero contrast in MUM."
            )

        nonzero_index = np.nonzero(self.contrast_vector)[1].item()
        if self.link_func == "log":
            MU = np.exp(self.Z @ self.beta)  # (n_subject, n_voxel)
            FI = np.einsum('im,ij,ik->jmk', self.Z, MU, self.Z)  # (N, R, R)
        elif self.link_func == "logit":
            MU = 1 / (1 + np.exp(-(self.Z @ self.beta)))  # (n_subject, n_voxel)
            FI = np.einsum('im,ij,ik->jmk', self.Z, MU * (1 - MU), self.Z)  # (N, R, R)
        else:
            raise ValueError(f"Link function {self.link_func} not supported.")

        Cov_beta = np.linalg.pinv(FI)  # (N, R, R)
        var_beta = Cov_beta[:, nonzero_index, nonzero_index]  # (n_voxel,)
        contrast_std_beta = np.sqrt(var_beta)  # (n_voxel,)

        plot_brain(p=contrast_beta_covariates.flatten(), brain_mask=lesion_mask,
                   threshold=0, vmax=None, output_filename="numerator_map_MUM.png")
        plot_brain(p=contrast_std_beta.flatten(), brain_mask=lesion_mask,
                   threshold=0, vmax=None, output_filename="denominator_map_MUM.png")

        z_stats = (contrast_beta_covariates / contrast_std_beta).reshape(-1)
        logger.info("z stats range: min=%.4g, max=%.4g", np.min(z_stats), np.max(z_stats))

        p_vals = 2 * scipy.stats.norm.sf(np.abs(z_stats))
        logger.info(
            "p-vals: min=%.4g, max=%.4g, significant (p<0.05)=%d / %d",
            np.min(p_vals), np.max(p_vals),
            np.count_nonzero(p_vals < 0.05), p_vals.size,
        )

        return p_vals, z_stats
    
    def meat_term(self, Z, B, MU, Y, batch_M=1000):
        """Compute or load the meat term of the sandwich estimator."""
        if MU.shape != Y.shape:
            MU = MU.reshape(Y.shape) # shape: (_M, _N)
        if not os.path.exists(self.meat_term_filename):
            meat_term_1 = np.zeros((self._P * self._R, self._P * self._R)) # shape: (_P*_R, _P*_R)
            W = Y - MU
            BW = W.dot(B)    # shape (M, P)
            T = (Z[:, :, None] * BW[:, None, :]).reshape(self._M, self._P * self._R)  # shape (M, PR)
            meat_term = T.T.dot(T)   # shape (PR, PR)
            del W, BW, T
            gc.collect()
            np.savez(self.meat_term_filename, meat_term=meat_term)
        else:
            logger.info("Loading precomputed meat term...")
            meat_term = np.load(self.meat_term_filename)["meat_term"]

        return meat_term

    def bread_term(self, Z, B, MU, Y, dtype=np.float64, chunk_rows=256, epsilon=1e-6):
        """Compute or load the bread term of the sandwich estimator."""
        if MU.shape != Y.shape:
            MU = MU.reshape(Y.shape)
        if not os.path.exists(self.bread_term_filename):
            logger.info("Computing bread term...")
            start_time = time.time()
            bread_term = np.zeros((self._P * self._R, self._P * self._R)) # shape: (_P*_R, _P*_R)

            for i in range(self._M):
                zi = Z[i, :]                    # shape: (R,)
                mu_i = MU[i, :]
                G_B = B.T @ (mu_i[:, None] * B)
                G_z = np.outer(zi, zi)          # (R, R)
                bread_term += np.kron(G_z, G_B)

            logger.info(
                "Bread term diag stats: min=%.4g, mean=%.4g, max=%.4g",
                np.min(np.diag(bread_term)), np.mean(np.diag(bread_term)),
                np.max(np.diag(bread_term)),
            )
            logger.info("Bread term computation took %.1f s", time.time() - start_time)
            del Z, B, MU, Y
            gc.collect()
            np.savez(self.bread_term_filename, bread_term=bread_term)
        else:
            logger.info("Loading precomputed bread term...")
            bread_term = np.load(self.bread_term_filename)["bread_term"]

        return bread_term

    def poisson_sandwich_kron(self,
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
            L, low = scipy.linalg.cho_factor(A)
            if meat_kind == "cluster":
                Y = scipy.linalg.cho_solve((L, low), C)           # A^{-1} U,  (p, M)
                cov = Y @ Y.T
            else:
                D   = scipy.linalg.cho_solve((L, low), Bmeat)     # A^{-1} Bmeat
                cov = scipy.linalg.cho_solve((L, low), D.T).T     # (A^{-1} Bmeat) A^{-1}
        except np.linalg.LinAlgError:
            logger.warning("Cholesky failed — falling back to pseudo-inverse")
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

    def histogram_z_stats(self, z_stats, filename):
        """Save a histogram of z-statistics to *filename*."""
        plt.figure(figsize=(10, 6))
        plt.hist(z_stats.flatten(), bins=100, color='blue', alpha=0.7, edgecolor='black')
        plt.title('Histogram of Z-statistics', fontsize=16)
        plt.xlabel('Z-statistic', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.grid(axis='y', alpha=0.75)
        plt.savefig(filename)
        plt.close()