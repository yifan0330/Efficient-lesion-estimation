"""Regression fitting for spatial brain-lesion models.

Provides two classes:
  * BrainRegression_full      – PyTorch-based L-BFGS optimisation
  * BrainRegression_Approximate – Closed-form / iterative NumPy solver
"""

import logging
import os
import time

import numpy as np
import scipy
from scipy.optimize import minimize
import torch
from tqdm import tqdm

from bspline import B_spline_bases
from model import SpatialBrainLesionModel, MassUnivariateRegression
from util import (
    fit_multiplicative_log_glm,
    fit_MUM_log_glm,
    SpatialGLM_compute_mu_mean,
    SpatialGLM_compute_P_mean,
)

logger = logging.getLogger(__name__)

class BrainRegression_full:
    """Full-data regression via PyTorch L-BFGS optimisation."""

    def __init__(self, dtype=torch.float64, device='cpu'):
        """Initialise with computation dtype and device."""
        self.dtype = dtype
        self.device = device
        self._kwargs = {"dtype": self.dtype, "device": self.device}

    def load_data(self, data):
        """Load and prepare data tensors (Y, B with intercept, Z with intercept)."""
        # load MU, X, Y, Z
        if "MU" in data and data["MU"] is not None:
            self.MU = torch.tensor(data["MU"], **self._kwargs)
            # self.MU = self.MU[subsampled_subjects, :]
        # Load X_spatial and add intercept
        B = torch.tensor(data["X_spatial"], **self._kwargs)
        self.B = torch.cat([B, torch.ones((B.shape[0], 1), **self._kwargs)], dim=1)
        Z = torch.tensor(data["Z"], **self._kwargs)
        self.Z = torch.cat([Z, torch.ones((Z.shape[0], 1), **self._kwargs)], dim=1)
        self.Y = torch.tensor(data["Y"], **self._kwargs)
        # Dimensions
        self.n_subjects, self.n_covariates = self.Z.shape
        self.n_voxels, self.n_bases = self.B.shape

    def init_model(self, model_name, **kwargs):
        """Instantiate the specified model with the given keyword arguments."""
        if model_name == "SpatialBrainLesion":
            self.model = SpatialBrainLesionModel(n_covariates=self.n_covariates, 
                                                n_auxiliary=kwargs["n_auxiliary"], 
                                                std_auxiliary=kwargs["std_auxiliary"],
                                                n_samples=kwargs["n_samples"],
                                                regression_terms=kwargs["regression_terms"],
                                                link_func=kwargs["link_func"],
                                                marginal_dist=kwargs["marginal_dist"],
                                                n_bases=self.n_bases,
                                                device=self.device, 
                                                dtype=self.dtype)
        elif model_name == "MassUnivariateRegression":
            self.model = MassUnivariateRegression(n_covariates=self.n_covariates, 
                                                n_auxiliary=kwargs["n_auxiliary"], 
                                                std_auxiliary=kwargs["std_auxiliary"],
                                                n_samples=kwargs["n_samples"],
                                                regression_terms=kwargs["regression_terms"],
                                                link_func=kwargs["link_func"],
                                                marginal_dist=kwargs["marginal_dist"],
                                                firth_penalty=kwargs['firth_penalty'],
                                                n_voxels=self.n_voxels,
                                                device=self.device, 
                                                dtype=self.dtype)
        else:
            raise ValueError(f"Model {model_name} not implemented")
    
    def optimize_model(self, lr, iter, tolerance_change, tolerance_grad=1e-7, 
                       history_size=100, line_search_fn="strong_wolfe"):
        """Run L-BFGS optimisation on the loaded model and data."""
        start_time = time.time()
        # Initialize iteration counter
        self.iteration = 0
        # lbfgs verbose model
        optimizer = torch.optim.LBFGS(params=self.model.parameters(), 
                                            lr=lr, 
                                            max_iter=iter,
                                            tolerance_grad=tolerance_grad, 
                                            tolerance_change=tolerance_change,
                                            history_size=history_size, 
                                            line_search_fn=line_search_fn)

        def closure():
            optimizer.zero_grad()
            preds = self.model(self.B, self.Y, self.Z)
            loss = self.model.get_loss(preds, self.Y, self.Z)
            logger.info("Iteration %d: Loss: %.6f", self.iteration, loss.item())
            self.iteration += 1
            loss.backward()
            return loss
        optimizer.step(closure)

        logger.info("Optimisation took %.1f s", time.time() - start_time)
        return
    
class BrainRegression_Approximate:
    """Approximate regression using closed-form / iterative NumPy solvers."""

    def __init__(self, simulated_dset, dtype=torch.float64, device='cpu'):
        """Initialise with dataset flag, dtype, and device."""
        self.simulated_dset = simulated_dset
        self.dtype = dtype
        self.device = device  

    def load_data(self, data, model):
        """Load and prepare data arrays (Y, B with intercept, Z with intercept)."""
        B, Z = data["X_spatial"], data["Z"]
        B = B.astype(np.float64)
        B = B * 50 / B.shape[0]
        Z = Z * 50 / Z.shape[0]
        self.B = np.concatenate([B, np.ones((B.shape[0], 1))], axis=1)
        self.Y = data["Y"]
        self.Z = np.concatenate([Z, np.ones((Z.shape[0], 1))], axis=1)
        # Dimensions
        self._M, self._R = self.Z.shape
        self._N, self._P = self.B.shape
        
    def run_regression(self, 
                       model: str, 
                       marginal_dist: str,
                       link_func: str,
                       tol: float = 1e-10,
                       max_iter: int = 1000, 
                       alpha: float = 1.0,
                       gradient_mode: str = "dask", 
                       preconditioner_mode: str = "approximate", 
                       nll_mode: int = "dask",
                       block_size: int = 10000, 
                       compute_nll: bool = False):
        """Fit the regression model and return estimated coefficients."""
        start = time.time()
        if model == "SpatialBrainLesion":
            beta = fit_multiplicative_log_glm(
                self.Z, self.B, self.Y, tol=tol,
                max_iter=max_iter, alpha=alpha,
                gradient_mode=gradient_mode,     
                preconditioner_mode=preconditioner_mode,
                nll_mode=nll_mode, block_size=block_size,
                compute_nll=compute_nll)
        elif model == "MassUnivariateRegression":
            beta = fit_MUM_log_glm(
                self.Z, self.B, self.Y, marginal_dist, 
                link_func, tol=tol, 
                max_iter=max_iter, alpha=alpha,
                nll_mode=nll_mode, block_size=block_size,
                compute_nll=compute_nll)
        else:
            raise ValueError(f"Model {model} not implemented")
        logger.info("Regression completed in %.1f s", time.time() - start)
        return beta

    def goodness_of_fit(self, beta, model, mode="dask", block_size=100):
        """Compute goodness-of-fit statistics (mean/std of MU, mean of P)."""
        if model == "SpatialBrainLesion":
            MU_mean, MU_std = SpatialGLM_compute_mu_mean(self.Z, self.B, beta, mode=mode, block_size=block_size)
            P_mean = SpatialGLM_compute_P_mean(self.Z, self.B, beta, mode=mode, block_size=block_size)
            return MU_mean, MU_std, P_mean
        elif model == "MassUnivariateRegression":
            MU = np.exp(self.Z @ beta)
            P = MU * np.exp(-MU)
            P_mean = np.mean(P, axis=0)
            return None, None, P_mean