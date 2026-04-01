from bspline import B_spline_bases
from model import SpatialBrainLesionModel, MassUnivariateRegression
import numpy as np
import scipy
from scipy.optimize import minimize, line_search
from util import fit_multiplicative_log_glm, fit_MUM_log_glm, SpatialGLM_compute_mu_mean, SpatialGLM_compute_P_mean
from tqdm import tqdm
import torch
import logging
import time
import os

class BrainRegression_full(object):

    def __init__(self, dtype=torch.float64, device='cpu'):
        self.dtype = dtype
        self.device = device
        self._kwargs = {"dtype": self.dtype, "device": self.device}

    def load_data(self, data):
        n_subjects = data["Y"].shape
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
        self.model = dict()
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
        # optimization 
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

        # optimizer = torch.optim.SGD(params=self.model.parameters(), 
        #                                 lr=1,
        #                                 weight_decay=1e-5)
        def closure():
            optimizer.zero_grad()
            preds = self.model(self.B, self.Y, self.Z)
            loss = self.model.get_loss(preds, self.Y, self.Z)
            print(f"Iteration {self.iteration}: Loss: {loss.item()}")
            self.iteration += 1
            loss.backward()
            return loss
        optimizer.step(closure)

        # # SGD optimization loop
        # for i in range(int(iter)):
        #     optimizer.zero_grad()
        #     preds = self.model(self.B, self.Y, self.Z)
        #     loss = self.model.get_loss(preds, self.Y, self.Z)
        #     print(f"Iteration {self.iteration}: Loss: {loss.item()}")
        #     self.iteration += 1
        #     loss.backward()
        #     optimizer.step()
            
        #     # Optional: check for convergence
        #     if i > 0 and abs(prev_loss - loss.item()) < tolerance_change:
        #         print(f"Converged at iteration {i}")
        #         break
        #     prev_loss = loss.item()
        print(f"Time taken for optimization: {time.time() - start_time} seconds")
        return
    
class BrainRegression_Approximate(object):

    def __init__(self, simulated_dset, dtype=torch.float64, device='cpu'):
        self.simulated_dset = simulated_dset
        self.dtype = dtype
        self.device = device  

    def load_data(self, data, model):
        self.group_names = list(data.keys())
        B = data[self.group_names[0]]["X_spatial"]
        B = B.astype(np.float64)
        Z = {}
        for group_name in self.group_names:
            # load MU, X, Y, Z
            Z_group = data[group_name]["Z"]
            Z_group = Z_group * 50 / Z_group.shape[0]
            Z[group_name] = np.concatenate([Z_group, np.ones((Z_group.shape[0], 1))], axis=1) 
        self.Z = Z
        B = B * 50 / B.shape[0]
        self.B = np.concatenate([B, np.ones((B.shape[0], 1))], axis=1)
        self.Y = {}
        for group_name in self.group_names:
            self.Y[group_name] = data[group_name]["Y"]
        # Dimensions
        self._M, self._R = self.Z[self.group_names[0]].shape
        self._N, self._P = self.B.shape
        
    def run_regression(self, 
                       model: str, 
                       marginal_dist: str,
                       link_func: str,
                       tol: float = 1e-10,
                       max_iter: int = 100, 
                       lr: float = 1.0,
                       gradient_mode: str = "dask", 
                       preconditioner_mode: str = "approximate", 
                       nll_mode: int = "dask",
                       block_size: int = 10000, 
                       compute_nll: bool = False):
        start = time.time()
        beta = {}
        if model == "SpatialBrainLesion":
            for group_name in self.group_names:
                beta_group = fit_multiplicative_log_glm(
                            self.Z[group_name], self.B, self.Y[group_name], marginal_dist=marginal_dist,
                            tol=tol,max_iter=max_iter, lr=lr, gradient_mode=gradient_mode, preconditioner_mode=preconditioner_mode, 
                            nll_mode=nll_mode, block_size=block_size, compute_nll=compute_nll)
            beta[group_name] = beta_group
        elif model == "MassUnivariateRegression":
            for group_name in self.group_names:
                beta_group = fit_MUM_log_glm(
                            self.Z[group_name], self.B, self.Y[group_name], marginal_dist, 
                            link_func, tol=tol, 
                            max_iter=max_iter,
                            nll_mode=nll_mode, block_size=block_size,
                            compute_nll=compute_nll)
                beta[group_name] = beta_group
        else:
            raise ValueError(f"Model {model} not implemented")
        print(f"Time: {time.time() - start}")
        return beta

    def goodness_of_fit(self, beta, model, mode="dask", block_size=100):
        if model == "SpatialBrainLesion":
            MU_mean, MU_std, P_mean = {}, {}, {}
            for group_name in self.group_names:
                MU_mean[group_name], MU_std[group_name] = SpatialGLM_compute_mu_mean(self.Z[group_name], self.B, beta[group_name], mode=mode, block_size=block_size)
                P_mean[group_name] = SpatialGLM_compute_P_mean(self.Z[group_name], self.B, beta[group_name], mode=mode, block_size=block_size)
            return MU_mean, MU_std, P_mean
        elif model == "MassUnivariateRegression":
            MU_mean, P_mean = {}, {}
            for group_name in self.group_names:
                MU_group = np.exp(self.Z[group_name] @ beta[group_name])
                P_group = MU_group * np.exp(-MU_group)
                MU_mean[group_name] = np.mean(MU_group, axis=0)
                P_mean[group_name] = np.mean(P_group, axis=0)
                print(np.min(P_mean[group_name]), np.max(P_mean[group_name]), "P mean range")

            return MU_mean, None, P_mean