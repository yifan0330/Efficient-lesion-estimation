from typing import List
import torch
import torch.nn as nn
from torch.distributions import Bernoulli

class ApproximatePoissonModel(nn.Module):
    def __init__(self,
                 n_covariates: int,
                 n_auxiliary: int,
                 n_bases: int,
                 n_samples: int = 100,
                 std_params: float = .1,
                 std_auxiliary: float = 1.0,
                 link_func: str = "logit",
                 marginal_dist: str = "Bernoulli",
                 regression_terms: List[str] = ["multiplicative", "additive"],
                 device: str = "cpu",
                 dtype = torch.float32):
        super().__init__()
        self.n_covariates = n_covariates
        self.n_bases = n_bases
        self.device = torch.device(device)
        self.dtype = dtype
        self._kwargs = {"device": self.device, "dtype": self.dtype}
        # link function and its inverse
        link_func == "log"
        self.inverse_link_func = torch.exp
        # beta: spatial regression coefficients
        self.beta = nn.Parameter(0.01*torch.randn(self.n_bases, 1, **self._kwargs))
        # gamma: regression coefficients for covariates
        self.gamma = nn.Parameter(0.01*torch.randn(self.n_covariates, 1, **self._kwargs))

    def forward(self, X, Y, Z):
        # # Produce intensity function parameters
        # W = self.bias_beta.T
        # W = W + Z @ self.beta.T # shape: (n_subject, n_bases)
        # b = self.bias_gamma.T
        # b = b + Z @ self.gamma.T # shape: (n_subject, 1)
        # # Compute intensity function
        # P = self.inverse_link_func(W @ X.T + b) # shape: (n_subject, n_voxel)
        y_g = torch.sum(Y, dim=0).reshape(-1, 1) # shape: (n_voxel, 1)
        y_t = torch.sum(Y, dim=1).reshape(-1, 1) # shape: (n_subject, 1)
        # spatial mu effect
        log_mu_spatial = X @ self.beta 
        mu_spatial = torch.exp(log_mu_spatial) # shape: (n_voxel, 1)
        # covariates mu effect
        log_mu_covariates = Z @ self.gamma
        mu_covariates = torch.exp(log_mu_covariates)
        # log likelihood
        log_l = torch.sum(torch.mul(y_g, log_mu_spatial)) + torch.sum(torch.mul(y_t, log_mu_covariates)) \
                            - torch.sum(mu_spatial) * torch.sum(mu_covariates) 

        return -log_l