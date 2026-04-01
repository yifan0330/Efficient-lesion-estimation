from typing import List
import torch
import torch.nn as nn
from torch.distributions import Bernoulli

class SpatialBrainLesionModel(nn.Module):

    def __init__(self,
                 n_covariates: int,
                 n_auxiliary: int,
                 n_bases: int,
                 n_samples: int = 100,
                 std_params: float = .1,
                 std_auxiliary: float = 1.0,
                 link_func: str = "logit",
                 polynomial_order: int = 1,
                 marginal_dist: str = "Bernoulli",
                 regression_terms: List[str] = ["multiplicative", "additive"],
                 device: str = "cpu",
                 dtype = torch.float32):
        """ Spatial brain lesion model

        Args:
            n_covariates: Number of covariates
            n_auxiliary: Number of auxiliary variables
            n_bases: Number of bases for spatial representations
            n_samples: Number of samples for Monte Carlo approximation
            std_params: Standard deviation of Gaussian parameters
            std_auxiliary: Standard deviation of Gaussian auxiliary variables
            link_func: Link function for intensity function, options: "logit", "log"
            polynomial_order: Polynomial order for GLM regression
            marginal_dist: Marginal distribution at each spatial location, options: "Bernoulli", "Poisson"
            regression_terms: Regression terms, options: ["multiplicative", "additive"]

        X: Spatial design matrix of shape (n_voxel, n_bases)
        Y: Binary lesion mask of shape (n_subject, n_voxel)
        Z: Covariates matrix of shape (n_subject, n_covariates)
        A: Random auxiliary variables of shape (n_subject, n_auxiliary)
        """
        super().__init__()
        self.n_covariates = n_covariates
        self.n_auxiliary = n_auxiliary
        self.n_bases = n_bases
        self.n_samples = n_samples
        self.std_params = std_params
        self.std_auxiliary = std_auxiliary
        if link_func == "logit":
            self.inverse_link_func = nn.Sigmoid()
        elif link_func == "log":
            self.inverse_link_func = torch.exp
        elif link_func == "arctanh":
            self.inverse_link_func = lambda z: (nn.Tanh()(z) + 1.) / 2.
        else:
            raise ValueError(f"Link function {link_func} not implemented")
        self.polynomial_order = polynomial_order
        self.marginal_dist = marginal_dist
        self.regression_terms = regression_terms
        self.device = torch.device(device)
        self.dtype = dtype
        self._kwargs = {"device": self.device, "dtype": self.dtype}

        # Initialize parameters
        if "multiplicative" in self.regression_terms:
            self.beta_W = nn.Parameter(torch.randn(n_bases, n_covariates*self.polynomial_order, **self._kwargs) * self.std_params)
            self.alpha_W = nn.Parameter(torch.randn(n_bases, n_auxiliary, **self._kwargs) * self.std_params)
            self.bias_W = nn.Parameter(torch.zeros(n_bases, 1, **self._kwargs))
        if "additive" in self.regression_terms:
            self.beta_b = nn.Parameter(torch.zeros(1, n_covariates*self.polynomial_order, **self._kwargs) * self.std_params)
            self.alpha_b = nn.Parameter(torch.randn(1, n_auxiliary, **self._kwargs) * self.std_params)
            self.bias_b = nn.Parameter(torch.zeros(1, 1, **self._kwargs))
        if "multiplicative" not in self.regression_terms and "additive" not in self.regression_terms:
            raise ValueError("At least one of the regression terms should be included")

    def forward(self, X, Y, Z):
        n_subject = Z.shape[0]
        # create polynomial Z matrix
        Z = torch.cat([Z**i for i in range(self.polynomial_order)], dim=1) # shape: (n_subject, n_covariates * polynomial_order)
        # Sample auxiliary variables from unit normal distribution
        if self.n_auxiliary > 0:
            A = torch.randn(self.n_samples, n_subject, self.n_auxiliary, **self._kwargs) * self.std_auxiliary
        # Produce intensity function parameters
        if self.n_auxiliary > 0:
            W, b = self.bias_W.T[None, ...], self.bias_b.T[None, ...]
            if "multiplicative" in self.regression_terms:
                W = W + Z[None, ...] @ self.beta_W.T + A @ self.alpha_W.T 
            if "additive" in self.regression_terms:
                b = b + Z[None, ...] @ self.beta_b.T + A @ self.alpha_b.T 
        else:
            if "multiplicative" in self.regression_terms:
                W = self.bias_W.T 
                W = W + Z @ self.beta_W.T # shape: (n_subject, n_bases)
            if "additive" in self.regression_terms:
                b = self.bias_b.T
                b = b + Z @ self.beta_b.T # shape: (n_subject, 1)
            W = 0 if "multiplicative" not in self.regression_terms else W
            b = 0 if "additive" not in self.regression_terms else b
        # Compute intensity function
        P = self.inverse_link_func(W @ X.T + b) 
        if self.n_auxiliary > 0:
            P = P.mean(0)
        # shape: (n_subject, n_voxel)
        return P
    
    def get_loss(self, P, Y):
        # equal weights for all voxels
        weights = torch.ones(Y.shape[1], device=self.device)
        if self.marginal_dist == "Bernoulli":
            nll = -(weights*torch.log(P) * Y + weights*torch.log(1 - P) * (1 - Y)).mean()
        elif self.marginal_dist == "Poisson":
            nll = -(Y * torch.log(P) - P).mean()
        return nll

    def _neg_log_likelihood(marginal_dist, link_func, polynomial_order, regression_terms, 
                            X_spatial, Y, Z, bias_W, beta_W, bias_b, beta_b, device="cpu"):
        Z = torch.cat([Z**i for i in range(polynomial_order)], dim=1)
        if link_func == "logit":
            inverse_link_func = nn.Sigmoid()
        elif link_func == "log":
            inverse_link_func = torch.exp
        if "multiplicative" in regression_terms:
            W = bias_W.T
            W = W + Z @ beta_W.T # shape: (n_subject, n_bases)
        if "additive" in regression_terms:
            b = bias_b.T
            b = b + Z @ beta_b.T # shape: (n_subject, 1)
        W = 0 if "multiplicative" not in regression_terms else W
        b = 0 if "additive" not in regression_terms else b
        # Compute probability function
        P = inverse_link_func(W @ X_spatial.T + b) 
        # negative log-likelihood
        nll = 0
        if marginal_dist == "Bernoulli":
            nll += -(torch.log(P) * Y + torch.log(1 - P) * (1 - Y)).mean()
        elif marginal_dist == "Poisson":
            nll += -(Y * torch.log(P) - P).mean()
        
        return nll