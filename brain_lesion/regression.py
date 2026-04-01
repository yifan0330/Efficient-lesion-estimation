from bspline import B_spline_bases
from model import SpatialBrainLesionModel
import numpy as np
import torch
import logging
import time
import os

class BrainRegression(object):

    def __init__(self, dtype=torch.float64, device='cpu'):
        self.dtype = dtype
        self.device = device
        self._kwargs = {"dtype": self.dtype, "device": self.device}

    def load_data(self, data):
        # load MU, X, Y, Z
        self.G = data["G"]
        self.MU = torch.tensor(data["MU"], **self._kwargs)
        self.X = torch.tensor(data["X_spatial"], **self._kwargs)
        self.Y = torch.tensor(data["Y"], **self._kwargs)
        self.Z = torch.tensor(data["Z"], **self._kwargs)
        # Dimension
        self.group_names = list(self.G.keys())
        self.n_group = len(self.group_names)
        self.n_subject, self.n_covariates = self.Z.shape
        self.n_voxel, self.n_bases = self.X.shape

    def init_model(self, model_name, **kwargs):
        self.model = dict()
        if model_name == "SpatialBrainLesion":
            self.model = SpatialBrainLesionModel(n_covariates=self.n_covariates, 
                                                n_auxiliary=kwargs["n_auxiliary"], 
                                                std_auxiliary=kwargs["std_auxiliary"],
                                                n_samples=kwargs["n_samples"],
                                                regression_terms=kwargs["regression_terms"],
                                                link_func=kwargs["link_func"],
                                                polynomial_order=kwargs["polynomial_order"],
                                                marginal_dist=kwargs["marginal_dist"],
                                                n_bases=self.n_bases,
                                                device=self.device, 
                                                dtype=self.dtype)
        else:
            raise ValueError(f"Model {model_name} not implemented")
    
    def optimize_model(self, lr, iter, tolerance_change, tolerance_grad=1e-7, 
                       history_size=100, line_search_fn="strong_wolfe"):
        # optimization 
        start_time = time.time()
        # lbfgs verbose model
        optimizers, all_loss = dict(), dict()
        optimizer = torch.optim.LBFGS(params=self.model.parameters(), 
                                            lr=lr, 
                                            max_iter=iter,
                                            tolerance_grad=tolerance_grad, 
                                            tolerance_change=tolerance_change,
                                            history_size=history_size, 
                                            line_search_fn=line_search_fn)
        def closure():
            optimizer.zero_grad()
            preds = self.model(self.X, self.Y, self.Z)
            loss = self.model.get_loss(preds, self.Y)
            print(f"Loss: {loss.item()}")
            loss.backward()
            return loss
        optimizer.step(closure)

        return