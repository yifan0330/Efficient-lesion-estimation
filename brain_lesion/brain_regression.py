from bspline import B_spline_bases
from model import GLMPoisson, SpatialBrainLesionModel
import numpy as np
import torch
import logging
import time

class BrainRegression(object):

    def __init__(self, simulated_dset, homogeneous, dtype=torch.float64, device='cpu'):
        self.simulated_dset = simulated_dset
        self.homogeneous = homogeneous
        self.dtype = dtype
        self.device = device
        # load spatial parametrisation 
        self.load_data(dtype)

    def load_data(self, dtype):
        filename_0 = "_Simulation" if self.simulated_dset else "_RealDataset"
        filename_1 = "_Homogeneous" if self.homogeneous else "_BumpSignals"
        data = np.load(f"data{filename_0}{filename_1}.npz")
        # load MU, X, Y, Z
        self.MU = torch.tensor(data["MU"], dtype=self.dtype, device=self.device)
        self.X = torch.tensor(data["X_spatial"], dtype=self.dtype, device=self.device)
        self.Y = torch.tensor(data["Y"], dtype=self.dtype, device=self.device)
        self.Z = torch.tensor(data["Z"], dtype=self.dtype, device=self.device)
        # Dimension
        self.n_subject, self.n_voxel = self.Y.shape

        return 
    
    def model_structure(self, model):
        # auxiliary variable
        alpha = np.random.normal(loc=0, scale=1, size=(self.n_subject, 1)) 
        P, R = self.X.shape[1], self.Z.shape[1]
        beta_init = torch.log(torch.sum(self.Y) / (self.n_subject * self.n_voxel)).repeat(R, P) # shape: (n_covariate, n_bases)
        if model == "Poisson":
            model = GLMPoisson(beta_dim=P, beta_init=beta_init)
        elif model == "SpatialBrainLesion":
            model = SpatialBrainLesionModel(n_covariates=R, n_auxiliary=2, n_bases=P, 
                                            device=self.device, dtype=self.dtype)
        return model
    
    def _optimizer(self, model, Y, Z, lr, tol, iter, history_size=100,
                line_search_fn="strong_wolfe", tolerance_grad=1e-5):
        
        print(lr, tol, iter, history_size, line_search_fn)
        
        # optimization 
        start_time = time.time()
        # lbfgs verbose model
        optimizer = torch.optim.LBFGS(params=model.parameters(), lr=lr, max_iter=iter,
                                    tolerance_grad=tolerance_grad, tolerance_change=tol,
                                    history_size=history_size, line_search_fn=line_search_fn)
        all_loss = list()
        def closure():
            optimizer.zero_grad()
            loss = model(self.X, self.Y, self.Z)
            all_loss.append(loss.item())
            print(loss.item())
            loss.backward()
            return loss
        optimizer.step(closure)
        print("Time taken: ", time.time()-start_time)
        loss = model(self.X, self.Y, Z)
        return loss
    
    def train(self, model, iter=1500, lr=0.01, tol=1e-4):
        self.model = model
        logging.info(f"Run regression")
        start_time = time.time()
        # model
        brain_model = self.model_structure(model)
        # optimisation

        tol = 1e-7
        loss = self._optimizer(model=brain_model, Y=self.Y, Z=self.Z, lr=lr, tol=tol,
                               iter=iter)
        # beta_hat = brain_model.beta
        # eta_hat = self.Z @ beta_hat @ self.X.T
        # MU_hat = torch.exp(eta_hat)
        # print(MU_hat)
        # print(MU_hat.shape)
        # print(self.MU)
        # exit()
        return 
