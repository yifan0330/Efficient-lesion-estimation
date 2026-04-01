from bspline import B_spline_bases
from model import ApproximatePoissonModel
import numpy as np
import torch
import logging
import time
import os

class BrainRegression(object):

    def __init__(self, dtype=torch.float64, device='cpu'):
        self.dtype = dtype
        self.device = device

    def load_data(self, data):
        # load MU, X, Y, Z
        self.MU = {group_name: torch.tensor(group_MU, dtype=self.dtype, device=self.device) for group_name, group_MU in data["MU"].items()}
        self.B = torch.tensor(data["X_spatial"], dtype=self.dtype, device=self.device)
        self.Y = {group_name: torch.tensor(group_Y, dtype=self.dtype, device=self.device) for group_name, group_Y in data["Y"].items()}
        self.Z = {group_name: torch.tensor(group_Z, dtype=self.dtype, device=self.device) for group_name, group_Z in data["Z"].items()}
        # Dimension
        self.group_names = list(self.Y.keys())
        self.n_group = len(self.group_names)
        self.n_subject = [group_Y.shape[0] for group_Y in self.Y.values()]
    
        self.n_voxel = self.Y[self.group_names[0]].shape[1]
        self.n_covariates = self.Z[self.group_names[0]].shape[1]
        self.n_bases = self.B.shape[1]

    def init_model(self, model_name, **kwargs):
        self.model = dict()
        if model_name == "SpatialBrainLesion":
            for group_name in self.group_names:
                self.model[group_name] = SpatialBrainLesionModel(n_covariates=self.n_covariates, 
                                                    n_auxiliary=kwargs["n_auxiliary"], 
                                                    std_auxiliary=kwargs["std_auxiliary"],
                                                    n_samples=kwargs["n_samples"],
                                                    regression_terms=kwargs["regression_terms"],
                                                    link_func=kwargs["link_func"],
                                                    marginal_dist=kwargs["marginal_dist"],
                                                    n_bases=self.n_bases,
                                                    device=self.device, 
                                                    dtype=self.dtype)
        elif model_name == "ApproximatePoisson":
            for group_name in self.group_names:
                self.model[group_name] = ApproximatePoissonModel(n_covariates=self.n_covariates, 
                                                    n_auxiliary=kwargs["n_auxiliary"], 
                                                    std_auxiliary=kwargs["std_auxiliary"],
                                                    n_samples=kwargs["n_samples"],
                                                    regression_terms=kwargs["regression_terms"],
                                                    link_func=kwargs["link_func"],
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
        for group_name in self.group_names:
            B, Y, Z = self.B, self.Y[group_name], self.Z[group_name]
            model = self.model[group_name]
            optimizer = torch.optim.LBFGS(params=model.parameters(), 
                                                        lr=lr, 
                                                        max_iter=iter,
                                                        tolerance_grad=tolerance_grad, 
                                                        tolerance_change=tolerance_change,
                                                        history_size=history_size, 
                                                        line_search_fn=line_search_fn)
            group_loss = list()
            def closure():
                optimizer.zero_grad()
                loss = model(B, Y, Z)
                # loss = model.get_loss(preds, Y)
                group_loss.append(loss.item())
                print(f"Group: {group_name}, Loss: {loss.item()}")
                loss.backward()
                return loss
            optimizer.step(closure)
            all_loss[group_name] = group_loss

        return all_loss
    
    def irls_log_glm(self, beta_init, X, y, max_iter=100, tol=1e-6):
        """
        Fit a GLM with a log link (e.g., Poisson regression) using the IRLS algorithm.

        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            The design matrix.
        y : np.ndarray, shape (n_samples,)
            The response vector.
        max_iter : int, default=100
            Maximum number of iterations.
        tol : float, default=1e-6
            Convergence tolerance (based on the change in beta).

        Returns:
        --------
        beta : np.ndarray, shape (n_features,)
            The estimated coefficients.
        """
        n_samples, n_features = X.shape

        # Initialize beta (for example, with zeros)
        if beta_init is not None:
            beta = np.insert(beta_init, 0, 0, axis=0)
        else: 
            beta = np.zeros((n_features, 1))
        
        for iteration in range(max_iter):
            # Compute the linear predictor and the mean response
            eta = X.dot(beta)
            mu = np.exp(eta)  # Because log-link: mu = exp(eta)
            # Weight vector: For Poisson with log-link, the variance is mu,
            # and using Fisher scoring we have W_i = mu_i.
            W = mu  # (n_samples,) vector of weights
            
            # Compute the working response:
            # For a GLM, z = eta + (y - mu) / (dmu/deta).
            # For log link, dmu/deta = mu, so:
            z = eta + (y[:, None] - mu) / mu
        
            # Form the weighted normal equations.
            # Instead of constructing a full diagonal matrix, use broadcasting:
            XTWX = X.T.dot(W * X)
            XTWz = X.T.dot(W * z)[..., 0]
            
            # Solve for the updated beta.
            beta_new = np.linalg.solve(XTWX, XTWz)
            beta_new = beta_new[:, None]
            
            print(iteration, np.linalg.norm(beta_new - beta))

            # Check for convergence (using the Euclidean norm of the change)
            if np.linalg.norm(beta_new - beta) < tol:
                beta = beta_new
                print(f"Converged in {iteration + 1} iterations.")
                break
            
            beta = beta_new
        return beta

    def IRLS_update(self, beta, gamma, tolerance_change=None):
        B_array = self.B.detach().cpu().numpy() # shape: (n_voxel N, n_bases P)
        updated_beta = dict()
        for group_name in self.group_names:
            beta_meta = beta[group_name]
            gamma_meta = gamma[group_name]
            Y_group = self.Y[group_name].detach().cpu().numpy() # shape: (n_subject M, n_voxel N)
            Y_group = Y_group.reshape(-1,1) # shape: (M*N, 1)
            Z_group = self.Z[group_name].detach().cpu().numpy() # shape: (n_subject M, n_covariates R)
            M_group = Z_group.shape[0]
            X = np.kron(Z_group, B_array)

            # sanity check: without any memory consideration
            # X = np.concatenate((np.ones((M_group*self.n_voxel, 1)), X), axis=1) # shape: (M*N, P*R+1)
            # Y_group = Y_group.reshape(-1,) # shape: (M*N,)
            # beta_group = self.irls_log_glm(beta_group, X, Y_group, max_iter=100, tol=1e-6)
            
            ## calculate the Hessian (X_T WX)^-1, not updated in each iteration 
            # spatial mu effect
            log_mu_spatial = B_array @ beta_meta # shape: (n_voxel, 1)
            mu_spatial = np.exp(log_mu_spatial)
            # covariates mu effect
            log_mu_covatiate= Z_group @ gamma_meta
            mu_covariates = np.exp(log_mu_covatiate) # shape: (n_subject, 1)

            # (X^TWX)^{-1} = [Z^T W_Z Z]^{-1} otimes [B^T W_B B]^{-1}
            Z_TWZ = Z_group.T @ (mu_covariates * Z_group) # shape: (n_covariates, n_covariates)
            Z_TWZ_inv = np.linalg.inv(Z_TWZ)
            B_TWB = B_array.T @ (mu_spatial * B_array)
            B_TWB_inv = np.linalg.inv(B_TWB)
            X_TWX_inv = np.kron(Z_TWZ_inv, B_TWB_inv) # shape: (n_covariates*n_bases, n_covariates*n_bases)
            del log_mu_spatial, mu_spatial, log_mu_covatiate, mu_covariates
            del Z_TWZ, Z_TWZ_inv, B_TWB, B_TWB_inv

            # initisation
            beta_group = 0.01 * np.random.rand(self.n_covariates*self.n_bases, 1)
            b_group = np.log(np.mean(Y_group))
            diff_beta = np.inf

            while diff_beta > tolerance_change:
                # (X^TWX)^{-1} X^T Y is a fixed term
                # X^T mu = (Z^T otimes B^T) exp[(Z otimes B)beta]
                # bar{eta} = 1/M (1_M^T Z) otimes B beta
                bar_eta = 1/M_group * np.kron(np.sum(Z_group, axis=0).reshape(1, -1), B_array) @ beta_group + b_group# shape: (n_voxel, 1)
                one_M_exp_eta = np.broadcast_to(np.exp(bar_eta).T, (M_group, self.n_voxel)).reshape(-1,1) # shape: (M*N, 1)
                diff_term = np.kron((np.eye(M_group) - 1/M_group*np.ones((M_group, M_group))) @ Z_group, B_array) @ beta_group + b_group # shape: (M*N, 1)
                exp_ZB_beta = one_M_exp_eta + diff_term * one_M_exp_eta # shape: (M*N, 1)
                del one_M_exp_eta, diff_term

                # X^T mu = (Z^T otimes B^T) exp[(Z otimes B)beta]
                X_T_mu = X.T @ exp_ZB_beta # shape: (n_covariates*n_bases, 1)
                update_term = X_TWX_inv @ X_T_mu # shape: (n_covariates*n_bases, 1)
                fixed_term = X_TWX_inv @ X.T @ Y_group # shape: (n_covariates*n_bases, 1)
                beta_group = beta_group + fixed_term - update_term
                diff_beta = np.linalg.norm(update_term)
                print(f"Group: {group_name}, Diff beta: {diff_beta}")
                
                b_group_update = b_group + np.sum(Y_group - exp_ZB_beta) / np.sum(exp_ZB_beta)
                b_group = b_group_update
                del X_T_mu, update_term, fixed_term

                # If update X_TWX_inv in each iteration
                updated_mu = np.exp(X @ beta_group + b_group)
                X_TWX = X.T @ (updated_mu * X)
                X_TWX_inv = np.linalg.inv(X_TWX + 1e-8 * np.eye(X_TWX.shape[0]))
                del updated_mu, X_TWX

                # # sanity check: without any memory consideration
                # Y_group = Y_group.reshape(-1,1) # shape: (M*N,1)
                # mu = np.exp(X @ beta_group + b_group) # shape: (M*N, 1)
                # X_TWX = X.T @ (mu * X)
                # X_TWX_inv = np.linalg.inv(X_TWX + 1e-8 * np.eye(X_TWX.shape[0]))
                # beta_group_update = beta_group + 1.0 * X_TWX_inv @ X.T @ (Y_group - mu)
                # diff_beta = np.linalg.norm(beta_group_update - beta_group)
                # b_group_update = b_group + np.sum(Y_group - mu) / np.sum(mu)
                # print(f"Group: {group_name}, Diff beta: {diff_beta}")
                # beta_group = beta_group_update
                # b_group = b_group_update
                # del mu, X_TWX, X_TWX_inv, beta_group_update
                # # (X^TWX)^{-1} = [Z^T W_Z Z]^{-1} otimes [B^T W_B B]^{-1}

            updated_beta[group_name] = beta_group
            del beta_group, Y_group, Z_group, X

        return updated_beta
    
    def estimated_P(self, beta):
        P = dict()
        B_array = self.B.detach().cpu().numpy() # shape: (n_voxel N, n_bases P)
        for group_name in self.group_names:
            Y_group = self.Y[group_name].detach().cpu().numpy()
            Z_group = self.Z[group_name].detach().cpu().numpy() # [M, R]
            beta_group = beta[group_name] # [P*R, 1]
            b_group = np.log(np.mean(Y_group))
            X = np.kron(Z_group, B_array)
            # X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1) # shape: (M*N, P*R+1)
            log_P_spatial = X @ beta_group + b_group # shape: (M*N, 1)
            P_spatial = np.exp(log_P_spatial)
            print("P: ", np.min(P_spatial), np.max(P_spatial), np.mean(P_spatial))
            print("MU: ", torch.min(self.MU[group_name]), torch.max(self.MU[group_name]), torch.mean(self.MU[group_name]))
            P[group_name] = P_spatial.reshape(-1, self.n_voxel)
            del log_P_spatial, P_spatial, X, beta_group, Z_group

        return P