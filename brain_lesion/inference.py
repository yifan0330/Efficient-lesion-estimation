import torch
import numpy as np
import scipy
import time
import gc
import os
from tqdm import tqdm 
import matplotlib.pyplot as plt
from model import SpatialBrainLesionModel

class BrainInference(object):
    def __init__(self, model, marginal_dist, link_func, regression_terms, 
                polynomial_order, n_categorical_covariate, dtype=torch.float64, device='cpu'):
        self.model = model
        self.marginal_dist = marginal_dist
        self.link_func = link_func
        self.regression_terms = regression_terms
        self.polynomial_order = polynomial_order
        self.n_categorical_covariate = n_categorical_covariate
        self.dtype = dtype
        self.device = device
        self._kwargs = {"device": self.device, "dtype": self.dtype}
    
    def load_params(self, params, n_categorical_covariate=1):
        # load X_spatial, G, P, Y
        self.G = params["G"].item()
        self.X_spatial_array = params["X_spatial"]
        self.X_spatial = torch.tensor(self.X_spatial_array, **self._kwargs)
        # group names
        self.group_names = list(self.G.keys())
        self.n_group = len(self.group_names)
        # load P, Y, Z
        self.P = params["P"]
        self.eta = np.log(self.P)
        self.P_mean = np.stack([np.mean(self.P[self.G[group], :], axis=0) for group in self.group_names], axis=0) # shape: (n_group, n_voxel)
        self.eta_mean = np.stack([np.mean(self.eta[self.G[group], :], axis=0) for group in self.group_names], axis=0) # shape: (n_group, n_voxel)
        self.Y = params["Y"]
        self.Z = params["Z"]
        if "multiplicative" in self.regression_terms:
            self.bias_W = torch.tensor(params["bias_W"], **self._kwargs) 
            self.beta_W = torch.tensor(params["beta_W"], **self._kwargs)
        else:
            self.bias_W, self.beta_W = None, None
        if "additive" in self.regression_terms:
            self.bias_b = torch.tensor(params["bias_b"], **self._kwargs)
            self.beta_b = torch.tensor(params["beta_b"], **self._kwargs)
        else:
            self.bias_b, self.beta_b = None, None
        # spatial coefficient dimension
        self.n_voxel, self.n_bases = self.X_spatial.shape
        self.n_covariate = self.Z.shape[1]
        self.n_subject = [len(self.G[group]) for group in self.group_names]

    def create_contrast(self, t_con_groups=None, t_con_group_name=None):
        self.t_con_group_name = t_con_group_name
        # Preprocess the contrast vector
        self.t_con_groups = (
            np.eye(self.n_group)
            if t_con_groups is None
            else np.array(t_con_groups).reshape(1, -1)
        )
        # raise error if dimension of contrast vector doesn't match with number of groups
        if self.t_con_groups.shape[1] != self.n_group:
            raise ValueError(
                f"""The shape of contrast vector: {str(self.t_con_groups)}
                doesn't match with number of groups."""
            )
        # standardization (row sum 1)
        self.t_con_groups = self.t_con_groups / np.sum(np.abs(self.t_con_groups), axis=1).reshape((-1, 1))

    def run_inference(self, method="FI"):
        # Generalised linear hypothesis testing
        inference = self._glh_con_group(method)
        # Plot the estimated P, standard error of P, and p-values
        print(f"fig_{method}_{self.t_con_group_name}")
        self.plot_1d(inference, f"fig_{method}_{self.t_con_group_name}", 0.05)


    def _glh_con_group(self, method, batch_size=20):
        t_con_groups_involved_index = np.where(self.t_con_groups != 0)[1].tolist()
        t_con_groups_involved = [self.group_names[i] for i in t_con_groups_involved_index]
        self.n_con_group_involved = len(t_con_groups_involved)
        # Compute the contrast P_mean and eta_mean
        contrast_P = self.t_con_groups @ self.P_mean # shape: (1, n_voxel)
        contrast_eta = self.t_con_groups @ self.eta_mean # shape: (1, n_voxel)

        # Hessian matrix
        if method == "FI":
            _, F_beta_W = self._Fisher_info(t_con_groups_involved)
            cov_beta_W = np.linalg.inv(F_beta_W + 1e-8*np.eye(F_beta_W.shape[0]))
            # shape: (n_group*n_bases*polynomial_order, n_group*n_bases*polynomial_order)
            print("cov_beta_W", cov_beta_W.shape)
        elif method == "sandwich":
            cov_beta_dict = dict()
            for k in range(self.n_con_group_involved):
                group = t_con_groups_involved[k]
                subject_indices = self.G[group]
                Z_group = self.Z[subject_indices, :] # shape: (n_subject, n_covariates)
                P_group = self.P[subject_indices, :] # shape: (n_subject, n_voxel)
                Y_group = self.Y[subject_indices, :] # shape: (n_subject, n_voxel)
                bread_term_group = self.bread_term(Z_group, self.X_spatial_array, P_group, polynomial_order) # shape: (n_covariates*n_bases*poly_order, n_covariates*n_bases*poly_order)
                meat_term_group = self.meat_term(Z_group, self.X_spatial_array, P_group, Y_group, polynomial_order) # shape: (n_covariates*n_bases*poly_order, n_covariates*n_bases*poly_order)
                # sandwich estimator
                cov_beta = bread_term_group @ meat_term_group @ bread_term_group # shape: (n_covariates*n_bases, n_covariates*n_bases)
                cov_beta_dict[group] = [cov_beta[self.n_covariate*self.n_bases*i:self.n_covariate*self.n_bases*(i+1), 
                                                self.n_covariate*self.n_bases*i:self.n_covariate*self.n_bases*(i+1)] 
                                        for i in range(polynomial_order)]
                del bread_term_group, meat_term_group, cov_beta
        # Compute the variance of P, from the variance of beta_W
        var_bar_P, var_bar_eta = list(), list()
        for k in range(self.n_con_group_involved):
            group = t_con_groups_involved[k]
            group_index = self.group_names.index(group)
            print("index", group_index)
            subject_indices = self.G[group]
            print("group index k:", k, group)
            print(k, self.n_con_group_involved*self.n_bases*self.n_covariate)
            if method == "FI":
                cov_beta_W_group = cov_beta_W[
                    k*self.n_con_group_involved*self.n_bases*self.n_covariate : (k + 1)*self.n_con_group_involved*self.n_bases*self.n_covariate,
                    k*self.n_con_group_involved*self.n_bases*self.n_covariate : (k + 1)*self.n_con_group_involved*self.n_bases*self.n_covariate,
                ] # shape: (n_bases*polynomial_order, n_bases*polynomial_order)
            elif method == "sandwich":
                cov_beta_group = cov_beta_dict[group]
                print(len(cov_beta_group))
                print(cov_beta_group[0].shape, cov_beta_group[1].shape, cov_beta_group[2].shape)
            Z_group = self.Z[subject_indices, :] # shape: (n_subject, n_covariates)
            P_mean_group = self.P_mean[group_index] # shape: (n_voxel,)
            # eta_ij = Z_i^T beta_W^T X_j
            # average: bar_eta_j = bar_Z^T beta_W^T X_j = vec(beta_W) (X_j \otimes bar_Z^T)
            # COV(bar_eta_j, bar_eta_k) = (X_j \otimes bar_Z^T)^T COV(vec(beta_W)) (X_k \otimes bar_Z^T)
            # j^th row of A: A_j = (X_j \otimes bar_Z^T)^T 
            # Cov(bar_eta) = A COV(vec(beta_W)) A^T
            bar_Z_group = np.mean(Z_group, axis=0).reshape(-1,1) # shape: (n_covariates,1)
            A = np.kron(self.X_spatial_array, bar_Z_group.T) # shape: (n_voxel, n_covariates*n_bases)
            var_bar_eta_group = np.einsum('ik,kl,il->i', A, cov_beta_W_group, A) 
            del bar_Z_group_k, A_k

            print("var_bar_eta_group", var_bar_eta_group.shape)
            var_bar_P_group = P_mean_group**2*var_bar_eta_group # shape: (n_voxel,)
            print("var_bar_P_group", var_bar_P_group.shape)
            # save the variance of P and eta
            var_bar_eta.append(var_bar_eta_group)
            var_bar_P.append(var_bar_P_group)
        var_bar_eta = np.stack(var_bar_eta, axis=0) # shape: (n_con_group_involved, n_voxel)
        var_bar_P = np.stack(var_bar_P, axis=0) # shape: (n_con_group_involved, n_voxel)

        # Compute the numerator of the Z test
        involved_var_bar_eta = self.t_con_groups**2 @ var_bar_eta # shape: (1, n_voxel)
        involved_std_bar_eta = np.sqrt(involved_var_bar_eta) # shape: (1, n_voxel)
        involved_var_bar_P = self.t_con_groups**2 @ var_bar_P # shape: (1, n_voxel)
        involved_std_bar_P = np.sqrt(involved_var_bar_P) # shape: (1, n_voxel)
        # Conduct Wald test (Z test)
        z_stats_eta = contrast_eta / involved_std_bar_eta
        z_stats_P = contrast_P / involved_std_bar_P
        p_vals_eta = scipy.stats.norm.sf(z_stats_eta) # shape: (1, n_voxel)
        p_vals_P = scipy.stats.norm.sf(z_stats_P) # shape: (1, n_voxels)
        # Return the inference results
        eta_inference = (contrast_eta, involved_std_bar_eta, p_vals_eta)
        P_inference = (contrast_P, involved_std_bar_P, p_vals_P)

        return [eta_inference, P_inference]
    
    def _Fisher_info(self, t_con_groups_involved):
        # Load Y, Z
        Y = torch.tensor(self.Y, **self._kwargs)
        Z = torch.tensor(self.Z, **self._kwargs)
        # Compute the Fisher information matrix
        if self.model == "SpatialBrainLesion":
            # nll = SpatialBrainLesionModel._neg_log_likelihood(self.marginal_dist,
            #                                                 self.link_func,
            #                                                 self.polynomial_order,
            #                                                 self.regression_terms,
            #                                                 self.X_spatial,
            #                                                 Y,
            #                                                 Z,
            #                                                 self.bias_W,
            #                                                 self.beta_W,
            #                                                 self.bias_b,
            #                                                 self.beta_b,
            #                                                 self.device)
            nll = lambda beta_W: SpatialBrainLesionModel._neg_log_likelihood(
                                                                    self.marginal_dist,
                                                                    self.link_func,
                                                                    self.polynomial_order,
                                                                    self.regression_terms,
                                                                    self.X_spatial,
                                                                    Y,
                                                                    Z,
                                                                    self.bias_W,
                                                                    beta_W,
                                                                    self.bias_b,
                                                                    self.beta_b,
                                                                    self.device)
            params = (self.beta_W)
            # Jacobian
            J = torch.autograd.functional.jacobian(nll, params, create_graph=False) 
            J = J.view(self.n_group*self.n_bases*self.polynomial_order, -1) # shape: (n_group*n_bases*polynomial_order, n_group*n_bases*polynomial_order)
            # Hessian
            H = torch.autograd.functional.hessian(nll, params, create_graph=False)
            H = H.view(self.n_group*self.n_bases*self.polynomial_order, -1) # shape: (n_group*n_bases*polynomial_order, n_group*n_bases*polynomial_order)

            return J.detach().cpu().numpy(), H.detach().cpu().numpy()
    
    def bread_term(self, Z, X_spatial, P):
        start_time = time.time()
        n_subject = Z.shape[0]
        # bread term: (X^TWX)^-1
        dim = self.n_covariate * self.n_bases * polynomial_order
        H = np.zeros((dim, dim))
        # Reshape Z and X_spatial to add dimensions for broadcasting
        Z_expand = Z[:, np.newaxis, :, np.newaxis]  # Shape: (n_subject, 1, n_covariates, 1)
        X_expand = X_spatial[np.newaxis, :, np.newaxis, :]  # Shape: (1, n_voxel, 1, n_bases)
        # Compute Z_i X_j^T for all i, j at once
        # shape: (n_subject, n_voxel, n_covariates, n_bases, polynomial_order)
        Z_X = np.stack([Z_expand**i * X_expand for i in range(polynomial_order)], axis=-1)
        for i in range(n_subject):
            for j in range(self.n_voxel):
                # Flatten Z_i_X_j into a vector
                Z_X_flat = Z_X[i, j].ravel()  # shape: (n_covariates*n_bases*polynomial_order,)
                # Compute the Kronecker product with its transpose
                kron_prod = np.outer(Z_X_flat, Z_X_flat)  # shape: (n_covariates*n_bases*polynomial_order, n_covariates*n_bases*polynomial_order)
                # Accumulate into H with the corresponding weight
                H += -P[i, j] * kron_prod
                del Z_X_flat, kron_prod
        bread_term = np.linalg.inv(-H + 1e-8 * np.eye(dim))
        # shape: (n_covariates*n_bases, n_covariates*n_bases)
        del Z_expand, X_expand, Z_X, H
        gc.collect()
        print(f"Time taken for bread term: {time.time() - start_time}")
        return bread_term
    
    def meat_term(self, Z, X_spatial, P, Y):
        # meat term: sum_M [D_i^TV_i^{-1}(Y_i-P_i)]*[D_i^TV_i^{-1}(Y_i-P_i)]^T
        start_time = time.time()
        n_subject = Z.shape[0]
        start_time = time.time()
        dim = self.n_covariate * self.n_bases * polynomial_order
        # meat term: sum_M [D_i^TV_i^{-1}(Y_i-P_i)]*[D_i^TV_i^{-1}(Y_i-P_i)]^T
        meat_term = np.zeros((dim, dim))
        for i in range(n_subject):
            Z_i = np.stack([Z[i]**i for i in range(polynomial_order)], axis=-1).T # shape (polynomial_order, n_covariates)
            D_i = list()
            for j in range(self.n_voxel):
                X_j = X_spatial[j].reshape((1,-1)) # shape: (1, n_bases)
                D_ij = np.kron(Z_i, X_j).reshape(1, -1) # shape: (1, n_covariates*n_bases*polynomial_order)
                D_i.append(D_ij)
                del X_j, D_ij
            D_i = np.concatenate(D_i, axis=0) # shape: (n_voxel, n_covariates*n_bases*polynomial_order)
            residue = (Y[i] - P[i]).reshape((-1,1)) # shape: (N, 1)
            D_i_V_i_residue = D_i.T @ residue # shape: (n_covariates*n_bases, 1)
            # D_i.T @ V_i_inv @ residue # shape: (n_covariates*n_bases, 1)
            meat_term += D_i_V_i_residue @ D_i_V_i_residue.T # shape: (n_covariates*n_bases, n_covariates*n_bases)
            del Z_i, D_i, residue, D_i_V_i_residue
        gc.collect()
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
        unstacked_cov_beta_w  = np.stack(np.split(cov_beta_w, self.n_bases, axis=-1))
        unstacked_cov_beta_w = np.stack(np.split(unstacked_cov_beta_w, self.n_bases, axis=-2)) # [_P, _P, _R, _R]
        
        cov_A = unstacked_cov_beta_w @ Z.T[None, None, :, :] 
        cov_A = np.sum(cov_A * Z.T[None, None, :, :], axis=-2)
        cov_A = np.moveaxis(cov_A, -1, 0) # shape: (n_batch, n_bases, n_bases)
        var_eta = np.einsum('np,mpq,nq->mn', X, cov_A, X) # shape: (n_batch, n_voxel)
        var_P = P**2*var_eta # shape: (n_batch, n_voxel)
        # cov_eta = X[None, :, :] @ cov_A @ X.T[None, :, :] # shape: (n_batch, n_voxel, n_voxel)
        # cov_P = cov_eta * P[:, :, None] * P[:, None, :] # shape: (n_batch, n_voxel, n_voxel)
        del unstacked_cov_beta_w, P, cov_A, var_eta,
        gc.collect()
        
        return var_P
    
    def plot_1d(self, inference, filename, significance_level=0.05):
        variable_names = ["eta", "P"]
        # slice list
        fig, axes = plt.subplots(2, 4, figsize=(40, 22))
        
        print(len(inference))
        for i in range(len(inference)):
            contrast, std, p_vals = inference[i]
            var = variable_names[i]
            # Subplot 0
            axes[i,0].plot(contrast.squeeze(), label=f'Estimated {var} contrast')
            axes[i,0].axhline(y=0.0, color='red', linestyle='--', label='y=0')
            axes[i,0].set_xlabel("Voxel")
            axes[i,0].set_ylabel("Intensity diff")
            axes[i,0].set_title(f"Contrast {var}", fontsize=30)
            axes[i,0].legend()

            # Subplot 1
            axes[i,1].plot(std.squeeze(), label=f'Std {var}')
            axes[i,1].set_xlabel("Voxel")
            axes[i,1].set_ylabel("Standard Error")
            axes[i,1].set_title(f"Std {var}", fontsize=30)
            axes[i,1].legend()

            # Subplot 2
            axes[i,2].plot(p_vals.squeeze(), label=f'p values')
            axes[i,2].axhline(y=0.05, color='red', linestyle='--', label='alpha=0.05')
            axes[i,2].set_xlabel("Voxel")
            axes[i,2].set_ylabel("P-values")
            axes[i,2].set_title(f"P-values of {var}", fontsize=30)
            axes[i,2].legend()

            # Subplot 3
            N = p_vals.shape[1]
            # theoretical p-values 
            th_p = np.arange(1/float(N),1+1/float(N),1/float(N)) # shape: (n_voxel,)
            th_p_log = -np.log10(th_p)
            # kth order statistics
            k_array = np.arange(start=1, stop=N+1, step=1)
            # empirical confidence interval (estimated from p-values)
            z_1, z_2 = scipy.stats.norm.ppf(significance_level), scipy.stats.norm.ppf(1-significance_level)
            # sort the order of p-values under -log10 scale
            sorted_p_vals = np.sort(p_vals, axis=1).squeeze() # shape: (n_voxel,)
            # Add the Beta confidence interval
            CI_lower = scipy.stats.beta.ppf(significance_level/2, k_array, N - k_array + 1)
            CI_upper = scipy.stats.beta.ppf(1 - significance_level/2, k_array, N - k_array + 1)

            axes[i,3].fill_between(th_p_log, -np.log10(CI_lower), -np.log10(CI_upper), color='grey', alpha=0.5,
                    label=f'{int((1-significance_level)*100)}% Beta CI')
            axes[i,3].plot(th_p_log, np.repeat(-np.log10(0.05), N), color='y', linestyle='--', label='threshold at -log10(0.05)')
            axes[i,3].plot(th_p_log, -np.log10(th_p), color='orange', linestyle='--', label='y=x')
            axes[i,3].plot(th_p_log, -np.log10(significance_level * th_p), color='red', linestyle='-', label='FDR(BH) control')
            axes[i,3].scatter(th_p_log, -np.log10(sorted_p_vals), c='#1f77b4', s=4)
            print(np.max(-np.log10(k_array/N)))
            axes[i,3].set_xlim([0, np.max(-np.log10(k_array/N))])
            axes[i,3].set_ylim([0, np.max(-np.log10(k_array/N))]) 
            axes[i,3].set_xlabel("Expected -log10(P)")
            axes[i,3].set_ylabel("Observed -log10(P)")
            axes[i,3].set_title(f"PP-plot (-log10) of {var}", fontsize=30)
            axes[i,3].legend()

        # Save the figure
        fig.savefig(f"{os.getcwd()}/figures/{filename}.png")

