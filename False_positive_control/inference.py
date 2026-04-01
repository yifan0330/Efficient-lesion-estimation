import torch
import numpy as np
import scipy
import time
import gc
import os
from tqdm import tqdm 
import matplotlib.pyplot as plt
from model import SpatialBrainLesionModel, MassUnivariateRegression
from util import compute_eta_mean, compute_mu, SpatialGLM_compute_mu_mean, efficient_kronT_diag_kron, robust_inverse, robust_inverse_generalised, eigenspectrum
from plot import plot_brain
from statsmodels.stats.multitest import fdrcorrection

class BrainInference_full(object):
    def __init__(self, model,space_dim, marginal_dist, link_func, regression_terms, random_seed, fewer_voxels=False,
                dtype=torch.float64, device='cpu'):
        self.model = model
        self.space_dim=space_dim
        self.marginal_dist = marginal_dist
        self.link_func = link_func
        self.regression_terms = regression_terms
        self.random_seed = random_seed
        self.fewer_voxels = fewer_voxels
        self.dtype = dtype
        self.device = device
        self._kwargs = {"device": self.device, "dtype": self.dtype}
    
    def load_params(self, data, params):
        # load X_spatial, P, Y
        self.group_names = list(data.keys())
        self.n_group = len(self.group_names)
        # load data["X_spatial"] from any group
        X_spatial = data["X_spatial"]
        self.X_spatial_array = np.concatenate([X_spatial, np.ones((X_spatial.shape[0], 1))], 
                                              axis=1)
        self.X_spatial = torch.tensor(self.X_spatial_array, **self._kwargs)
        self.n_subject_per_group = data["Z"].shape[0]
        # load P, Y, Z
        P = params["P"] 
        self.P_mean = np.stack([np.mean(P_group, axis=0) for P_group in P], axis=0) # shape: (n_group, n_voxel)
        # if self.fewer_voxels:
        #     # select 100 voxels with largest mean P across subjects
        #     mean_P = np.mean(self.P, axis=0)  # shape: (n_voxel,)
        #     top_voxel_indices = np.argsort(mean_P)[-100:]  # indices of top 100 voxels
        #     self.P = self.P[:, top_voxel_indices]  # shape: (n_subject, 100)
        #     # Also need to subset X_spatial_array to match the selected voxels
        #     self.X_spatial_array = self.X_spatial_array[top_voxel_indices, :]
        #     self.X_spatial = torch.tensor(self.X_spatial_array, **self._kwargs)
        self.eta = np.log(self.P_mean)
        # After: Multi-line for clarity
        self.Y = torch.tensor(data["Y"], **self._kwargs)
        # Create intercept column
        intercept_col = np.ones((data["Z"].shape[0], 1))
        self.Z = np.concatenate([data["Z"], intercept_col], axis=1)
        self.Z = torch.tensor(self.Z, **self._kwargs)

        self.n_subject, self.n_covariates = self.Z.shape
        self.beta = torch.tensor(params["beta"], **self._kwargs)
        self.beta_array = params["beta"]
        # spatial coefficient dimension
        self.n_voxel, self.n_bases = self.X_spatial.shape

    def create_contrast(self, contrast_vector=None, contrast_name=None):
        self.contrast_vector = contrast_vector
        self.contrast_name = contrast_name
        # Preprocess the contrast vector
        self.contrast_vector = (
            np.eye(self.n_covariates)
            if contrast_vector is None
            else np.array(contrast_vector).reshape(1, -1)
        )
        # raise error if dimension of contrast vector doesn't match with number of groups
        if self.contrast_vector.shape[1] != self.n_covariates:
            raise ValueError(
                f"""The shape of contrast vector: {str(self.contrast_vector)}
                doesn't match with number of covariates."""
            )
        # # raise error if dimension of contrast vector doesn't match with number of groups
        # if self.contrast_vector.shape[1] != self.n_covariates:
        #     raise ValueError(
        #         f"""The shape of contrast vector: {str(self.contrast_vector)}
        #         doesn't match with number of groups."""
        #     )
        # standardization (row sum 1)
        self.contrast_vector = self.contrast_vector / np.sum(np.abs(self.contrast_vector), axis=1).reshape((-1, 1))

    def run_inference(self, method="FI", inference_filename=None, fig_filename=None, lesion_mask=None, alpha=0.05):
        z_threshold = scipy.stats.norm.ppf(1-alpha)
        # Generalised linear hypothesis testing
        if not os.path.exists(inference_filename):
            p_vals, z_stats = self._glh_con_group(method)
            np.savez(inference_filename, p_vals=p_vals, z_stats=z_stats)
        else:
            p_vals = np.load(inference_filename)["p_vals"]
        print(p_vals.shape, "p_vals shape")
        # Plot the estimated P, standard error of P, and p-values
        print(f"Plotting inference results to {fig_filename}")
        # self.plot_1d(p_vals, fig_filename, 0.05)
        # set z_stats values over 15 as NaN
        plot_brain(p=z_stats, brain_mask=lesion_mask, threshold=z_threshold, output_filename=fig_filename)

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
            # # sandwich estimator
            cov_beta = [B @ M @ B.T for B, M in zip(bread_term, meat_term)]
            del bread_term, meat_term
        print("Variance of beta computed")

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
        print(a.shape)

        contrast_var_bar_eta = self.contrast_vector**2 @ a # shape: (1, n_voxel)
        contrast_std_bar_eta = np.sqrt(contrast_var_bar_eta) # shape: (1, n_voxel)
        # Conduct Wald test (Z test)
        z_stats_eta = contrast_eta / contrast_std_bar_eta
        z_stats = np.concatenate([z_stats_eta, -z_stats_eta], axis=0) # shape: (2, n_voxel)
        p_vals = scipy.stats.norm.sf(z_stats) # shape: (2, n_voxel)
        print(np.min(p_vals), np.max(p_vals), np.count_nonzero(p_vals < 0.05), p_vals.shape)
        exit()
        
    # def _glh_con_group(self, method, batch_size=20):
    #     for group in self.group_names:
    #         bar_Z = np.mean(self.Z[group], axis=0) # shape: (n_covariates,)
    #         group_n_subjects = bar_Z[1:1+self.n_group]
    #         group_ratio = group_n_subjects / np.max(group_n_subjects)
    #         print(group_ratio, "group ratio")
    #         # for covariate s, bar_eta_s = bar_eta_s * X @ beta_s -- dim: (1, n_voxel)
    #         bar_eta_covariates = (bar_Z * self.beta_array[group]).T @ self.X_spatial_array.T # shape: (n_covariates, n_voxel)
    #         self.contrast_vector[:, 1:1+self.n_group:] /= group_ratio
    #         # self.contrast_vector[:, -self.n_group:] /= group_ratio
    #         print(self.contrast_vector)
    #         contrast_eta_covariates = self.contrast_vector @ bar_eta_covariates # shape: (1, n_voxel)
    #         print(np.min(contrast_eta_covariates), np.max(contrast_eta_covariates))
    #         del bar_eta_covariates
    #         # Estimate the variance of beta, from either FI or sandwich estimator
    #         if method == "FI":
    #             F_beta = self._Fisher_info()[group] # shape: (n_covariates_expand, n_covariates_expand)
    #             cov_beta = [np.linalg.inv(F_beta[:,i,:,i]+1e-6*np.eye(self.n_bases)) for i in range(self.n_covariates[group])]
    #             del F_beta
    #         elif method == "sandwich":
    #             bread_term = self.bread_term(self.Z["Group_1"], self.X_spatial_array, self.P) # list: len = n_covariates
    #             meat_term = self.meat_term(self.Z["Group_1"], self.X_spatial_array, self.P, self.Y) # list: len = n_covariates
    #             # # sandwich estimator
    #             cov_beta = [B @ M @ B.T for B, M in zip(bread_term, meat_term)]
    #             del bread_term, meat_term
    #         print("Variance of beta computed")
    #         var_bar_eta = list()
    #         for s in range(self.n_covariates[group]):
    #             # for covariate s, at voxel j
    #             # bar_eta_sj = bar_Z_s * X_j^T @ beta_s -- dim: (1,)
    #             # COV(bar_eta_sj) = bar_Z_s * X_j^T @ COV(beta_s) @ X_j -- dim: (1,)
    #             # COV(bar_eta_s) = bar_P_s**2 * X @ COV(beta_s) @ X^T -- dim: (n_voxel, n_voxel)
    #             print("bar_Z", bar_Z[s])
    #             var_bar_eta_s = bar_Z[s] * np.einsum('ij,jk,ik->i', self.X_spatial_array, cov_beta[s], self.X_spatial_array)
    #             var_bar_eta.append(var_bar_eta_s)
    #             del var_bar_eta_s
    #         var_bar_eta = np.stack(var_bar_eta, axis=0) # shape: (n_covariate, n_voxel)
    #         del cov_beta
    #         # Compute the numerator of the Z test
    #         contrast_var_bar_eta = self.contrast_vector**2 @ var_bar_eta # shape: (1, n_voxel)
    #         contrast_std_bar_eta = np.sqrt(contrast_var_bar_eta) # shape: (1, n_voxel)
    #         # Conduct Wald test (Z test)
    #         z_stats_eta = contrast_eta_covariates / contrast_std_bar_eta
    #         z_stats = np.concatenate([z_stats_eta, -z_stats_eta], axis=0) # shape: (2, n_voxel)
    #         p_vals = scipy.stats.norm.sf(z_stats) # shape: (2, n_voxel)

    #     return p_vals


    # def _glh_con_group(self, method, batch_size=20):
    #     bar_Z = np.mean(self.Z, axis=0) # shape: (n_covariates,)
    #     group_n_subjects = bar_Z[1:1+self.n_group]
    #     group_ratio = group_n_subjects / np.max(group_n_subjects)
    #     print(group_ratio, "group ratio")
    #     # for covariate s, bar_eta_s = bar_eta_s * X @ beta_s -- dim: (1, n_voxel)
    #     print(bar_Z.shape, self.beta_array.shape, self.X_spatial_array.shape)
    #     bar_eta_covariates = (bar_Z * self.beta_array).T @ self.X_spatial_array.T # shape: (n_covariates, n_voxel)
    #     self.contrast_vector[:, 1:1+self.n_group:] /= group_ratio
    #     # self.contrast_vector[:, -self.n_group:] /= group_ratio
    #     print(self.contrast_vector)
    #     contrast_eta_covariates = self.contrast_vector @ bar_eta_covariates # shape: (1, n_voxel)
    #     print(np.min(contrast_eta_covariates), np.max(contrast_eta_covariates))
    #     del bar_eta_covariates
    #     # Estimate the variance of beta, from either FI or sandwich estimator
    #     if method == "FI":
    #         F_beta = self._Fisher_info() # shape: (n_covariates_expand, n_covariates_expand)
    #         cov_beta = [np.linalg.inv(F_beta[:,i,:,i]+1e-6*np.eye(self.n_bases)) for i in range(self.n_covariates)]
    #         del F_beta
    #     elif method == "sandwich":
    #         bread_term = self.bread_term(self.Z, self.X_spatial_array, self.P) # list: len = n_covariates
    #         meat_term = self.meat_term(self.Z, self.X_spatial_array, self.P, self.Y) # list: len = n_covariates
    #         # # sandwich estimator
    #         cov_beta = [B @ M @ B.T for B, M in zip(bread_term, meat_term)]
    #         del bread_term, meat_term
    #     print("Variance of beta computed")
    #     var_bar_eta = list()
    #     for s in range(self.n_covariates):
    #         # for covariate s, at voxel j
    #         # bar_eta_sj = bar_Z_s * X_j^T @ beta_s -- dim: (1,)
    #         # COV(bar_eta_sj) = bar_Z_s * X_j^T @ COV(beta_s) @ X_j -- dim: (1,)
    #         # COV(bar_eta_s) = bar_P_s**2 * X @ COV(beta_s) @ X^T -- dim: (n_voxel, n_voxel)
    #         var_bar_eta_s = bar_Z[s] * np.einsum('ij,jk,ik->i', self.X_spatial_array, cov_beta[s], self.X_spatial_array)
    #         var_bar_eta.append(var_bar_eta_s)
    #         del var_bar_eta_s
    #     var_bar_eta = np.stack(var_bar_eta, axis=0) # shape: (n_covariate, n_voxel)
    #     print(np.min(var_bar_eta[1]), np.mean(var_bar_eta[1]), np.max(var_bar_eta[1]))
    #     print(np.min(var_bar_eta[2]), np.mean(var_bar_eta[2]), np.max(var_bar_eta[2]))
    #     del cov_beta
    #     # Compute the numerator of the Z test
    #     contrast_var_bar_eta = self.contrast_vector**2 @ var_bar_eta # shape: (1, n_voxel)
    #     contrast_std_bar_eta = np.sqrt(contrast_var_bar_eta) # shape: (1, n_voxel)
    #     # Conduct Wald test (Z test)
    #     z_stats_eta = contrast_eta_covariates / contrast_std_bar_eta
    #     z_stats = np.concatenate([z_stats_eta, -z_stats_eta], axis=0) # shape: (2, n_voxel)
    #     p_vals = scipy.stats.norm.sf(z_stats) # shape: (2, n_voxel)
    #     print(np.min(p_vals), np.max(p_vals), np.count_nonzero(p_vals < 0.05), p_vals.shape)
        
    #     return p_vals

    
    
    def _Fisher_info(self):
        Fisher_info_filename = os.getcwd() + f"/results/{self.space_dim}/GRF_{self.n_subject}/" \
                            f"{self.model}_{self.marginal_dist}_{self.link_func}/Fisher_info_{self.random_seed}.npz"
        print(os.path.dirname(Fisher_info_filename))
        os.makedirs(os.path.dirname(Fisher_info_filename), exist_ok=True) if not os.path.exists(os.path.dirname(Fisher_info_filename)) else None
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
            # np.savez(Fisher_info_filename, H=H.detach().cpu().numpy())
            print(f"Time taken for Fisher information: {time.time() - start_time}")

        return H.detach().cpu().numpy()

    def bread_term(self, Z, X_spatial, P):
        start_time = time.time()
        # Compute A: outer product for each subject, 
        # shape: (n_subject, n_covariates, n_covariates)
        # Z_i = Z[:, i].reshape(-1, 1) # shape: (n_subject, 1)
        A = np.einsum('ia,ib->iab', Z, Z)
        # Compute B: for each subject, sum_j P[i,j] * (X_spatial[j] outer X_spatial[j])
        # shape: (n_subject, n_bases, n_bases)
        B = np.einsum('ij,jk,jl->ikl', P, X_spatial, X_spatial)
        # Use the identity: kron(A, B) = einsum('ab,cd->acbd', A, B)
        # Summing over subjects gives H_tensor of shape (n_covariates, n_bases, n_covariates, n_bases)
        H = np.einsum('iab,icd->acbd', A, B)
        print(H.shape, "H shape")
        bread_term = [np.linalg.inv(H[i,:,i,:] + 1e-6*np.eye(self.n_bases)) for i in range(self.n_covariates)]
        # shape: (n_covariates*n_bases, n_covariates*n_bases)
        del H, Z, A, B
        gc.collect()
        print(f"Time taken for bread term: {time.time() - start_time}")
        return bread_term
    
    def meat_term(self, Z, X_spatial, P, Y):
        # meat term: sum_M [D_i^TV_i^{-1}(Y_i-P_i)]*[D_i^TV_i^{-1}(Y_i-P_i)]^T
        start_time = time.time()
        R = Y - P # shape: (n_subject, n_voxel)
        # 2. Compute the weighted spatial sum for each subject
        L = np.dot(R, X_spatial)  # shape: (n_subject, n_bases)
        # 3. For each subject, compute v_i = kron(Z[i], L[i])
        #    This uses einsum to compute the outer product for each subject,
        #    resulting in shape (n_subject, n_covariates, n_bases) and then reshapes it.
        V = [Z[:, i][:, None] * L for i in range(self.n_covariates)]
        # 4. Compute the meat term by summing the outer products of v_i
        meat_term = [Vi.T @ Vi for Vi in V]
        del R, L, V
        gc.collect()
        print(f"Time taken for meat term: {time.time() - start_time}")

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
    
    def plot_1d(self, p_vals, filename, significance_level=0.05):
        # slice list
        fig, axes = plt.subplots(1, 2, figsize=(23, 11))

        # Subplot 3
        M, N = p_vals.shape
        # theoretical p-values 
        th_p = np.arange(1/float(N),1+1/float(N),1/float(N)) # shape: (n_voxel,)
        th_p_log = -np.log10(th_p)
        # kth order statistics
        k_array = np.arange(start=1, stop=N+1, step=1)
        # empirical confidence interval (estimated from p-values)
        z_1, z_2 = scipy.stats.norm.ppf(significance_level), scipy.stats.norm.ppf(1-significance_level)
        # Add the Beta confidence interval
        CI_lower = scipy.stats.beta.ppf(significance_level/2, k_array, N - k_array + 1)
        CI_upper = scipy.stats.beta.ppf(1 - significance_level/2, k_array, N - k_array + 1)

        group_comparison = [[0, 1], [1, 0]]
        title_list = ["group_0 - group_1", "group_1 - group_0"]
        for i in range(M):
            # sort the order of p-values under -log10 scale
            sorted_p_vals = np.sort(p_vals[i, :]) # shape: (n_voxel,)
            significance_percentage = np.sum(sorted_p_vals < 0.05) / N
            print(significance_percentage)
            axes[i].fill_between(th_p_log, -np.log10(CI_lower), -np.log10(CI_upper), color='grey', alpha=0.5,
                    label=f'{int((1-significance_level)*100)}% Beta CI')
            axes[i].plot(th_p_log, np.repeat(-np.log10(0.05), N), color='y', linestyle='--', label='threshold at -log10(0.05)')
            axes[i].plot(th_p_log, -np.log10(th_p), color='orange', linestyle='--', label='y=x')
            axes[i].plot(th_p_log, -np.log10(significance_level * th_p), color='red', linestyle='-', label='FDR(BH) control')
            axes[i].scatter(th_p_log, -np.log10(sorted_p_vals), c='#1f77b4', s=4)
            axes[i].set_xlim([0, np.max(-np.log10(k_array/N))])
            axes[i].set_ylim([0, np.max(-np.log10(k_array/N))]) 
            axes[i].set_xlabel("Expected -log10(P)", fontsize=20)
            axes[i].set_ylabel("Observed -log10(P)", fontsize=20)
            axes[i].set_title(f"{title_list[i]}: {significance_percentage*100:.2f}% voxels rejected", fontsize=30)
            axes[i].legend()

        # Save the figure
        fig.savefig(filename)

class BrainInference_Approximate(object):
    def __init__(self, model, marginal_dist, link_func, regression_terms, 
                dtype=torch.float64, device='cpu'):
        self.model = model
        self.marginal_dist = marginal_dist
        self.link_func = link_func
        self.regression_terms = regression_terms
        self.dtype = dtype
        self.device = device
    
    def load_params(self, data, params):
        # load X_spatial, G, P, Y
        self.group_names = list(data.keys())
        self.n_subject_per_group = [len(data[group]["Y"]) for group in self.group_names]
        self.n_group = len(self.group_names)
        # load B
        B = data[self.group_names[0]]["X_spatial"]
        B = B.astype(np.float64)
        # load Z and B, add intercept column
        Z = {}
        for group_name in self.group_names:
            Z_group = data[group_name]["Z"]
            Z_group = Z_group * 50 / Z_group.shape[0]
            Z[group_name] = np.concatenate([Z_group, np.ones((Z_group.shape[0], 1))], axis=1) 
        self.Z = Z
        B = B * 50 / B.shape[0]
        self.B = np.concatenate([B, np.ones((B.shape[0], 1))], axis=1)
        self.Y, self.Y_reshape = {}, {}
        for group_name in self.group_names:
            self.Y[group_name] = data[group_name]["Y"]
            self.Y_reshape[group_name] = self.Y[group_name].reshape(-1, 1)
        # Dimensions
        self._M, self._R = self.Z[self.group_names[0]].shape
        self._N, self._P = self.B.shape
        # Load parameters and re-scale
        self.MU_mean = params["MU_mean"].item() if params["MU_mean"].ndim == 0 else params["MU_mean"]
        self.beta = {group_name: params["beta"].item()[group_name] for group_name in self.group_names}
        self.MU = {}
        for group_name in self.group_names:
            if self.model == "SpatialBrainLesion":
                MU_group = compute_mu(self.Z[group_name], self.B, self.beta[group_name]) # shape: (n_subject*n_voxel, 1)
                self.MU[group_name] = MU_group.reshape(self._M, self._N) # shape: (n_subject, n_voxel)
        
    def create_contrast(self, contrast_vector=None, contrast_name=None):
        self.contrast_vector = contrast_vector
        self.contrast_name = contrast_name
        # Preprocess the contrast vector
        self.contrast_vector = (
            np.eye(self.n_group)
            if contrast_vector is None
            else np.array(contrast_vector).reshape(1, -1)
        )
        # raise error if dimension of contrast vector doesn't match with number of groups
        if self.contrast_vector.shape[1] != self._R:
            raise ValueError(
                f"""The shape of contrast vector: {str(self.contrast_vector)}
                doesn't match with number of groups."""
            )
        # standardization (row sum 1)
        self.contrast_vector = self.contrast_vector / np.sum(np.abs(self.contrast_vector), axis=1).reshape((-1, 1))
        self._S = self.contrast_vector.shape[0]
        
    def run_inference(self, method="FI", lesion_mask=None, XTWX_filename=None, inference_filename=None, fig_filename=None):
        self.lesion_mask = lesion_mask
        self.XTWX_filename = XTWX_filename
        self.inference_filename = inference_filename
        # Generalised linear hypothesis testing
        if self.model == "SpatialBrainLesion":
            self.SpatialGLM_glh_con_group(method)
        elif self.model == "MassUnivariateRegression":
            self.MUM_glh_con_group()
        else:
            raise ValueError(f"Model {self.model} not supported for inference.")

    def SpatialGLM_glh_con_group(self, method, use_dask=True, block_size=1e6):
        all_MU = np.concatenate([self.MU_mean[group_name].reshape(-1, 1) for group_name in self.group_names], axis=0) # shape: (n_group*n_voxel, 1)
        all_beta = np.concatenate([self.beta[group_name] for group_name in self.group_names], axis=0) # shape: (n_group*n_covariates, 1)
        # Remove the groups that are not involved in the contrast to save computation
        group_involved_index = np.nonzero(self.contrast_vector)[0]
        group_involved = [self.group_names[i] for i in group_involved_index]
        n_group_involved = len(group_involved)
        # Estimate the variance of beta, from either FI or sandwich estimator
        # Compute the Fisher information matrix
        if method == "FI":
            if not os.path.exists(self.XTWX_filename):  
                XTWX = {}
                for group in group_involved:
                    XTWX_group = efficient_kronT_diag_kron(self.Z[group], self.B, self.MU[group], use_dask=use_dask, block_size=block_size) # shape: (n_covariates*n_bases, n_covariates*n_bases)
                    XTWX[group] = XTWX_group
                np.savez(self.XTWX_filename, XTWX=XTWX)
            else:
                XTWX = np.load(self.XTWX_filename, allow_pickle=True)["XTWX"].item()
            
        CB = np.einsum('ij,kl->ikjl', self.contrast_vector, self.B) # shape: (_S, _N, _R, _P)
        CB_flat = CB.reshape(self._S, self._N, -1) # shape: (_S, _N, _R*_P)
        # (C \otimes B) \beta
        CB_beta = CB_flat @ all_beta  # shape: (_S, _N, 1)
        CB_beta = CB_beta.squeeze(-1) # shape: (_S, _N)
        # plot_brain(p=CB_beta.flatten(), brain_mask=lesion_mask, threshold=1, vmax=None, output_filename="numerator_map.png")
        # shape: (_S, _N) 
        if method == "FI":
            contrast_var_eta = robust_inverse_generalised(XTWX=XTWX, Q=CB_flat) # shape: (_N, 1)
        elif method == "sandwich":
            all_cov_beta = []
            for group in group_involved:
                cov_beta, diag = self.poisson_sandwich_kron(self.Z[group], self.B, self.Y[group], self.MU[group], meat="iid", ridge=0, return_diagnostics=True)
                all_cov_beta.append(cov_beta)
                del diag, cov_beta
                gc.collect()
            all_cov_beta = np.concatenate(all_cov_beta, axis=0) # shape: (n_group_involved, _R*_P, _R*_P)
            tmp = np.einsum('snk,kl->snl', CB_flat, all_cov_beta)         # (S, N, K)
            contrast_var_eta = np.sum(tmp * CB_flat, axis=-1, keepdims=True)  # (S, N, 1)
            if self._S == 1:
                contrast_std_eta = np.sqrt(contrast_var_eta) # shape: (S, N, 1)
                # contrast_std_eta = np.clip(contrast_std_eta, a_min=1e-6, a_max=None)
                # Conduct Wald test (Z test)
                z_stats = 1/1.4*CB_beta.reshape(-1, 1) / contrast_std_eta.reshape(-1, 1) # shape: (_N, 1)
                print(np.min(z_stats), np.max(z_stats), "z stats range")
                # convert z_stats to two-sided p-values
                p_vals = 2 * scipy.stats.norm.sf(np.abs(z_stats)) # shape: (_N, 1)
                print(np.count_nonzero(p_vals < 0.05), p_vals.shape, "p vals shape")
                # p_vals = scipy.stats.norm.sf(z_stats) # shape: (_N, 1)
            np.savez(self.inference_filename, z_stats=z_stats, p_vals=p_vals)
        return p_vals
    
    def MUM_glh_con_group(self):
        all_beta = np.concatenate([self.beta[group_name] for group_name in self.group_names], axis=0) # shape: (n_group, _N)
        contrast_beta_covariates = self.contrast_vector @ all_beta # shape: (1, _N)
        # Remove the groups that are not involved in the contrast to save computation
        group_involved_index = np.nonzero(self.contrast_vector)[0]
        group_involved = [self.group_names[i] for i in group_involved_index]
        n_group_involved = len(group_involved)
        if np.count_nonzero(self.contrast_vector) == 1:
            nonzero_index = np.nonzero(self.contrast_vector)[1].item()
            all_cov_beta = []
            if self.link_func == "log":
                for group in group_involved:
                    Y_group = self.Y[group] # shape: (n_subject, n_voxel)
                    p_empirical = np.mean(Y_group, axis=0) # shape: (n_voxel,)
                    # only keep the voxels with top 1000 highest empirical mean, 
                    # but also make sure the elements involved are same
                    # idx = np.argpartition(p_empirical, -1000)[-1000:]
                    # valid_inference_mask = np.zeros_like(p_empirical, dtype=bool)
                    # valid_inference_mask[idx] = True
                    MU_group = np.exp(self.Z[group] @ self.beta[group]) # shape: (n_subject, n_voxel)
                    FI = np.einsum('im,ij,ik->jmk', self.Z[group], MU_group, self.Z[group])  # shape: (N, R, R)
                    Cov_beta = np.linalg.pinv(FI+1e-6*np.eye(self._R)) # shape: (N, R, R)
                    all_cov_beta.append(Cov_beta)
            elif self.link_func == "logit":
                MU = 1 / (1 + np.exp(-(self.Z @ self.beta))) # shape: (n_subject, n_voxel)
                FI = np.einsum('im,ij,ik->jmk', self.Z, MU * (1 - MU), self.Z)  # shape: (N, R, R)
                Cov_beta = np.linalg.pinv(FI) # shape: (N, R, R)
            else:
                raise ValueError(f"Link function {self.link_func} not supported.")
        else:
            raise NotImplementedError("FI method only implemented for single non-zero contrast in MUM.")
        all_cov_beta = np.concatenate(all_cov_beta, axis=0) # shape: (N, R, R)
        var_beta = all_cov_beta[:, nonzero_index, nonzero_index] # shape: (N,)
        # Compute the numerator of the Z test
        contrast_std_beta = np.sqrt(var_beta) # shape: (1, n_voxel)
        # Conduct Wald test (Z test)
        z_stats = (contrast_beta_covariates / contrast_std_beta).flatten() # shape: (n_voxel,)
        print(contrast_beta_covariates)
        print(np.min(contrast_beta_covariates), np.mean(contrast_beta_covariates), np.max(contrast_beta_covariates), "contrast beta range")
        print(contrast_std_beta)
        print(np.min(contrast_std_beta), np.mean(contrast_std_beta), np.max(contrast_std_beta), "contrast std beta range")
        p_vals = 2 * scipy.stats.norm.sf(np.abs(z_stats.flatten()))
        print(p_vals.shape, z_stats.shape)

        # z_valid = z_stats[valid_inference_mask]
        # p_valid = p_vals[valid_inference_mask]
        print(np.count_nonzero(p_vals < 0.05), p_vals.shape, "p vals shape")
        # print(np.count_nonzero(p_valid < 0.05), p_valid.shape, "valid p vals shape")
        # plot_brain(p=z_stats.flatten(), brain_mask=self.lesion_mask, threshold=0.05, output_filename="z_stats_map.png")
        # np.savez(self.inference_filename, z_stats=z_valid, p_vals=p_valid)
        np.savez(self.inference_filename, z_stats=z_stats, p_vals=p_vals)
        return p_vals, z_stats

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

        alpha = 1.5
        # r = y - mu                                          # (M, N)
        r = y - mu / (1 + alpha * mu)                                  # (M, N)  # variance stabilising residuals for Poisson

        # ------------------------------------------------------------------
        # Bread:  A  (p x p)
        # w[k,l,j] = sum_i Z[i,k]*Z[i,l]*mu[i,j]
        # ------------------------------------------------------------------
        # w_bread = np.einsum('ik,il,ij->klj', Z, Z, mu)     # (R, R, N)
        w_bread = np.einsum('ik,il,ij->klj', Z, Z, mu/(1 + alpha * mu))     # (R, R, N)

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

    def plot_1d(self, p_vals, filename, significance_level=0.05):
        # slice list
        fig, axes = plt.subplots(1, 2, figsize=(23, 11))

        # Subplot 3
        M, N = p_vals.shape
        # theoretical p-values 
        th_p = np.arange(1/float(N),1+1/float(N),1/float(N)) # shape: (n_voxel,)
        th_p_log = -np.log10(th_p)
        # kth order statistics
        k_array = np.arange(start=1, stop=N+1, step=1)
        # empirical confidence interval (estimated from p-values)
        z_1, z_2 = scipy.stats.norm.ppf(significance_level), scipy.stats.norm.ppf(1-significance_level)
        # Add the Beta confidence interval
        CI_lower = scipy.stats.beta.ppf(significance_level/2, k_array, N - k_array + 1)
        CI_upper = scipy.stats.beta.ppf(1 - significance_level/2, k_array, N - k_array + 1)

        group_comparison = [[0, 1], [1, 0]]
        title_list = ["group_0 - group_1", "group_1 - group_0"]
        for i in range(M):
            # sort the order of p-values under -log10 scale
            sorted_p_vals = np.sort(p_vals[i, :]) # shape: (n_voxel,)
            significance_percentage = np.sum(sorted_p_vals < 0.05) / N
            print(significance_percentage)
            axes[i].fill_between(th_p_log, -np.log10(CI_lower), -np.log10(CI_upper), color='grey', alpha=0.5,
                    label=f'{int((1-significance_level)*100)}% Beta CI')
            axes[i].plot(th_p_log, np.repeat(-np.log10(0.05), N), color='y', linestyle='--', label='threshold at -log10(0.05)')
            axes[i].plot(th_p_log, -np.log10(th_p), color='orange', linestyle='--', label='y=x')
            axes[i].plot(th_p_log, -np.log10(significance_level * th_p), color='red', linestyle='-', label='FDR(BH) control')
            axes[i].scatter(th_p_log, -np.log10(sorted_p_vals), c='#1f77b4', s=4)
            axes[i].set_xlim([0, np.max(-np.log10(k_array/N))])
            axes[i].set_ylim([0, np.max(-np.log10(k_array/N))]) 
            axes[i].set_xlabel("Expected -log10(P)", fontsize=20)
            axes[i].set_ylabel("Observed -log10(P)", fontsize=20)
            axes[i].set_title(f"{title_list[i]}: {significance_percentage*100:.2f}% voxels rejected", fontsize=30)
            axes[i].legend()

        # Save the figure
        fig.savefig(filename)

class BrainInference_UKB(object):
    def __init__(self, model, marginal_dist, link_func, regression_terms, 
                dtype=torch.float64, device='cpu'):
        self.model = model
        self.marginal_dist = marginal_dist
        self.link_func = link_func
        self.regression_terms = regression_terms
        self.dtype = dtype
        self.device = device

    def load_params(self, data, params):
        # Load data
        B, Z = data["X_spatial"], data["Z"]
        if self.model == "SpatialBrainLesion":
            B = B * 50 / B.shape[0]
            Z = Z * 50 / Z.shape[0]
        self.B = np.concatenate([B, 0.01*np.ones((B.shape[0], 1))], axis=1)
        self.Y = data["Y"]
        self.Z = np.concatenate([Z, 0.01*np.ones((Z.shape[0], 1))], axis=1)
        self._M, self._R = self.Z.shape
        self._N, self._P = self.B.shape
        # Load parameters and re-scale
        self.beta = params["beta"]
        # MU
        if self.model == "SpatialBrainLesion":
            self.MU = compute_mu(self.Z, self.B, self.beta, mode="dask", block_size=5000) # shape: (n_subject*n_voxel, 1)

    def create_contrast(self, contrast_vector=None, contrast_name=None, polynomial_order=1):
        self.contrast_vector = contrast_vector
        self.contrast_name = contrast_name
        # Preprocess the contrast vector
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
        self.lesion_mask = lesion_mask
        self.XTWX_filename = XTWX_filename
        self.Fisher_info_filename = Fisher_info_filename
        self.meat_term_filename = meat_term_filename
        self.bread_term_filename = bread_term_filename
        self.p_vals_filename = p_vals_filename
        self.z_vals_filename = z_vals_filename
        z_threshold = scipy.stats.norm.ppf(1-alpha)
        # Generalised linear hypothesis testing
        if os.path.exists(self.p_vals_filename) and os.path.exists(self.z_vals_filename):
            p_vals = np.load(self.p_vals_filename)["p_vals"]
            z_stats = np.load(self.z_vals_filename)["z_stats"]
            print("loaded p-values and z-stats from file.")
        else:
            if self.model == "SpatialBrainLesion":
                p_vals, z_stats = self.SpatialGLM_glh_con_group(method, lesion_mask, True, 1e4)
            elif self.model == "MassUnivariateRegression":
                p_vals, z_stats = self.MUM_glh_con_group(lesion_mask)
            else:
                raise ValueError(f"Model {self.model} not supported for inference.")
            np.savez(self.p_vals_filename, p_vals=p_vals)
            np.savez(self.z_vals_filename, z_stats=z_stats)
            print("saved p-values and z-stats to file.")
        # Plot the estimated P, standard error of P, and p-values
        self.histogram_z_stats(z_stats, fig_filename.replace(".png", "_z_stats_histogram.png"))
        plot_brain(p=z_stats, brain_mask=lesion_mask, threshold=z_threshold, vmax=None, output_filename=fig_filename)
        # # FDR correction
        # rejected, corr_p = fdrcor
        # rection(p_vals.flatten(), alpha=0.05, method='indep')
        # # Clip to avoid 0 or 1 which produce +/-inf.
        # eps = 1e-300  # safe tiny number to avoid exact 0
        # corr_p_clipped = np.clip(corr_p, eps, 1.0 - 1e-16)
        # # Convert two-sided corrected p to a *signed* z:
        # corr_z = scipy.stats.norm.isf(corr_p_clipped) * np.sign(z_stats.flatten())
        # plot_brain(p=corr_z, brain_mask=lesion_mask, threshold=z_threshold, vmax=None, output_filename=fig_filename.replace(".png", "_FDR.png"))
    
    def SpatialGLM_glh_con_group(self, method, lesion_mask, use_dask=True, block_size=1e6):
        # Estimate the variance of beta, from either FI or sandwich estimator
        # Compute the Fisher information matrix
        if not os.path.exists(self.XTWX_filename):
            XTWX = efficient_kronT_diag_kron(self.Z, self.B, self.MU, use_dask=use_dask, block_size=block_size) # shape: (n_covariates*n_bases, n_covariates*n_bases)
            np.savez(self.XTWX_filename, XTWX=XTWX)
        else:
            XTWX = np.load(self.XTWX_filename)["XTWX"]
        # covariance of contrast eta: (C \otimes B) Cov(\beta) (C \otimes B)^T
        # Reshape operations to avoid explicit kron
        # print(self.Z)
        # Z_mean = np.mean(self.Z, axis=0) # shape: (n_covariates,)
        # delta = np.zeros_like(Z_mean)
        # delta[2] = 1  # add 1 to the 3rd covariate
        # Z_new = (Z_mean + delta).reshape((1, -1))
        # Z_new *= 50 / self.Z.shape[0]
        # a = np.kron(delta, self.B) @ self.beta
        # plot_brain(p=a, brain_mask=lesion_mask, threshold=0, vmax=None, output_filename="numerator_test.png")
        # exit()
        CB = np.einsum('ij,kl->ikjl', self.contrast_vector, self.B) # shape: (_S, _N, _R, _P)
        CB_flat = CB.reshape(self._S, self._N, -1) # shape: (_S, _N, _R*_P)
        # (C \otimes B) \beta
        CB_beta = CB_flat @ self.beta  # shape: (_S, _N, 1)
        CB_beta = CB_beta.squeeze(-1) # shape: (_S, _N)
        # plot_brain(p=CB_beta.flatten(), brain_mask=lesion_mask, threshold=1, vmax=None, output_filename="numerator_map.png")
        # shape: (_S, _N) 
        if method == "FI":
            contrast_var_eta = robust_inverse_generalised(XTWX=XTWX, Q=CB_flat) # shape: (_N, 1)
        elif method == "sandwich":
            meat_term = self.meat_term(self.Z, self.B, self.MU, self.Y) 
            bread_term = self.bread_term(self.Z, self.B, self.MU, self.Y)
            eigenspectrum(bread_term, save_path="bread_term_eigenspectrum.png")
            # Back transfer scaled bread term
            # sz = np.sqrt((self.Z**2).sum(axis=0)) 
            # mu_mean = self.MU.reshape((self._M, self._N)).mean(axis=0)               # (N,)
            # sb = np.sqrt((mu_mean[:, None] * self.B**2).sum(axis=0))  # (P,)
            # d = np.kron(sz, sb)
            # # # add a small epsilon to diagonal elements
            # bread_term += np.eye(bread_term.shape[0]) * 1e-4
            # bread_inv = np.linalg.pinv(bread_term) # shape: (n_covariates*n_bases, n_covariates*n_bases)
            # Use SVD to compute the inverse
            bread_inv = robust_inverse(XTWX=bread_term, eps=1e-6)
            eigenspectrum(bread_inv, save_path="bread_inv_eigenspectrum.png")
            cov_beta = bread_inv @ meat_term @ bread_inv
            tmp = np.einsum('snk,kl->snl', CB_flat, cov_beta)         # (S, N, K)
            contrast_var_eta = np.sum(tmp * CB_flat, axis=-1, keepdims=True)  # (S, N, 1)
            del bread_term, meat_term, cov_beta
        if self._S == 1:
            contrast_std_eta = np.sqrt(contrast_var_eta) # shape: (_N, 1)
            plot_brain(p=contrast_std_eta.flatten(), brain_mask=lesion_mask, threshold=0, vmax=None, output_filename="denominator_map_SGLM.png")
            # contrast_std_eta = np.clip(contrast_std_eta, a_min=1e-6, a_max=None)
            # Conduct Wald test (Z test)
            z_stats = CB_beta.reshape(-1, 1) / contrast_std_eta.reshape(-1, 1) # shape: (_N, 1)
            print(np.min(z_stats), np.max(z_stats), "z stats range")
            p_vals = scipy.stats.norm.sf(z_stats) # shape: (_N, 1)
            print(p_vals.shape, np.count_nonzero(p_vals<0.05))
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
            print(p_vals.shape, np.count_nonzero(p_vals < 0.05))
            rejected, p_vals = fdrcorrection(p_vals.flatten(), alpha=0.05, method='indep')
            p_vals = p_vals.reshape((1,-1))
            print(p_vals.shape, np.count_nonzero(p_vals < 0.05))
            z_stats = scipy.stats.norm.isf(p_vals / 2)

        return p_vals, z_stats

    def MUM_glh_con_group(self, lesion_mask):
        contrast_beta_covariates = self.contrast_vector @ self.beta # shape: (1, n_voxel)
        # Estimate the variance of beta, from either FI or sandwich estimator
        # check if there is only one non-zero contrast
        if np.count_nonzero(self.contrast_vector) == 1:
            nonzero_index = np.nonzero(self.contrast_vector)[1].item()
            if self.link_func == "log":
                P = np.exp(self.Z @ self.beta) # shape: (n_subject, n_voxel)
                # Reshape to enable broadcasting: (n_subject,) -> (n_subject, 1)
                Z_covariate_index = self.Z[:, nonzero_index].reshape(-1, 1)  # shape: (n_subject, 1)
                H_diag = np.sum(Z_covariate_index**2 * P, axis=0)  # shape: (n_voxel,)
            elif self.link_func == "logit":
                P = 1 / (1 + np.exp(-(self.Z @ self.beta))) # shape: (n_subject, n_voxel)
                Z_covariate_index = self.Z[:, nonzero_index].reshape(-1, 1)  # shape: (n_subject, 1)
                H_diag = np.sum(Z_covariate_index**2 * P * (1 - P), axis=0)  # shape: (n_voxel,)
            else:
                raise ValueError(f"Link function {self.link_func} not supported.")
        else:
            raise NotImplementedError("FI method only implemented for single non-zero contrast in MUM.")
        var_beta = 1.0 / (H_diag + 1e-6)  # shape: (n_voxel,)
        print(np.min(var_beta), np.mean(var_beta), np.max(var_beta), "variance of beta")
        # Compute the numerator of the Z test
        contrast_std_beta = np.sqrt(var_beta) # shape: (1, n_voxel)
        # plot_brain(p=contrast_std_beta.flatten(), brain_mask=lesion_mask, threshold=0, vmax=None, output_filename="denominator_map_MUM.png")
        # Conduct Wald test (Z test)
        z_stats_eta = contrast_beta_covariates / contrast_std_beta
        z_stats = z_stats_eta.reshape(-1)
        print(np.min(z_stats), np.max(z_stats), "z stats range")
        # z_stats = np.concatenate([z_stats_eta, -z_stats_eta], axis=0) # shape: (2, n_voxel)
        p_vals = 2 * scipy.stats.norm.sf(abs(z_stats))
        print(p_vals.shape, z_stats.shape)
        print(np.min(p_vals), np.max(p_vals), np.count_nonzero(p_vals < 0.05), p_vals.shape)

        return p_vals, z_stats
    
    def meat_term(self, Z, B, MU, Y, batch_M=1000):
        if MU.shape != Y.shape:
            MU = MU.reshape(Y.shape) # shape: (_M, _N)
        if not os.path.exists(self.meat_term_filename):
            W = Y - MU
            BW = W.dot(B)    # shape (M, P)
            T = (Z[:, :, None] * BW[:, None, :]).reshape(self._M, self._P * self._R)  # shape (M, PR)
            meat_term = T.T.dot(T)   # shape (PR, PR)
            del W, BW, T
            gc.collect()
            np.savez(self.meat_term_filename, meat_term=meat_term)
        else:
            print("Loading precomputed meat term...")
            meat_term = np.load(self.meat_term_filename)["meat_term"]

        return meat_term
    
    def bread_term(self, Z, B, MU, Y, dtype=np.float64, chunk_rows=256, epsilon=1e-6):
        if MU.shape != Y.shape:
            MU = MU.reshape(Y.shape)
        if not os.path.exists(self.bread_term_filename):
            print("Computing bread term...")
            start_time = time.time()
            bread_term = np.zeros((self._P * self._R, self._P * self._R)) # shape: (_P*_R, _P*_R)
            # # scale Z columns
            # sz = np.sqrt((Z**2).sum(axis=0))        # (R,)
            # Z_scaled = Z / szs
            # # scale B columns with average MU (or global MU)
            # mu_mean = MU.mean(axis=0)               # (N,)
            # sb = np.sqrt((mu_mean[:, None] * B**2).sum(axis=0))  # (P,)
            # B_scaled = B / sb

            for i in range(self._M): 
                print(f"Processing subject {i+1}/{self._M}", end='\r')
                # X_i = np.kron(Z[i,:], B) # shape: (_N, _P*_R) 
                # U_i = X_i.T * np.sqrt(MU[i, :]) # shape: (_P*_R, _N) 
                # bread_term += U_i @ U_i.T # shape: (_P*_R, _P*_R)
                zi = Z[i, :]                    # shape: (R,)
                mu_i = MU[i, :]          
                G_B = B.T @ (mu_i[:, None] * B)
                G_z = np.outer(zi, zi)          # (R, R)
                # Accumulate
                bread_term += np.kron(G_z, G_B)
            # print(np.min(np.diag(bread_term)), np.mean(np.diag(bread_term)), np.max(np.diag(bread_term)), "bread term diag stats")
            # bread_term += epsilon * np.eye(self._P * self._R)
            # print("Added epsilon {} to bread term".format(epsilon))
            print(np.min(np.diag(bread_term)), np.mean(np.diag(bread_term)), np.max(np.diag(bread_term)), "bread term diag stats")
            print("Time taken for bread term computation:", time.time() - start_time)
            del Z, B, MU, Y
            gc.collect()
            np.savez(self.bread_term_filename, bread_term=bread_term)
        else:
            print("Loading precomputed bread term...")
            bread_term = np.load(self.bread_term_filename)["bread_term"]
            print(np.min(np.diag(bread_term)), np.mean(np.diag(bread_term)), np.max(np.diag(bread_term)), "bread term diag stats")
            exit()
        return bread_term

    def histogram_z_stats(self, z_stats, filename):
        plt.figure(figsize=(10, 6))
        plt.hist(z_stats.flatten(), bins=100, color='blue', alpha=0.7, edgecolor='black')
        plt.title('Histogram of Z-statistics', fontsize=16)
        plt.xlabel('Z-statistic', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.grid(axis='y', alpha=0.75)
        plt.savefig(filename)
        plt.close()