import torch
import numpy as np
import scipy
import time
import gc
import os
from tqdm import tqdm 
import matplotlib.pyplot as plt
from model import ApproximatePoissonModel

class BrainInference(object):
    def __init__(self, model, marginal_dist, link_func, regression_terms, dtype=torch.float64, device='cpu'):
        self.model = model
        self.marginal_dist = marginal_dist
        self.link_func = link_func
        self.regression_terms = regression_terms
        self.dtype = dtype
        self.device = device
    
    def load_params(self, params):
        # load X_spatial, P, Y
        self.X_spatial = torch.tensor(params["X_spatial"], dtype=self.dtype, device=self.device)
        self.P = {group:torch.tensor(params["P"].item()[group], dtype=self.dtype, device=self.device) for group in params["P"].item().keys()}
        self.Y = {group:torch.tensor(params["Y"].item()[group], dtype=self.dtype, device=self.device) for group in params["Y"].item().keys()}
        self.Z = {group:torch.tensor(params["Z"].item()[group], dtype=self.dtype, device=self.device) for group in params["Z"].item().keys()}
        if "multiplicative" in self.regression_terms:
            self.bias_W = {group: torch.tensor(params["bias_W"].item()[group], dtype=self.dtype, device=self.device) for group in params["bias_W"].item().keys()}
            self.beta_W = {group: torch.tensor(params["beta_W"].item()[group], dtype=self.dtype, device=self.device) for group in params["beta_W"].item().keys()}
        if "additive" in self.regression_terms:
            self.bias_b = {group: torch.tensor(params["bias_b"].item()[group], dtype=self.dtype, device=self.device) for group in params["bias_b"].item().keys()}
            self.beta_b = {group: torch.tensor(params["beta_b"].item()[group], dtype=self.dtype, device=self.device) for group in params["beta_b"].item().keys()}
        # group names
        self.group_names = list(self.P.keys())
        self.n_group = len(self.group_names)
        # spatial coefficient dimension
        self.n_voxel, self.n_bases = self.X_spatial.shape
        self.n_subject = [self.P[group].shape[0] for group in self.group_names]

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
        contrast_P, std_P, p_vals = self._glh_con_group(method)
        # Plot the estimated P, standard error of P, and p-values
        print(f"fig_{method}_{self.t_con_group_name}")
        self.plot_1d(contrast_P, std_P, p_vals, f"fig_{method}_{self.t_con_group_name}")


    def _glh_con_group(self, method, batch_size=20):
        t_con_groups_involved_index = np.where(self.t_con_groups != 0)[1].tolist()
        t_con_groups_involved = [self.group_names[i] for i in t_con_groups_involved_index]
        self.n_con_group_involved = len(t_con_groups_involved)
        # Compute the contrast P
        if np.count_nonzero(self.t_con_groups, axis=1) == 1:
            P_array = list()
            for group in t_con_groups_involved:
                P_group_0 = np.mean(self.Y[group].detach().cpu().numpy(), axis=1)
                P_array.append(self.P[group].detach().cpu().numpy() - P_group_0[:, None])
            P_array = np.stack(P_array, axis=0) # shape: (n_con_group_involved, n_subject, n_voxel)
        else:
            P_array = np.stack([self.P[group].detach().cpu().numpy() for group in t_con_groups_involved], axis=0) # shape: (n_con_group_involved, n_subject, n_voxel) 
        contrast_P = np.einsum('ij,jkl->ikl', self.t_con_groups, P_array)
        X_spatial_array = self.X_spatial.detach().cpu().numpy() # shape: (n_voxel, n_bases)
        # Hessian matrix
        if method == "FI":
            _, F_beta_W = self._Fisher_info(t_con_groups_involved)
            cov_beta_W = np.linalg.inv(F_beta_W)
        elif method == "sandwich":
            print(self.n_con_group_involved)
            for k in range(self.n_con_group_involved):
                group = t_con_groups_involved[k]
                Z_group = self.Z[group].detach().cpu().numpy()
                P_group = self.P[group].detach().cpu().numpy()
                Y_group = self.Y[group].detach().cpu().numpy()
                bread_term_group = self.bread_term(Z_group, X_spatial_array, P_group) # shape: (n_covariates*n_bases, n_covariates*n_bases)
                meat_term_group = self.meat_term(Z_group, X_spatial_array, P_group, Y_group) # shape: (n_covariates*n_bases, n_covariates*n_bases)
                # sandwich estimator
                cov_beta_W = bread_term_group @ meat_term_group @ bread_term_group # shape: (n_covariates*n_bases, n_covariates*n_bases)
        # Compute the variance of P, from the variance of beta_W
        var_P = list()
        for k in range(self.n_con_group_involved):
            group = t_con_groups_involved[k]
            # create a group-specific memmap file to store the variance of P
            memmap_path = f"{os.getcwd()}/results/var_P_{group}.dat"
            os.makedirs(os.path.dirname(memmap_path), exist_ok=True)
            with open(memmap_path, 'wb') as f:
                pass
            var_P_group = np.memmap(memmap_path, mode="r+", shape=(self.n_subject[k],self.n_voxel), dtype='float64')
            print("group index k:", k, group)
            print(k * self.n_bases, (k + 1) * self.n_bases)
            cov_beta_W_group = cov_beta_W[
                k * self.n_bases : (k + 1) * self.n_bases,
                k * self.n_bases : (k + 1) * self.n_bases,
            ]
            Z_group = self.Z[group].detach().cpu().numpy() # shape: (n_subject, n_covariates)
            P_group = self.P[group].detach().cpu().numpy() # shape: (n_subject, n_voxel)
            self.batch_compute_covariance(var_P_group, Z_group, X_spatial_array, P_group, cov_beta_W_group, batch_size=1000) # shape: (n_subject, n_voxel)
            var_P.append(var_P_group)
            del cov_beta_W_group, Z_group, P_group, var_P_group
            gc.collect()
        var_P = np.stack(var_P, axis=0) # shape: (n_con_group_involved, n_subject, n_voxel)
        # Compute the numerator of the Z test
        involved_var_P = np.einsum('ij,jkl->ikl', self.t_con_groups**2, var_P)
        involved_std_P = np.sqrt(involved_var_P) # shape: (1, n_subject, n_voxel)
        # Conduct Wald test (Z test)
        z_stats = contrast_P / involved_std_P
        if self.n_con_group_involved == 1: # one-tailed test
            p_vals = scipy.stats.norm.sf(z_stats) # shape: (1, n_subject, n_voxels)
        else: 
            p_vals = 2 * scipy.stats.norm.sf(np.abs(z_stats))
        return contrast_P, involved_std_P, p_vals
    
    def _Fisher_info(self, t_con_groups_involved):
        # Load Y, Z for the involved groups
        involved_Y_group = {group: self.Y[group] for group in t_con_groups_involved}
        involved_Z = {group: self.Z[group] for group in t_con_groups_involved}
        if "multiplicative" in self.regression_terms:
            involved_beta_W = torch.cat([self.beta_W[group][None, :] for group in t_con_groups_involved], dim=0) # shape: (n_involved_groups, n_bases, n_covariates)
            involved_bias_W = torch.cat([self.bias_W[group][None, :] for group in t_con_groups_involved], dim=0) # shape: (n_involved_groups, n_bases, n_covariates)
        else: 
            involved_beta_W = None
            involved_bias_W = None
        if "additive" in self.regression_terms:
            involved_beta_b = torch.cat([self.beta_b[group][None, :] for group in t_con_groups_involved], dim=0) # shape: (n_involved_groups, 1, n_covariates)
            involved_bias_b = torch.cat([self.bias_b[group][None, :] for group in t_con_groups_involved], dim=0) # shape: (n_involved_groups, 1, n_covariates)
        else:
            involved_beta_b = None
            involved_bias_b = None
        # Compute the Fisher information matrix
        if self.model == "SpatialBrainLesion":
            # a = SpatialBrainLesionModel._log_likelihood(self.marginal_dist,
            #                                             self.link_func,
            #                                             self.regression_terms,
            #                                             self.X_spatial,
            #                                             self.Y,
            #                                             involved_Z,
            #                                             involved_beta_W,
            #                                             involved_bias_W,
            #                                             involved_beta_b,
            #                                             involved_bias_b,
            #                                             self.device)
            nll = lambda involved_bias_W: SpatialBrainLesionModel._neg_log_likelihood(t_con_groups_involved,
                                                                        self.marginal_dist,
                                                                        self.link_func,
                                                                        self.regression_terms,
                                                                        self.X_spatial,
                                                                        self.Y,
                                                                        involved_Z,
                                                                        involved_beta_W,
                                                                        involved_bias_W,
                                                                        involved_beta_b,
                                                                        involved_bias_b,
                                                                        self.device)
            params = (involved_bias_W)
            # Jacobian
            J = torch.autograd.functional.jacobian(nll, params, create_graph=False) 
            # shape: [2, 103, 1]
            J = J.view(self.n_con_group_involved * self.n_bases, -1)
            # Hessian
            H = torch.autograd.functional.hessian(nll, params, create_graph=False)
            H = H.view(self.n_con_group_involved * self.n_bases, -1)

            return J.detach().cpu().numpy(), H.detach().cpu().numpy()
    
    def bread_term(self, Z, X_spatial, P):
        n_subject, n_covariates = Z.shape
        # bread term: (X^TWX)^-1
        H = np.zeros((n_covariates*self.n_bases, n_covariates*self.n_bases))
        # Reshape Z and X_spatial to add dimensions for broadcasting
        Z_expand = Z[:, :, np.newaxis]  # Shape: (n_subject, n_covariates, 1)
        X_expand = X_spatial[:, np.newaxis, :]  # Shape: (n_voxel, 1, n_bases)
        # Compute Z_i X_j^T for all i, j at once
        Z_X = Z_expand[:, :, None] * X_expand[None, :, :, :] # shape: (n_subject, n_covariates, 1, n_bases)
        for i in range(n_subject):
            for j in range(self.n_voxel):
                # Flatten Z_i_X_j into a vector
                Z_X_flat = Z_X[i, j].ravel()  # shape: (n_covariates * n_bases,)
                # Compute the Kronecker product with its transpose
                kron_prod = np.outer(Z_X_flat, Z_X_flat)  # shape: (n_covariates * n_bases, n_covariates * n_bases)
                # Accumulate into H with the corresponding weight
                H += -P[i, j] * kron_prod
                del Z_X_flat, kron_prod
        bread_term = np.linalg.inv(-H) # shape: (n_covariates*n_bases, n_covariates*n_bases)
        del Z_expand, X_expand, Z_X
        gc.collect()
        return bread_term
    
    def meat_term(self, Z, X_spatial, P, Y):
        start_time = time.time()
        n_subject, n_covariates = Z.shape
        # meat term: sum_M [D_i^TV_i^{-1}(Y_i-P_i)]*[D_i^TV_i^{-1}(Y_i-P_i)]^T
        # meat_term = np.zeros((n_covariates*self.n_bases, n_covariates*self.n_bases))
        # Z_expand = Z[:, :, np.newaxis]  # Shape: (n_subject, n_covariates, 1)
        # X_expand = X_spatial[:, np.newaxis, :]  # Shape: (n_voxel, 1, n_bases)
        # # Compute D_ij for all subjects and voxels
        # D_ij = P[:, :, np.newaxis, np.newaxis] * np.einsum('sci,vbn->svcb', Z_expand, X_expand)  # shape: (n_subject, n_voxel, n_covariates, n_bases)
        # # Flatten D_ij for efficient matrix operations
        # D_ij_flat = D_ij.reshape(n_subject, self.n_voxel, -1)  # shape: (n_subject, n_voxel, n_covariates * n_bases)
        # # Compute residue
        # residue = (Y - P).reshape(n_subject, self.n_voxel, 1)  # Shape: (n_subject, n_voxel, 1)
        # # Compute D_i^T V_i^-1 residue for all subjects
        # D_i_V_i_residue = np.einsum('sni,sn,snj->si', D_ij_flat, 1 / P, residue.squeeze(-1))
        # # D_i_V_i_residue shape: (n_subject, n_covariates * n_bases)
        # # Update meat_term with contributions from each subject
        # meat_term = np.einsum("si,sj->ij", D_i_V_i_residue, D_i_V_i_residue)
        # del Z_expand, X_expand, D_ij, D_ij_flat, residue, P_inv, D_i_V_i_residue
        # gc.collect()
        # print(meat_term)
        # print("Time taken: ", time.time() - start_time)
        # print("---------------------")
        start_time = time.time()
        # meat term: sum_M [D_i^TV_i^{-1}(Y_i-P_i)]*[D_i^TV_i^{-1}(Y_i-P_i)]^T
        meat_term = np.zeros((n_covariates*self.n_bases, n_covariates*self.n_bases))
        for i in range(n_subject):
            Z_i = Z[i].reshape((1,-1)) # shape: (1, n_covariates)
            D_i = list()
            for j in range(self.n_voxel):
                X_j = X_spatial[j].reshape((1,-1)) # shape: (1, n_bases)
                D_ij = P[i,j]*np.kron(Z_i, X_j)
                D_i.append(D_ij)
            D_i = np.concatenate(D_i, axis=0) # shape: (N, n_covariates*n_bases)
            V_i = np.diag(P[i]) # shape: (N, N)
            V_i_inv = np.diag(1/P[i]) # shape: (N, N)
            residue = (Y[i] - P[i]).reshape((-1,1)) # shape: (N, 1)
            D_i_V_i_residue = D_i.T @ V_i_inv @ residue # shape: (n_covariates*n_bases, 1)
            meat_term += D_i_V_i_residue @ D_i_V_i_residue.T # shape: (n_covariates*n_bases, n_covariates*n_bases)
        print("Time taken: ", time.time() - start_time)
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
    
    def plot_1d(self, P, std_P, p_vals, filename):
        n_subject = p_vals.shape[1]
        # slice list
        q1 = 0
        q2 = int(np.percentile(np.arange(n_subject), 25))
        q3 = int(np.percentile(np.arange(n_subject), 50))
        q4 = int(np.percentile(np.arange(n_subject), 75))
        q5 = n_subject - 1
        slice_list = [q1, q2, q3, q4, q5]
        fig, axes = plt.subplots(5, 3, figsize=(30, 50))
        for i in range(len(slice_list)):
            slice = slice_list[i]
            print("slice: ", slice)
            # Subplot 0
            axes[i,0].plot(P[:, slice, :].squeeze(), label=f'Estimated P')
            axes[i,0].axhline(y=0.0, color='red', linestyle='--', label='y=P_0')
            axes[i,0].set_xlabel("Voxel")
            axes[i,0].set_ylabel("Variance")
            axes[i,0].set_title(f"Slice: {slice_list[i]}", fontsize=30)
            axes[i,0].legend()

            # Subplot 2
            axes[i,1].plot(std_P[:, slice, :].squeeze(), label=f'Var P')
            axes[i,1].set_xlabel("Voxel")
            axes[i,1].set_ylabel("Variance")
            axes[i,1].legend()

            # Subplot 3
            axes[i,2].plot(p_vals[:, slice, :].squeeze(), label=f'p values')
            axes[i,2].axhline(y=0.05, color='red', linestyle='--', label='alpha=0.05')
            axes[i,2].set_xlabel("Voxel")
            axes[i,2].set_ylabel("Variance")
            axes[i,2].legend()

        # Save the figure
        fig.savefig(f"{os.getcwd()}/figures/{filename}.png")

