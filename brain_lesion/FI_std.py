import numpy as np
import scipy
import matplotlib.pyplot as plt

# general parameters
space_dim = 1
simulated_dset = True
homogeneous = True
n_subject = 2000
n_group = 1
marginal_dist = "Poisson"
link_func = "log"

filename_0 = "_Simulation" if simulated_dset else "_RealDataset"
filename_1 = "_Homogeneous" if homogeneous else "_BumpSignals"

# Load the results
results_filename = f"results/{space_dim}D_Probability_comparison{filename_0}{filename_1}_{n_group}_group_{marginal_dist}_{link_func}_link_func.npz"
results = np.load(results_filename, allow_pickle=True)
P = results["P"].item()
X_spatial = results["X_spatial"]
Y = results["Y"].item()
Z = results["Z"].item()
bias_b = results["bias_b"].item()
beta_b = results["beta_b"].item()
bias_W = results["bias_W"].item()
beta_W = results["beta_W"].item()

group = list(Z.keys())[0]

n_covariates = Z[group].shape[1]
n_bases = X_spatial.shape[1]

# W = bias_W.T
# W = W + Z["group_0"] @ beta_W.T
# b = bias_b.T
# b = b + Z["group_0"] @ beta_b.T
# inversed_link_func = np.exp if link_func == "log" else None
# P = inversed_link_func(W @ X_spatial.T + b)

H = np.zeros((n_covariates*n_bases, n_covariates*n_bases))
M, N = P[group].shape
for i in range(M):
    for j in range(N):
        Z_i = Z[group][i].reshape((-1,1)) # shape: (n_covariates, 1)
        X_j = X_spatial[j].reshape((-1,1)) # shape: (n_bases, 1)
        # Z_i X_j^T
        Z_i_X_j = np.matmul(Z_i, X_j.T) # shape: (n_covariates, n_bases)
        kron_prod = np.kron(Z_i_X_j, Z_i_X_j.T) # shape: (n_covariates*n_bases, n_covariates*n_bases)
        H += -P[group][i,j] * kron_prod
COV_beta_W = np.linalg.inv(-H) # shape: (n_covariates*n_bases, n_covariates*n_bases)
# Var_beta_W = np.diag(COV_beta_W)

slice_list = [0, 499, 999, 1499, 1999]
SE_P_list, P_0_list, Z_list, p_vals_list = list(),list(),list(),list()
for i in slice_list:
    Z_i = Z[group][i].reshape((-1,1)) # shape: (n_covariates, 1)
    # intermediate param: A = beta_w * Z_i
    Cov_A = np.kron(Z_i.T, np.eye(n_bases)) @ COV_beta_W @ np.kron(Z_i, np.eye(n_bases)).T # shape: (n_covariates, n_bases)
    # eta_i = X*Cov(A)*X^T
    Cov_eta_i = X_spatial @ Cov_A @ X_spatial.T
    # Cov_P = diag(P) * Cov_eta * diag(P), where P = exp(eta)
    Cov_P_i = (P[group][i] * Cov_eta_i) * P[group][i]
    Var_P_i = np.diag(Cov_P_i)
    SE_P_i = np.sqrt(Var_P_i)
    SE_P_list.append(SE_P_i)
    # Wald test
    P_i_0 = np.mean(Y[group][i])
    Z_i = (P[group][i] - P_i_0) / SE_P_i
    p_vals_i = 1 - scipy.stats.norm.cdf(Z_i)
    P_0_list.append(P_i_0)
    Z_list.append(Z_i)
    p_vals_list.append(p_vals_i)

fig, axes = plt.subplots(5, 3, figsize=(30, 50))
for i in range(len(slice_list)):
    slice = slice_list[i]
    print("slice: ", slice)
    # Subplot 0
    axes[i,0].plot(P[group][slice], label=f'Estimated P')
    axes[i,0].axhline(y=P_0_list[i], color='red', linestyle='--', label='y=P_0')
    axes[i,0].set_xlabel("Voxel")
    axes[i,0].set_ylabel("Variance")
    axes[i,0].set_title(f"Slice {slice_list[i]}", fontsize=30)
    axes[i,0].legend()

    # Subplot 2
    axes[i,1].plot(SE_P_list[i], label=f'Var P')
    axes[i,1].set_xlabel("Voxel")
    axes[i,1].set_ylabel("Variance")
    axes[i,1].legend()

    # Subplot 3
    axes[i,2].plot(p_vals_list[i], label=f'p values')
    axes[i,2].axhline(y=0.05, color='red', linestyle='--', label='alpha=0.05')
    axes[i,2].set_xlabel("Voxel")
    axes[i,2].set_ylabel("Variance")
    axes[i,2].legend()

# Save the figure
fig.savefig(f"figures/FI_SE_map{filename_1}.png")

