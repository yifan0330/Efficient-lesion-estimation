import numpy as np
import os 
import matplotlib.pyplot as plt

space_dim = 1
n_group = 2
n_subject = [2000, 2000]
simulated_dset = True
homogeneous = True
full_model = False
marginal_dist = "Poisson"
link_func = "log"
model_comb = [["approximate", "approximate"], ["dask", "approximate"], ["dask", "exact"], ["offload", "approximate"]]

filename_0 = "_Simulation" if simulated_dset else "_RealDataset"
filename_1 = "_Homogeneous" if homogeneous else "_BumpSignals"
filename_2 = "_full_model" if full_model else "_approximate_model"
# data filename
data_filename = f"{os.getcwd()}/data/{space_dim}D/{space_dim}D_data{filename_0}{filename_1}{filename_2}_{n_group}_group_{n_subject}_{marginal_dist}_{link_func}_link_func.npz"
data = np.load(data_filename, allow_pickle=True)
MU = data['MU']
MU_mean = MU.mean(axis=0)
MU_std = MU.std(axis=0)

# plot 
P_mean_figure = f"{os.getcwd()}/figures/{space_dim}D/{space_dim}D_P_mean_comparison{filename_0}{filename_1}{filename_2}_{n_group}_group_{n_subject}_{marginal_dist}_{link_func}_link_func.png"
fig, axs = plt.subplots(2, 2, figsize=(20, 20))
i = 0
for mode in model_comb:
    gradient_mode, preconditioner_mode = mode
    results_filename = f"{os.getcwd()}/results/{space_dim}D/{space_dim}D_Probability_comparison{filename_0}{filename_1}{filename_2}_{gradient_mode}_{preconditioner_mode}_{n_group}_group_{n_subject}_{marginal_dist}_{link_func}_link_func.npz"
    results = np.load(results_filename, allow_pickle=True)
    # beta = results['beta']
    P_mean = results['P_mean']
    # Subplot 
    axs[i//2, i%2].plot(MU_mean, label='actual P')
    axs[i//2, i%2].plot(P_mean, label='estimated P')
    axs[i//2, i%2].set_title(f"Gradient Mode: {gradient_mode}, Preconditioner Mode: {preconditioner_mode}", fontsize=20)
    axs[i//2, i%2].legend()
    i += 1
plt.savefig(P_mean_figure)
exit()

# # baseline regression results filename
# baseline_gradient_mode = "offload"
# baseline_preconditioner_mode = "exact"
# baseline_filename = f"{os.getcwd()}/results/{space_dim}D/{space_dim}D_Probability_comparison{filename_0}{filename_1}{filename_2}_{baseline_gradient_mode}_{baseline_preconditioner_mode}_{n_group}_group_{n_subject}_{marginal_dist}_{link_func}_link_func.npz"
# baseline_results = np.load(baseline_filename, allow_pickle=True)
# baseline_P_mean = baseline_results['P_mean']
# baseline_P_std = baseline_results['P_std']

# gradient_mode_list = ["offload", "approximate"]
# preconditioner_mode_list = ["approximate", "exact"]

# for gradient_mode in gradient_mode_list:
#     for preconditioner_mode in preconditioner_mode_list:
#         results_filename = f"{os.getcwd()}/results/{space_dim}D/{space_dim}D_Probability_comparison{filename_0}{filename_1}{filename_2}_{gradient_mode}_{preconditioner_mode}_{n_group}_group_{n_subject}_{marginal_dist}_{link_func}_link_func.npz"
#         results = np.load(results_filename, allow_pickle=True)
#         # beta = results['beta']
#         P_mean = results['P_mean']
#         P_std = results['P_std']

#         diff_ratio_P_mean = P_mean - baseline_P_mean
#         diff_ratio_P_std = P_std - baseline_P_std
#         print(f"Gradient Mode: {gradient_mode}, Preconditioner Mode: {preconditioner_mode}")
#         print(np.min(P_mean), np.mean(P_mean), np.max(P_mean))
#         print(np.min(baseline_P_mean), np.mean(baseline_P_mean), np.max(baseline_P_mean))
#         print(np.min(diff_ratio_P_mean), np.mean(diff_ratio_P_mean), np.max(diff_ratio_P_mean))
#         print(np.min(diff_ratio_P_std), np.mean(diff_ratio_P_std), np.max(diff_ratio_P_std))
#         print("-" * 50)