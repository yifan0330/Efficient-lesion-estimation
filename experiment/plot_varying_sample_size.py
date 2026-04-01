import os
import numpy as np
import nibabel as nib  # For reading NIfTI files
from nilearn.plotting import plot_stat_map
import matplotlib.pyplot as plt
from data_simulation import GRF_simulated_data
import time

smooth_lesion_mask_path = os.getcwd() + "/data/brain/smooth_lesion_mask_Simulation.nii.gz"
smooth_lesion_mask = nib.load(smooth_lesion_mask_path)
n_voxels = np.count_nonzero(smooth_lesion_mask.get_fdata())

M = 100  # number of datasets
N_list = [50, 100, 500, 1000]

Y = np.empty((M, 1000, n_voxels), dtype=np.float16)
for random_seed in range(M):
    simulated_data_path = os.getcwd() + f"/data/brain/GRF_1000/data_Simulation_random_seed_{random_seed}.npz"
    simulated_data = np.load(simulated_data_path)
    Y[random_seed, :, :] = simulated_data['Y']  # shape: (n_subjects, n_lesion_voxel)
empirical_P_mean = np.mean(Y, axis=(0,1))  # shape: (n_lesion_voxel,)

# p_hat \in (0.005, 1]
indices_type_1 = np.where((empirical_P_mean > 0.005) & (empirical_P_mean <= 1))[0]
# p_hat \in (0.005, 0.01]
indices_type_2 = np.where((empirical_P_mean > 0.005) & (empirical_P_mean <= 0.01))[0]
# p_hat \in (0.01, 0.05]
indices_type_3 = np.where((empirical_P_mean > 0.01) & (empirical_P_mean <= 0.05))[0]
# p_hat \in (0.05, 0.1]
indices_type_4 = np.where((empirical_P_mean > 0.05) & (empirical_P_mean <= 0.1))[0]
# p_hat \in (0.1, 1]
indices_type_5 = np.where((empirical_P_mean > 0.1) & (empirical_P_mean <= 1))[0]
all_indices_types = [indices_type_1, indices_type_2, indices_type_3, indices_type_4, indices_type_5]
n_all_indices_types = len(all_indices_types)
all_subfig_names = ["0.005<p<1", "0.005<p<=0.01", "0.01<p<=0.05", "0.05<p<=0.1", "0.1<p<=1"]

models = ["SpatialBrainLesion","MassUnivariateRegression"]
dist_links = [["Bernoulli", "logit"], ["Poisson", "log"]]
full_model = True
filename_0 = "full_model" if full_model else "approximate_model"

for distribution, link_func in dist_links:
    for model in models:
        # path to save evaluation metrics
        eval_metric_path = os.getcwd() + f"/results/brain/eval_metric_{model}_{distribution}_{link_func}.npz"
        if os.path.exists(eval_metric_path):
            # eval_metric_dict = np.load(eval_metric_path, allow_pickle=True)
            # for key in eval_metric_dict:
            #     print(f"{key}:")
            #     print(eval_metric_dict[key])
            print("Evaluation metrics already computed for", model, distribution, link_func)
            pass
        else:
            eval_metric_dict = dict()
            for N in N_list:
                print("Processing N =", N)
                data_folder = os.getcwd() + "/data/brain/GRF_{}/".format(N)
                results_folder = os.getcwd() + "/results/brain/GRF_{}/".format(N)
                eval_metric_mask_type = dict()
                for i, indices_type in enumerate(all_indices_types):
                    n_lesion_voxel = len(indices_type)
                    all_P, all_empirical_P = np.empty((M, n_lesion_voxel)), np.empty((M, n_lesion_voxel))
                    for random_seed in range(M):
                        simulated_data_path = data_folder + f"data_Simulation_random_seed_{random_seed}.npz"
                        simulated_data = np.load(simulated_data_path)
                        Y = simulated_data['Y']  # shape: (n_subjects, n_lesion_voxel)
                        empirical_P_mean = np.mean(Y, axis=0)[indices_type] # shape: (n_lesion_voxel,)
                        all_empirical_P[random_seed, :] = empirical_P_mean
                        results_path = results_folder + f"{model}_{distribution}_{link_func}/brain_Probability_comparison_Simulation_{filename_0}_linear_random_seed_{random_seed}.npz"
                        results = np.load(results_path)
                        if full_model:
                            P = results['P']  # shape: (n_simulations, n_lesion_voxel)
                            P_hat = np.mean(P, axis=0)[indices_type]
                        else:
                            P_hat = results['P_mean'][indices_type]
                        all_P[random_seed, :] = P_hat
                    # remove nan rows in all_P
                    nan_indices = np.isnan(all_P).any(axis=1)
                    all_empirical_P = all_empirical_P[~nan_indices]
                    all_P = all_P[~np.isnan(all_P).any(axis=1)]
                    print(all_P.shape)
                    # Bias
                    bias = np.mean(all_P - all_empirical_P, axis=0)
                    # Variance
                    var = np.var(all_P, axis=0, ddof=1)
                    # MSE
                    mse = 1/M * np.sum((all_P - all_empirical_P) ** 2, axis=0)
                    # PU: Probability of underestimation
                    PU = 1/M * np.sum((all_P < all_empirical_P).astype(np.float32), axis=0)

                    eval_metric = np.array([np.mean(bias), np.std(bias),
                                            np.mean(var), np.std(var),
                                            np.mean(mse), np.std(mse),
                                            np.mean(PU), np.std(PU)]).reshape(4,2)
                    eval_metric_mask_type[f"lesion_mask_{i+1}"] = eval_metric
                eval_metric_dict["N={}".format(N)] = eval_metric_mask_type
            np.savez(eval_metric_path, **eval_metric_dict)

models = ["SpatialBrainLesion","MassUnivariateRegression"]
dist_links = [["Bernoulli", "logit"], ["Poisson", "log"]]

# create a plot with 3 subplots in a row (Bias, Variance, MSE)
subfig_name = ["Bias", "Variance", "MSE"]
n_subfig = len(subfig_name)

# Improved figure setup with better styling
plt.style.use('seaborn-v0_8')  # Use seaborn style for better aesthetics
fig, axs = plt.subplots(n_all_indices_types, n_subfig, figsize=(24, 32))
fig.suptitle('Model Performance Comparison', fontsize=24, fontweight='bold', y=0.98)

# Color palette for better distinction
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Professional color scheme
markers = ['o', 's', '^', 'D']  # Different markers for each combination
line_styles = ['-', '--', '-.', ':']

for row, indices_type in enumerate(all_indices_types):
    for i in range(n_subfig):
        ax = axs[row, i]
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_title(f"{subfig_name[i]}--({all_subfig_names[row]}, n_voxel={len(indices_type)})", fontsize=18, fontweight='bold', pad=8)
        ax.set_xlabel("Sample Size", fontsize=15, fontweight='semibold')
        ax.set_ylabel(subfig_name[i], fontsize=15, fontweight='semibold')
        ax.tick_params(labelsize=13)
        
        # Improve axis appearance
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)

color_idx = 0
for dist, link in dist_links:
    for model in models:
        eval_metric_path = os.getcwd() + f"/results/brain/eval_metric_{model}_{dist}_{link}.npz"
        eval_metric_dict = np.load(eval_metric_path, allow_pickle=True)
        # Improved plotting with better styling
        label = f"{dist} + {link}"
        color = colors[color_idx % len(colors)]
        marker = markers[color_idx % len(markers)]
        linestyle = line_styles[color_idx % len(line_styles)]
        for index_type_idx, indices_type in enumerate(all_indices_types):
            bias_vals = [eval_metric_dict[f"N={N}"].item()[f"lesion_mask_{index_type_idx+1}"][0, 0] for N in N_list]
            var_vals  = [eval_metric_dict[f"N={N}"].item()[f"lesion_mask_{index_type_idx+1}"][1, 0] for N in N_list]
            mse_vals  = [eval_metric_dict[f"N={N}"].item()[f"lesion_mask_{index_type_idx+1}"][2, 0] for N in N_list]
            # plotting
            axs_row = axs[index_type_idx]
            for i, (ax, vals, name) in enumerate(zip(axs_row, [bias_vals, var_vals, mse_vals], subfig_name)):
                ax.plot(N_list, vals, 
                       marker=marker, 
                       color=color,
                       linestyle=linestyle,
                       linewidth=2.5, 
                       markersize=8,
                       markeredgewidth=1.5,
                       markeredgecolor='white',
                       label=label,
                       alpha=0.8)

                
                legend = ax.legend(loc="center right",
                                    frameon=True, 
                                    fancybox=True, 
                                    shadow=True,
                                    fontsize=9,
                                    framealpha=0.9)
                legend.get_frame().set_facecolor('white')
            
        color_idx += 1

# Adjust layout for better spacing with reduced gap between title and subplots
plt.tight_layout(rect=[0.04, 0.08, 0.96, 0.92])

# Save with higher quality
fig.savefig(os.getcwd() + "/model_comparison_empirical.png", dpi=300, bbox_inches='tight', facecolor='white')