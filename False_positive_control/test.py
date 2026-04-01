import numpy as np
import matplotlib.pyplot as plt
import scipy.stats


model = "Poisson"

all_z_stats, all_p_vals = [], []
for seed in range(1, 101):
    load_test_statistics = np.load(f"inference/brain/GRF_[10000]/MassUnivariateRegression_{model}_log/brain_Inference_sandwich_Simulation_approximate_model_linear_random_seed_{seed}.npz", allow_pickle=True)
    z_stats, p_vals = load_test_statistics["z_stats"], load_test_statistics["p_vals"]
    all_z_stats.append(z_stats)
    all_p_vals.append(p_vals)
all_z_stats = np.stack(all_z_stats, axis=0)
all_p_vals = np.stack(all_p_vals, axis=0)
# generate QQ plot for all p-values
def qq_plot(p_vals, save_to=None):
    M, N = p_vals.shape
    expected = -np.log10(np.arange(1, N + 1) / (N + 1))
    # observed values: taking average of p-values across seeds for each voxel, then sort
    observed = -np.log10(np.sort(p_vals, axis=1))
    # sort in descending order for QQ plot
    observed = np.mean(observed, axis=0).flatten()#[::-1]
    # confidence intervals for the expected values under the null hypothesis
    ci_lower = -np.log10(scipy.stats.beta.ppf(0.025, np.arange(1, N + 1), np.arange(N, 0, -1)))
    ci_upper = -np.log10(scipy.stats.beta.ppf(0.975, np.arange(1, N + 1), np.arange(N, 0, -1)))

    plt.figure(figsize=(6, 6))
    plt.plot(expected, observed, marker='o', linestyle='', label='Observed P-values')
    plt.plot([0, max(expected)], [0, max(expected)], color='red', linestyle='--', label='Expected under null')
    plt.fill_between(expected, ci_lower, ci_upper, color='gray', alpha=0.2, label='95% Confidence Interval')
    plt.xlabel('Expected -log10(P-values)')
    plt.ylabel('Observed -log10(P-values)')
    # Include the rate of significant voxels in the title
    significant_rate = np.mean(p_vals < 0.05)
    plt.title(f'PP-plot of P-values, {significant_rate:.2%} significant voxels under alpha=0.05')
    plt.legend()
    plt.grid()
    
    if save_to is not None:
        plt.savefig(save_to)
    else:
        plt.show()
qq_plot(all_p_vals, save_to=f"qq_plot_MUM_{model}.png")
exit()

# # load_test_statistics = np.load(f"inference/brain/GRF_[10000]/SpatialBrainLesion_{model}_log/brain_Inference_sandwich_Simulation_approximate_model_linear_random_seed_1.npz", allow_pickle=True)
# load_test_statistics = np.load(f"inference/brain/GRF_[10000]/MassUnivariateRegression_Poisson_log/brain_Inference_sandwich_Simulation_approximate_model_linear_random_seed_1.npz", allow_pickle=True)
# z_stats, p_vals = load_test_statistics["z_stats"], load_test_statistics["p_vals"]
# print(np.count_nonzero(p_vals<0.05), p_vals.shape, "number of significant voxels")
# # generate PP plot
# def pp_plot(p_vals, save_to=None):
#     p_vals = p_vals.flatten()
#     p_vals = p_vals[~np.isnan(p_vals)]
#     p_vals = p_vals[p_vals > 0]
#     p_vals = p_vals[p_vals < 1]
#     n = len(p_vals)
#     expected = - np.log10(np.arange(1, n + 1) / (n + 1))
#     observed = - np.log10(np.sort(p_vals))
    
#     plt.figure(figsize=(6, 6))
#     plt.plot(expected, observed, marker='o', linestyle='', label='Observed P-values')
#     plt.plot([0, max(expected)], [0, max(expected)], color='red', linestyle='--', label='Expected under null')
#     plt.xlabel('Expected -log10(P-values)')
#     plt.ylabel('Observed -log10(P-values)')
#     plt.title('PP Plot of P-values')
#     plt.legend()
#     plt.grid()
    
#     if save_to is not None:
#         plt.savefig(save_to)
#     else:
#         plt.show()

# # pp_plot(p_vals, save_to=f"pp_plot_{model}.png")
# pp_plot(p_vals, save_to=f"pp_plot_MUM.png")