import numpy as np
import matplotlib.pyplot as plt

results_bernoulli = np.load("results/1D_Probability_comparison_Simulation_Homogeneous_1_group_Bernoulli_logit_link_func.npz", allow_pickle=True)
P_bernoulli = results_bernoulli["P"]

results_poission = np.load("results/1D_Probability_comparison_Simulation_Homogeneous_1_group_Poisson_log_link_func.npz", allow_pickle=True)
P_poission = results_poission["P"]

P_diff = P_bernoulli - P_poission

n_subject = P_bernoulli.shape[0]

q1 = 0
q2 = int(np.percentile(np.arange(n_subject), 25))
q3 = int(np.percentile(np.arange(n_subject), 75))
q4 = n_subject - 1

# Bias plot
fig, axs = plt.subplots(2, 2, figsize=(10, 11))
# Subplot 1
axs[0, 0].plot(P_diff[q1], label=f'bias P')
axs[0, 0].set_title(f"Bias at subject {q1+1}")
axs[0, 0].legend()
# Subplot 2
axs[0, 1].plot(P_diff[q2], label='bias P')
axs[0, 1].set_title(f"Bias at subject {q2+1}")
axs[0, 1].legend()
# Subplot 3
axs[1, 0].plot(P_diff[q3], label=f'bias P')
axs[1, 0].set_title(f"Bias at subject {q3+1}")
axs[1, 0].legend()
# Subplot 4
axs[1, 1].plot(P_diff[q4], label='bias P')
axs[1, 1].set_title(f"Bias at subject {q4+1}")
axs[1, 1].legend()
# Adjust layout to prevent overlapping
plt.tight_layout()
# Save the figure
fig.savefig("figures/test.png")

