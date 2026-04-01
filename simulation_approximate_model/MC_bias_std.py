import numpy as np
import matplotlib.pyplot as plt

n_group = 100
n_subject = 2000
results = np.load("results/results_bump.npz", allow_pickle=True)
P = results["P"].item()
MU = results["MU"].item()

all_MU, all_P = list(), list()
for i in range(n_group):
    P_i = P[f"group_{i}"].reshape(1, 2000, 1000)
    MU_i = MU[f"group_{i}"].reshape(1, 2000, 1000)
    all_P.append(P_i)
    all_MU.append(MU_i)
all_P = np.concatenate(all_P, axis=0)
all_MU = np.concatenate(all_MU, axis=0)

bias = np.mean(all_P-all_MU, axis=0)
std = np.std(all_P, axis=0)

q1 = 0
q2 = int(np.percentile(np.arange(n_subject), 25))
q3 = int(np.percentile(np.arange(n_subject), 75))
q4 = n_subject - 1

# Bias plot
fig, axs = plt.subplots(2, 2, figsize=(10, 11))
# Subplot 1
axs[0, 0].plot(bias[q1], label=f'bias P')
axs[0, 0].set_title(f"Bias at subject {q1+1}")
axs[0, 0].legend()
# Subplot 2
axs[0, 1].plot(bias[q2], label='bias P')
axs[0, 1].set_title(f"Bias at subject {q2+1}")
axs[0, 1].legend()
# Subplot 3
axs[1, 0].plot(bias[q3], label=f'bias P')
axs[1, 0].set_title(f"Bias at subject {q3+1}")
axs[1, 0].legend()
# Subplot 4
axs[1, 1].plot(bias[q4], label='bias P')
axs[1, 1].set_title(f"Bias at subject {q4+1}")
axs[1, 1].legend()
# Adjust layout to prevent overlapping
plt.tight_layout()
# Save the figure
fig.savefig("figures/bias_bump.png")


# Std plot
fig, axs = plt.subplots(2, 2, figsize=(10, 11))
# Subplot 1
axs[0, 0].plot(std[q1], label=f'std P')
axs[0, 0].set_title(f"Standard error at subject {q1+1}")
axs[0, 0].legend()
# Subplot 2
axs[0, 1].plot(std[q2], label='std P')
axs[0, 1].set_title(f"Standard error at subject {q2+1}")
axs[0, 1].legend()
# Subplot 3
axs[1, 0].plot(std[q3], label=f'std P')
axs[1, 0].set_title(f"Standard error at subject {q3+1}")
axs[1, 0].legend()
# Subplot 4
axs[1, 1].plot(std[q4], label='std P')
axs[1, 1].set_title(f"Standard error at subject {q4+1}")
axs[1, 1].legend()
# Adjust layout to prevent overlapping
plt.tight_layout()
# Save the figure
fig.savefig("figures/std_bump.png")