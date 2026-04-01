import numpy as np
import matplotlib.pyplot as plt

data = np.load(f"data/UKB/masked_data_RealDataset_approximate_model_Poisson_log_link_func_spacing_5.npz")
Y, Z = data["Y"], data["Z"]
Y_mean = np.mean(Y, axis=0)

# Find indices of the largest 100 elements
indices = np.argpartition(Y_mean, -100)[-100:]
# Create a boolean mask of the same shape as arr, initialized to False
mask = np.zeros_like(Y_mean, dtype=bool)
# Set the positions of the largest 100 elements to True
mask[indices] = True

p_empirical = []
age_range = np.arange(np.ceil(np.min(Z[:, 2])), np.ceil(np.max(Z[:, 2])))
print(age_range)
exit()
for age in age_range:
    indices = np.where((Z[:, 2] >= age) & (Z[:, 2] < age+1))
    p_empirical.append(np.mean(Y[indices][:, mask]))
p_empirical = np.array(p_empirical)
log_p_empirical = np.log(p_empirical)


# Figure setup
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

# Create the bar plot
bars = ax.scatter(age_range, log_p_empirical, color='skyblue', edgecolor='black', linewidth=1.2, label='empirical log(p)')

# Add a fitted line across the bars
fit_linear = np.polyfit(age_range, log_p_empirical, 1)
fit_linear_fn = np.poly1d(fit_linear)
ax.plot(age_range, fit_linear_fn(age_range), color='red', linestyle='--', linewidth=2, label='Linear fit')

fit_cubic = np.polyfit(age_range, log_p_empirical, 3)
fit_cubic_fn = np.poly1d(fit_cubic)
ax.plot(age_range, fit_cubic_fn(age_range), color='blue', linestyle='--', linewidth=2, label='Cubic fit')

# Add legend
ax.legend(fontsize=14)

# Aesthetics: labels, title, axes limits, and ticks
ax.set_ylabel('Log empirical lesion probability', fontsize=14)
ax.set_xlabel('Age', fontsize=14)
# ax.set_ylim(0, 0.5)  # Adjust based on your data range

# Optional: add gridlines for readability
ax.yaxis.grid(True, linestyle='--', alpha=0.7)

# Remove unnecessary borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Adjust tick parameters
ax.tick_params(axis='both', which='major', labelsize=12)

# Tight layout to prevent overlap
plt.tight_layout()

# Save the figure (high resolution suitable for papers)
plt.savefig('scatter_plot_age.png', dpi=300)

# p_empirical = []
# CVR_range = np.arange(np.floor(np.min(Z[:, 4])), np.ceil(np.max(Z[:, 4])))
# for CVR in CVR_range:
#     indices = np.where(Z[:, 4] == CVR)
#     p_empirical.append(np.mean(Y[indices][:, mask]))
# p_empirical = np.array(p_empirical)

# # Figure setup
# fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

# # Create the bar plot
# bars = ax.bar(CVR_range, p_empirical, yerr=0, capsize=5, color='skyblue', edgecolor='black', linewidth=1.2)

# # Add a fitted line across the bars
# fit = np.polyfit(CVR_range, p_empirical, 3)
# fit_fn = np.poly1d(fit)
# ax.plot(CVR_range, fit_fn(CVR_range), color='red', linestyle='--', linewidth=2)

# # Aesthetics: labels, title, axes limits, and ticks
# ax.set_ylabel('Empirical lesion probability', fontsize=14)
# ax.set_xlabel('Age', fontsize=14)
# ax.set_ylim(0, 0.5)  # Adjust based on your data range

# # Optional: add gridlines for readability
# ax.yaxis.grid(True, linestyle='--', alpha=0.7)

# # Remove unnecessary borders
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

# # Adjust tick parameters
# ax.tick_params(axis='both', which='major', labelsize=12)

# # Tight layout to prevent overlap
# plt.tight_layout()

# # Save the figure (high resolution suitable for papers)
# plt.savefig('test.png', dpi=300)

