import os
import numpy as np
import nibabel as nib  # For reading NIfTI files
from nilearn.plotting import plot_stat_map
from plot import plot_brain
import time

smooth_lesion_mask_path = os.getcwd() + "/data/brain/smooth_lesion_mask_Simulation.nii.gz"
smooth_lesion_mask = nib.load(smooth_lesion_mask_path)
n_voxels = np.count_nonzero(smooth_lesion_mask.get_fdata())

empirical_mask_path = os.getcwd() + "/data/brain/empir_prob_mask.nii.gz"
empirical_mask = nib.load(empirical_mask_path)
empirical_data = empirical_mask.get_fdata().astype(np.float32) # shape: (91, 109, 91)
empirical_data = empirical_data[smooth_lesion_mask.get_fdata().astype(bool)] # shape: (n_voxels, )

# coef_age_img_path = os.getcwd() + "/data/brain/coef_age_nvars_1_method_2.nii.gz"
# coef_age_img = nib.load(coef_age_img_path).get_fdata().astype(np.float32)
# coef_age_img = coef_age_img[smooth_lesion_mask.get_fdata().astype(bool)]
# coef_intercept_img_path = os.getcwd() + "/data/brain/coef_Intercept_nvars_1_method_2.nii.gz"
# coef_intercept_img = nib.load(coef_intercept_img_path).get_fdata().astype(np.float32)
# coef_intercept_img = coef_intercept_img[smooth_lesion_mask.get_fdata().astype(bool)]
# coef = np.stack([coef_age_img, coef_intercept_img], axis=1)

# all voxels within the empirical lesion mask
indices_type_0 = np.arange(n_voxels)
# p_hat \in (0.005, 1]
indices_type_1 = np.where((empirical_data > 0.005) & (empirical_data <= 1))[0]
# p_hat \in (0.005, 0.01]
indices_type_2 = np.where((empirical_data > 0.005) & (empirical_data <= 0.01))[0]
# p_hat \in (0.01, 0.05]
indices_type_3 = np.where((empirical_data > 0.01) & (empirical_data <= 0.05))[0]
# p_hat \in (0.05, 0.1]
indices_type_4 = np.where((empirical_data > 0.05) & (empirical_data <= 0.1))[0]
# p_hat \in (0.1, 1]
indices_type_5 = np.where((empirical_data > 0.1) & (empirical_data <= 1))[0]

M = 100  # number of datasets
N = 1000  # number of subjects
model = "MassUnivariateRegression" #"SpatialBrainLesion"
distribution = "Poisson" #"Bernoulli"
link_func = "log" #"logit"
full_model = True
filename_0 = "full_model" if full_model else "approximate_model"
indices_type = indices_type_5
n_lesion_voxel = len(indices_type)

all_P, all_empirical_P = np.empty((M, n_lesion_voxel)), np.empty((M, n_lesion_voxel))
start_time = time.time()
# load simulated data
for random_seed in range(M):
    simulated_data_path = os.getcwd() + f"/data/brain/data_Simulation/data_Simulation_random_seed_{random_seed}.npz"
    simulated_data = np.load(simulated_data_path)
    Y = simulated_data['Y']  # shape: (n_subjects, n_lesion_voxel)
    empirical_P_mean = np.mean(Y, axis=0)[indices_type] # shape: (n_lesion_voxel,)
    all_empirical_P[random_seed, :] = empirical_P_mean
    results_path = os.getcwd()+f"/results/brain/{model}_{distribution}_{link_func}/brain_Probability_comparison_Simulation_{filename_0}_linear_random_seed_{random_seed}.npz"
    results = np.load(results_path)
    if full_model:
        P = results['P']  # shape: (n_simulations, n_lesion_voxel)
        P_hat = np.mean(P, axis=0)[indices_type]
    else:
        P_hat = results['P_mean'][indices_type]
    all_P[random_seed, :] = P_hat
# remove nan rows in all_P
print("all_P shape before removing NaNs:", all_P.shape)
all_P = all_P[~np.isnan(all_P).any(axis=1)]
print("all_P shape after removing NaNs:", all_P.shape)
# Bias
Bias = np.mean(all_P - all_empirical_P, axis=0)
print("Bias", np.mean(Bias), np.std(Bias))

# # convert Bias back to brain image
# Bias_full = np.zeros(smooth_lesion_mask.shape)
# Bias_full[smooth_lesion_mask.get_fdata().astype(bool)] = Bias.flatten()
# Bias_full[~smooth_lesion_mask.get_fdata().astype(bool)] = np.nan
# Bias_image = nib.Nifti1Image(Bias_full, affine=smooth_lesion_mask.affine, header=smooth_lesion_mask.header)
# nib.save(Bias_image, os.getcwd()+f"/results/brain/brain_Bias_{model}_{distribution}_{link_func}_{filename_0}_linear.nii.gz")
# plot Bias
# plot_stat_map(
#     Bias_image, 
#     cut_coords=[0,6,12,18,24,30,36],
#     display_mode='z', 
#     draw_cross=False, 
#     cmap='inferno',
#     colorbar=True,
#     vmax=5e-3,
#     vmin=-5e-3,
#     output_file=os.getcwd()+f"/results/brain/P_bias_{model}_{distribution}_{link_func}_{filename_0}_linear.png"
#     )

# Var
Var = np.var(all_P, axis=0, ddof=1)
print("Var", np.mean(Var), np.std(Var))

# MSE
MSE = 1/M * np.sum((all_P - all_empirical_P) ** 2, axis=0)
print("MSE", np.mean(MSE), np.std(MSE))
# PU: Probability of underestimation
PU = 1/M * np.sum((all_P < all_empirical_P).astype(np.float32), axis=0)
print("PU", np.mean(PU), np.std(PU))
# # Pearson correlation
# Pearson_corr = np.corrcoef(all_P, all_P)
exit()

# empirical_P = np.mean(empirical_data['Y'], axis=0)  # shape: (n_voxels)
# # plot_brain(np.sqrt(empirical_P), smooth_lesion_mask, output_filename=os.getcwd() + "/empirical_probability.png")

# full_model = True
# model = "SpatialBrainLesion" 
# distribution = "Bernoulli"  #"Poisson"
# link_func = "logit"
# filename_0 = "full_model" if full_model else "approximate_model"

# result_path = os.getcwd()+"/results/brain/brain_Probability_comparison_Simulation_{}_linear_{}_{}_{}_link_func.npz".format(filename_0, model, distribution, link_func)
# results = np.load(result_path)
# P = results['P']
# avg_P = np.mean(P, axis=0)  # shape: (n_voxel, )

# err = avg_P - empirical_P
# rel_bias_P = np.mean(err) / np.mean(empirical_P)  # relative bias
# rel_std_P = np.std(err, ddof=1) / np.std(empirical_P, ddof=1)  # relative standard deviation
# rel_mse_P = np.mean(err**2) / np.var(empirical_P, ddof=1)  # relative mean squared error
# print(f"Relative Bias: {rel_bias_P}, Relative Std: {rel_std_P}, Relative MSE: {rel_mse_P}")

# exit()

# print(rel_bias_P, rel_std_P, rel_mse_P)

# =========================================================


empirical_data_path = os.getcwd() + "/data/brain/data_Simulation.npz"
empirical_data = np.load(empirical_data_path, allow_pickle=True)
empirical_P = np.mean(empirical_data['Y'], axis=0)  # shape: (n_voxels)

model = "SpatialBrainLesion"
distribution = "Bernoulli" 
link_func = "logit"
# baseline
full_model = True
filename_0 = "full_model" if full_model else "approximate_model"
baseline_results_path = os.getcwd() + f"/results/brain/brain_Probability_comparison_Simulation_{filename_0}_linear_SpatialBrainLesion_{distribution}_{link_func}_link_func.npz"
baseline_results = np.load(baseline_results_path, allow_pickle=True)
baseline_P = baseline_results['P']
print(baseline_P[0])
print(baseline_P[0].shape)
exit()
baseline_avg_P = np.mean(baseline_P, axis=0)  # shape: (n_voxel, )

rel_bias_P = np.mean(baseline_avg_P - empirical_P) / np.mean(empirical_P)  # relative bias
rel_std_P = np.std(baseline_avg_P - empirical_P) / np.mean(empirical_P)  # relative standard deviation
rel_mse_P = np.mean((baseline_avg_P - empirical_P) ** 2) / np.mean(empirical_P)  # relative mean squared error
print(f"Baseline - Relative Bias: {rel_bias_P}, Relative Std: {rel_std_P}, Relative MSE: {rel_mse_P}")

# comparison
full_model = False
filename_0 = "full_model" if full_model else "approximate_model"
results_path = os.getcwd() + f"/results/brain/brain_Probability_comparison_Simulation_{filename_0}_linear_SpatialBrainLesion_{distribution}_{link_func}_link_func.npz"
results = np.load(results_path, allow_pickle=True)
P_mean = results['MU_mean']

rel_bias_P = np.mean(1.29*P_mean - empirical_P) / np.mean(empirical_P)  # relative bias
rel_std_P = np.std(1.29*P_mean - empirical_P) / np.mean(empirical_P)  # relative standard deviation
rel_mse_P = np.mean((1.29*P_mean - empirical_P) ** 2) / np.mean(empirical_P)  # relative mean squared error
print(f"Relative Bias: {rel_bias_P}, Relative Std: {rel_std_P}, Relative MSE: {rel_mse_P}")
