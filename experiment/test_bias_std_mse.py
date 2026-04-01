import numpy as np

data = np.load("data/3D/3D_data_Simulation_Homogeneous_full_model_2_group_[2000, 4000]_Poisson_log_link_func.npz")
MU = data["MU"] # shape: (6000, 125000)

results = np.load("results/3D/3D_Probability_comparison_Simulation_Homogeneous_full_model_dask_approximate_2_group_[2000, 4000]_Poisson_log_link_func.npz")
P = results["P"] # shape: (6000, 125000)

bias = np.mean(P - MU)
std = np.std(P - MU)
MSE = np.mean((P - MU)**2)
print(bias, std, MSE)

relative_bias = bias / np.mean(MU)
relative_std = std / np.mean(MU)
relative_MSE = MSE / np.mean(MU**2)
print(relative_bias, relative_std, relative_MSE)