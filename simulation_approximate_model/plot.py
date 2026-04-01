import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_intensity_1d(p, p_hat, filename):
    group_names = list(p.keys())
    n_group = len(group_names)
    n_subject = [p[group_name].shape[0] for group_name in group_names]
    n_voxel = p[group_names[0]].shape[1]

    fig, axs = plt.subplots(2*n_group, 2, figsize=(10, 10*n_group+1))
    for i, group_name in enumerate(group_names):
        q1 = 0
        q2 = int(np.percentile(np.arange(n_subject[i]), 25))
        q3 = int(np.percentile(np.arange(n_subject[i]), 75))
        q4 = n_subject[i] - 1

        # Subplot 1
        axs[2*i, 0].plot(p_hat[group_name][q1], label=f'estimated P')
        axs[2*i, 0].plot(p[group_name][q1], label='actual P')
        axs[2*i, 0].set_title(f"Probability at subject {q1+1} for group {group_name}")
        axs[2*i, 0].legend()

        # Subplot 2
        axs[2*i, 1].plot(p_hat[group_name][q2], label='estimated P')
        axs[2*i, 1].plot(p[group_name][q2], label='actual P')
        axs[2*i, 1].set_title(f"Probability at subject {q2+1} for group {group_name}")
        axs[2*i, 1].legend()

        # Subplot 3
        axs[2*i+1, 0].plot(p_hat[group_name][q3], label=f'estimated P')
        axs[2*i+1, 0].plot(p[group_name][q3], label=f'actual P')
        axs[2*i+1, 0].set_title(f"Probability at subject {q3+1} for group {group_name}")
        axs[2*i+1, 0].legend()

        # Subplot 4
        axs[2*i+1, 1].plot(p_hat[group_name][q4], label='estimated P')
        axs[2*i+1, 1].plot(p[group_name][q4], label='actual P')
        axs[2*i+1, 1].set_title(f"Probability at subject {q4+1} for group {group_name}")
        axs[2*i+1, 1].legend()

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Save the figure
    fig.savefig(filename)
    
    return 

def plot_intensity_2d(p, p_hat, n_voxel, filename):
    group_names = list(p.keys())
    n_group = len(group_names)
    n_subject = [p[group_name].shape[0] for group_name in group_names]

    fig, axs = plt.subplots(2*n_group, 4, figsize=(46, 20*n_group))
    for i, group_name in enumerate(group_names):
        q1 = 0
        q2 = int(np.percentile(np.arange(n_subject[i]), 25))
        q3 = int(np.percentile(np.arange(n_subject[i]), 75))
        q4 = n_subject[i] - 1

        min_value1 = min(p[group_name][q1,:].min(), p_hat[group_name][q1,:].min())
        max_value1 = max(p[group_name][q1,:].max(), p_hat[group_name][q1,:].max())
        # Subplot 1
        p_1 = p[group_name][q1,:].reshape((n_voxel))
        heatmap_1 = axs[2*i, 0].imshow(p_1, cmap='viridis', aspect='equal',vmin=min_value1, vmax=max_value1)
        axs[2*i, 0].set_title(f"Actual probability at subject {q1+1} for group {group_name}", fontsize=30)

        # Subplot 2
        p_hat_1 = p_hat[group_name][q1,:].reshape((n_voxel))
        heatmap_2 = axs[2*i, 1].imshow(p_hat_1, cmap='viridis', aspect='equal', vmin=min_value1, vmax=max_value1)
        axs[2*i, 1].set_title(f"Estimated probability at subject {q1+1} for group {group_name}", fontsize=30)
        plt.colorbar(heatmap_2, label='Probability')

        min_value2 = min(p[group_name][q2,:].min(), p_hat[group_name][q2,:].min())
        max_value2 = max(p[group_name][q2,:].max(), p_hat[group_name][q2,:].max())
        # Subplot 3
        p_2 = p[group_name][q2,:].reshape((n_voxel))
        heatmap_3 = axs[2*i, 2].imshow(p_2, cmap='viridis', aspect='equal', vmin=min_value2, vmax=max_value2)
        axs[2*i, 2].set_title(f"Actual probability at subject {q2+1} for group {group_name}", fontsize=30)

        # Subplot 4
        p_hat_2 = p_hat[group_name][q2,:].reshape((n_voxel))
        heatmap_4 = axs[2*i, 3].imshow(p_hat_2, cmap='viridis', aspect='equal', vmin=min_value2, vmax=max_value2)
        axs[2*i, 3].set_title(f"Estimated probability at subject {q2+1} for group {group_name}", fontsize=30)
        plt.colorbar(heatmap_4, label='Probability')

        min_value3 = min(p[group_name][q3,:].min(), p_hat[group_name][q3,:].min())
        max_value3 = max(p[group_name][q3,:].max(), p_hat[group_name][q3,:].max())
        # Subplot 5
        p_3 = p[group_name][q3,:].reshape((n_voxel))
        heatmap_5 = axs[2*i+1, 0].imshow(p_3, cmap='viridis', aspect='equal', vmin=min_value3, vmax=max_value3)
        axs[2*i+1, 0].set_title(f"Actual probability at subject {q3+1} for group {group_name}", fontsize=30)

        # Subplot 6
        p_hat_3 = p_hat[group_name][q3,:].reshape((n_voxel))
        heatmap_6 = axs[2*i+1, 1].imshow(p_hat_3, cmap='viridis', aspect='equal', vmin=min_value3, vmax=max_value3)
        axs[2*i+1, 1].set_title(f"Estimated probability at subject {q3+1} for group {group_name}", fontsize=30)
        plt.colorbar(heatmap_6, label='Probability')

        min_value4 = min(p[group_name][q4,:].min(), p_hat[group_name][q4,:].min())
        max_value4 = max(p[group_name][q4,:].max(), p_hat[group_name][q4,:].max())
        # Subplot 7
        p_4 = p[group_name][q4,:].reshape((n_voxel))
        heatmap_7 = axs[2*i+1, 2].imshow(p_4, cmap='viridis', aspect='equal', vmin=min_value4, vmax=max_value4)
        axs[2*i+1, 2].set_title(f"Actual probability at subject {q4+1} for group {group_name}", fontsize=30)

        # Subplot 8
        p_hat_4 = p_hat[group_name][q4,:].reshape((n_voxel))
        heatmap_8 = axs[2*i+1, 3].imshow(p_hat_4, cmap='viridis', aspect='equal', vmin=min_value4, vmax=max_value4)
        axs[2*i+1, 3].set_title(f"Estimated probability at subject {q4+1} for group {group_name}", fontsize=30)
        plt.colorbar(heatmap_8, label='Probability')

    # Adjust layout to prevent overlapping
    plt.tight_layout()
    # Save the figure
    fig.savefig(filename)
    
    return 

def plot_intensity_3d(p, p_hat, n_voxel, filename, slice_idx=None):
    group_names = list(p.keys())
    n_group = len(group_names)
    n_subject = [p[group_name].shape[0] for group_name in group_names]

    fig, axs = plt.subplots(2*n_group, 4, figsize=(46, 20*n_group))
    for i, group_name in enumerate(group_names):
        q1 = 0
        q2 = int(np.percentile(np.arange(n_subject[i]), 25))
        q3 = int(np.percentile(np.arange(n_subject[i]), 75))
        q4 = n_subject[i] - 1

        slice_idx = round(0.5*n_voxel[2]) if slice_idx is None else slice_idx
        print("slice_idx", slice_idx)

        # Subplot 1
        p_1 = p[group_name][q1,:].reshape((n_voxel))
        heatmap_1 = axs[2*i, 0].imshow(p_1[:, :, slice_idx], cmap='viridis', aspect='equal')
        axs[2*i, 0].set_title(f"Actual probability at subject {q1+1} for group {group_name}")
        plt.colorbar(heatmap_1, label='Probability')

        # Subplot 2
        p_hat_1 = p_hat[group_name][q1,:].reshape((n_voxel))
        heatmap_2 = axs[2*i, 1].imshow(p_hat_1[:, :, slice_idx], cmap='viridis', aspect='equal')
        axs[2*i, 1].set_title(f"Estimated probability at subject {q1+1} for group {group_name}")
        plt.colorbar(heatmap_2, label='Probability')

        # Subplot 3
        p_2 = p[group_name][q2,:].reshape((n_voxel))
        heatmap_3 = axs[2*i, 2].imshow(p_2[:, :, slice_idx], cmap='viridis', aspect='equal')
        axs[2*i, 2].set_title(f"Actual probability at subject {q2+1} for group {group_name}")
        plt.colorbar(heatmap_3, label='Probability')

        # Subplot 4
        p_hat_2 = p_hat[group_name][q2,:].reshape((n_voxel))
        heatmap_4 = axs[2*i, 3].imshow(p_hat_2[:, :, slice_idx], cmap='viridis', aspect='equal')
        axs[2*i, 3].set_title(f"Estimated probability at subject {q2+1} for group {group_name}")
        plt.colorbar(heatmap_4, label='Probability')


        # Subplot 5
        p_3 = p[group_name][q3,:].reshape((n_voxel))
        heatmap_5 = axs[2*i+1, 0].imshow(p_3[:, :, slice_idx], cmap='viridis', aspect='equal')
        axs[2*i+1, 0].set_title(f"Actual probability at subject {q3+1} for group {group_name}")
        plt.colorbar(heatmap_5, label='Probability')

        # Subplot 6
        p_hat_3 = p_hat[group_name][q3,:].reshape((n_voxel))
        heatmap_6 = axs[2*i+1, 1].imshow(p_hat_3[:, :, slice_idx], cmap='viridis', aspect='equal')
        axs[2*i+1, 1].set_title(f"Estimated probability at subject {q3+1} for group {group_name}")
        plt.colorbar(heatmap_6, label='Probability')

        # Subplot 7
        p_4 = p[group_name][q4,:].reshape((n_voxel))
        heatmap_7 = axs[2*i+1, 2].imshow(p_4[:, :, slice_idx], cmap='viridis', aspect='equal')
        axs[2*i+1, 2].set_title(f"Actual probability at subject {q4+1} for group {group_name}")
        plt.colorbar(heatmap_7, label='Probability')

        # Subplot 8
        p_hat_4 = p_hat[group_name][q4,:].reshape((n_voxel))
        heatmap_8 = axs[2*i+1, 3].imshow(p_hat_4[:, :, slice_idx], cmap='viridis', aspect='equal')
        axs[2*i+1, 3].set_title(f"Estimated probability at subject {q4+1} for group {group_name}")
        plt.colorbar(heatmap_8, label='Probability')

    # Adjust layout to prevent overlapping
    plt.tight_layout()
    # Save the figure
    fig.savefig(filename)
    
    return 

def plot_intensity_brain(p, p_hat, n_voxel, filename, slice_idx=None):
    return 