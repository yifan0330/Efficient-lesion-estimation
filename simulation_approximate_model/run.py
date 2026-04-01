from data_simulation import simulated_data
from bspline import B_spline_bases
from regression import BrainRegression
from inference import BrainInference
from nilearn.datasets import load_mni152_template
from absl import logging 
import numpy as np
import argparse
import torch 
import sys
import os

# Example usage:
# python run.py --n_auxiliary=0 --simulated_dset True --covariate True --model="SpatialBrainLesion" --regression_terms 'multiplicative' 'additive' --n_group=2 --n_subject 1000 2000 --spacing=10 --space_dim=2 --n_voxel 100 100 --lesion_per_subject 10 10 --homogeneous True

def parse_int_or_str(value):
    try:
        return int(value)
    except ValueError:
        return value
        
def get_args():
    parser = argparse.ArgumentParser(description="Arguments for data generation, regression, and inference")
    # Boolean flags
    parser.add_argument('--simulated_dset', type=lambda x: x.lower() == 'true', default=True,
                            help="Use simulationed dataset (True or False, default: True)")
    parser.add_argument('--homogeneous', type=lambda x: x.lower() == 'true', default=True,
                            help="Set homogeneous underlying function (True or False, default: True)")
    parser.add_argument('--covariate', type=lambda x: x.lower() == 'true', default=True,
                            help="Include spatially varying covariate effect (default: True)")
    # modelling stages
    parser.add_argument('--run_data_generation', type=lambda x: x.lower() == 'true', default=True,
                        help="Run data generation (default: True)")
    parser.add_argument('--run_regression', type=lambda x: x.lower() == 'true', default=True,
                        help="Run regression (default: True)")
    parser.add_argument('--run_inference', type=lambda x: x.lower() == 'true', default=True,
                        help="Run inference (default: True)")
    # Model parameters
    parser.add_argument('--model', type=str, default="SpatialBrainLesion",
                        help="Type of stochastic model (default: Poisson)")
    parser.add_argument('--regression_terms', nargs='+', type=str, help="Regression terms (default: ['multiplicative', 'additive'])")
    parser.add_argument('--link_func', type=str, default="logit", help="Link function for intensity function (default: logit)")
    parser.add_argument('--marginal_dist', type=str, default="Bernoulli", help="Marginal distribution at each spatial location (default: Bernoulli)")
    parser.add_argument('--std_params', type=float, default=0.1, help="Standard deviation of Gaussian parameters (default: 0.1)")
    parser.add_argument('--lr', type=float, default=1, help="Learning rate for optimisation (default: 0.1)")
    parser.add_argument('--tol', type=float, default=1e-7, help="Tolerance for optimisation (default: 1e-7)")
    parser.add_argument('--iter', type=int, default=1e4, help="Number of iterations for optimisation (default: 100)")

    # Inference parameters
    parser.add_argument('--t_con_groups', nargs='+', type=int, default=None, help="Contrast groups for t-test (default: None)")
    parser.add_argument('--t_con_group_name', type=str, default=None, help="Contrast groups names for t-test (default: None)")

    # General options
    parser.add_argument('--gpus', type=str, default="0", help="GPU device (default: 0)")
    parser.add_argument('--space_dim', type=parse_int_or_str, default=1, 
                        help="Dimension of simulation space (default: 1)")
    parser.add_argument('--n_group', type=int, default=1,
                        help="Number of groups (default: 1)")
    parser.add_argument('--group_names', nargs='+', type=str, default=None,
                        help="Name of groups (default: Group_1)")
    parser.add_argument('--n_subject', nargs='+',type=int, default=[0],
                        help="Number of subjects (default: 0)")
    parser.add_argument('--n_voxel', nargs='+', type=int, default=[0],
                        help="Number of voxel (default: 0). Accepts a single integer or a comma-separated list of integers.")
    parser.add_argument('--spacing', type=int, default=10,
                        help="Spacing for B-spline basis (default: 10)")
    parser.add_argument('--lesion_per_subject', nargs='+', type=int, default=[10],
                        help="Number of lesions per subject (default: 10). Accepts a single integer or a comma-separated list of integers.")

    # Auxiliary variables
    parser.add_argument('--n_auxiliary', type=int, default=2,
                        help="Number of auxiliary variables (default: 2)")
    parser.add_argument('--std_auxiliary', type=float, default=1.0, 
                        help="Standard deviation of auxiliary variables (default: 1.0)")
    parser.add_argument('--n_samples', type=int, default=100, help="Number of samples for Monte Carlo approximation (default: 100)")
    return parser.parse_args()

args = get_args()

simulated_dset = args.simulated_dset
homogeneous = args.homogeneous
covariate = args.covariate
space_dim = args.space_dim
n_group = args.n_group
group_names = args.group_names
n_subject = args.n_subject
n_voxel = args.n_voxel
lesion_per_subject = args.lesion_per_subject
model = args.model
lr = args.lr
tolerance_change = args.tol
iter = args.iter

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
if torch.cuda.is_available() and torch.cuda.device_count() >= 1:
    device = 'cuda'
else:
    device = 'cpu'

filename_0 = "_Simulation" if simulated_dset else "_RealDataset"
filename_1 = "_Homogeneous" if homogeneous else "_BumpSignals"

logging.info(f"Construct spatial design matrix ...")
spacing = args.spacing
brain_mask = load_mni152_template(resolution=2) if space_dim == "brain" else None
X_spatial = B_spline_bases(space_dim=space_dim, dim=n_voxel, brain_mask=brain_mask, spacing=spacing)
P = X_spatial.shape[1] # Number of spatial features, dimension of beta

if args.run_data_generation:
    logging.info("Generate data ...")
    lesion_size_mapping = {
        1: [1, 3],
        2: [1, 8],
        3: [1, 16],
        "brain": [1, 16]
    }
    lesion_size_range = lesion_size_mapping.get(space_dim, None)

    data_simulation = simulated_data(space_dim=space_dim, n_group=n_group, n_subject=n_subject, n_voxel=n_voxel, brain_mask=brain_mask,
                                    group_names=group_names, homogeneous_intensity=homogeneous, lesion_per_subject=lesion_per_subject)
    MU, Y, Z = data_simulation.generate_data(covariate=covariate, lesion_size_range=lesion_size_range)
    data = dict(MU=MU, X_spatial=X_spatial, Y=Y, Z=Z)

if args.run_regression:
    logging.info("Setup model and optimise regression coefficients")
    BR = BrainRegression(dtype=torch.float64, device=device)
    BR.load_data(data)
    BR.init_model(model, 
                n_auxiliary=args.n_auxiliary, 
                std_auxiliary=args.std_auxiliary,
                n_samples=args.n_samples,
                regression_terms=args.regression_terms,
                link_func=args.link_func,
                marginal_dist=args.marginal_dist,
                std_params=args.std_params)

    BR.optimize_model(lr, iter, tolerance_change)

    beta = dict()
    for group_name in BR.group_names:
        beta[group_name] = BR.model[group_name].beta.detach().cpu().numpy()

    gamma = dict()
    for group_name in BR.group_names:
        gamma[group_name] = BR.model[group_name].gamma.detach().cpu().numpy()

    beta = BR.IRLS_update(beta, gamma, tolerance_change)

    P = BR.estimated_P(beta)

    # P = BR.estimated_P(beta)

    # np.savez(f"{os.getcwd()}/results/{space_dim}D_Probability_comparison{filename_0}{filename_1}_{n_group}_group_{args.marginal_dist}_{args.link_func}_link_func.npz", P=P, 
    #         Y=Y, X_spatial=X_spatial, Z=Z, beta=beta)

    import matplotlib.pyplot as plt
    from plot import plot_intensity_1d, plot_intensity_2d, plot_intensity_3d, plot_intensity_brain
    fig_filename = f"{os.getcwd()}/figures/{space_dim}D_Probability_comparison{filename_0}{filename_1}_{n_group}_group_{args.marginal_dist}_{args.link_func}_link_func.png"

    if space_dim == 1:
        plot_intensity_1d(MU, P, fig_filename)
    elif space_dim == 2:    
        plot_intensity_2d(MU, P, n_voxel, fig_filename)
    elif space_dim == 3:
        plot_intensity_3d(MU, P, n_voxel, fig_filename)
    elif space_dim == "brain":
        plot_intensity_brain(MU, P, n_voxel, fig_filename)

    
if args.run_inference:
    # load optimised params
    results_filename = f"{os.getcwd()}/results/{space_dim}D_Probability_comparison{filename_0}{filename_1}_{n_group}_group_{args.marginal_dist}_{args.link_func}_link_func.npz"
    results = np.load(results_filename, allow_pickle=True)
    # BrainInference
    BI = BrainInference(model=model, marginal_dist=args.marginal_dist, 
                        link_func=args.link_func, regression_terms=args.regression_terms,
                        dtype=torch.float64, device=device)
    BI.load_params(params=results)
    BI.create_contrast(t_con_groups=args.t_con_groups, t_con_group_name=args.t_con_group_name)
    BI.run_inference(method="sandwich")




