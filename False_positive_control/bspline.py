import nilearn.plotting
import nilearn
import scipy.stats as stats
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
import time

def w(t, i, k, u):
    if u[i] != u[i + k - 1]:
        wt = (t - u[i]) / (u[i + k - 1] - u[i])
    else:
        wt = 0
    return wt

def recu(t, i, k, u):
    if k == 1:
        if u[i] == u[i + 1]:
            B = np.zeros_like(t)
        else:
            B = np.logical_and(u[i] <= t, t < u[i + 1])
    else:
        B = w(t, i, k, u) * recu(t, i, k - 1, u) + (1 - w(t, i + 1, k, u)) * recu(t, i + 1, k - 1, u)
    return B

def Bspline(t, k, u, v=None, ForceSup=1):
    n = len(u)
    if k + 1 > n:
        raise ValueError("u must be at least length k + 1")
    if (v is not None) and (len(v) + k != n) and (ForceSup):
        raise ValueError("{} knots requires {} control vertices".format(n, n - k))

    t = np.array(t).reshape(-1, 1)
    u = np.array(u).reshape(-1, 1)
    nBasis = n - k
    B = np.zeros((len(t), nBasis))
    iB = np.zeros(nBasis)
    for i in range(nBasis):
        B[:, i] = recu(t, i, k, u).flatten()
        iB[i] = (u[i + k] - u[i]) / k
    if n >= 2 * k:
        if ForceSup:
            bool_vec = np.logical_or(t < u[k], t >= u[n - k + 1]).flatten()
            B[bool_vec, :] = 0
    else:
        print("Insufficient knots to be a proper spline basis")
    if v is not None:
        B = np.dot(B, v.reshape(-1, 1))
    return B

def B_spline_bases(space_dim, dim, brain_mask=None, spacing=10, margin=20, dtype=np.float64):
    if space_dim == 1:
        H = dim[0]
        if not isinstance(H, int):
            raise ValueError("Invalid input: dim must be an integer for 1D simulation.")
    elif space_dim == 2:
        if isinstance(dim, list) and len(dim) == 2:
            H, D = dim
        else:
            raise ValueError(f"Invalid input: dim must be a list with two elements for 2D simulation."
                             f"Received: {type(dim).__name__} with value {dim}.")
    elif space_dim == 3:
        if isinstance(dim, list) and len(dim) == 3:
            H, D, W = dim
        else:
            raise ValueError(f"Invalid input: dim must be a list with three elements for 3D simulation."
                             f"Received: {type(dim).__name__} with value {dim}.")
    elif space_dim == "brain":
        # nilearn.plotting.plot_anat(mni152_2mm_template, output_file="template.png")
        H, D, W = brain_mask.shape
    else:
        raise ValueError(f"Unsupported space_dim value: {space_dim}. space_dim must be 1 or 2 or 3 or brain template.")
    
    if isinstance(space_dim, int):
        xx = np.arange(H)
        if space_dim >= 2:
            yy = np.arange(D)
            if space_dim == 3:
                zz = np.arange(W)
    elif space_dim == "brain":
        masker_voxels = brain_mask.get_fdata()
        dim_mask = masker_voxels.shape
        # remove the blank space around the brain mask
        xx = np.where(np.apply_over_axes(np.sum, masker_voxels, [1, 2]) > 0)[0]
        yy = np.where(np.apply_over_axes(np.sum, masker_voxels, [0, 2]) > 0)[1]
        zz = np.where(np.apply_over_axes(np.sum, masker_voxels, [0, 1]) > 0)[2]
    # X direction
    wider_xx = np.arange(np.min(xx) - margin, np.max(xx) + margin)
    xx_knots = np.arange(np.min(wider_xx), np.max(wider_xx), step=spacing)
    xx_knots = np.concatenate(([xx_knots[0]]*2, xx_knots, [xx_knots[-1]]*2), axis=0)
    x_spline = Bspline(t=xx, k=4, u=xx_knots, ForceSup=1)
    x_support_basis = np.sum(x_spline, axis=0) > 0.1
    x_spline = x_spline[:, x_support_basis]
    del wider_xx, xx_knots, x_support_basis
    # data type
    x_spline = x_spline.astype(dtype)
    if space_dim == 1:
        X = x_spline.copy()
    elif space_dim in [2, 3, "brain"]:
        # y direction
        wider_yy = np.arange(np.min(yy) - margin, np.max(yy) + margin)
        yy_knots = np.arange(np.min(wider_yy), np.max(wider_yy), step=spacing)
        yy_knots = np.concatenate(([yy_knots[0]]*2, yy_knots, [yy_knots[-1]]*2), axis=0)
        y_spline = Bspline(t=yy, k=4, u=yy_knots, ForceSup=1)
        y_support_basis = np.sum(y_spline, axis=0) > 0.1
        y_spline = y_spline[:, y_support_basis]  
        # data type
        y_spline = y_spline.astype(dtype)
        del wider_yy, yy_knots, y_support_basis
        if space_dim == 2:
            # create spatial design matrix by tensor product of spline bases in 2 dimesions
            X = np.einsum('ab,cd->acbd', x_spline, y_spline).reshape(
                x_spline.shape[0] * y_spline.shape[0],
                x_spline.shape[1] * y_spline.shape[1]
            )
        elif space_dim in [3, "brain"]:
            # z direction
            wider_zz = np.arange(np.min(zz) - margin, np.max(zz) + margin)
            zz_knots = np.arange(np.min(wider_zz), np.max(wider_zz), step=spacing)
            zz_knots = np.concatenate(([zz_knots[0]]*2, zz_knots, [zz_knots[-1]]*2), axis=0)
            z_spline = Bspline(t=zz, k=4, u=zz_knots, ForceSup=1)
            z_support_basis = np.sum(z_spline, axis=0) > 0.1
            z_spline = z_spline[:, z_support_basis] 
            # data type
            z_spline = z_spline.astype(dtype)
            del wider_zz, zz_knots, z_support_basis
            # create spatial design matrix by tensor product of spline bases in 3 dimesions
            X = np.einsum('ab,cd,ef->acebdf', x_spline, y_spline, z_spline).reshape(
                x_spline.shape[0] * y_spline.shape[0] * z_spline.shape[0],
                x_spline.shape[1] * y_spline.shape[1] * z_spline.shape[1]
            )
            del x_spline, y_spline, z_spline
    if space_dim == "brain":
        # remove the voxels outside brain mask
        axis_dim = [xx.shape[0], yy.shape[0], zz.shape[0]]
        sub_mask = masker_voxels[np.ix_(xx, yy, zz)] > 0
        xv = xx[:, None, None]
        yv = yy[None, :, None]
        zv = zz[None, None, :]
        # Compute the linear indices for each voxel in the subvolume.
        # The formula adjusts the coordinates by subtracting the minimum and then uses the dimensions from axis_dim.
        linear_indices = ((zv - np.min(zz)) + axis_dim[2] * (yv - np.min(yy)) + axis_dim[1] * axis_dim[2] * (xv - np.min(xx)))
        # Apply the mask to select only the brain voxel indices.
        brain_voxels_index = linear_indices[sub_mask]
        del sub_mask, xv, yv, zv, linear_indices
        X = X[brain_voxels_index, :]
        del brain_voxels_index

    # remove weakly supported spline bases
    support_basis = np.sum(X, axis=0) >= 0.1
    X = X[:, support_basis]
    
    return X

def RandomFourierFeatures_3D(space_dim, dim, brain_mask=None, n_features=100, sigma=1, random_state=42):
    """
    Apply Random Fourier Features (RFF) to a 3D input tensor.

    Args:
        X (np.ndarray): Input tensor of shape (n_samples, n_features).
        n_features (int): Number of random features to generate.

    Returns:
        np.ndarray: Transformed tensor with random Fourier features.
    """
    masker_voxels = brain_mask.get_fdata()
    dim_mask = masker_voxels.shape # (91, 109, 91)
    coords = np.column_stack(np.nonzero(masker_voxels > 0))
    # optional: normalize coordinates to have roughly unit scale (helps with sigma)
    # For example, center and scale to unit variance along each dimension:
    coords = coords.astype(float)
    coords -= coords.mean(axis=0, keepdims=True)
    stds = coords.std(axis=0, keepdims=True)
    stds[stds == 0] = 1.0
    coords /= stds

    n, d = coords.shape
    # reproducibility
    rng = np.random.default_rng(random_state)
    # Random Fourier Features
    # sample W ~ N(0, 1/sigma^2 I) -> std = 1/sigma
    W = rng.normal(loc=0.0, scale=1.0 / float(sigma), size=(n_features, d))

    # sample random phases b ~ Uniform(0, 2pi)
    b = rng.uniform(low=0.0, high=2.0 * np.pi, size=(n_features,))

    # projections: (n_voxels x n_features)
    proj = coords @ W.T  # shape (n, n_features)

    # feature map: sqrt(2/D) * cos(proj + b)
    gamma_x = np.sqrt(2.0 / n_features) * np.cos(proj + b[np.newaxis, :])
    

    return gamma_x

def QMCFeatures_3D(brain_mask=None, n_features=100, length_scale=1.0, random_state=42):
    if brain_mask is None:
        raise ValueError("brain_mask must be provided (NIfTI-like object with get_fdata()).")
    if not length_scale > 0:
        raise ValueError("All elements in length_scale must be > 0.")
    if n_features <= 0:
        raise ValueError("n_features must be > 0.")
    masker_voxels = brain_mask.get_fdata()
    dim_mask = masker_voxels.shape # (91, 109, 91)
    coords = np.column_stack(np.nonzero(masker_voxels > 0))

    n_dims = coords.shape[1]
    
    # length_scale is now a scalar integer
    ls = float(length_scale)
    # Initialize Sobol sampler
    sampler = stats.qmc.Sobol(d=n_dims, scramble=True, seed=random_state)
    sample_points = sampler.random(n=n_features)
    # knots = stats.qmc.scale(sample_points, min_bounds, max_bounds)
    omega_base = stats.norm.ppf(sample_points) / ls
    # Compute projection: coords @ omega.T
    projection = coords @ omega_base.T

    # Use sin and cos components to satisfy shift-invariance
    X_fourier = np.hstack([
        np.cos(projection),
        np.sin(projection),
    ])

    # Normalize (RFF normalization)
    X_fourier *= np.sqrt(2.0 / n_features)

    return X_fourier


