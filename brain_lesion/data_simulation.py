import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.stats
import os

class simulated_data(object):
    def __init__(self, space_dim, n_group, n_subject=2000, n_voxel=1000, brain_mask=None, 
                 group_names=None, homogeneous_intensity=True, lesion_per_subject=10):
        self.space_dim = space_dim
        self.group_names = group_names if group_names is not None else [f"group_{i}" for i in range(n_group)]
        self.n_group = n_group
        self.n_subject = n_subject
        self.n_voxel = brain_mask._dataobj.shape if space_dim == "brain" else n_voxel
        self.brain_mask = brain_mask
        self.homogeneous_intensity = homogeneous_intensity
        if len(lesion_per_subject) != n_group:
            raise ValueError(f"Length of lesion per subject: {len(lesion_per_subject)} doesn't equal to number of groups: {n_group}")
        # create underlying intensity funtion 
        self.background_intensity_func = self.create_background_intensity_func(lesion_per_subject, brain_mask)
        self.covariate_intensity_func = self.create_covariate_intensity_func(lesion_per_subject, brain_mask)

    def create_background_intensity_func(self, lesion_per_subject, brain_mask, cov_scale=100):
        background_intensity_func = dict()
        if self.homogeneous_intensity:
            if self.space_dim in [1,2,3]:
                for i in range(self.n_group):
                    group_name = self.group_names[i]
                    background_intensity_func[group_name] = lesion_per_subject[i]/np.prod(self.n_voxel)*np.ones(self.n_voxel)
            elif self.space_dim == "brain":
                for i in range(self.n_group):
                    group_name = self.group_names[i]
                    background_intensity_func[group_name] = 2*lesion_per_subject[i]/np.prod(self.n_voxel)*np.ones(self.n_voxel)
        else:
            if self.space_dim == 1:
                x = np.linspace(0,self.n_voxel[0]-1, self.n_voxel[0])
                for i in range(self.n_group):
                    group_name = self.group_names[i]
                    background_intensity_func[group_name] = lesion_per_subject[i]*scipy.stats.norm.pdf(x, loc=np.mean(x), scale=cov_scale)
            elif self.space_dim == 2:
                x = np.linspace(0,self.n_voxel[0]-1, self.n_voxel[0])
                y = np.linspace(0,self.n_voxel[1]-1, self.n_voxel[1])
                X, Y = np.meshgrid(x, y, indexing='ij')
                coordinates = np.stack([X.ravel(), Y.ravel()], axis=-1)
                bump_mean = [round(np.mean(x)), round(np.mean(y))]
                bump_cov = cov_scale*np.eye(self.space_dim)
                # background intensity function
                for i in range(self.n_group):
                    group_name = self.group_names[i]
                    background_intensity_func[group_name] = lesion_per_subject[i]*(scipy.stats.multivariate_normal.cdf(coordinates+[0.5,0.5], mean=bump_mean, cov=bump_cov) \
                        - scipy.stats.multivariate_normal.cdf(coordinates+[0.5,-0.5], mean=bump_mean, cov=bump_cov) \
                        - scipy.stats.multivariate_normal.cdf(coordinates+[-0.5,0.5], mean=bump_mean, cov=bump_cov) \
                        + scipy.stats.multivariate_normal.cdf(coordinates+[-0.5,-0.5], mean=bump_mean, cov=bump_cov))
                    
            elif self.space_dim == 3:
                filename = f"probability_function/{self.space_dim}D_{self.n_group}_group_bump_background_intensity_func.npz"
                if os.path.exists(filename):
                    background_intensity_func = np.load(filename)
                    background_intensity_func = {key: background_intensity_func[key] for key in background_intensity_func.files}
                else:
                    x = np.linspace(0,self.n_voxel[0]-1, self.n_voxel[0])
                    y = np.linspace(0,self.n_voxel[1]-1, self.n_voxel[1])
                    z = np.linspace(0,self.n_voxel[2]-1, self.n_voxel[2])
                    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
                    coordinates = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
                    bump_mean = [round(np.mean(x)), round(np.mean(y)), round(np.mean(z))]
                    bump_cov = cov_scale*np.eye(self.space_dim)
                    # background intensity function
                    for i in range(self.n_group):
                        group_name = self.group_names[i]
                        background_intensity_func[group_name] = lesion_per_subject[i]*(scipy.stats.multivariate_normal.cdf(coordinates+[0.5,0.5,0.5], mean=bump_mean, cov=bump_cov) \
                            - scipy.stats.multivariate_normal.cdf(coordinates+[0.5,0.5,-0.5], mean=bump_mean, cov=bump_cov) \
                            - scipy.stats.multivariate_normal.cdf(coordinates+[0.5,-0.5,0.5], mean=bump_mean, cov=bump_cov) \
                            - scipy.stats.multivariate_normal.cdf(coordinates+[-0.5,0.5,0.5], mean=bump_mean, cov=bump_cov) \
                            - scipy.stats.multivariate_normal.cdf(coordinates+[-0.5,-0.5,-0.5], mean=bump_mean, cov=bump_cov) \
                            + scipy.stats.multivariate_normal.cdf(coordinates+[0.5,-0.5,-0.5], mean=bump_mean, cov=bump_cov) \
                            + scipy.stats.multivariate_normal.cdf(coordinates+[-0.5,0.5,-0.5], mean=bump_mean, cov=bump_cov) \
                            + scipy.stats.multivariate_normal.cdf(coordinates+[-0.5,-0.5,0.5], mean=bump_mean, cov=bump_cov))
                    np.savez(filename, **background_intensity_func)
            elif self.space_dim == "brain":
                raise NotImplementedError("Brain template not implemented")

        return background_intensity_func
    
    def create_covariate_intensity_func(self, lesion_per_subject, brain_mask, cov_scale=100):
        covariate_intensity_func = dict()
        if self.homogeneous_intensity:
            if self.space_dim == 1:
                start_index, end_index = round(0.25*self.n_voxel[0]), round(0.75*self.n_voxel[0])
                for i in range(self.n_group):
                    group_name = self.group_names[i]
                    covariate_intensity_func[group_name] = np.zeros(self.n_voxel)
                    covariate_intensity_func[group_name][start_index:end_index] = 0.5*2**self.space_dim*lesion_per_subject[i]/np.prod(self.n_voxel)
            elif self.space_dim == 2:
                start_index_0, end_index_0 = round(0.25*self.n_voxel[0]), round(0.75*self.n_voxel[0])
                start_index_1, end_index_1 = round(0.4*self.n_voxel[1]), round(0.6*self.n_voxel[1])
                for i in range(self.n_group):
                    group_name = self.group_names[i]
                    covariate_intensity_func[group_name] = np.zeros(self.n_voxel)
                    covariate_intensity_func[group_name][start_index_0:end_index_0, start_index_1:end_index_1] = 0.5*2**self.space_dim*lesion_per_subject[i]/np.prod(self.n_voxel)
            elif self.space_dim == 3:
                start_index_0, end_index_0 = round(0.25*self.n_voxel[0]), round(0.75*self.n_voxel[0])
                start_index_1, end_index_1 = round(0.4*self.n_voxel[1]), round(0.6*self.n_voxel[1])
                start_index_2, end_index_2 = round(0.25*self.n_voxel[2]), round(0.75*self.n_voxel[2])
                for i in range(self.n_group):
                    group_name = self.group_names[i]
                    covariate_intensity_func[group_name] = np.zeros(self.n_voxel)
                    covariate_intensity_func[group_name][start_index_0:end_index_0, start_index_1:end_index_1, start_index_2:end_index_2] = 0.5*2**self.space_dim*lesion_per_subject[i]/np.prod(self.n_voxel)
            elif self.space_dim == "brain":
                start_index_0, end_index_0 = round(0.25*self.n_voxel[0]), round(0.75*self.n_voxel[0])
                start_index_1, end_index_1 = round(0.4*self.n_voxel[1]), round(0.6*self.n_voxel[1])
                start_index_2, end_index_2 = round(0.25*self.n_voxel[2]), round(0.75*self.n_voxel[2])
                covariate_intensity_func = np.zeros((self.n_voxel))
                covariate_intensity_func[start_index_0:end_index_0, start_index_1:end_index_1, start_index_2:end_index_2] = 8*lesion_per_subject/np.prod(self.n_voxel)
        else:
            if self.space_dim == 1:
                x = np.linspace(0,self.n_voxel[0]-1, self.n_voxel[0])
                for i in range(self.n_group):
                    group_name = self.group_names[i]
                    covariate_intensity_func[group_name] = 0.5*lesion_per_subject[i]*scipy.stats.norm.pdf(x, loc=np.mean(x), scale=0.5*cov_scale)
            elif self.space_dim == 2:
                x = np.linspace(0,self.n_voxel[0]-1, self.n_voxel[0])
                y = np.linspace(0,self.n_voxel[1]-1, self.n_voxel[1])
                X, Y = np.meshgrid(x, y, indexing='ij')
                coordinates = np.stack([X.ravel(), Y.ravel()], axis=-1)
                bump_mean = [round(np.mean(x)), round(np.mean(y))]
                bump_cov = 0.5*cov_scale*np.eye(self.space_dim)
                for i in range(self.n_group):
                    group_name = self.group_names[i]
                    covariate_intensity_func[group_name] = 0.25*lesion_per_subject[i]*(scipy.stats.multivariate_normal.cdf(coordinates+[0.5,0.5], mean=bump_mean, cov=bump_cov) \
                        - scipy.stats.multivariate_normal.cdf(coordinates+[0.5,-0.5], mean=bump_mean, cov=bump_cov) \
                        - scipy.stats.multivariate_normal.cdf(coordinates+[-0.5,0.5], mean=bump_mean, cov=bump_cov) \
                        + scipy.stats.multivariate_normal.cdf(coordinates+[-0.5,-0.5], mean=bump_mean, cov=bump_cov))
            elif self.space_dim == 3:
                filename = f"probability_function/{self.space_dim}D_{self.n_group}_group_bump_covariate_intensity_func.npy"
                if os.path.exists(filename):
                    covariate_intensity_func = np.load(filename)
                    covariate_intensity_func = {key: background_intensity_func[key] for key in covariate_intensity_func.files}
                else:
                    x = np.linspace(0,self.n_voxel[0]-1, self.n_voxel[0])
                    y = np.linspace(0,self.n_voxel[1]-1, self.n_voxel[1])
                    z = np.linspace(0,self.n_voxel[2]-1, self.n_voxel[2])
                    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
                    coordinates = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
                    bump_mean = [round(np.mean(x)), round(np.mean(y)), round(np.mean(z))]
                    bump_cov = 0.5*cov_scale*np.eye(self.space_dim)
                    for i in range(self.n_group):
                        group_name = self.group_names[i]
                        covariate_intensity_func[group_name] = 0.25*lesion_per_subject[i]*(scipy.stats.multivariate_normal.cdf(coordinates+[0.5,0.5,0.5], mean=bump_mean, cov=bump_cov) \
                            - scipy.stats.multivariate_normal.cdf(coordinates+[0.5,0.5,-0.5], mean=bump_mean, cov=bump_cov) \
                            - scipy.stats.multivariate_normal.cdf(coordinates+[0.5,-0.5,0.5], mean=bump_mean, cov=bump_cov) \
                            - scipy.stats.multivariate_normal.cdf(coordinates+[-0.5,0.5,0.5], mean=bump_mean, cov=bump_cov) \
                            - scipy.stats.multivariate_normal.cdf(coordinates+[-0.5,-0.5,-0.5], mean=bump_mean, cov=bump_cov) \
                            + scipy.stats.multivariate_normal.cdf(coordinates+[0.5,-0.5,-0.5], mean=bump_mean, cov=bump_cov) \
                            + scipy.stats.multivariate_normal.cdf(coordinates+[-0.5,0.5,-0.5], mean=bump_mean, cov=bump_cov) \
                            + scipy.stats.multivariate_normal.cdf(coordinates+[-0.5,-0.5,0.5], mean=bump_mean, cov=bump_cov))
                    np.savez(filename, **covariate_intensity_func)
            elif self.space_dim == "brain":
                raise NotImplementedError("Brain template not implemented")
        return covariate_intensity_func

    def scale_constant(self, space_dim, n_neighbor, lesion_size_max):
        space_dim = 3 if self.space_dim == "brain" else self.space_dim
        sequence = [1 / lesion_size_max * (i+1)/3**space_dim for i in range(lesion_size_max)]
        sequence_sum = sum(sequence)
        scale_constant = sequence_sum * n_neighbor

        return scale_constant
    
    def generate_data(self, lesion_size_range):
        # covariate effect 
        Y, Z, total_intensity_func = list(), list(), list()
        for i in range(self.n_group):
            # subject index as covariate
            subject_idx = np.arange(start=0, stop=1, step=1/self.n_subject[i], dtype=np.float64)
            # subject group as covariate
            subject_group = np.repeat(i, self.n_subject[i])
            Z_i = np.stack([subject_idx, subject_group], axis=-1) # shape: (n_subject, 2)
            Z.append(Z_i)
        Z = np.concatenate(Z, axis=0)
        # subject indices in different groups based on categorical covariate
        group_subjects = {f"group_{int(key)}": list(np.where(Z[:, 1] == key)[0]) for key in np.unique(Z[:, 1])}
        # other spatial effect shared by subjects
        seed = 0
        for k in range(self.n_group):
            group_name = self.group_names[k]
            subject_indices = group_subjects[group_name]
            for s in subject_indices:
                np.random.seed(seed)
                # combine background intensity and spatially varying covariate intensity
                sum_intensity_func_s = self.background_intensity_func[group_name] + Z[s][0]*self.covariate_intensity_func[group_name]
                sum_intensity_func_s = np.clip(sum_intensity_func_s, a_min=0, a_max=1)
                # generate brain lesion for each subject
                sum_intensity_func_s = sum_intensity_func_s.reshape((self.n_voxel))
                lesion_centre = np.random.binomial(n=1, p=sum_intensity_func_s, size=self.n_voxel)
                n_lesion_centre = np.sum(lesion_centre)
                if self.space_dim == 1:
                    lesion_centriod = np.where(lesion_centre==1)[0]
                elif self.space_dim == 2:
                    lesion_centriod_x, lesion_centriod_y = np.where(lesion_centre==1)
                elif self.space_dim in [3, "brain"]:
                    lesion_index_x, lesion_index_y, lesion_index_z = np.where(lesion_centre==1)
                Y_s = np.zeros(self.n_voxel).astype(np.int32)
                # Randomly generate lesion at neighbouring voxels based on lesion size
                lesion_size = np.random.randint(lesion_size_range[0],lesion_size_range[1]+1,size=n_lesion_centre)
                if self.space_dim == 1:
                    offset = [-1, 0, 1]
                    n_neighbor = len(offset)
                    for i in range(n_lesion_centre):
                        sampled_offsets = np.random.choice(n_neighbor, size=lesion_size[i], replace=False)
                        sampled_neighbors = [lesion_centriod[i] + offset[j] for j in sampled_offsets]
                        for neighbor in sampled_neighbors:
                            if 0 <= neighbor < self.n_voxel:
                                Y_s[neighbor] += 1
                elif self.space_dim == 2:
                    # Define all possible neighbor offsets (3x3 square, excluding center)
                    offsets = [(dx, dy)
                                for dx in [-1, 0, 1]
                                for dy in [-1, 0, 1]]
                    n_neighbor = len(offsets)
                    for i in range(n_lesion_centre):
                        sampled_offsets = np.random.choice(n_neighbor, size=lesion_size[i], replace=False)
                        sampled_neighbors = [tuple(np.add([lesion_centriod_x[i], lesion_centriod_y[i]], offsets[j])) for j in sampled_offsets]
                        for neighbor in sampled_neighbors:
                            if 0 <= neighbor[0] < self.n_voxel[0] and 0 <= neighbor[1] < self.n_voxel[1]:
                                Y_s[neighbor] += 1
                elif self.space_dim in [3, "brain"]:
                    # Define all possible neighbor offsets (3x3x3 cube, excluding center)
                    offsets = [(dx, dy, dz)
                                for dx in [-1, 0, 1]
                                for dy in [-1, 0, 1]
                                for dz in [-1, 0, 1]]
                    n_neighbor = len(offsets)
                    for i in range(n_lesion_centre):
                        sampled_offsets = np.random.choice(n_neighbor, size=lesion_size[i], replace=False)
                        sampled_neighbors = [tuple(np.add([lesion_index_x[i], lesion_index_y[i], lesion_index_z[i]], offsets[j])) for j in sampled_offsets]
                        for neighbor in sampled_neighbors:
                            if 0 <= neighbor[0] < self.n_voxel[0] and 0 <= neighbor[1] < self.n_voxel[1] and 0 <= neighbor[2] < self.n_voxel[2]:
                                Y_s[neighbor] += 1
                Y_s = np.clip(Y_s, a_min=0, a_max=1)
                # Scale intensity function by corresponding constant
                C = self.scale_constant(self.space_dim, n_neighbor, lesion_size_range[1])
                total_intensity_func_s = C*sum_intensity_func_s
                # Reshape data
                Y_s = Y_s.reshape((1, -1)) if self.space_dim != "brain" else Y_s[None, ...]
                total_intensity_func_s = total_intensity_func_s.reshape((1, -1)) if self.space_dim != "brain" else total_intensity_func_s[None, ...]
                # Store data
                Y.append(Y_s)
                total_intensity_func.append(total_intensity_func_s)
                seed += 1
                # # 1D Visualisation
                # plt.figure(figsize=(100, 2))
                # # plt.step(range(self.n_voxel), Y_s, where="mid")
                # plt.step(range(self.n_voxel), 0.5*lesion_size_range[1]*sum_intensity_func_s, where="mid")
                # plt.xlabel("Voxel location")
                # plt.ylabel("Brain lesion")
                # plt.title("1D simulation of brain lesion on 1000 voxels")
                # plt.savefig("test.png")
                # # 2D Visualisation
                # plt.figure(figsize=(8, 8))
                # plt.imshow(total_intensity_func_s, cmap='viridis', aspect='equal')
                # # plt.imshow(Y_s, cmap='viridis', aspect='equal')
                # plt.colorbar(label='Intensity')  # Add a color bar with a label
                # plt.xlabel('X-axis')
                # plt.ylabel('Y-axis')
                # plt.savefig("test.png")
                # 3D Visualisation
                # plt.figure(figsize=(8, 8))
                # plt.imshow(total_intensity_func_s[:,:,10], cmap='viridis', aspect='equal')
                # # plt.imshow(total_intensity_func_s[:,:,10].reshape((self.n_voxel[0], self.n_voxel[1])), cmap='viridis', aspect='equal')
                # # plt.imshow(Y_s.reshape(self.n_voxel)[:,:,10], cmap='viridis', aspect='equal')
                # plt.colorbar(label='Intensity')  # Add a color bar with a label
                # plt.xlabel('X-axis')
                # plt.ylabel('Y-axis')
                # plt.savefig("test.png")
        Y = np.concatenate(Y, axis=0)
        total_intensity_func = np.concatenate(total_intensity_func, axis=0)
        # if self.space_dim == "brain": 
        #     Y = Y[:,self.brain_mask._dataobj>0]
        #     total_intensity_func = total_intensity_func[:,self.brain_mask._dataobj>0]
        
        return group_subjects, total_intensity_func, Y, Z
            

        

