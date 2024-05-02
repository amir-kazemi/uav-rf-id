import os
from collections import defaultdict
import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import wasserstein_distance
from tqdm import tqdm
from skimage.util import view_as_windows
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
import random

def set_seeds(seed_value=0):
    """
    Set seeds for reproducibility.
    :param seed_value: The seed value to use for random number generation.
    """
    # Set the seed for NumPy
    np.random.seed(seed_value)

    # Set the seed for PyTorch
    torch.manual_seed(seed_value)
    
     # Set the seed for Python
    random.seed(seed_value)

    # If using a GPU (CUDA), set the seed for all GPUs
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

        # Ensuring further reproducibility when using GPUs
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
def get_sorted_paths_dict(data_directory  = './data/', substring_to_replace = "train_subsampled"):
    # Create a defaultdict to store the paths, where new keys will be automatically initialized with an empty list
    paths_dict = defaultdict(list)

    # List all files in the data directory and its subdirectories
    for root, _, filenames in os.walk(data_directory):
        for filename in filenames:
            if filename.endswith(f'X_{substring_to_replace}.npy'):
                # Extract the C and P values from the folder name
                folder_name = os.path.basename(root)
                parts = folder_name.split('C')[1].split('P')
                C_value = int(parts[0])
                P_value = int(parts[1].split('K')[0]) if len(parts) > 1 else 0  # Set P_value to 0 if there's no 'P' part

                # Get the full file paths to X_{substring_to_replace}.npy and Y_{substring_to_replace}.npy
                x_file_path = os.path.join(root, filename)
                y_file_path = os.path.join(root, filename.replace(f'X_{substring_to_replace}.npy', f'Y_{substring_to_replace}.npy'))

                # Append the tuple of file paths (X_{substring_to_replace}, Y_{substring_to_replace}) to the list under the corresponding key
                paths_dict[(C_value, P_value)].append((x_file_path, y_file_path))

    # Sort the lists of file paths in ascending order for each key in the dictionary
    for key in paths_dict:
        paths_dict[key].sort()

    # Sort the paths_dict based on the keys (C, P)
    sorted_items = sorted(paths_dict.items(), key=lambda item: item[0])  # item[0] is the key (C, P)
    sorted_paths_dict = dict(sorted_items)

    return sorted_paths_dict

class OutlierRemover:
    def __init__(self, n_mad=3):
        """
        Initialize the outlier remover.
        :param n_mad: Number of MADs to use as the threshold for defining outliers.
        """
        self.n_mad = n_mad

    def remove_outliers(self, X, Y):
        """
        Remove outliers from X based on the norms of samples using Median and MAD.
        :param X: Feature matrix (sample_size, feature_dim).
        :param Y: Labels (sample_size,).
        :return: Cleaned X and Y without outliers.
        """
        if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
            raise ValueError("X and Y must be numpy arrays.")

        clean_X, clean_Y = [], []
        for label in np.unique(Y):
            indices = np.where(Y == label)[0]
            label_data = X[indices]

            # Compute norms
            norms = np.linalg.norm(label_data, axis=1)

            # Calculate median and MAD
            median_norm = np.median(norms)
            mad = np.median(np.abs(norms - median_norm))

            # Identify outliers
            outliers = np.abs(norms - median_norm) > self.n_mad * mad
            
            # Append non-outlier data to the clean lists
            clean_X.append(label_data[~outliers])
            clean_Y.append(Y[indices][~outliers])

        return np.vstack(clean_X), np.concatenate(clean_Y)

def generate_mixup_samples(X, Y, target_size=1000, alpha=0.2):
    """
    Generates synthetic mixup samples to reach a specified target size.
    
    Parameters:
    - X: Original data samples (Tensor).
    - Y: Original labels (Tensor).
    - target_size: The desired number of synthetic samples.
    - alpha: The alpha parameter for the Beta distribution.
    
    Returns:
    - Synthetic data and labels tensors.
    """
    current_size = X.size(0)
    
    # Calculate the number of batches needed
    batch_size = current_size  # You can adjust this based on your memory capacity
    num_batches = target_size // batch_size + (1 if target_size % batch_size != 0 else 0)
    
    synthetic_X_batches = []
    synthetic_Y_batches = []
    
    for _ in range(num_batches):
        # Generate random indices for mixing
        index = torch.randperm(current_size)
        
        # Generate lambda for each sample in the batch
        lam = np.random.beta(alpha, alpha, size=(batch_size, 1)).astype(np.float32)
        lam = torch.from_numpy(lam)
        
        # Ensure lam and 1-lam are broadcastable to the shape of X and Y
        lam_X = lam.view(-1, 1).to(X.device)  # Adjust shape for X
        lam_Y = lam.to(Y.device)  # Adjust shape for Y if necessary
        
        # Mixing
        mixed_X = lam_X * X + (1 - lam_X) * X[index, :]
        mixed_Y = lam_Y * Y + (1 - lam_Y) * Y[index, :]
        
        synthetic_X_batches.append(mixed_X)
        synthetic_Y_batches.append(mixed_Y)
    
    # Concatenate all batches
    synthetic_X = torch.cat(synthetic_X_batches, dim=0)[:target_size]
    synthetic_Y = torch.cat(synthetic_Y_batches, dim=0)[:target_size]
    
    return synthetic_X, synthetic_Y


        


def stratified_split_loader(data_loader, validation_split):
    """
    Splits a DataLoader into training and validation sets with stratified sampling for one-hot encoded labels.

    Args:
    data_loader (DataLoader): The DataLoader to split.
    validation_split (float): The fraction of the dataset to use for validation.

    Returns:
    DataLoader: DataLoader for the training set.
    DataLoader: DataLoader for the validation set.
    """
    # Extract class for stratification from one-hot encoded labels
    stratify_classes = [np.argmax(label.numpy(), axis=0) for _, label in data_loader.dataset]

    # Split the indices in a stratified way
    train_idx, val_idx = train_test_split(
        range(len(data_loader.dataset)),
        test_size=validation_split,
        stratify=stratify_classes,
        random_state=0  # Ensures reproducibility, you can change or remove this
    )

    # Creating PyTorch subsets for training and validation
    train_dataset = Subset(data_loader.dataset, train_idx)
    val_dataset = Subset(data_loader.dataset, val_idx)

    # Creating data loaders for the subsets
    train_loader = DataLoader(train_dataset, batch_size=data_loader.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=data_loader.batch_size)  # Adjust batch size if needed

    return train_loader, val_loader

class Normalizer:
    def __init__(self, min_value, max_value, symmetry=False):
        self.symmetry = symmetry
        self.min_value = min_value
        self.max_value = max_value

    def normalize(self, x):
        if self.symmetry:
            x = 2 * (x - self.min_value) / (self.max_value - self.min_value) - 1
        else:
            x = (x - self.min_value) / (self.max_value - self.min_value)
            
        np.nan_to_num(x, copy=False, nan=0.5)
        return x

    def denormalize(self, x):
        if self.symmetry:
            x = 0.5 * (x + 1) * (self.max_value - self.min_value) + self.min_value
        else:
            x = x * (self.max_value - self.min_value) + self.min_value
        return x
    

class ClasswiseNormalizer:

    def __init__(self, symmetry=False):
        self.symmetry = symmetry
        self.classwise_normalizers = []

    def normalize(self, X, y):
        """
        Normalize the data X class-wise based on the labels y.
        """
        
        unique_classes = np.unique(y)
        X = X.astype(float)
        # Result placeholder for normalized values
        X_normalized = np.zeros_like(X)
        
        # Clear existing normalizers
        self.classwise_normalizers = []
        
        # Iterate over each unique class
        for u_class in unique_classes:
            # Get the indices of rows that belong to the current class
            class_indices = np.where(y == u_class)[0]
            
            # Calculate min and max for all features of the current class at once
            min_values, max_values = np.min(X[class_indices, :], axis=0), np.max(X[class_indices, :], axis=0)

            # Generate instance for normalizer
            normalizer = Normalizer(min_values, max_values, symmetry=self.symmetry)  # Assuming Normalizer is in the same scope
            
            self.classwise_normalizers.append(normalizer)
            
            # Normalize the values
            X_normalized[class_indices, :] = normalizer.normalize(X[class_indices, :])

        return X_normalized

    def denormalize(self, X_normalized, y):
        """
        Denormalize the data X_normalized class-wise based on the labels y.
        """
        unique_classes = np.unique(y)
        
        # Result placeholder for denormalized values
        X_denormalized = np.zeros_like(X_normalized)
        
        # Iterate over each unique class
        for idx, u_class in enumerate(unique_classes):
            # Get the indices of rows that belong to the current class
            class_indices = np.where(y == u_class)[0]
            
            # Use the appropriate normalizer for the current class
            normalizer = self.classwise_normalizers[idx]
            
            # Denormalize the values
            X_denormalized[class_indices, :] = normalizer.denormalize(X_normalized[class_indices, :])

        return X_denormalized
    
    



def np2tensor_onehot(X_data, y_data, num_classes):
    """
    Converts NumPy arrays to PyTorch tensors and performs one-hot encoding.

    Args:
        X_data (numpy.ndarray): Input data.
        y_data (numpy.ndarray): Labels to be one-hot encoded.
        num_classes (int): Total number of classes.

    Returns:
        torch.Tensor: One-hot encoded labels.
    """
    X_tensor = torch.tensor(X_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_data, dtype=torch.long)
    y_tensor = F.one_hot(y_tensor, num_classes=num_classes)
    
    return X_tensor,y_tensor

class PatchUtils:
    def __init__(self):
        pass
    
    def __extract_patches(self, image, patch_size):
        # Using view_as_windows from skimage to extract patches
        patches = view_as_windows(image, (patch_size, patch_size))
        return patches.reshape(-1, patch_size, patch_size)

    def __extract_patches_datasets(self, X_r_2d, X_g_2d, patch_dim):
        num_samples = X_r_2d.shape[0]
        p_r_list = []
        p_g_list = []
        for sample_idx in range(num_samples):
            x_r_2d = X_r_2d[sample_idx]
            x_g_2d = X_g_2d[sample_idx]
            p_r_2d = self.__extract_patches(x_r_2d, patch_dim)
            p_g_2d = self.__extract_patches(x_g_2d, patch_dim)
            p_r_list.append(p_r_2d.reshape(p_r_2d.shape[0], -1))
            p_g_list.append(p_g_2d.reshape(p_g_2d.shape[0], -1))

        P_r = np.stack(p_r_list, axis=0)
        P_g = np.stack(p_g_list, axis=0)
        return P_r, P_g

    def sliced_wasserstein_distance(self, x, y, num_projections=128):
        """
        Calculates the Sliced Wasserstein Distance between two datasets x and y
        using random projections onto one-dimensional subspaces.

        Args:
            x: numpy array with shape (n_samples, n_features) containing the first dataset
            y: numpy array with shape (n_samples, n_features) containing the second dataset
            num_projections: number of random projections to use (default=100)

        Returns:
            sliced_wd: the Sliced Wasserstein Distance between the two datasets
        """
        set_seeds(seed_value=0)
        n_samples, n_features = x.shape
        projections = np.random.normal(size=(num_projections, n_features))
        sliced_distances = []
        for i in range(num_projections):
            x_proj = np.dot(x, projections[i])
            y_proj = np.dot(y, projections[i])
            sliced_distances.append(wasserstein_distance(x_proj, y_proj))
        sliced_wd = np.sqrt(np.mean(sliced_distances)**2)
        return sliced_wd

    def expected_patch_distribution_distance(self, X_r_2d, X_g_2d, patch_dim):
        P_r, P_g = self.__extract_patches_datasets(X_r_2d, X_g_2d, patch_dim)

        distances = np.zeros((P_r.shape[1]))
        for L_index in range(P_r.shape[1]):
            distances[L_index]= self.sliced_wasserstein_distance(P_r[:,L_index,:], P_g[:,L_index,:])
        d = np.mean(distances)
        return d
    