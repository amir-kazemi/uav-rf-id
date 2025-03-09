import numpy as np
from collections import Counter
from sklearn.utils import shuffle
import itertools
from tqdm import tqdm
import torch
import torch.nn as nn


from .utils import Normalizer,set_seeds
from .cvae import CVAE
from .cgan import CGAN

set_seeds(seed_value=0)
#### vicinal packages
#from .gpdm import PatchSWDLoss, ImageGenerator, ImageProcessor
from . import gpdm, gpnn



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GenerativeModel:
    def generate_samples(self, X, Y, num_samples):
        raise NotImplementedError

class Conditional(nn.Module):
    def __init__(self, model_type, n_features, n_classes, latent_dim, **kwargs):
        super(Conditional, self).__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.latent_dim = latent_dim

        if model_type == 'CVAE':
            self.model = CVAE(n_features=n_features, n_classes=n_classes, latent_dim=latent_dim, **kwargs)
        elif model_type == 'CGAN':
            self.model = CGAN(n_features=n_features, n_classes=n_classes, latent_dim=latent_dim, **kwargs)
        else:
            raise ValueError(f"Invalid model type: {model_type}")

    def forward(self, x, y):
        return self.model(x, y)

    def generate(self, num_samples):
        # Calculate the base number of samples per class
        base_samples_per_label = num_samples // self.n_classes

        # Calculate the remainder
        remainder = num_samples % self.n_classes

        # Distribute the base number and the remainder to get the number of samples per label
        num_samples_per_label = [base_samples_per_label + (1 if i < remainder else 0) for i in range(self.n_classes)]

        generated_samples = []
        generated_labels = []

        for label, num_samples_for_label in enumerate(num_samples_per_label):
            # Create label tensor
            label_tensor = torch.zeros((num_samples_for_label, self.n_classes))
            label_tensor[:, label] = 1

            # Generate samples
            samples_for_label = self.model.eval_model(num_samples_for_label, label_tensor).detach().cpu()
            generated_samples.append(samples_for_label.detach())
            generated_labels.extend([label] * num_samples_for_label)

        return torch.cat(generated_samples, dim=0).numpy(), np.array(generated_labels)
    
class Vicinal(GenerativeModel):
    def __init__(self, patch_dim, min_height, algorithm):
        super(Vicinal, self).__init__()
        self.patch_dim = patch_dim
        self.min_height = min_height
        self.algorithm = algorithm
    
    def generate(self, X, Y, num_samples):
        pbar = tqdm(total=num_samples, desc="Generating synthetic samples")
        counter = 0
        # Calculate label proportions in given dataset
        label_counts = Counter(Y)
        label_proportions = {label: count / len(Y) for label, count in label_counts.items()}

        # Create lists to hold the synthetic samples and their labels
        synthetic_samples = []
        synthetic_labels = []

        # Generate the rounded down number of samples
        for label, proportion in label_proportions.items():
            # Determine how many synthetic samples to generate for this label
            num_synthetic_samples = int(np.floor(proportion * num_samples))
            # Select samples of the current label
            samples_with_label = X[Y == label]

            circular_iterator = itertools.cycle(list(range(len(samples_with_label))))

            for _ in range(num_synthetic_samples):
                sample = samples_with_label[next(circular_iterator)]
                # Generate a vicinal synthetic sample and add it to the list
                synthetic_sample = self.generate_vicinal_sample(sample)
                

                synthetic_samples.append(synthetic_sample)
                synthetic_labels.append(label)
                
                counter += 1
                if counter % 100 == 0:  # Update the progress bar every 10 iterations
                    pbar.update(100)
                    counter = 0

        # Distribute remaining samples based on the label proportions
        remaining_samples = num_samples - len(synthetic_samples)
        remaining_labels = np.random.choice(list(label_proportions.keys()), remaining_samples, p=list(label_proportions.values()))
        
        for label in remaining_labels:
            # Select samples of the current label
            samples_with_label = X[Y == label]

            circular_iterator = itertools.cycle(list(range(len(samples_with_label))))
            sample = samples_with_label[next(circular_iterator)]
            
            # Generate a vicinal synthetic sample and add it to the list
            synthetic_sample = self.generate_vicinal_sample(sample)

            synthetic_samples.append(synthetic_sample)
            synthetic_labels.append(label)
            
            counter += 1
            if counter % 100 == 0:  # Update the progress bar every 10 iterations
                pbar.update(100)
                counter = 0

        # Convert lists to numpy arrays for consistency with input format
        synthetic_samples = np.array(synthetic_samples)
        synthetic_labels = np.array(synthetic_labels)

        pbar.close()
        return synthetic_samples, synthetic_labels

    def generate_vicinal_sample(self, sample):
        # Implement the method for generating a single vicinal sample
        x = sample[:2025].reshape(45,45)
        normalizer = Normalizer(np.min(x), np.max(x),symmetry=True)
        x_normalized = normalizer.normalize(x)
        x_tensor = torch.from_numpy(x_normalized).float().unsqueeze(0).unsqueeze(0)
        n_images = 1
        reference_images = x_tensor.repeat(n_images, 1, 1, 1)
        if self.algorithm == 'gpdm':
            criteria = gpdm.PatchSWDLoss(patch_size=self.patch_dim, stride=1, num_proj=256)
            pyramid_scales = gpdm.ImageProcessor.get_pyramid_scales(reference_images.shape[-2], self.min_height, .85)
            generator = gpdm.ImageGenerator()
            new_images = generator.generate(reference_images=reference_images,
                                  criteria=criteria,
                                  pyramid_scales=pyramid_scales,
                                  aspect_ratio=(1, 1),
                                  init_from="target",
                                  additive_noise_sigma=0,
                                  lr=0.01,
                                  num_steps=500,
                                  debug_dir=None)
            
        elif self.algorithm == 'gpnn':
            nn_module = gpnn.get_nn_module('Exact', None, True)
            pyramid_scales = gpnn.get_pyramid_scales(reference_images.shape[-2], self.min_height, 0.85)
            generator = gpnn.ImageGenerator()
            new_images = generator.generate(reference_images,
                                    nn_module,
                                    patch_size=self.patch_dim,
                                    pyramid_scales=pyramid_scales,
                                    aspect_ratio=(1., 1.),
                                    init_from='target',
                                    num_iters=10,
                                    initial_level_num_iters=1,
                                    keys_blur_factor=1,
                                    additive_noise_sigma=0,
            )
            
            
        
        new_images_np = new_images.detach().cpu().numpy()
        new_images_denorm = normalizer.denormalize(new_images_np)
        new_images_denorm = new_images_denorm.reshape(45*45,)
        return new_images_denorm


