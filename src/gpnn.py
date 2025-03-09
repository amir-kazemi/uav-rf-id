"""
This code is based on the implementation from:
https://github.com/ariel415el/Efficient-GPNN

For more details, visit the GitHub page above.

Efficient-GPNN is an optimized implementation of the method described in:
"Drop the GAN: In Defense of Patches Nearest Neighbors as Single Image Generative Models"
https://arxiv.org/abs/2103.15545

This implementation was developed for comparison with GPDM, introduced in:
"Generating Natural Images with Direct Patch Distributions Matching"
https://arxiv.org/abs/2203.11862

If you use this code, please cite the above references.
"""
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
from os.path import join, basename, dirname, abspath
import sys
import numpy as np
import cv2
import torch
from torchvision.transforms import transforms
from torchvision.transforms import Resize as tv_resize
from torchvision.utils import save_image
import torch.nn.functional as F
import faiss





## image tools

def load_image(path):
    return cv2pt(cv2.imread(path))


def dump_images(images, out_dir):
    if os.path.exists(out_dir):
        i = len(os.listdir(out_dir))
    else:
        i = 0
        os.makedirs(out_dir)
    for j in range(images.shape[0]):
        save_image(images[j], os.path.join(out_dir, f"{i}.png"), normalize=True)
        i += 1


def get_pyramid_scales(max_height, min_height, step):
    cur_scale = max_height
    scales = [cur_scale]
    while cur_scale > min_height:
        if type(step) == float:
            cur_scale = int(cur_scale * step)
        else:
            cur_scale -= step
        scales.append(cur_scale)

    return scales[::-1]


def cv2pt(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float64) / 255.
    img = img * 2 - 1
    img = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)

    return img


def downscale(img, pyr_factor):
    assert 0 < pyr_factor < 1
    new_w = int(pyr_factor * img.shape[-1])
    new_h = int(pyr_factor * img.shape[-2])
    return transforms.Resize((new_h, new_w), antialias=True)(img)


def blur(img, pyr_factor):
    """Blur image by downscaling and then upscaling it back to original size"""
    if pyr_factor < 1:
        d_img = downscale(img, pyr_factor)
        img = transforms.Resize(img.shape[-2:], antialias=True)(d_img)
    return img

def extract_patches(src_img, patch_size, stride):
    """
    Splits the image to overlapping patches and returns a pytorch tensor of size (N_patches, 3*patch_size**2)
    """
    channels = src_img.shape[1]
    patches = F.unfold(src_img, kernel_size=patch_size, stride=stride) # shape (b, 3*p*p, N_patches)
    patches = patches.squeeze(dim=0).permute((1, 0)).reshape(-1, channels * patch_size**2)
    return patches


def combine_patches(patches, patch_size, stride, img_shape):
    """
    Combines patches into an image by averaging overlapping pixels
    :param patches: patches to be combined. pytorch tensor of shape (N_patches, 3*patch_size**2)
    :param img_shape: an image of a shape that if split into patches with the given stride and patch_size will give
                      the same number of patches N_patches
    returns an image of shape img_shape
    """
    patches = patches.permute(1,0).unsqueeze(0)
    combined = F.fold(patches, output_size=img_shape[-2:], kernel_size=patch_size, stride=stride)

    # normal fold matrix
    input_ones = torch.ones(img_shape, dtype=patches.dtype, device=patches.device)
    divisor = F.unfold(input_ones, kernel_size=patch_size, dilation=(1, 1), stride=stride, padding=(0, 0))
    divisor = F.fold(divisor, output_size=img_shape[-2:], kernel_size=patch_size, stride=stride)

    divisor[divisor == 0] = 1.0
    return (combined / divisor).squeeze(dim=0).unsqueeze(0)

## NN tools

def get_NN_indices(X, Y, alpha, b=128):
    """
    Get the nearest neighbor index from Y for each X. Use batches to save memory
    :param X:  (n1, d) tensor
    :param Y:  (n2, d) tensor
    Returns a n2 n1 of indices
    """
    dist = compute_distances_batch(X, Y, b=b)
    # dist = torch.cdist(X.view(len(X), -1), Y.view(len(Y), -1)) # Not enough memory
    dist = (dist / (torch.min(dist, dim=0)[0] + alpha)) # compute_normalized_scores
    NNs = torch.argmin(dist, dim=1)  # find_NNs
    return NNs

def compute_distances_batch(X, Y, b):
    """
    Computes distance matrix in batches of rows to reduce memory consumption from (n1 * n2 * d) to (d * n2 * d)
    :param X:  (n1, d) tensor
    :param Y:  (n2, d) tensor
    :param b: rows batch size
    Returns a (n2, n1) matrix of L2 distances
    """
    """"""
    b = min(b, len(X))
    dist_mat = torch.zeros((X.shape[0], Y.shape[0]), dtype=torch.float16, device=X.device)
    n_batches = len(X) // b
    for i in range(n_batches):
        dist_mat[i * b:(i + 1) * b] = efficient_compute_distances(X[i * b:(i + 1) * b], Y)
    if len(X) % b != 0:
        dist_mat[n_batches * b:] = efficient_compute_distances(X[n_batches * b:], Y)

    return dist_mat

def efficient_compute_distances(X, Y):
    """
    Pytorch efficient way of computing distances between all vectors in X and Y, i.e (X[:, None] - Y[None, :])**2
    Get the nearest neighbor index from Y for each X
    :param X:  (n1, d) tensor
    :param Y:  (n2, d) tensor
    Returns a n2 n1 of indices
    """
    dist = (X * X).sum(1)[:, None] + (Y * Y).sum(1)[None, :] - 2.0 * torch.mm(X, torch.transpose(Y, 0, 1))
    d = X.shape[1]
    dist /= d # normalize by size of vector to make dists independent of the size of d ( use same alpha for all patche-sizes)
    return dist


def get_NN_indices_low_memory(X, Y, alpha, b):
    """
    Get the nearest neighbor index from Y for each X.
    Avoids holding a (n1 * n2) amtrix in order to reducing memory footprint to (b * max(n1,n2)).
    :param X:  (n1, d) tensor
    :param Y:  (n2, d) tensor
    Returns a n2 n1 of indices
    """
    if alpha is not None:
        normalizing_row = get_col_mins_efficient(X, Y, b=b)
        normalizing_row = alpha + normalizing_row[None, :]
    else:
        normalizing_row = 1

    NNs = torch.zeros(X.shape[0], dtype=torch.long, device=X.device)
    n_batches = len(X) // b
    for i in range(n_batches):
        dists = efficient_compute_distances(X[i * b:(i + 1) * b], Y) / normalizing_row
        NNs[i * b:(i + 1) * b] = dists.min(1)[1]
    if len(X) % b != 0:
        dists = efficient_compute_distances(X[n_batches * b:], Y) / normalizing_row
        NNs[n_batches * b:] = dists.min(1)[1]
    return NNs


def get_col_mins_efficient(X, Y, b):
    """
    Computes the l2 distance to the closest x or each y.
    :param X:  (n1, d) tensor
    :param Y:  (n2, d) tensor
    Returns n1 long array of L2 distances
    """
    mins = torch.zeros(Y.shape[0], dtype=X.dtype, device=X.device)
    n_batches = len(Y) // b
    for i in range(n_batches):
        mins[i * b:(i + 1) * b] = efficient_compute_distances(X, Y[i * b:(i + 1) * b]).min(0)[0]
    if len(Y) % b != 0:
        mins[n_batches * b:] = efficient_compute_distances(X, Y[n_batches * b:]).min(0)[0]

    return mins

## NN Modules

class FaissNNModule:
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu
        self.index = None

    def _get_index(self, n, d):
        raise NotImplemented

    def init_index(self, index_vectors):
        self.index_vectors = np.ascontiguousarray(index_vectors.cpu().numpy(), dtype='float32')
        self.index = self._get_index(*self.index_vectors.shape)

        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        if not self.index.is_trained:
            self.index.train(self.index_vectors)

        self.index.add(self.index_vectors)

    def search(self, queries):
        assert self.index is not None
        queries_np = np.ascontiguousarray(queries.cpu().numpy(), dtype='float32')
        _, I = self.index.search(queries_np, 1)  # actual search

        NNs = torch.from_numpy(I[:, 0])

        return NNs

class FaissFlat(FaissNNModule):
    def __str__(self):
        return "FaissFlat(" + ("GPU" if self.use_gpu else "CPU") + ")"
        
    def _get_index(self, n, d):
        return faiss.IndexFlatL2(d)


class FaissIVF(FaissNNModule):
    def __str__(self):
        return "FaisIVF(" + ("GPU" if self.use_gpu else "CPU") + ")"
        
    def _get_index(self, n, d):
        return faiss.IndexIVFFlat(faiss.IndexFlat(d), d, int(np.sqrt(n)))


class FaissIVFPQ(FaissNNModule):
    def __str__(self):
        return "FaisIVFPQ(" + ("GPU" if self.use_gpu else "CPU") + ")"
        
    def _get_index(self, n, d):
        return faiss.IndexIVFPQ(faiss.IndexFlatL2(d), d, int(np.sqrt(n)), 8, 8)


class PytorchNN:
    def __init__(self, batch_size=256, alpha=None, use_gpu=False):
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.alpha = alpha
        self.device = torch.device("cuda:0" if use_gpu else 'cpu')

    def init_index(self, index_vectors):
        self.index_vectors = index_vectors.to(self.device)

    def search(self, queries):
        return get_NN_indices(queries.to(self.device), self.index_vectors, self.alpha, self.batch_size).cpu()

    def __str__(self):
        return "PytorchNN(" + ("GPU" if self.use_gpu else "CPU") + f",alpha={self.alpha})"

class PytorchNNLowMemory:
    def __init__(self, batch_size=256, alpha=None, use_gpu=False):
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.alpha = alpha
        self.device = torch.device("cuda:0" if use_gpu else 'cpu')

    def init_index(self, index_vectors):
        self.index_vectors = index_vectors.to(self.device)
        
    def search(self, queries):
        return get_NN_indices_low_memory(queries.to(self.device), self.index_vectors, self.alpha, self.batch_size).cpu()

    def __str__(self):
        return "PytorchNNLowMem(" + ("GPU" if self.use_gpu else "CPU") + f",alpha={self.alpha})"


def get_nn_module(NN_type, alpha, use_gpu):
    if NN_type == "Exact":
        nn_module = PytorchNNLowMemory(alpha=alpha, use_gpu=use_gpu)
    elif NN_type == "Exact-low-memory":
        nn_module = PytorchNNLowMemory(alpha=alpha, use_gpu=use_gpu)
    else:
        if alpha is not None:
            raise ValueError("Can't use an alpha parameter with approximate nearest neighbor")
        nn_module = FaissIVF(use_gpu=use_gpu)

    return nn_module

## GPNN

class ImageGenerator:
    def __init__(self, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        self.device = device
        
    def generate(self,
                 reference_images,
                 nn_module,
                 patch_size=7,
                 stride=1,
                 init_from: str = 'zeros',
                 pyramid_scales=(32, 64, 128, 256),
                 aspect_ratio=(1, 1),
                 additive_noise_sigma=0.0,
                 num_iters: int = 10,
                 initial_level_num_iters: int = 1,
                 keys_blur_factor=1):
        """
        Run the GPNN model to generate an image using coarse to fine patch replacements.
        """
        reference_images = reference_images.to(self.device)
        synthesized_images = self.get_first_initial_guess(reference_images, init_from, additive_noise_sigma).to(self.device)
        original_image_shape = synthesized_images.shape[-2:]

        for i, scale in enumerate(pyramid_scales):
            lvl_references = tv_resize(scale, antialias=True)(reference_images)
            lvl_output_shape = self.get_output_shape(original_image_shape, scale, aspect_ratio)
            synthesized_images = tv_resize(lvl_output_shape, antialias=True)(synthesized_images)

            synthesized_images = self.replace_patches(synthesized_images, lvl_references, nn_module,
                                                 patch_size,
                                                 stride,
                                                 initial_level_num_iters if i == 0 else num_iters,
                                                 keys_blur_factor=keys_blur_factor)


        return synthesized_images


    def replace_patches(self,queries_image, values_image, nn_module, patch_size, stride, num_iters, keys_blur_factor=1):
        """
        Repeats n_steps iterations of repalcing the patches in "queries_image" by thier nearest neighbors from "values_image".
        The NN matrix is calculated with "keys" wich are a possibly blurred version of the patches from "values_image"
        :param values_image: The target patches to extract possible pathces or replacement
        :param queries_image: The synthesized image who's patches are to be replaced
        :param num_iters: number of repeated replacements for each patch
        :param keys_blur_factor: the factor with which to blur the values to get keys (image is downscaled and then upscaled with this factor)
        """
        keys_image = blur(values_image, keys_blur_factor)
        keys = extract_patches(keys_image, patch_size, stride)

        nn_module.init_index(keys)

        values = extract_patches(values_image, patch_size, stride)
        for i in range(num_iters):
            queries = extract_patches(queries_image, patch_size, stride)

            NNs = nn_module.search(queries)

            queries_image = combine_patches(values[NNs], patch_size, stride, queries_image.shape)
        return queries_image

    @staticmethod
    def get_output_shape(initial_image_shape, size, aspect_ratio):
        """Get the size of the output pyramid level"""
        h, w = initial_image_shape
        h, w = int(size * aspect_ratio[0]), int((w * size / h) * aspect_ratio[1])
        return h, w


    def get_first_initial_guess(self,reference_images, init_from, additive_noise_sigma):
        if init_from == "zeros":
            synthesized_images = torch.zeros_like(reference_images)
        elif init_from == "target":
            synthesized_images = reference_images.clone()
            import torchvision
            synthesized_images = torchvision.transforms.GaussianBlur(7, sigma=7)(synthesized_images)
        elif os.path.exists(init_from):
            synthesized_images = load_image(init_from)
        else:
            raise ValueError("Bad init mode", init_from)
        if additive_noise_sigma:
            synthesized_images += torch.randn_like(synthesized_images) * additive_noise_sigma

        return synthesized_images