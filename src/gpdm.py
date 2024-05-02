# Based on the work "Generating natural images with direct Patch Distributions Matching" by Ariel Elnekave and Yair Weiss (https://github.com/ariel415el/GPDM)
# 
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


import torch
import torch.nn.functional as F
import torchvision
from torchvision.utils import save_image
from torchvision.transforms import Resize as tv_resize
import cv2
import torch
import numpy as np
import os
from PIL import Image


class ImageProcessor:
    def __init__(self):
        pass

    @staticmethod
    def load_image(path):
        return ImageProcessor.cv2pt(cv2.imread(path))

    @staticmethod
    def dump_images(images, out_dir):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        for i, img in enumerate(images):
            save_image(img, os.path.join(out_dir, f"{i}.png"), normalize=True)

    @staticmethod
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

    
    @staticmethod
    def cv2pt(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float64) / 255.
        img = img * 2 - 1
        img = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
        return img

    
class PatchSWDLoss(torch.nn.Module):
    def __init__(self, patch_size=7, stride=1, num_proj=256):
        super(PatchSWDLoss, self).__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.num_proj = num_proj

    def forward(self, x, y):
        b, c, h, w = x.shape

        # Sample random normalized projections
        rand = torch.randn(self.num_proj, c*self.patch_size**2).to(x.device) # (slice_size**2*ch)
        rand = rand / torch.norm(rand, dim=1, keepdim=True)  # normalize to unit directions
        rand = rand.reshape(self.num_proj, c, self.patch_size, self.patch_size)

        # Project patches
        projx = F.conv2d(x, rand).transpose(1,0).reshape(self.num_proj, -1)
        projy = F.conv2d(y, rand).transpose(1,0).reshape(self.num_proj, -1)

        # Duplicate patches if number does not equal
        projx, projy = self.duplicate_to_match_lengths(projx, projy)

        # Sort and compute L1 loss
        projx, _ = torch.sort(projx, dim=1)
        projy, _ = torch.sort(projy, dim=1)

        loss = torch.abs(projx - projy).mean()

        return loss

    @staticmethod
    def duplicate_to_match_lengths(arr1, arr2):
        """
        Duplicates randomly selected entries from the smaller array to match its size to the bigger one
        :param arr1: (r, n) torch tensor
        :param arr2: (r, m) torch tensor
        :return: (r,max(n,m)) torch tensor
        """
        if arr1.shape[1] == arr2.shape[1]:
            return arr1, arr2
        elif arr1.shape[1] < arr2.shape[1]:
            arr1, arr2 = arr2, arr1

        b = arr1.shape[1] // arr2.shape[1]
        arr2 = torch.cat([arr2] * b, dim=1)
        if arr1.shape[1] > arr2.shape[1]:
            indices = torch.randperm(arr2.shape[1])[:arr1.shape[1] - arr2.shape[1]]
            arr2 = torch.cat([arr2, arr2[:, indices]], dim=1)

        return arr1, arr2
    

class ImageGenerator:
    def __init__(self, device='cuda:0'):
        self.device = device

    def generate(self, reference_images, criteria, init_from='zeros', pyramid_scales=(32, 64, 128, 256), lr=0.01,
                 num_steps=300, aspect_ratio=(1, 1), additive_noise_sigma=0.0, debug_dir=None):
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)

        criteria = criteria.to(self.device)
        reference_images = reference_images.to(self.device)
        synthesized_images = self.get_fist_initial_guess(reference_images, init_from, additive_noise_sigma).to(self.device)
        original_image_shape = synthesized_images.shape[-2:]
        all_losses = []
        for scale in pyramid_scales:
            lvl_references = tv_resize(scale)(reference_images)

            lvl_output_shape = self.get_output_shape(original_image_shape, scale, aspect_ratio)
            synthesized_images = tv_resize(lvl_output_shape)(synthesized_images)

            synthesized_images, losses = self._match_patch_distributions(synthesized_images, lvl_references, criteria,
                                                                         num_steps, lr)
            all_losses += losses

        return synthesized_images

    def _match_patch_distributions(self, synthesized_images, reference_images, criteria, num_steps, lr):
        synthesized_images.requires_grad_(True)
        optim = torch.optim.Adam([synthesized_images], lr=lr)
        losses = []
        for _ in range(num_steps):
            optim.zero_grad()
            loss = criteria(synthesized_images, reference_images)
            loss.backward()
            optim.step()
            losses.append(loss.item())

        return torch.clip(synthesized_images.detach(), -1, 1), losses

    def get_fist_initial_guess(self, reference_images, init_from, additive_noise_sigma):
        if init_from == "zeros":
            synthesized_images = torch.zeros_like(reference_images)
        elif init_from == "target":
            synthesized_images = reference_images.clone()
            synthesized_images = torchvision.transforms.GaussianBlur(7, sigma=7)(synthesized_images)
        elif os.path.exists(init_from):
            synthesized_images = ImageProcessor.load_image(init_from)
            synthesized_images = synthesized_images.repeat(reference_images.shape[0], 1, 1, 1)
        else:
            raise ValueError("Bad init mode", init_from)
        if additive_noise_sigma:
            synthesized_images += torch.randn_like(synthesized_images) * additive_noise_sigma

        return synthesized_images

    @staticmethod
    def get_output_shape(initial_image_shape, size, aspect_ratio):
        h, w = initial_image_shape
        h, w = int(size * aspect_ratio[0]), int((w * size / h) * aspect_ratio[1])
        return h, w