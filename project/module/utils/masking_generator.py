# ref (UM-MAE): https://github.com/implus/UM-MAE/blob/main/mask_transform.py

import random
import math
from einops.einops import rearrange
import torch
import numpy as np
import torchvision.transforms as transforms
from monai.utils import ensure_tuple_rep


class RandomMaskingGenerator:
    def __init__(self, input_size=None, mask_ratio=0.875, masking_type='tube'):
        """
        ** Implementation detail
        1. random masking is applied randomly to 3D volumes for each time point.
        2. tube masking divides time into blocks of the same size as the time window for masking.

        input_size (default) = (96 // spatial_stride, 96 // spatial_stride, 96 // spatial_stride, 20 // temporal stride): spatial_stride=2*patch_size, temporal_stride=patch_size(time dim) 
        """
        self.masking_type = masking_type
        if masking_type == 'random':
            raise ValueError('Random masking every single frame is not implemented yet')
        self.depth, self.height, self.width, self.time = input_size
        # count the number of spatial patches 
        self.num_spatial_patches = self.depth * self.height * self.width


        assert mask_ratio == 0.875  # In case of 3D image: 1/8.
        candidate_list_t = []
        ## spatial dimension
        while True:
            for j in range(4):
                candidate = torch.ones(8)
                candidate[j] = 0   # masking ratio = 0.875
                #candidate[j] = candidate[j + 4]= 0  # masking ratio = 0.75
                #candidate[j] = candidate[j + 1] = candidate[j + 2] = candidate[j + 3] = 0  # masking ratio = 0.5
                candidate_list_t.append(candidate)
            if len(candidate_list_t) * 4 >= self.num_spatial_patches * 2:
                break
        self.mask_candidate = torch.vstack(candidate_list_t)
        print(f'using {masking_type}, mask_candidate shape = {self.mask_candidate.shape}')

    def __repr__(self):
        repr_str = "Mask: total patches {}, mask patches {}, masking type {}".format(
            self.num_spatial_patches, self.num_mask, self.masking_type
        )
        return repr_str

    def __call__(self):
        if self.masking_type == 'random':
            raise ValueError('Random masking every single frame is not implemented yet')

        elif self.masking_type == 'tube':
            mask = self.mask_candidate.clone()
            mask_shuffle = torch.from_numpy(np.random.permutation(mask))
            mask_shuffle = rearrange(mask_shuffle[:self.num_spatial_patches // 8], '(d h w) (p1 p2 p3) -> (d p1) (h p2) (w p3)',
                                     d=self.depth // 2, h=self.height // 2, w=self.width // 2, p1=2, p2=2, p3=2)
            mask_shuffle = torch.tile(mask_shuffle, (self.time, 1, 1, 1))
            mask_shuffle = rearrange(mask_shuffle, 'T D H W -> D H W T', T=self.time, D=self.depth, H=self.height, W=self.width)

        return mask_shuffle  # D H W T



"""
## numpy version
class RandomMaskingGenerator:
    def __init__(self, input_size=(96,96,96,20), window_size=[4,4,4,4], mask_ratio=0.75, masking_type='random'):
        self.depth, self.height, self.width, self.time = input_size
        self.num_spatial_patches = self.depth * self.height * self.width
        self.time_window_size = window_size[-1]
        self.masking_type = masking_type

        assert mask_ratio == 0.75   # In case of 2D image: 1/4; In case of 3D image: 2/8. 
        candidate_list = []
        ## time dimension 
        for t in range(self.time):
            candidate_list_t = []
            ## spatial dimension
            while True: 
                for j in range(4):
                    candidate = np.ones(8)
                    candidate[j] = candidate[j+4] = 0
                    candidate_list_t.append(candidate)
                if len(candidate_list_t) * 4 >= self.num_spatial_patches * 2: 
                    break 
            candidate_list.append(np.expand_dims(np.vstack(candidate_list_t), axis=0))
        self.mask_candidate = np.vstack(candidate_list)
        print(f'using {masking_type}, mask_candidate shape = {self.mask_candidate.shape}')
                

    def __repr__(self):
        repr_str = "Mask: total patches {}, mask patches {}, masking type {}".format(
            self.num_spatial_patches, self.num_mask, self.masking_type
        )
        return repr_str

    def __call__(self):
        if self.masking_type == 'random':
            mask = self.mask_candidate.copy()
            mask_shuffle_tmp = self.mask_candidate.copy()
            mask_shuffle = []
            # shuffle
            for t in range(self.time):
                mask_shuffle_tmp[t] = np.random.permutation(mask[t])
                # rearrange
                mask_tmp = rearrange(mask_shuffle_tmp[t][:self.num_spatial_patches//8], '(d h w) (p1 p2 p3) -> (d p1) (h p2) (w p3)', 
                                     d=self.depth//2, h=self.height//2, w=self.width//2, p1=2, p2=2, p3=2)
                mask_shuffle.append(np.expand_dims(mask_tmp, axis=0))
            mask_shuffle = np.vstack(mask_shuffle)  # T, D//p1, H//p2, W//p3
            mask_shuffle = rearrange(mask_shuffle, 'T D H W -> D H W T', T= self.time, D=self.depth, H=self.height, W=self.width) 
            
        elif self.masking_type == 'tube':
            mask = self.mask_candidate.copy()
            mask_shuffle_tmp = self.mask_candidate.copy()
            mask_shuffle = []
            # shuffle
            for t in range(self.time):
                # apply the same shuffle index within the window
                if t % self.time_window_size == 0: 
                    idx = np.random.permutation(np.arange(mask.shape[-1]))
                for i in range(mask.shape[1]):
                    mask_shuffle_tmp[t][i] = mask[t][i][idx]
                # rearrange
                mask_tmp = rearrange(mask_shuffle_tmp[t][:self.num_spatial_patches//8], '(d h w) (p1 p2 p3) -> (d p1) (h p2) (w p3)', 
                                     d=self.depth//2, h=self.height//2, w=self.width//2, p1=2, p2=2, p3=2)
                mask_shuffle.append(np.expand_dims(mask_tmp, axis=0))
            mask_shuffle = np.vstack(mask_shuffle)  # T, D//p1, H//p2, W//p3 
            mask_shuffle = rearrange(mask_shuffle, 'T D H W -> D H W T', T= self.time, D=self.depth, H=self.height, W=self.width)
                
        return mask_shuffle     # D H W T
"""




class simmim_MaskGenerator:
    def __init__(self, input_size=[96, 96, 96, 20], mask_patch_size=(12, 12, 12, 2), model_patch_size=[6, 6, 6, 2], mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        assert self.input_size[0] % self.mask_patch_size[0] == 0 and self.input_size[1] % self.mask_patch_size[1] == 0 and self.input_size[2] % self.mask_patch_size[2] == 0 and self.input_size[3] % self.mask_patch_size[3] == 0 
        assert self.mask_patch_size[0] % self.model_patch_size[0] == 0 and self.mask_patch_size[1] % self.model_patch_size[1] == 0 and self.mask_patch_size[2] % self.model_patch_size[2] == 0 and self.mask_patch_size[3] % self.model_patch_size[3] == 0
        
        self.rand_size = (self.input_size[0]//self.mask_patch_size[0], self.input_size[1]//self.mask_patch_size[1], self.input_size[2]//self.mask_patch_size[2], self.input_size[3]//self.mask_patch_size[3])
        self.scale = (self.mask_patch_size[0]//self.model_patch_size[0], self.mask_patch_size[1]//self.model_patch_size[1], self.mask_patch_size[2]//self.model_patch_size[2], self.mask_patch_size[3]//self.model_patch_size[3])
        
        self.token_count = self.rand_size[0] * self.rand_size[1] * self.rand_size[2] * self.rand_size[3]
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
            
        mask = mask.reshape((self.rand_size[0], self.rand_size[1], self.rand_size[2], self.rand_size[3]))
        mask = mask.repeat(self.scale[0], axis=0).repeat(self.scale[1], axis=1).repeat(self.scale[2], axis=2).repeat(self.scale[3], axis=3)    
        return torch.tensor(mask)