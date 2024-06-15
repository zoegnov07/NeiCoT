# --------------------------------------------------------
# Based on MAE, CPC, BEiT, timm, DINO and DeiT code bases
# https://github.com/pengzhiliang/MAE-pytorch
# https://github.com/jefflai108/Contrastive-Predictive-Coding-PyTorch
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import random
import math
import numpy as np

class RandomMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        # if not isinstance(input_size, tuple):
        #     input_size = (input_size,) *2
        # self.height, self.width = input_size
        # self.num_patches = self.height * self.width
        
        self.num_patches = input_size        
        self.num_mask = int(mask_ratio * self.num_patches)
        
        
    def __repr__(self):
        repr_str = "Maks: total band {}, mask band {}".format(
            self.num_patches, self.num_mask
        )
        return repr_str

    def __call__(self):
        mask = np.hstack([
            np.zeros(self.num_patches - self.num_mask),
            np.ones(self.num_mask),
        ])
        np.random.shuffle(mask)
        # print("mask1_print:::::::::::::::::::::",mask)
        return mask # [196]