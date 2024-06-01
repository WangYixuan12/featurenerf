import os.path
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import vision_transformer_flexible as vits


class DINOv2(nn.Module):
    def __init__(self, patch_h, patch_w):
        super().__init__()
        self.device = 'cuda'
        self.dtype= torch.float16
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14").to(self.device, dtype=self.dtype)
        self.patch_h = patch_h
        self.patch_w = patch_w

    def forward(self, imgs):
        B = imgs.shape[0]
        patch_h = self.patch_h
        patch_w = self.patch_w
        # feat_dim = 384 # vits14
        # feat_dim = 768 # vitb14
        feat_dim = 1024  # vitl14
        # feat_dim = 1536 # vitg14

        with torch.no_grad():
            features_dict = self.model.forward_features(
                imgs.to(dtype=self.dtype)
            )
            features = features_dict["x_norm_patchtokens"]
            features = features.reshape((B, patch_h, patch_w, feat_dim))
            features = features.permute(0, 3, 1, 2)
        return features
