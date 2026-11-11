import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layer import DownsampleLayer



class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        # Create a list of downsampling layers according to args.layer_num
        encoder_layers = []
        for i in range(args.layer_num):
            encoder_layers.append(DownsampleLayer(args, i))
        self.encoder_layers = nn.ModuleList(encoder_layers)  # Register as ModuleList

    def forward(self, xyzs, feats, p_tensor):
        # Forward pass of the encoder
        # Input:
        #   xyzs: point coordinates, shape (B, 3, N)
        #   feats: point features, shape (B, C, N)
        #   p_tensor: tensor representing current packet loss rate, used for conditional encoding

        gt_xyzs = []   # Ground-truth coordinates at each downsampling layer (folded points / centroids)
        gt_dnums = []  # Number of original points represented by each sampled point
        gt_mdis = []   # Mean distance from original points to corresponding sampled centroid
        gt_std = []    # Standard deviation of distances within each sampled cluster

        # Pass through each downsampling layer sequentially
        for encoder_layer in self.encoder_layers:
            gt_xyzs.append(xyzs)  # Store coordinates before downsampling
            # Downsample points and features
            xyzs, feats, downsample_num, mean_distance, std_distance = encoder_layer(xyzs, feats, p_tensor)
            # Store statistics for loss computation and reconstruction
            gt_dnums.append(downsample_num)
            gt_mdis.append(mean_distance)
            gt_std.append(std_distance)

        latent_xyzs = xyzs    # Final latent coordinates after all downsampling layers
        latent_feats = feats  # Final latent features after all downsampling layers

        # Return ground-truth statistics for all layers and final latent representation
        return gt_xyzs, gt_dnums, gt_mdis, gt_std, latent_xyzs, latent_feats
