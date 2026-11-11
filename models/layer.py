import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.pointops.functions import pointops
from models.Chamfer3D.dist_chamfer_3D import chamfer_3DDist
from einops import rearrange, repeat
from models.utils import index_points
import math
from models.pointops.functions import pointops
from models.icosahedron2sphere import icosahedron2sphere
chamfer_dist = chamfer_3DDist()

#[b,c,m]
class PointTransformerLayer(nn.Module):
    def __init__(self, in_fdim, out_fdim, args):
        super(PointTransformerLayer, self).__init__()

        self.w_qs = nn.Conv1d(in_fdim, args.hidden_dim, 1)
        self.w_ks = nn.Conv1d(in_fdim, args.hidden_dim, 1)
        self.w_vs = nn.Conv1d(in_fdim, args.hidden_dim, 1)

        self.conv_delta = nn.Sequential(
            nn.Conv2d(3, args.hidden_dim, 1),
            nn.GroupNorm(args.ngroups, args.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(args.hidden_dim, args.hidden_dim, 1),
            nn.GroupNorm(args.ngroups, args.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(args.hidden_dim, args.hidden_dim, 1)
        )

        self.conv_gamma = nn.Sequential(
            nn.Conv2d(args.hidden_dim, args.hidden_dim, 1),
            nn.GroupNorm(args.ngroups, args.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(args.hidden_dim, args.hidden_dim, 1),
            nn.GroupNorm(args.ngroups, args.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(args.hidden_dim, args.hidden_dim, 1)
        )

        self.post_conv = nn.Conv1d(args.hidden_dim, out_fdim, 1)

    def forward(self, q_xyzs, k_xyzs, q_feats, k_feats, v_feats, knn_idx, mask):
        # q: (b, c, m), k: (b, c, n), knn_idx: (b, m, k), mask: (b, m, k)

        # (b, 3, m, k) 
        knn_xyzs = index_points(k_xyzs, knn_idx)

        # note it's q_feats but not v_feats
        pre = q_feats
        # (b, c, m) -> (b,hidden_dim,m)
        query = self.w_qs(q_feats)
        # (b, c, m, k) 
        key = index_points(self.w_ks(k_feats), knn_idx)
        value = index_points(self.w_vs(v_feats), knn_idx)

        # (b, c, m, k) 
        pos_enc = self.conv_delta(q_xyzs.unsqueeze(-1) - knn_xyzs)
        # attention (b,hidden_dim,m,k)
        attn = self.conv_gamma(query.unsqueeze(-1) - key + pos_enc)
        # (b, c, m, k) 
        attn = attn / math.sqrt(key.shape[1])

        # mask
        mask_value = -(torch.finfo(attn.dtype).max) 
        attn.masked_fill_(~mask[:, None], mask_value)
        attn = F.softmax(attn, dim=-1)
        res = torch.einsum('bcmk, bcmk->bcm', attn, value + pos_enc) 
        res = self.post_conv(res) + pre

        return res
#[b,4,m,k]->[b,dim,m,k]->[b,dim,m]
class PositionEmbeddingLayer(nn.Module):
    def __init__(self, args):
        super(PositionEmbeddingLayer, self).__init__()

        self.pre_nn = nn.Sequential(
            nn.Conv2d(4, args.hidden_dim, 1),
            nn.GroupNorm(args.ngroups, args.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(args.hidden_dim, args.hidden_dim, 1),
            nn.GroupNorm(args.ngroups, args.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(args.hidden_dim, args.dim, 1)
        )

        self.attn_nn = nn.Sequential(
            nn.Conv2d(args.dim, args.hidden_dim, 1),
            nn.GroupNorm(args.ngroups, args.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(args.hidden_dim, args.hidden_dim, 1),
            nn.GroupNorm(args.ngroups, args.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(args.hidden_dim, args.dim, 1)
        )


    def forward(self, q_xyzs, k_xyzs, knn_idx, mask):
        # q_xyzs: (b, 3, m), k_xyzs: (b, 3, n), knn_idx and mask: (b, m, k)

        # (b, 3, m, k)
        knn_xyzs = index_points(k_xyzs, knn_idx)
        # (b, 3, m, k)
        k = knn_xyzs.shape[-1]
        #repeated_xyzs = q_xyzs[..., None].repeat(1, 1, 1, k)
        repeated_xyzs = q_xyzs.unsqueeze(-1).expand(-1, -1, -1, k)

        # (b, 3, m, k) 
        direction = F.normalize(knn_xyzs - repeated_xyzs, p=2, dim=1) 
        # (b, 1, m, k) 
        distance = torch.norm(knn_xyzs - repeated_xyzs, p=2, dim=1, keepdim=True)
        # (b, 4, m, k)
        local_pattern = torch.cat((direction, distance), dim=1)

        # (b, c, m, k)
        position_embedding = self.pre_nn(local_pattern)

        # (b, c, m, k)
        attn = self.attn_nn(position_embedding)
        # mask
        mask_value = -(torch.finfo(attn.dtype).max)
        attn.masked_fill_(~mask[:, None], mask_value) 
        attn = F.softmax(attn, dim=-1) 

        position_embedding = position_embedding * attn
        # (b, c, m)
        position_embedding = position_embedding.sum(dim=-1)

        return position_embedding

#[b,m]->[b,dim,m]
class DensityEmbeddingLayer(nn.Module):
    def __init__(self, args):
        super(DensityEmbeddingLayer, self).__init__()

        self.pre_nn = nn.Sequential(
            nn.Conv1d(1, args.hidden_dim, 1),
            nn.GroupNorm(args.ngroups, args.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(args.hidden_dim, args.hidden_dim, 1),
            nn.GroupNorm(args.ngroups, args.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(args.hidden_dim, args.dim, 1)
        )


    def forward(self, downsample_num):
        # input: (b, 1, n)

        density_embedding = self.pre_nn(downsample_num)

        return density_embedding

class ChannelAwareConditioning(nn.Module):
    """
    Channel-Aware Conditioning module for point cloud features.
    It encodes the channel condition
    and generates loss rate features and branch gates.
    """
    def __init__(self, args):
        super(ChannelAwareConditioning, self).__init__()
        self.dim = args.dim
        hidden = args.hidden_dim
        # Frequency embedding: map scalar channel parameter p_tensor -> high-dimensional hidden feature
        self.frequency_embedding = nn.Sequential(
            nn.Linear(1, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, hidden)
        )
        #generate loss rate features
        self.base_p_nn = nn.Sequential(
            nn.Conv1d(hidden, hidden, 1),
            nn.GroupNorm(args.ngroups, hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, 1),
            nn.GroupNorm(args.ngroups, hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, self.dim, 1) 
        )
        #generate branch gates for adaptive feature fusion
        self.branch_gates = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, 3), 
            nn.Sigmoid() 
        )

    def forward(self, p_tensor, sample_num):

        p_enc = self.frequency_embedding(p_tensor) # [B, hidden]
        
        gates = self.branch_gates(p_enc) # [B, 3]

        p_feat = self.base_p_nn(p_enc.unsqueeze(-1)) # [B, dim, 1]
        p_feat = p_feat.expand(-1, -1, sample_num)   # [B, dim, M]
        
        return p_feat, gates

class DownsampleLayer(nn.Module):
    """
    Point cloud downsampling layer 
    """
    def __init__(self, args, layer_idx):
        super(DownsampleLayer, self).__init__()

        self.args = args
        self.k = args.k # number of nearest neighbors
        self.downsample_rate = args.downsample_rate[layer_idx]
        self.dim = args.dim
        self.pre_conv = nn.Conv1d(args.dim, args.dim, 1)
        
        #ancestor feature aggregation
        self.feats_agg_nn = PointTransformerLayer(args.dim, args.dim, args)
        #position embedding
        self.position_embedding_nn = PositionEmbeddingLayer(args)
        #density embedding
        self.density_embedding_nn = DensityEmbeddingLayer(args) 
        #channel-aware conditioning
        self.p_adaptive_nn = ChannelAwareConditioning(args)
        

        self.post_conv = nn.Conv1d(args.dim * 4, args.dim, 1) 


    def get_density(self, sampled_xyzs, xyzs):
        # input: (b, 3, m), (b, 3, n)
        batch_size = xyzs.shape[0]
        sample_num = sampled_xyzs.shape[2]
        # (b, n, 3)
        xyzs_trans = xyzs.permute(0, 2, 1).contiguous()
        # (b, sample_num, 3)
        sampled_xyzs_trans = sampled_xyzs.permute(0, 2, 1).contiguous()

        # find the nearest neighbor in sampled_xyzs_trans: (b, n, 1)
        ori2sample_idx = pointops.knnquery_heap(1, sampled_xyzs_trans, xyzs_trans) 

        # (b, sample_num)
        downsample_num = torch.zeros((batch_size, sample_num)).cuda()
        for i in range(batch_size):
            uniques, counts = torch.unique(ori2sample_idx[i], return_counts=True)
            downsample_num[i][uniques.long()] = counts.float() 

        # (b, m, k)
        knn_idx = pointops.knnquery_heap(self.k, xyzs_trans, sampled_xyzs_trans).long() 
        torch.cuda.empty_cache()

        # mask: (m)
        expect_center = torch.arange(0, sample_num).cuda()
        # (b, m, k)
        expect_center = expect_center.view(1, -1, 1)
        expect_center = expect_center.expand(batch_size, -1, self.k)
        # (b, 1, m, k) 
        real_center = index_points(ori2sample_idx.permute(0, 2, 1).contiguous(), knn_idx) 
        # (b, m, k)
        real_center = real_center.squeeze(1)
        # mask those points that not belong to collapsed points set
        mask = torch.eq(expect_center, real_center)

        # (b, 3, m, k) sampled_xyzs(b,3,m)
        knn_xyzs = index_points(xyzs, knn_idx)
        # (b, 1, m, k) 
        distance = torch.norm(knn_xyzs - sampled_xyzs[..., None], p=2, dim=1, keepdim=True)
        # mask 
        mask_value = 0
        distance.masked_fill_(~mask[:, None], mask_value)

        # (b, m)
        distance_sum = distance.sum(dim=-1).squeeze(1)
        dist_sq_sum = (distance ** 2).sum(dim=-1).squeeze(1) # (b, m)
        # (b, m)
        mean_distance = distance_sum / downsample_num
        mean_sq_dist = dist_sq_sum / downsample_num 
        variance = mean_sq_dist - (mean_distance ** 2)
        std_distance = torch.sqrt(torch.clamp(variance, min=1e-10))

        return downsample_num, mean_distance,std_distance,mask, knn_idx


    def forward(self, xyzs, feats,p_tensor):
        """
        Forward pass: farthest point sampling + feature aggregation + embeddings + residual connection.
        
        Args:
            xyzs: (b, 3, n) input point coordinates
            feats: (b, c, n) input features
            p_tensor: (b, 1) channel parameter
        Returns:
            sampled_xyzs: (b, 3, m) downsampled points
            sampled_feats: (b, c, m) aggregated features
            downsample_num: (b, m) number of original points per sampled point
            mean_distance: (b, m) mean distance to sampled point
            std_distance: (b, m) std of distances
        """
        # xyzs: (b, 3, n), features: (b, cin, n)
        if self.k > xyzs.shape[2]:
            self.k = xyzs.shape[2]
        sample_num = round(xyzs.shape[2] * self.downsample_rate)
        # (b, n, 3)
        xyzs_trans = xyzs.permute(0, 2, 1).contiguous()
        # FPS, (b, sample_num)
        sample_idx = pointops.furthestsampling(xyzs_trans, sample_num).long()
        # (b, 3, sample_num)
        sampled_xyzs = index_points(xyzs, sample_idx)

        # get density
        downsample_num, mean_distance,std_distance, mask, knn_idx = self.get_density(sampled_xyzs, xyzs)

        identity = feats
        feats = self.pre_conv(feats)
        # (b, cin, sample_num)
        sampled_feats = index_points(feats, sample_idx)
        # embedding
        ancestor_embedding = self.feats_agg_nn(sampled_xyzs, xyzs, sampled_feats, feats, feats, knn_idx, mask)
        position_embedding = self.position_embedding_nn(sampled_xyzs, xyzs, knn_idx, mask)
        density_embedding = self.density_embedding_nn(downsample_num.unsqueeze(1))
        #apply adaptive gating
        p_feat, gates = self.p_adaptive_nn(p_tensor, sample_num)
        gate_trans = gates[:, 0].view(-1, 1, 1)
        gate_pos = gates[:, 1].view(-1, 1, 1)
        gate_dens = gates[:, 2].view(-1, 1, 1)
        # (b, 3c, m)
        agg_embedding = torch.cat((
            ancestor_embedding * gate_trans,
            position_embedding * gate_pos,
            density_embedding * gate_dens,
            p_feat
        ), dim=1)
        # (b, c, m)
        agg_embedding = self.post_conv(agg_embedding)

        # residual connection: (b, c, m) 
        sampled_feats = agg_embedding + index_points(identity, sample_idx) 

        return sampled_xyzs, sampled_feats, downsample_num, mean_distance,std_distance

class SubPointConv(nn.Module):
    def __init__(self, args, in_fdim, out_fdim, group_num):
        super(SubPointConv, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.group_num = group_num
        self.group_in_fdim = in_fdim // group_num
        self.group_out_fdim = out_fdim // group_num
        # mlp
        mid_dim = self.hidden_dim // 2 
    
        self.mlp = nn.Sequential(
            nn.Conv2d(self.group_in_fdim, mid_dim, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_dim, self.hidden_dim, 1),
            nn.GroupNorm(args.ngroups, self.hidden_dim), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.hidden_dim, self.group_out_fdim, 1)
        )

    def forward(self, feats):
        # per-group conv: (b, cin, n, g)
        feats = rearrange(feats, 'b (c g) n -> b c n g', g=self.group_num).contiguous()
        # (b, cout, n, g)
        expanded_feats = self.mlp(feats)
        return expanded_feats

#upsampled_xyzs  (b, 3, n, u) 
class XyzsUpsampleLayer(nn.Module):
    def __init__(self, args, layer_idx, upsample_rate=None):
        super(XyzsUpsampleLayer, self).__init__()

        if upsample_rate == None:
            self.upsample_rate = args.max_upsample_num[layer_idx]
        else:
            self.upsample_rate = upsample_rate

        # each point has fixed 43 candidate directions, size (43, 3)
        hypothesis, _ = icosahedron2sphere(1) 
        hypothesis = np.append(np.zeros((1,3)), hypothesis, axis=0) 
        self.hypothesis = torch.from_numpy(hypothesis).float().cuda()

        # weights 
        self.weight_nn = SubPointConv(args, args.dim, 43*self.upsample_rate, self.upsample_rate)

        # scales 
        self.scale_nn = SubPointConv(args, args.dim, 1*self.upsample_rate, self.upsample_rate)


    def forward(self, xyzs, feats):
        # xyzs: (b, 3, n)  feats (b, c, n)
        batch_size = xyzs.shape[0]
        points_num = xyzs.shape[2]

        # return upsampled_xyzs
        # (b, 43, n，u)
        weights = self.weight_nn(feats)
        weights = rearrange(weights, 'b h n u -> b u h n', h=43, u=self.upsample_rate)
        weights = F.softmax(weights, dim=2) 

        # (b, 1, n, u)
        scales = self.scale_nn(feats)
        directions = torch.einsum('buhn,hc->bucn', weights, self.hypothesis)


        # (b, u, c, n) -> (b, c, n, u)
        directions = rearrange(directions, 'b u c n -> b c n u')
        directions = F.normalize(directions, p=2, dim=1)
    
        # (b, 3, n, u)
        deltas = directions * scales

        repeated_xyzs = xyzs.unsqueeze(-1) # (b, 3, n, 1)
        upsampled_xyzs = repeated_xyzs + deltas # 利用广播机制，无需 expand

        return upsampled_xyzs

class FeatsUpsampleLayer(nn.Module):
    def __init__(self, args, layer_idx, upsample_rate=None):
        super(FeatsUpsampleLayer, self).__init__()

        if upsample_rate == None:
            self.upsample_rate = args.max_upsample_num[layer_idx]
        else:
            self.upsample_rate = upsample_rate

        self.out_fdim = args.dim

        self.feats_nn = SubPointConv(args, args.dim, self.out_fdim * self.upsample_rate, self.upsample_rate)


    def forward(self, feats):
        # (b, c, n, u)
        upsampled_feats = self.feats_nn(feats)

        if self.decompress_normal == False:
            repeated_feats = feats.unsqueeze(-1).expand(-1, -1, -1, self.upsample_rate)

            # (b, c, n, u)
            upsampled_feats = upsampled_feats + repeated_feats

        return upsampled_feats


class UpsampleLayer(nn.Module):
    def __init__(self, args, layer_idx):
        super(UpsampleLayer, self).__init__()

        self.xyzs_upsample_nn = XyzsUpsampleLayer(args, layer_idx)
        self.feats_upsample_nn = FeatsUpsampleLayer(args, layer_idx)


    def forward(self, xyzs, feats):
        upsampled_xyzs = self.xyzs_upsample_nn(xyzs, feats)
        upsampled_feats = self.feats_upsample_nn(feats)

        return upsampled_xyzs, upsampled_feats


class KNNInterpolationRestoration(nn.Module):
    def __init__(self, k=8):
        super().__init__()
        self.k = k

    def forward(self, xyz, feats_corrupted, mask):

        B, C, N = feats_corrupted.shape
        xyz_t = xyz.permute(0, 2, 1).contiguous() # [B, N, 3]

        knn_idx = pointops.knnquery_heap(self.k, xyz_t, xyz_t).long()

        # knn_xyz: [B, 3, N, K]
        knn_xyz = index_points(xyz, knn_idx) 
  
        diff = knn_xyz - xyz.unsqueeze(-1) # [B, 3, N, K]
        dist = torch.norm(diff, dim=1, p=2) # [B, N, K]
        

        # knn_feats: [B, C, N, K]
        knn_feats = index_points(feats_corrupted, knn_idx)
        # knn_mask: [B, C, N, K]
        knn_mask = index_points(mask, knn_idx)

        eps = 1e-6
        weights = 1.0 / (dist.unsqueeze(1) + eps) # [B, 1, N, K]
        valid_weights = weights * knn_mask # [B, C, N, K]
        
        # sum(feature * weight) / sum(weight)
        weighted_sum = torch.sum(knn_feats * valid_weights, dim=-1) # [B, C, N]
        weight_total = torch.sum(valid_weights, dim=-1) + eps
        
        restored_feats = weighted_sum / weight_total

        out = feats_corrupted * mask + restored_feats * (1.0 - mask)
        
        return out


class GeometryGuidedRestorationNet(nn.Module):
    """
    Geometry-guided feature restoration network for corrupted point cloud features.
    Uses local neighborhoods, relative geometry, and masks to restore features.
    """
    def __init__(self, args, feat_dim):
        super(GeometryGuidedRestorationNet, self).__init__()
        self.args = args
        self.k = args.k # number of neighbors
        self.feat_dim = feat_dim
        xyz_dim = 3 
        hidden_dim = 128
        # input channels: center+neighbor xyz, features, mask
        in_channel = 2 * xyz_dim + 2 * feat_dim + 2 * feat_dim
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, hidden_dim, 1, bias=False),
            nn.GroupNorm(args.ngroups, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
            nn.GroupNorm(args.ngroups, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.attn_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.Sigmoid() 
        )
        # channel-wise MLP for post-aggregation refinement
        self.channel_mlp = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 1, bias=False),
            nn.GroupNorm(args.ngroups, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, 1, bias=False),
            nn.GroupNorm(args.ngroups, hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        self.conv_out = nn.Conv1d(hidden_dim, feat_dim, 1)
        
        nn.init.zeros_(self.conv_out.weight)
        if self.conv_out.bias is not None:
            nn.init.zeros_(self.conv_out.bias)

    def forward(self, xyz, feats_corrupted, mask):
        """
        Forward pass to restore corrupted features using geometry and mask guidance.
        
        Args:
            xyz: (B, 3, N) point coordinates
            feats_corrupted: (B, C, N) corrupted features
            mask: (B, C, N) mask indicating valid points (1 = valid, 0 = corrupted)
        Returns:
        restored: (B, C, N) restored features
        """
        B, C, N = feats_corrupted.shape
        k = min(self.k, xyz.shape[2])
        # compute k-nearest neighbors
        xyz_trans = xyz.permute(0, 2, 1).contiguous()
        idx = pointops.knnquery_heap(k, xyz_trans, xyz_trans).long()
        actual_k = idx.shape[-1]
        # gather neighbor info
        xyz_neighbors = index_points(xyz, idx) # [B, 3, N, K]
        feats_neighbors = index_points(feats_corrupted, idx)
        mask_neighbors = index_points(mask, idx)
        # expand center info for relative computation
        xyz_center = xyz.unsqueeze(-1).expand(-1, -1, -1, actual_k)
        feats_center = feats_corrupted.unsqueeze(-1).expand(-1, -1, -1, actual_k)
        mask_center = mask.unsqueeze(-1).expand(-1, -1, -1, actual_k)
        diff_xyz = xyz_neighbors - xyz_center
        dists = torch.norm(diff_xyz, dim=1, p=2, keepdim=True)
        feature_concat = torch.cat([
            xyz_center, diff_xyz,
            feats_center, feats_neighbors - feats_center,
            mask_center, mask_neighbors - mask_center
        ], dim=1)
        
        feat = self.conv1(feature_concat) # [B, 128, N, K]
        feat = self.conv2(feat)
        attn = self.attn_conv(feat) 
        feat = feat * attn
        
        #weight dist and reliability
        sigma = dists.mean(dim=-1, keepdim=True) + 1e-6
        weight_dist = torch.exp(-(dists**2) / (2 * sigma**2))
        neighbor_reliability = mask_neighbors.mean(dim=1, keepdim=True) 
        feat = feat * neighbor_reliability * weight_dist
        
        feat = feat.max(dim=-1)[0]        
        feat = self.channel_mlp(feat)     
        delta = self.conv_out(feat)    
        # restore features at corrupted positions   
        restored = feats_corrupted + (1.0 - mask) * delta
        
        return restored


class FeatureRefinementModule(nn.Module):
    """
    Feature refinement using codebook-based attention.
    Maps input features to a learnable codebook, applies attention, and outputs residual-corrected features.
    """
    def __init__(self, channels, num_codebook=128): 
        super(FeatureRefinementModule, self).__init__()
        self.channels = channels
        self.num_codebook = num_codebook
        # normalization and learnable codebook
        self.ln = nn.LayerNorm(channels)
        self.codebook = nn.Parameter(torch.randn(1, channels, num_codebook) * 0.1)
    
        self.q_conv = nn.Conv1d(channels, channels, 1)
        self.k_conv = nn.Conv1d(channels, channels, 1)
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.out_conv = nn.Conv1d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        nn.init.zeros_(self.out_conv.weight)
        if self.out_conv.bias is not None:
            nn.init.zeros_(self.out_conv.bias)

    def forward(self, x):
        """
        Args:
            x: (B, C, N) input features
        Returns:
            out: (B, C, N) refined features
            usage_ratio: codebook usage ratio [0,1]
        """
        B, C, N = x.shape

        x_in = x.permute(0, 2, 1) 
        x_norm = self.ln(x_in).permute(0, 2, 1) # [B, C, N]
        
        q = self.q_conv(x_norm).permute(0, 2, 1) # [B, N, C]b,c,m
        cb_norm = F.normalize(self.codebook, p=2, dim=1)
        k = self.k_conv(cb_norm).expand(B, -1, -1).permute(0, 2, 1) # [B, M, C]
        v = self.v_conv(cb_norm).expand(B, -1, -1).permute(0, 2, 1) # [B, M, C]
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / (C**0.5 * 0.1)
        attn = self.softmax(attn_scores)
        refine = torch.matmul(attn, v).permute(0, 2, 1) # [B, C, N]
        correction = self.out_conv(refine)
        out = x + self.gamma * correction
        avg_probs = torch.mean(attn, dim=(0, 1))
        entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        perplexity = torch.exp(entropy)
        usage_ratio = perplexity / self.num_codebook
        return out, usage_ratio
