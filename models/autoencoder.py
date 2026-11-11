import torch
import torch.nn as nn
from models.encoder import  Encoder
from models.decoder import Decoder
from compressai.entropy_models import EntropyBottleneck,GaussianConditional
import math
import numpy as np
from models.layer import GeometryGuidedRestorationNet,FeatureRefinementModule,KNNInterpolationRestoration
from models.utils import pad_and_reshape, swap_element, inv_swap_element, get_z_order_indices_torch,LowerBound
from models.loss import get_chamfer_loss, get_density_loss, get_pts_num_loss
from compressai.models.utils import update_registered_buffers
import random
import torch.nn.functional as F


def Simulated_Packet_Loss(loss_rate_percent, channel_num, device):
    if loss_rate_percent <= 0:
        return torch.ones(channel_num, device=device)

    prob_threshold = loss_rate_percent / 100.0
    pkt_size = 2
    num_packets = channel_num // pkt_size 
    packet_mask = (torch.rand(num_packets, device=device) >= prob_threshold).float()
    mask = torch.repeat_interleave(packet_mask, pkt_size)

    return mask


class AutoEncoder(nn.Module): 
    def __init__(self, args):
        super(AutoEncoder, self).__init__() 
        self.args = args
        self.target_loss_rate = 10
        self.lower_bound = LowerBound(1e-4)
        self.pre_conv = nn.Sequential(
            nn.Conv1d(args.in_fdim, args.hidden_dim, 1), 
            nn.GroupNorm(args.ngroups, args.hidden_dim), 
            nn.ReLU(inplace=True),
            nn.Conv1d(args.hidden_dim, args.dim, 1) 
        ) 
        self.encoder = Encoder(args) # Latent encoding network
        N = args.hyper_channels # Hyper latent channels
        M = args.dim # Feature dimension
        self.N = N
        self.M = M

        # Hyperprior encoder 
        self.h_a = nn.Sequential(
            nn.Conv2d(M, N, kernel_size=5, stride=2, padding=2),  
            nn.ReLU(inplace=True),
            nn.Conv2d(N, N, kernel_size=5, stride=2, padding=2),           
            nn.ReLU(inplace=True),
            nn.Conv2d(N, N, kernel_size=5, stride=1, padding=2),           
        )
        #hyperprior entropy model
        self.z_eblock = EntropyBottleneck(self.N)
        # Hyperprior decoder
        self.h_s_main = nn.Sequential(
            nn.ConvTranspose2d(N, N, kernel_size=5, stride=1, padding=2),          
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(N, N, kernel_size=5, stride=2, padding=2, output_padding=1),  
            nn.ReLU(inplace=True),
        )
        # predict Gaussian parameters for latent features
        self.h_s_feats_means_head = nn.ConvTranspose2d(
            N, M, kernel_size=5, stride=2, padding=2, output_padding=1
        )
        self.h_s_feats_scales_head = nn.Sequential(
            nn.ConvTranspose2d(N, M, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Softplus(), 
            self.lower_bound
        )
        self.gaussian_conditional_feats = GaussianConditional(None)
        #restoration network for feature reconstruction after simulated packet loss
        self.restoration_net = GeometryGuidedRestorationNet(args=args, feat_dim=self.M)
        #feature refinement network 
        self.refinement_net = FeatureRefinementModule(channels=self.M, num_codebook=512)
        self.decoder = Decoder(args)
           
    def get_loss(self, gt_xyzs,gt_normals, gt_dnums, gt_mdis,gt_std, pred_xyzs, pred_unums, pred_mdis,pred_std):
        # Compute Chamfer distance, density loss, and points number loss
        chamfer_loss, all_pred2gt_idx = get_chamfer_loss(gt_xyzs, pred_xyzs, self.args) 
        density_loss = get_density_loss(gt_dnums, gt_mdis,gt_std, pred_unums, pred_mdis,pred_std, all_pred2gt_idx, self.args) 
        pts_num_loss = get_pts_num_loss(gt_xyzs, pred_unums, self.args) 

        loss = chamfer_loss + density_loss + pts_num_loss 

        loss_items = {
            'chamfer_loss': chamfer_loss.item(),
            'density_loss': density_loss.item(),
            'pts_num_loss': pts_num_loss.item(),
        }

        return loss, loss_items, all_pred2gt_idx

    def forward(self, input,loss_rate=None):
        # input: (b, c, n)
        points_num = input.shape[0] * input.shape[2]
        xyzs = input[:, :3, :].contiguous() 
        gt_normals = input[:, 3:, :].contiguous() 

        # raise dimension
        feats = self.pre_conv(feats) 
        # Simulate packet loss during training
        if self.training:
            if random.random() < 0.50:
                current_loss_rate = 0.0
            else:
                p_sample = np.random.beta(1.1, 2)
                current_loss_rate = p_sample * 50.0 
        else:
            current_loss_rate = float(self.target_loss_rate)

        B = xyzs.shape[0]
        p_tensor = torch.tensor([[current_loss_rate / 100.0]], device=xyzs.device).expand(B, 1) 

        #encoding
        gt_xyzs, gt_dnums, gt_mdis,gt_std, latent_xyzs_ori, latent_feats_ori = self.encoder(xyzs, feats,p_tensor)
        points = latent_xyzs_ori.shape[2]

        # sort points by z-order
        sorted_indices = get_z_order_indices_torch(latent_xyzs_ori) 
        B, C_xyzs, N = latent_xyzs_ori.shape
        _, C_feats, _ = latent_feats_ori.shape
        sorted_indices_xyzs = sorted_indices.unsqueeze(1).expand(B, C_xyzs, N)
        sorted_indices_feats = sorted_indices.unsqueeze(1).expand(B, C_feats, N)
        sorted_indices_1d = sorted_indices.unsqueeze(1) 
        latent_xyzs = torch.gather(latent_xyzs_ori, 2, sorted_indices_xyzs)
        latent_feats = torch.gather(latent_feats_ori, 2, sorted_indices_feats)
        gt_dnums[-1] = torch.gather(gt_dnums[-1].unsqueeze(1), 2, sorted_indices_1d).squeeze(1)
        gt_mdis[-1] = torch.gather(gt_mdis[-1].unsqueeze(1), 2, sorted_indices_1d).squeeze(1)
        gt_std[-1] = torch.gather(gt_std[-1].unsqueeze(1), 2, sorted_indices_1d).squeeze(1)
        del latent_xyzs_ori, latent_feats_ori

        #coordinates bpp calculation
        gt_latent_xyzs = latent_xyzs
        pred_latent_xyzs = latent_xyzs.half().float() 
        xyzs_size = B * points * 3 * 16
        xyzs_bpp = torch.tensor(xyzs_size / points_num, device=xyzs.device) 
        
        #reshape to 2D and interleave
        latent_feats_2d= pad_and_reshape(latent_feats)
        latent_feats_2d_chunk = latent_feats_2d.chunk(self.M // 4, dim=1)
        swapped_chunks_feats = [swap_element(chunk) for chunk in latent_feats_2d_chunk]
        latent_feats_2d = torch.cat(swapped_chunks_feats, dim=1)
    
        #hyperprior bpp calculation
        z = self.h_a(latent_feats_2d)
        z_hat, z_likelihoods = self.z_eblock(z)
        z_size = (torch.log(z_likelihoods).sum()) / (-math.log(2))
        z_bpp = z_size / points_num

        #entropy coding for latent features
        shared_params = self.h_s_main(z_hat)
        means_hat_feats_2d = self.h_s_feats_means_head(shared_params)
        scales_hat_feats_2d = self.h_s_feats_scales_head(shared_params)
        latent_feats_hat_2d, feats_likelihoods_2d = self.gaussian_conditional_feats(latent_feats_2d, scales_hat_feats_2d, means=means_hat_feats_2d)
        
        feats_size = (torch.log(feats_likelihoods_2d).sum()) / (-math.log(2))
        feats_bpp = feats_size / points_num

        # Simulate packet loss on latent features
        device = latent_feats_hat_2d.device
        B, _, H, W = latent_feats_hat_2d.shape
        mask_feats_1d = Simulated_Packet_Loss(current_loss_rate, self.M, device)
        mask_feats_4d = mask_feats_1d.view(1, self.M, 1, 1).expand(B, self.M, H, W)
        latent_feats_hat_2d_loss = latent_feats_hat_2d * mask_feats_4d

        # inverse interleave and reshape to 1D
        latent_feats_hat_2d_chunk = latent_feats_hat_2d_loss.chunk(self.M // 4, dim=1)
        inv_swapped_chunks_feats = [inv_swap_element(chunk) for chunk in latent_feats_hat_2d_chunk]
        latent_feats_hat_2d_inv = torch.cat(inv_swapped_chunks_feats, dim=1)
        mask_feats_chunk = mask_feats_4d.chunk(self.M // 4, dim=1)
        inv_swapped_mask_feats = [inv_swap_element(chunk) for chunk in mask_feats_chunk]
        mask_feats_inv = torch.cat(inv_swapped_mask_feats, dim=1)

        #restore and refine features 
        B, C_f, H, W = latent_feats_hat_2d_inv.shape
        latent_feats_1d = latent_feats_hat_2d_inv.view(B, C_f, -1)[:, :, :points]
        mask_feats_1d = mask_feats_inv.view(B, C_f, -1)[:, :, :points]
        restored_feats_1d = self.restoration_net(
            xyz=pred_latent_xyzs,       
            feats_corrupted=latent_feats_1d, 
            mask=mask_feats_1d          
        )
        restored_feats_1d,usage = self.refinement_net(restored_feats_1d)
        del latent_feats_hat_2d
        del latent_feats_hat_2d_inv
        del scales_hat_feats_2d, means_hat_feats_2d

        #decoding
        pred_xyzs, pred_unums, pred_mdis,pred_std, upsampled_feats = self.decoder(pred_latent_xyzs, restored_feats_1d)

        #loss
        loss, loss_items, all_pred2gt_idx = self.get_loss(gt_xyzs,gt_normals, gt_dnums, gt_mdis,gt_std, pred_xyzs, pred_unums, pred_mdis,pred_std)

        p = current_loss_rate / 100.0   
        fec_factor = 1.0 / (1.0 - p + 1e-6)

        loss_xyz_bpp = xyzs_bpp * self.args.bpp_lambda * fec_factor 
        loss_feats_bpp = feats_bpp * self.args.bpp_lambda
        loss_zbpp = z_bpp * self.args.bpp_lambda * fec_factor
        bpp_loss = loss_xyz_bpp + loss_feats_bpp + loss_zbpp
        loss = loss + bpp_loss
        loss_items['bpp_loss'] = bpp_loss.item()
        total_real_bpp = feats_bpp + xyzs_bpp *fec_factor + z_bpp * fec_factor
        loss_items['real_bpp'] = total_real_bpp.item()  
        loss_items['xyz_bpp'] = xyzs_bpp.item() * fec_factor          
        loss_items['feat_bpp'] = feats_bpp.item()        

        decompressed_xyzs = pred_xyzs[-1]

        return decompressed_xyzs, loss, loss_items, total_real_bpp

    def load_state_dict(self, state_dict, strict=True):
        for name, module in self.named_modules():
            if not any(x.startswith(name) for x in state_dict.keys()):
                continue

            if isinstance(module, EntropyBottleneck):
                update_registered_buffers(
                    module, name, ["_quantized_cdf", "_offset", "_cdf_length"],
                    state_dict,
                )

            if isinstance(module, GaussianConditional):
                update_registered_buffers(
                    module, name, ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
                    state_dict,
                )

        return super().load_state_dict(state_dict, strict=strict)

    def update(self, scale_table_feats=None, force=True):

        if scale_table_feats is None:
            scale_table_feats = np.exp(np.linspace(np.log(0.001), np.log(64), 512))


        device = next(self.parameters()).device
        scale_table_feats = torch.from_numpy(scale_table_feats).float().to(device)

        updated = True  
        
        updated |= self.z_eblock.update(force=force)

        updated |= self.gaussian_conditional_feats.update_scale_table(scale_table_feats, force=force)
        
        print("Entropy models updated with custom scale tables.")
        return updated

            
