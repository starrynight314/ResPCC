import torch
from torch import nn
from models.utils import index_points
from models.Chamfer3D.dist_chamfer_3D import chamfer_3DDist
chamfer_dist = chamfer_3DDist()


def get_chamfer_loss(gt_xyzs, pred_xyzs, args):
    # list: (b, 3, n)
    chamfer_loss = 0.0
    all_pred2gt_idx = []

    # for each stage
    for i in range(args.layer_num):
        d1, d2, _, pred2gt_idx = chamfer_dist(gt_xyzs[args.layer_num-1-i].permute(0, 2, 1).contiguous(),
                                    pred_xyzs[i].permute(0, 2, 1).contiguous())

        layer_chamfer = torch.mean(d1) + torch.mean(d2)

        # (b, n)
        all_pred2gt_idx.append(pred2gt_idx.long())

        if i == args.layer_num-1:
            # final point cloud
            chamfer_loss = chamfer_loss + layer_chamfer
        else:
            # intermediate point cloud
            chamfer_loss = chamfer_loss + args.chamfer_coe * layer_chamfer

    return chamfer_loss, all_pred2gt_idx



def get_density_loss(gt_dnums, gt_mdis, gt_stds,pred_unums, pred_mdis, pred_stds,all_pred2gt_idx, args):
    # input: (b, n)

    l1_loss = nn.L1Loss(reduction='mean') 
    density_loss = 0.0

    for i in range(args.layer_num): 
        cur_pred_unum = pred_unums[i] 
        cur_gt_dnum = gt_dnums[args.layer_num-1-i]
        
        cur_pred_std  = pred_stds[i] 
        cur_gt_std  = gt_stds[args.layer_num-1-i] 

        cur_pred_mdis = pred_mdis[i]
        cur_gt_mdis = gt_mdis[args.layer_num-1-i]

        if i == 0: 
            cur_nearest_dnum = cur_gt_dnum
            cur_nearest_mdis = cur_gt_mdis
            cur_nearest_std  = cur_gt_std
        else:
            cur_nearest_dnum = index_points(cur_gt_dnum.unsqueeze(1), all_pred2gt_idx[i-1]).squeeze(1) 
            cur_nearest_mdis = index_points(cur_gt_mdis.unsqueeze(1), all_pred2gt_idx[i-1]).squeeze(1)
            cur_nearest_std  = index_points(cur_gt_std.unsqueeze(1), all_pred2gt_idx[i-1]).squeeze(1)

        cur_upsample_num_loss = l1_loss(cur_pred_unum, cur_nearest_dnum)
        cur_mean_distance_loss = l1_loss(cur_pred_mdis, cur_nearest_mdis)
        cur_std_loss = l1_loss(cur_pred_std, cur_nearest_std)

        cur_density_loss = cur_upsample_num_loss + args.mean_distance_coe * cur_mean_distance_loss+ args.std_distance_coe * cur_std_loss
        density_loss = density_loss + cur_density_loss

    density_loss = density_loss * args.density_coe

    return density_loss



def get_pts_num_loss(gt_xyzs, pred_unums, args):
    # input: list
    batch_size = gt_xyzs[0].shape[0]
    pts_num_loss = 0.0

    for i in range(args.layer_num):
        cur_pts_num_loss = torch.abs(pred_unums[i].sum() - gt_xyzs[args.layer_num-1-i].shape[2]*batch_size)
        pts_num_loss = pts_num_loss + cur_pts_num_loss

    pts_num_loss = pts_num_loss * args.pts_num_coe

    return pts_num_loss



