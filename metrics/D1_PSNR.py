import torch
import os
from models.utils import index_points
from models.pointops.functions import pointops

def sum_d1(p1, p2):
    """
    compute squared L2 distance sum between two point sets
    p1, p2: (N,3) tensors of points
    """
    return torch.sum(torch.sum((p1 - p2) ** 2, dim=1))

def psnr(peak_point, mse):
    """
    compute PSNR given peak point and MSE
    returns 0 if MSE is zero or NaN
    """
    if mse <= 0 or torch.isnan(mse):
        return torch.tensor(0.0).to(mse.device) 
    max_energy = peak_point * peak_point
    return 10 * torch.log10(max_energy / mse)

def get_d1_psnr(gt_xyzs, pred_xyzs, args):
    """
    compute D1 PSNR between ground-truth and predicted points
    gt_xyzs: (1,N,3) tensor, ground-truth point cloud
    pred_xyzs: (1,M,3) tensor, predicted point cloud
    args.peak_point: peak point value for PSNR calculation
    returns: D1 PSNR (scalar tensor)
    """
    peak_point = args.peak_point

    # nearest neighbor indices for both directions
    gt2pred_idx = pointops.knnquery_heap(1, pred_xyzs, gt_xyzs).view(-1).long()
    pred2gt_idx = pointops.knnquery_heap(1, gt_xyzs, pred_xyzs).view(-1).long()

    # remove batch dimension
    gt_xyzs = gt_xyzs.squeeze(0)      
    pred_xyzs = pred_xyzs.squeeze(0)  

    # map each point to nearest neighbor
    gt_nearest_xyzs = pred_xyzs[gt2pred_idx, :] 
    pred_nearest_xyzs = gt_xyzs[pred2gt_idx, :] 

    # compute D1 MSE in both directions
    d1_sum_gt2pred = sum_d1(gt_xyzs, gt_nearest_xyzs)
    d1_mse_gt2pred = d1_sum_gt2pred / gt_xyzs.shape[0]
    d1_sum_pred2gt = sum_d1(pred_xyzs, pred_nearest_xyzs)
    d1_mse_pred2gt = d1_sum_pred2gt / pred_xyzs.shape[0]

    # select maximum MSE for PSNR calculation
    d1_max_mse = max(d1_mse_gt2pred, d1_mse_pred2gt)
    
    # compute PSNR
    d1_psnr = psnr(peak_point, d1_max_mse)

    return d1_psnr