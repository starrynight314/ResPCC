import torch
import os
from models.utils import index_points
from models.pointops.functions import pointops

def sum_d2(p1, p2, normal):
    """
    compute squared point-to-plane distance sum
    p1, p2: (N,3) point sets
    normal: (N,3) corresponding normals for p1
    """
    return torch.sum(torch.sum((p1 - p2) * normal, dim=1) ** 2)

def psnr(peak_point, mse):
    """
    compute PSNR given peak point and MSE
    returns 0 if MSE is negative or NaN
    """
    if mse < 0 or torch.isnan(mse):
        print("error")
        return torch.tensor(0.0).to(mse.device) 
    max_energy = peak_point * peak_point
    return 10 * torch.log10(max_energy / mse)

def get_psnr(gt_xyzs, gt_normals, pred_xyzs, args):
    """
    compute D2 PSNR (point-to-plane) between ground-truth and predicted points
    gt_xyzs: (1,N,3) ground-truth points
    gt_normals: (1,N,3) ground-truth normals
    pred_xyzs: (1,M,3) predicted points
    args.peak_point: peak point value for PSNR
    returns: D2 PSNR (scalar tensor)
    """

    peak_point = args.peak_point

    # nearest neighbor indices
    gt2pred_idx = pointops.knnquery_heap(1, pred_xyzs, gt_xyzs).view(-1).long() 
    pred2gt_idx = pointops.knnquery_heap(1, gt_xyzs, pred_xyzs).view(-1).long() 

    # remove batch dimension
    gt_xyzs = gt_xyzs.squeeze(0)
    gt_normals = gt_normals.squeeze(0)
    pred_xyzs = pred_xyzs.squeeze(0)

    # map each point to nearest neighbor
    gt_nearest_xyzs = pred_xyzs[gt2pred_idx, :] 
    pred_nearest_xyzs = gt_xyzs[pred2gt_idx, :]

    # corresponding normals
    gt_nearest_normals = gt_normals
    pred_nearest_normals = gt_normals[pred2gt_idx, :]

    # compute point-to-plane MSE
    d2_sum_gt2pred = sum_d2(gt_xyzs, gt_nearest_xyzs, gt_nearest_normals)
    d2_sum_pred2gt = sum_d2(pred_xyzs, pred_nearest_xyzs, pred_nearest_normals)
    d2_mse_gt2pred = d2_sum_gt2pred / gt_xyzs.shape[0]
    d2_mse_pred2gt = d2_sum_pred2gt / pred_xyzs.shape[0]

    # select maximum MSE for PSNR calculation
    d2_max_mse = max(d2_mse_gt2pred, d2_mse_pred2gt)
    d2_psnr = psnr(peak_point, d2_max_mse)

    return d2_psnr