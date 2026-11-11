import math
import os
import struct
os.environ['CUDA_VISIBLE_DEVICES'] = '7' 
import torch
import sys
sys.path.append(os.getcwd())
import numpy as np
from models.autoencoder import AutoEncoder
import time
import argparse
from args.shapenet_args import parse_shapenet_args
from args.semantickitti_args import parse_semantickitti_args
from models.utils import save_pcd, AverageMeter, str2bool, get_z_order_indices_torch, pad_and_reshape, swap_element, inv_swap_element, load_loss_trace
from dataset.dataset import CompressDataset
from metrics.PSNR import get_psnr
from metrics.D1_PSNR import get_d1_psnr
from models.Chamfer3D.dist_chamfer_3D import chamfer_3DDist
import logging
from plyfile import PlyData
chamfer_dist = chamfer_3DDist()


def make_dirs(save_dir):
    # Create folders for GT/pred patches and merged point clouds
    gt_patch_dir = os.path.join(save_dir, 'patch/gt')
    os.makedirs(gt_patch_dir, exist_ok=True)
    pred_patch_dir = os.path.join(save_dir, 'patch/pred')
    os.makedirs(pred_patch_dir, exist_ok=True)
    gt_merge_dir = os.path.join(save_dir, 'merge/gt')
    os.makedirs(gt_merge_dir, exist_ok=True)
    pred_merge_dir = os.path.join(save_dir, 'merge/pred')
    os.makedirs(pred_merge_dir, exist_ok=True)
    return gt_patch_dir, pred_patch_dir, gt_merge_dir, pred_merge_dir

def load_model(args, model_path):
    # Load pretrained model
    model = AutoEncoder(args).cuda()
    checkpoint = torch.load(model_path, map_location='cuda')
    model_state = checkpoint['model_state_dict']
    model.load_state_dict(model_state, strict=False) 
    model.update() 
    model.eval()
    return model

def setup_logger(name, log_file, level=logging.INFO):
    # Setup logger to write messages with timestamp
    handler = logging.FileHandler(log_file, mode='a')        
    handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


def compress(args, model, xyzs, feats, test_loss_rate):
    torch.cuda.synchronize()
    B, _, N_orig = xyzs.shape # B=1
    device = xyzs.device
    p_tensor = torch.tensor([[test_loss_rate / 100.0]], device=xyzs.device).expand(B, 1) 
    temp_dir = "temp "
    os.makedirs(temp_dir, exist_ok=True)
    
    # encode
    feats = model.pre_conv(feats)
    gt_xyzs, gt_dnums, gt_mdis, gt_std, l_xyzs_ori, l_feats_ori = model.encoder(xyzs, feats,p_tensor)
    points_m = l_xyzs_ori.shape[2] 

    # Z-order sorting
    sorted_indices = get_z_order_indices_torch(l_xyzs_ori) 
    s_idx_xyz = sorted_indices.unsqueeze(1).expand(B, 3, points_m)
    s_idx_feat = sorted_indices.unsqueeze(1).expand(B, model.M, points_m)
    l_xyzs = torch.gather(l_xyzs_ori, 2, s_idx_xyz)
    l_feats = torch.gather(l_feats_ori, 2, s_idx_feat)

    # save xyzs as half precision binary
    pred_l_xyzs = l_xyzs.half() 
    xyz_path = os.path.join(temp_dir, "xyz.bin")
    with open(xyz_path, "wb") as f:
        f.write(pred_l_xyzs.cpu().numpy().tobytes())

    # reshape to 2D and interleave
    y_feat_2d = pad_and_reshape(l_feats)
    y_feat_2d = torch.cat([swap_element(c) for c in y_feat_2d.chunk(model.M//4, dim=1)], dim=1)

    # hyperprior encoding and save compressed strings
    z = model.h_a(y_feat_2d) 
    z_strings = model.z_eblock.compress(z)
    z_shape = z.size()[2:] # [H_z, W_z]
    meta_path = os.path.join(temp_dir, "meta.bin")
    with open(meta_path, "wb") as f:
        f.write(struct.pack(">IHH", points_m, z_shape[0], z_shape[1]))
        for s in z_strings:
            f.write(s)

    # generate means and scales for feature compression
    z_hat = model.z_eblock.decompress(z_strings, z_shape)
    params = model.h_s_main(z_hat)
    f_means = model.h_s_feats_means_head(params)
    f_scales = model.h_s_feats_scales_head(params)
    f_idx = model.gaussian_conditional_feats.build_indexes(f_scales)

    # compress features into packets and save
    feat_packet_paths = []
    for k in range(32):
        i = k * 2
        t0 = time.time()
        s = model.gaussian_conditional_feats.compress(
            y_feat_2d[:, i:i+2, :, :], f_idx[:, i:i+2, :, :], f_means[:, i:i+2, :, :]
        )

        pkt_path = os.path.join(temp_dir, f"feats_p{k}.bin")
        with open(pkt_path, "wb") as f:
            header = struct.pack(">BH", k, len(s[0]))
            f.write(header)
            f.write(s[0])
        feat_packet_paths.append(pkt_path)

    # calculate BPP
    p = test_loss_rate / 100.0
    fec_multiplier = 1.0 / (1.0 - p + 1e-6) 
    base_size_bytes = os.path.getsize(xyz_path) + os.path.getsize(meta_path)
    protected_base_bits = base_size_bytes * 8 * fec_multiplier
    total_feat_bits = 0
    for pkt_path in feat_packet_paths:
        total_feat_bits += os.path.getsize(pkt_path) * 8
    total_bits = protected_base_bits + total_feat_bits
    actual_bpp = total_bits / N_orig
    xyz_bpp = protected_base_bits / N_orig 
    feats_bpp = total_feat_bits / N_orig      


    compressed_data = {
        "xyz_bin_path": xyz_path,
        "meta_bin_path": meta_path,
        "feat_bin_paths": feat_packet_paths,
        "points_m": points_m,
        "z_shape": z_shape,
    }
    
    return compressed_data, actual_bpp, xyz_bpp, feats_bpp

def simulate_loss_and_load_feats(feat_packet_paths, loss_trace, global_trace_idx):
    # Simulate packet loss and load surviving feature packets
    survived_data = {}
    loss_mask = []
    num_packets = len(feat_packet_paths) 
    ptr = global_trace_idx
    for pkt_id in range(num_packets):
        is_lost = loss_trace[ptr % len(loss_trace)]
        ptr += 1
        
        if is_lost == 0:
            path = feat_packet_paths[pkt_id]
            if os.path.exists(path):
                with open(path, "rb") as f:
                    header = f.read(3)
                    actual_id, data_len = struct.unpack(">BH", header)
                    payload = f.read(data_len)
                    survived_data[actual_id] = payload
                    loss_mask.append(1.0)
            else:
                loss_mask.append(0.0) 
        else:
            loss_mask.append(0.0)
            
    return survived_data, loss_mask, ptr

def decompress(args, model, compressed_data, loss_trace, global_trace_idx):
    device = next(model.parameters()).device
    B = 1 # batch size
    #get hyperprior
    meta_path = compressed_data["meta_bin_path"]
    with open(meta_path, "rb") as f:
        header = f.read(8)
        points_m, z_h, z_w = struct.unpack(">IHH", header)
        z_shape = (z_h, z_w)
        z_strings = [f.read()]
    #get xyz
    xyz_path = compressed_data["xyz_bin_path"]
    with open(xyz_path, "rb") as f:
        xyz_bytes = f.read()
        pred_l_xyzs_np = np.frombuffer(xyz_bytes, dtype=np.float16).copy()
        pred_l_xyzs = torch.from_numpy(pred_l_xyzs_np).to(device).float().view(B, 3, points_m)

    #generate means and scales for feature decompression
    z_hat = model.z_eblock.decompress(z_strings, z_shape)
    shared_params = model.h_s_main(z_hat)
    f_means = model.h_s_feats_means_head(shared_params)
    f_scales = model.h_s_feats_scales_head(shared_params)
    f_idx = model.gaussian_conditional_feats.build_indexes(f_scales)
    
    #simulate packet loss and load surviving feature packets
    _, _, H_feat, W_feat = f_means.shape 
    survived_feats, pkt_loss_mask, global_trace_idx = simulate_loss_and_load_feats(
        compressed_data["feat_bin_paths"], loss_trace, global_trace_idx
    )
    y_feats_hat_2d = torch.zeros_like(f_means)
    mask_feats_2d = torch.zeros_like(f_means)
    # decompress surviving packets
    for k in range(32):
        ch_start = k * 2
        ch_end = ch_start + 2
        
        if k in survived_feats:
            s_part = [survived_feats[k]]
            y_part = model.gaussian_conditional_feats.decompress(
                s_part, f_idx[:, ch_start:ch_end, :, :], 
                means=f_means[:, ch_start:ch_end, :, :]
            )
            if y_part.dim() == 3:
                y_part = y_part.unsqueeze(0)
            y_feats_hat_2d[:, ch_start:ch_end, :, :] = y_part
            mask_feats_2d[:, ch_start:ch_end, :, :] = 1.0
        else:
            y_feats_hat_2d[:, ch_start:ch_end, :, :] = 0.0
            mask_feats_2d[:, ch_start:ch_end, :, :] = 0.0
    # inverse interleave and reshape to 1D
    y_feats_hat_2d_chunk = y_feats_hat_2d.chunk(model.M // 4, dim=1)
    y_feats_inv = torch.cat([inv_swap_element(chunk) for chunk in y_feats_hat_2d_chunk], dim=1)
    mask_chunk = mask_feats_2d.chunk(model.M // 4, dim=1)
    mask_inv = torch.cat([inv_swap_element(chunk) for chunk in mask_chunk], dim=1)
    latent_feats_1d = y_feats_inv.view(B, model.M, -1)[:, :, :points_m]
    mask_feats_1d = mask_inv.view(B, model.M, -1)[:, :, :points_m]
    #restore and refinefeatures and decode
    restored_feats = model.restoration_net(xyz=pred_l_xyzs, feats_corrupted=latent_feats_1d, mask=mask_feats_1d)
    restored_feats, _ = model.refinement_net(restored_feats)
    pred_xyzs, _, _, _, _ = model.decoder(pred_l_xyzs, restored_feats)
    final_pcd = pred_xyzs[-1]
    
    return final_pcd, global_trace_idx

def test_xyzs(args):
    test_dataset = CompressDataset(
        data_path=args.test_data_path,
        map_size=args.test_map_size,  
        cube_size=args.test_cube_size,  
        batch_size=1   
    )
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    pcd_last_patch_num = test_dataset.pcd_last_patch_num
    total_patches_num = len(test_loader)

    # Paths
    model_path = args.model_path
    save_dir = os.path.join(args.results, '')
    save_dir_forlog = os.path.join(args.results, '')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir_forlog, exist_ok=True)
    gt_patch_dir, pred_patch_dir, gt_merge_dir, pred_merge_dir = make_dirs(save_dir)
    patch_logger = setup_logger('patch_logger', os.path.join(save_dir_forlog, 'patch_metrics.log'))
    pcd_logger = setup_logger('pcd_logger', os.path.join(save_dir_forlog, 'full_pcd_metrics.log'))
    patch_logger.info(f"--- Starting test ---")

    # Trace
    loss_file_path = "./exp/gilbert_channel/30/channel0.txt" 
    if not os.path.exists(loss_file_path):
        loss_file_path = args.loss_file if hasattr(args, 'loss_file') else loss_file_path
    loss_trace = load_loss_trace(loss_file_path)
    global_trace_idx = 0

    # Load model
    model = load_model(args, model_path)
    
    pcd_metrics_storage = {
        'bpp': [],
        'bpp_xyz': [],
        'bpp_feats': [],
        'chamfer': [],
        'psnr': [],
        'psnr_d1': []
    }
    # Metrics
    patch_bpp = AverageMeter()
    patch_xyz_bpp=AverageMeter()
    patch_feats_bpp=AverageMeter()
    patch_chamfer_loss = AverageMeter()

    pcd_num = 0
    pcd_bpp_mean = AverageMeter()
    pcd_bpp_mean_xyz=AverageMeter()
    pcd_bpp_mean_feats=AverageMeter()
    pcd_chamfer_loss = AverageMeter()
    pcd_psnr = AverageMeter()
    pcd_psnr_d1 = AverageMeter()

    pcd_gt_patches = []
    pcd_pred_patches = []
    pcd_gt_normal_patches = []
    
    # Global BPP (for final summary)
    global_bpp_meter = AverageMeter()
    global_bpp_meter_xyz = AverageMeter()
    global_bpp_meter_feats = AverageMeter()

    with torch.no_grad():
        for i, input_dict in enumerate(test_loader):
            try:
                input_tensor = input_dict['xyzs'].cuda()
                gt_normals = input_dict['normals'].cuda() 
                input_tensor = input_tensor.permute(0, 2, 1).contiguous() 
                xyzs = input_tensor[:, :3, :].contiguous()
                feats = input_tensor 
                device = xyzs.device
            
                current_file_name = input_dict['file_name'][0]
                points_num = input_tensor.shape[0] * xyzs.shape[2]
                    
                # Compress
                compressed_data, actual_bpp ,xyz_bpp,feats_bpp= compress(args, model, xyzs, feats,args.loss_rate)
                patch_bpp.update(actual_bpp)
                patch_xyz_bpp.update(xyz_bpp)
                patch_feats_bpp.update(feats_bpp)
                pcd_bpp_mean.update(actual_bpp, points_num)
                pcd_bpp_mean_xyz.update(xyz_bpp, points_num)
                pcd_bpp_mean_feats.update(feats_bpp, points_num)

                # Decompress
                pred_patches, global_trace_idx = decompress(
                    args, model, compressed_data,loss_trace, global_trace_idx
                )

                # Metrics
                gt_patches_metric = xyzs.permute(0, 2, 1).contiguous()
                pred_patches_metric = pred_patches.permute(0, 2, 1).contiguous()
            
                gt2pred_loss, pred2gt_loss, _, _ = chamfer_dist(gt_patches_metric, pred_patches_metric)
                chamfer_loss = gt2pred_loss.mean() + pred2gt_loss.mean()
                patch_chamfer_loss.update(chamfer_loss.item())

                # Log
                print("patch:", i, "patch bpp:", actual_bpp, "chamfer loss:", chamfer_loss.item(),
                  "xyz_bpp:", xyz_bpp,"feats bpp:", feats_bpp)
                patch_log_str = (
                f"patch: {i}, "
                f"patch bpp: {actual_bpp:.6f}, "
                f"chamfer loss: {chamfer_loss.item():.6f}, "
                f"xyz bpp: {xyz_bpp:.6f},"
                f"feats bpp: {feats_bpp:.6f},"
                )
                patch_logger.info(patch_log_str)

                # Cache for Merge
                original_gt_patches = test_dataset.scale_to_origin(gt_patches_metric.detach().cpu(), i).squeeze(0).numpy()
                original_pred_patches = test_dataset.scale_to_origin(pred_patches_metric.detach().cpu(), i).squeeze(0).numpy()
                original_gt_normals = gt_normals.detach().cpu().squeeze(0).numpy()

                pcd_gt_patches.append(original_gt_patches)
                pcd_pred_patches.append(original_pred_patches)
                pcd_gt_normal_patches.append(original_gt_normals)

                # Merge
                if i == pcd_last_patch_num[pcd_num] - 1:
                    gt_pcd_np = np.concatenate(pcd_gt_patches, axis=0)
                    pred_pcd_np = np.concatenate(pcd_pred_patches, axis=0)
                    gt_normal_np = np.concatenate(pcd_gt_normal_patches, axis=0)

                    gt_pcd_tensor = torch.from_numpy(gt_pcd_np).unsqueeze(0).float().cuda()
                    pred_pcd_tensor = torch.from_numpy(pred_pcd_np).unsqueeze(0).float().cuda()
                    gt_normal_tensor = torch.from_numpy(gt_normal_np).unsqueeze(0).float().cuda()

                    gt2pred_loss_full, pred2gt_loss_full, _, _ = chamfer_dist(gt_pcd_tensor, pred_pcd_tensor)
                    pcd_actual_chamfer = gt2pred_loss_full.mean().item() + pred2gt_loss_full.mean().item()
                    pcd_actual_psnr = get_psnr(gt_pcd_tensor, gt_normal_tensor, pred_pcd_tensor, args).item()
                    pcd_actual_psnr_d1 = get_d1_psnr(gt_pcd_tensor, pred_pcd_tensor, args).item()

                    pcd_chamfer_loss.update(pcd_actual_chamfer)
                    pcd_psnr.update(pcd_actual_psnr)
                    pcd_psnr_d1.update(pcd_actual_psnr_d1)
                    global_bpp_meter.update(pcd_bpp_mean.get_avg())
                    global_bpp_meter_xyz.update(pcd_bpp_mean_xyz.get_avg())
                    global_bpp_meter_feats.update(pcd_bpp_mean_feats.get_avg())
                    
                    pcd_metrics_storage['bpp'].append(pcd_bpp_mean.get_avg())
                    pcd_metrics_storage['bpp_xyz'].append(pcd_bpp_mean_xyz.get_avg())
                    pcd_metrics_storage['bpp_feats'].append(pcd_bpp_mean_feats.get_avg())
                    pcd_metrics_storage['chamfer'].append(pcd_actual_chamfer)
                    pcd_metrics_storage['psnr'].append(pcd_actual_psnr)
                    pcd_metrics_storage['psnr_d1'].append(pcd_actual_psnr_d1)

                    print("pcd:", current_file_name, "pcd bpp:", pcd_bpp_mean.get_avg(),  "xyz bpp:", pcd_bpp_mean_xyz.get_avg(), "feats bpp:", pcd_bpp_mean_feats.get_avg(),"pcd chamfer loss:", pcd_actual_chamfer,
                         "pcd psnr:", pcd_actual_psnr, "pcd psnr_d1:", pcd_actual_psnr_d1)
                    log_message_pcd = (
                        f"pcd: {current_file_name}, "
                        f"pcd bpp: {pcd_bpp_mean.get_avg():.10f}, "
                        f"xyz bpp: {pcd_bpp_mean_xyz.get_avg():.10f}, "
                        f"feats bpp: {pcd_bpp_mean_feats.get_avg():.10f}, "
                        f"pcd chamfer loss: {pcd_actual_chamfer:.10f}, "
                        f"pcd psnr: {pcd_actual_psnr:.10f}, "
                        f"pcd psnr_d1: {pcd_actual_psnr_d1:.10f} "

                    )   
                    pcd_logger.info(log_message_pcd)
                    save_pcd(gt_merge_dir, f'{current_file_name}.ply', gt_pcd_np,gt_normal_np)
                    save_pcd(pred_merge_dir, f'{current_file_name}.ply', pred_pcd_np)

                    pcd_num += 1
                    pcd_gt_patches = []
                    pcd_pred_patches = []
                    pcd_gt_normal_patches = []
                    pcd_bpp_mean.reset() 
                    pcd_bpp_mean_xyz.reset()
                    pcd_bpp_mean_feats.reset()
            except Exception as e:
                log_message_pcd = (
                    f"!!! Error at patch {i} ({current_file_name}): {e}"
                )  
                pcd_logger.info(log_message_pcd)
                print(f"!!! Error at patch {i} ({current_file_name}): {e}")
                continue

    # Final Summary
    print("\n" + "="*60)
    print(f" FINAL TEST REPORT (Dataset: {args.dataset}) ")
    print("="*60)
    print(f"Avg Patch BPP:       {patch_bpp.get_avg():.4f}")
    print(f"Avg Patch CD:        {patch_chamfer_loss.get_avg():.6f}")
    print(f"Avg Whole PCD BPP:   {global_bpp_meter.get_avg():.10f}")
    print(f"Avg Whole PCD BPP xyz:   {global_bpp_meter_xyz.get_avg():.10f}")
    print(f"Avg Whole PCD BPP feats:   {global_bpp_meter_feats.get_avg():.10f}")
    print(f"Avg Whole PCD CD:    {pcd_chamfer_loss.get_avg():.10f}")
    print(f"Avg Whole PCD PSNR:  {pcd_psnr.get_avg():.10f}")
    print(f"Avg Whole PCD PSNR D1:  {pcd_psnr_d1.get_avg():.10f}")


def reset_model_args(test_args, model_args):
    for arg in vars(test_args):
        setattr(model_args, arg, getattr(test_args, arg))


def parse_test_args():
    parser = argparse.ArgumentParser(description='Test Arguments')

    # dataset
    parser.add_argument('--dataset', default='shapenet', type=str, help='shapenet or semantickitti')
    parser.add_argument('--model_path', default='', type=str, help='path to ckpt')
    parser.add_argument('--batch_size', default=1, type=int, help='the test batch_size must be 1')
    parser.add_argument('--downsample_rate', default=[1/5, 1/5, 1/5], nargs='+', type=float, help='downsample rate')
    parser.add_argument('--max_upsample_num', default=[16, 16, 16], nargs='+', type=int, help='max upsmaple number, reversely symmetric with downsample_rate')
    parser.add_argument('--bpp_lambda', default=5e-4, type=float, help='bpp loss coefficient')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    test_args = parse_test_args()
    assert test_args.dataset in ['shapenet', 'semantickitti']
    # the test batch_size must be 1
    assert test_args.batch_size == 1

    if test_args.dataset == 'shapenet':
        model_args = parse_shapenet_args()
    else:
        model_args = parse_semantickitti_args()

    reset_model_args(test_args, model_args)

    if model_args.compress_normal == False:
        test_xyzs(model_args)
