import numpy as np
import os
import torch
from einops import rearrange
import open3d as o3d
import random
import argparse
import pickle
import torch.nn as nn
import struct
import math
import torch.nn.functional as F



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
   
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = 1.0 * self.sum / self.count

    def get_avg(self):
        return self.avg

    def get_count(self):
        return self.count


def index_points(xyzs, idx):
    """
    Input:
        xyzs: input points data, [B, C, N]
        idx: sample index data, [B, S, [K]] 
    Return:
        new_points:, indexed points data, [B, C, S, [K]]
    """
    batch_size = idx.shape[0]
    sample_num = idx.shape[1]
    fdim = xyzs.shape[1]

    reshape = False
    if len(idx.shape) == 3:
        reshape = True
        idx = idx.reshape(batch_size, -1) #[B,S[K]] 

    # (b, c, (s k))
    res = torch.gather(xyzs, 2, idx[:, None].repeat(1, fdim, 1))

    if reshape:
        res = rearrange(res, 'b c (s k) -> b c s k', s=sample_num)

    return res


def save_pcd(dir, name, xyzs, normals=None):
    # input: (n, 3)
    path = os.path.join(dir, name)
    f = open(path, 'w')
    f.write("ply\n") 
    f.write("format ascii 1.0\n") 
    f.write("element vertex " + str(xyzs.shape[0]) + "\n") 
    f.write("property float x\n") 
    f.write("property float y\n")
    f.write("property float z\n")
    if isinstance(normals, np.ndarray):
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
    f.write("element face 0\n") 
    f.write("property list uchar int vertex_indices\n")
    f.write("end_header\n")
    f.close()


    with open(path, 'ab') as f:  
        if isinstance(normals, np.ndarray):
            # (n, 6)
            xyzs_and_normals = np.concatenate((xyzs, normals), axis=1)
            np.savetxt(f, xyzs_and_normals, fmt='%s')
        else:
            np.savetxt(f, xyzs, fmt='%s')



def str2bool(val):
    if isinstance(val, bool): 
        return val
    if val.lower() in ['yes', 'true', 't', 'y']:
        return True
    elif val.lower() in ['no', 'false', 'f', 'n']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def _load_lookup_table(path="hw_lookup.pkl"):
    # hw_lookup.pkl is a dictionary of M to (H, W)
    try:
        with open(path, 'rb') as f:
            table = pickle.load(f)
        print("Lookup table loaded successfully.")
        return table
    except FileNotFoundError:
        print(f"Error: Lookup table file not found at '{path}'")
        raise

HW_LOOKUP_TABLE = _load_lookup_table()


def pad_and_reshape(tensor_1x64xM):
    """
    input: tensor_1x64xM  (torch.Tensor)
    output: padded_tensor_1x64xHxW, mask, (H, W)
    """
    B, C, M = tensor_1x64xM.shape

    if M not in HW_LOOKUP_TABLE:
        raise KeyError(f"M={M} not in lookup table")

    H, W = HW_LOOKUP_TABLE[M]
    pad_len = H * W - M

    # padding mask
    mask = torch.zeros(H * W, dtype=torch.bool)
    mask[:M] = True

    # padding 
    padded = torch.nn.functional.pad(tensor_1x64xM, (0, pad_len), mode="constant", value=0.0)

    # reshape
    padded = padded.view(1, C, H, W)

    return padded


def swap_element(tensor):
    #interleave
    tensor_return = tensor.clone()

    tensor_return[:, 0, 0::2, 1::2] = tensor[:, 1, 0::2, 0::2]
    tensor_return[:, 0, 1::2, 0::2] = tensor[:, 2, 0::2, 0::2]
    tensor_return[:, 0, 1::2, 1::2] = tensor[:, 3, 0::2, 0::2]

    tensor_return[:, 1, 0::2, 0::2] = tensor[:, 0, 0::2, 1::2]
    tensor_return[:, 1, 1::2, 0::2] = tensor[:, 2, 0::2, 1::2]
    tensor_return[:, 1, 1::2, 1::2] = tensor[:, 3, 0::2, 1::2]

    tensor_return[:, 2, 0::2, 0::2] = tensor[:, 0, 1::2, 0::2]
    tensor_return[:, 2, 0::2, 1::2] = tensor[:, 1, 1::2, 0::2]
    tensor_return[:, 2, 1::2, 1::2] = tensor[:, 3, 1::2, 0::2]

    tensor_return[:, 3, 0::2, 0::2] = tensor[:, 0, 1::2, 1::2]
    tensor_return[:, 3, 0::2, 1::2] = tensor[:, 1, 1::2, 1::2]
    tensor_return[:, 3, 1::2, 0::2] = tensor[:, 2, 1::2, 1::2]

    return tensor_return

def inv_swap_element(tensor_swaped):
    tensor_return = tensor_swaped.clone()
    tensor_return[:, 1, 0::2, 0::2] = tensor_swaped[:, 0, 0::2, 1::2]
    tensor_return[:, 2, 0::2, 0::2] = tensor_swaped[:, 0, 1::2, 0::2]
    tensor_return[:, 3, 0::2, 0::2] = tensor_swaped[:, 0, 1::2, 1::2]

    tensor_return[:, 0, 0::2, 1::2] = tensor_swaped[:, 1, 0::2, 0::2]
    tensor_return[:, 2, 0::2, 1::2] = tensor_swaped[:, 1, 1::2, 0::2]
    tensor_return[:, 3, 0::2, 1::2] = tensor_swaped[:, 1, 1::2, 1::2]

    tensor_return[:, 0, 1::2, 0::2] = tensor_swaped[:, 2, 0::2, 0::2]
    tensor_return[:, 1, 1::2, 0::2] = tensor_swaped[:, 2, 0::2, 1::2]
    tensor_return[:, 3, 1::2, 0::2] = tensor_swaped[:, 2, 1::2, 1::2]

    tensor_return[:, 0, 1::2, 1::2] = tensor_swaped[:, 3, 0::2, 0::2]
    tensor_return[:, 1, 1::2, 1::2] = tensor_swaped[:, 3, 0::2, 1::2]
    tensor_return[:, 2, 1::2, 1::2] = tensor_swaped[:, 3, 1::2, 0::2]
    return tensor_return


#[1,3,n]->[1,n]
def get_z_order_indices_torch(xyzs):
    #get z-order indices
    B, _, N = xyzs.shape
    coords = xyzs.permute(0, 2, 1)
    coords_min, _ = torch.min(coords, dim=1, keepdim=True)
    coords_max, _ = torch.max(coords, dim=1, keepdim=True)
    scale = coords_max - coords_min
    scale[scale == 0] = 1
    coords_normalized = (coords - coords_min) / scale
    coords_int = (coords_normalized * ((1 << 10) - 1)).to(torch.int64)

    def spread_bits(x):
        x = (x | (x << 10)) & 0x000F801F
        x = (x | (x << 4))  & 0x00E181C3
        x = (x | (x << 2))  & 0x0249249
        x = (x | (x << 1))  & 0x049249249
        x = (x | (x << 1))  & 0x09249249249
        return x

    morton_codes = (spread_bits(coords_int[..., 0]) |
                    (spread_bits(coords_int[..., 1]) << 1) |
                    (spread_bits(coords_int[..., 2]) << 2))
    sorted_indices = torch.argsort(morton_codes, dim=1)
    return sorted_indices

class LowerBound(nn.Module):
    def __init__(self, bound: float):
        super().__init__()
        self.register_buffer("bound", torch.Tensor([bound]))

    def forward(self, x):
        return torch.max(x, self.bound)


def load_loss_trace(file_path):
    # Load loss trace from a text file, where the content is a list of integers in string format, e.g., "[0, 1, 2, 3]"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Loss trace file not found: {file_path}")
        
    with open(file_path, 'r') as f:
        content = f.read().strip()
        content = content.replace('[', '').replace(']', '')
        trace = [int(x.strip()) for x in content.split(',') if x.strip()]
        
    print(f"Loaded loss trace from {file_path}, length: {len(trace)}")
    return trace