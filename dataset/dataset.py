
import glob
import torch
import numpy as np
import os
from torch.utils.data import Dataset
import pickle as pkl
import random
import shelve

class ShelveWrapper:
    def __init__(self, db_path):
        self.db = shelve.open(db_path, flag='r')
        # 预取所有的 key，并转回 int，因为 init_data 可能需要用 int 进行排序或计算
        try:
            self._keys = sorted([int(k) for k in self.db.keys()])
        except ValueError:
            self._keys = list(self.db.keys())

    def __getitem__(self, key):
        return self.db[str(key)]

    def keys(self):
        return self._keys

    def __contains__(self, key):
        return str(key) in self.db


class CompressDataset(Dataset):
    def __init__(self, data_path, map_size=100, cube_size=12, batch_size=1, points_num=1024):
        """
        参数说明:
        map_size: 必须与数据准备(prepare_data.py)时使用的 map_size 一致
        cube_size: 必须与数据准备时的一致
        """
        self.data_path = data_path
        with open(self.data_path, 'rb') as f:
            self.data = pkl.load(f)
        #self.data = ShelveWrapper(self.data_path)
        self.init_data()

        self.map_size = map_size
        self.cube_size = cube_size
        # 计算网格维度，逻辑与数据准备代码保持一致
        self.dim_cube_num = np.ceil(map_size / cube_size).astype(int)
        self.batch_size = batch_size
        self.points_num = points_num
        

    def init_data(self):
        self.patch_num = [] 
        for pcd_idx in self.data.keys():
            # 获取每个点云包含的 Cube 数量
            cur_patch_num = len(self.data[pcd_idx]['points'].keys())
            self.patch_num.append(cur_patch_num)
        # 计算累积和，用于快速定位 idx 属于哪个点云
        self.pcd_last_patch_num = np.cumsum(self.patch_num)

    def get_pcd_and_patch(self, idx):
        # 查找 idx 落在哪个区间
        diff = idx + 1 - self.pcd_last_patch_num
        pcd_idx = np.where(diff <= 0)[0][0]
        if pcd_idx > 0:
            patch_idx = idx - self.pcd_last_patch_num[pcd_idx-1]
        else:
            patch_idx = idx

        return pcd_idx, patch_idx 

    def __getitem__(self, idx):
        pcd_idx, patch_idx = self.get_pcd_and_patch(idx)
        cubes = list(self.data[pcd_idx]['points'].keys())
        
        # 解析 Cube 的三维索引 (x, y, z)
        # 这里的计算依赖于 self.dim_cube_num，而它依赖于 self.map_size
        cube_x = cubes[patch_idx] // self.dim_cube_num ** 2
        cube_y  = (cubes[patch_idx] - cube_x * self.dim_cube_num ** 2) // self.dim_cube_num
        cube_z = cubes[patch_idx] % self.dim_cube_num
        
        # 计算 Cube 的中心坐标 (在 map_size 尺度下)
        center = [(cube_x + 0.5) * self.cube_size, (cube_y + 0.5) * self.cube_size, (cube_z + 0.5) * self.cube_size]
        
        # 获取点和法向量
        # 注意：这里的点坐标是在 [0, map_size] 范围内的浮点数
        xyzs = self.data[pcd_idx]['points'][cubes[patch_idx]][:, :3] 
        normals = self.data[pcd_idx]['points'][cubes[patch_idx]][:, 3:]
        
        # 归一化到 [-1, 1] 供网络输入
        # (xyz - center) 得到相对于 cube 中心的偏移
        xyzs = 2 * (xyzs - center) / self.cube_size
        
        xyzs = torch.tensor(xyzs).float()
        normals = torch.tensor(normals).float()
        
        file_name = self.data[pcd_idx].get('name', str(pcd_idx))
        file_path = self.data[pcd_idx].get('file_path', '') 
        
        input_dict = {}
        if self.batch_size == 1:
            input_dict['xyzs'] = xyzs
            input_dict['normals'] = normals
        else:
            # 随机采样以对齐 Batch 维度
            if xyzs.shape[0] >= self.points_num:
                sample_idx = random.sample(range(xyzs.shape[0]), self.points_num)
            else:
                # 如果点数不够(理论上被数据准备过滤了，但也防万一)，允许重复采样
                sample_idx = random.choices(range(xyzs.shape[0]), k=self.points_num)
                
            sample_idx = torch.tensor(sample_idx).long()
            input_dict['xyzs'] = xyzs[sample_idx, :]
            input_dict['normals'] = normals[sample_idx, :]
            
        input_dict['file_name'] = file_name
        input_dict['file_path'] = file_path
        return input_dict 

    def __len__(self):
        return sum(self.patch_num)

    def scale_to_origin(self, xyzs, idx):
        """
        将网络输出的 Patch 恢复到原始世界坐标系
        """
        pcd_idx, patch_idx = self.get_pcd_and_patch(idx)
        cubes = list(self.data[pcd_idx]['points'].keys())
        
        # 1. 重新计算中心点
        cube_x = cubes[patch_idx] // self.dim_cube_num ** 2
        cube_y  = (cubes[patch_idx] - cube_x * self.dim_cube_num ** 2) // self.dim_cube_num
        cube_z = cubes[patch_idx] % self.dim_cube_num
        
        center = [(cube_x + 0.5) * self.cube_size, (cube_y + 0.5) * self.cube_size, (cube_z + 0.5) * self.cube_size]
        center = torch.tensor(center).float().to(xyzs.device)
        
        # 2. 从 [-1, 1] 恢复到 Cube 局部坐标，再加 Center 恢复到 [0, map_size]
        xyzs = xyzs * self.cube_size / 2 + center 
        
        # 3. 
        # 从 [0, map_size] 恢复到归一化的 [0, 1]
        # 之前是 / 100，现在改为动态除以 self.map_size
        xyzs = xyzs / self.map_size 
        
        # 4. 反归一化：从 [0, 1] 恢复到原始世界坐标
        meta_data = self.data[pcd_idx]['meta_data']
        
        # 对应的归一化逻辑是 (raw - shift - min) / (max - min)
        # 所以反变换是: norm * (max - min) + min + shift
        shift = torch.tensor(meta_data['shift']).float().to(xyzs.device)
        max_coord = meta_data['max_coord']
        min_coord = meta_data['min_coord']
        
        xyzs = xyzs * (max_coord - min_coord) + min_coord + shift

        return xyzs