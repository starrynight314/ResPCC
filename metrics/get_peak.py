import os
import torch
import sys
sys.path.append(os.getcwd())
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import numpy as np
import open3d as o3d
from tqdm import tqdm
from models.pointops.functions import pointops
from models.utils import index_points

def calculate_global_pcd_peak(ply_dir):
    """
    遍历目录下的所有 .ply 文件，计算全局最大最近邻距离作为 peak
    ply_dir: 完整点云存放路径 (merge/gt/)
    """
    print(f"开始扫描目录: {ply_dir}")
    
    # 1. 获取所有 ply 文件路径
    ply_files = [f for f in os.listdir(ply_dir) if f.endswith('.ply')]
    if not ply_files:
        print("错误: 没找到任何 .ply 文件")
        return None

    global_max_dist = 0.0
    device = torch.device('cuda')

    # 2. 遍历每个完整点云
    for file_name in tqdm(ply_files, desc="计算全局 Peak"):
        file_path = os.path.join(ply_dir, file_name)
        
        # 加载点云
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points)
        
        # 转换为 Tensor 并送入 GPU: [1, N, 3]
        xyzs = torch.from_numpy(points).float().unsqueeze(0).to(device)
        
        # 3. 
        # 寻找最近的 2 个点 (index 0 是点自己，index 1 是最近邻居)
        # 对于 20w 点，这个操作在 GPU 上很快
        with torch.no_grad():
            # knnquery_heap 预期输入: (B, N, 3)
            idx = pointops.knnquery_heap(2, xyzs, xyzs)[..., 1].long() 
            
            # 提取最近邻居的坐标
            xyzs_trans = xyzs.permute(0, 2, 1).contiguous()
            nearest_xyzs = index_points(xyzs_trans, idx).permute(0, 2, 1).contiguous()
            
            # 计算距离: [1, N]
            dists = torch.norm(nearest_xyzs - xyzs, dim=-1)
            
            # 获取当前点云的最大最近邻距离
            current_max = dists.max().item()
            
            # 更新全局最大值
            if current_max > global_max_dist:
                global_max_dist = current_max

    print("-" * 30)
    print(f"处理了 {len(ply_files)} 个完整点云")
    print(f"全局 Peak 值 (Max NN Distance): {global_max_dist:.10f}")
    print("-" * 30)
    
    return global_max_dist

if __name__ == "__main__":
    # 指向你合并后的 GT 点云目录
    gt_merge_dir = "/home/niuxueqin/D-PCC/output_test_semantickitti/test10/merge/gt"
    
    # 确保此脚本能访问你的模型组件（如 pointops）
    # 如果路径不对，请手动 sys.path.append
    
    global_peak = calculate_global_pcd_peak(gt_merge_dir)
    print(global_peak)

# import os
# import numpy as np
# import open3d as o3d
# from tqdm import tqdm
# import math

# def calculate_global_diagonal_peak(ply_dir):
#     """
#     计算整个数据集的全局包围盒对角线长度作为 PSNR 的 Peak 值
#     逻辑：找出所有点云中 X, Y, Z 的全局最大最小值，计算对角线。
#     """
#     print(f"开始分析目录: {ply_dir}")
    
#     ply_files = [f for f in os.listdir(ply_dir) if f.endswith('.ply')]
#     if not ply_files:
#         print("错误: 没找到任何 .ply 文件")
#         return None

#     # 初始化全局极值
#     # 使用极大的正数和极小的负数初始化
#     global_min = np.array([float('inf'), float('inf'), float('inf')])
#     global_max = np.array([float('-inf'), float('-inf'), float('-inf')])

#     # 1. 第一遍扫描：找出坐标轴的全局范围
#     for file_name in tqdm(ply_files, desc="扫描全局坐标范围"):
#         file_path = os.path.join(ply_dir, file_name)
        
#         # 加载点云 (使用 Open3D 读取)
#         pcd = o3d.io.read_point_cloud(file_path)
#         if not pcd.has_points():
#             continue
            
#         points = np.asarray(pcd.points)
        
#         # 获取当前点云的极值
#         current_min = points.min(axis=0)
#         current_max = points.max(axis=0)
        
#         # 更新全局极值
#         global_min = np.minimum(global_min, current_min)
#         global_max = np.maximum(global_max, current_max)

#     # 2. 计算三个轴的跨度 (Range)
#     ranges = global_max - global_min
#     dx, dy, dz = ranges[0], ranges[1], ranges[2]
    
#     # 3. 计算对角线长度 (Diagonal)
#     global_diagonal = math.sqrt(dx**2 + dy**2 + dz**2)

#     print("\n" + "="*40)
#     print("📊 数据集坐标统计报告")
#     print("-"*40)
#     print(f"X 轴范围: [{global_min[0]:.4f}, {global_max[0]:.4f}] (跨度: {dx:.4f})")
#     print(f"Y 轴范围: [{global_min[1]:.4f}, {global_max[1]:.4f}] (跨度: {dy:.4f})")
#     print(f"Z 轴范围: [{global_min[2]:.4f}, {global_max[2]:.4f}] (跨度: {dz:.4f})")
#     print("-"*40)
#     print(f"🔥 全局包围盒对角线 (Global Diagonal): {global_diagonal:.10f}")
#     print(f"💡 建议在测试代码中使用的 peak_point 设定为该值。")
#     print("="*40 + "\n")
    
#     return global_diagonal

# if __name__ == "__main__":
#     # 指向你合并后的真实点云目录 (GT)
#     # 注意：一定要用原始的、还原后的米制单位点云来算
#     gt_merge_dir = "/home/niuxueqin/D-PCC/output_test_semantickitti/test10/merge/gt"
    
#     if os.path.exists(gt_merge_dir):
#         global_peak = calculate_global_diagonal_peak(gt_merge_dir)
#     else:
#         print(f"路径不存在: {gt_merge_dir}")