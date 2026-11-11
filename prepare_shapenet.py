import os
import sys
import random
import numpy as np
import pickle as pkl
import open3d as o3d
import argparse
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Paths for external tools and temporary directory
MANIFOLD_SCRIPT_PATH = './dataset/Manifold/build/manifold'
TEMP_DIR = './temp_processing'
os.makedirs(TEMP_DIR, exist_ok=True)

def normalize_pcd_compatible(xyzs):
    """
    Normalize point cloud coordinates to [0,1] range and store shift/scale metadata.
    Returns:
        xyzs_norm: normalized points
        meta_data: dict containing original shift and scale information
    """
    shift = np.mean(xyzs, axis=0)
    xyzs_shifted = xyzs - shift
    
    max_coord = np.max(xyzs_shifted)
    min_coord = np.min(xyzs_shifted)
    scale_span = max_coord - min_coord
    
    if scale_span == 0: scale_span = 1
    
    xyzs_norm = (xyzs_shifted - min_coord) / scale_span
    
    meta_data = {
        'shift': shift,
        'max_coord': max_coord,
        'min_coord': min_coord
    }
    return xyzs_norm, meta_data

def random_rotate(xyzs):
    """
    Apply a random rotation to the point cloud (data augmentation).
    Rotation angles are sampled uniformly from [0, 2π] for each axis.
    """
    centroid = np.mean(xyzs, axis=0)
    xyzs -= centroid
    angles = np.random.uniform(0, 2 * np.pi, 3)
    Rx = np.array([[1, 0, 0], [0, np.cos(angles[0]), -np.sin(angles[0])], [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])], [0, 1, 0], [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0], [np.sin(angles[2]), np.cos(angles[2]), 0], [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    xyzs_rot = np.dot(xyzs, R.T)
    xyzs_rot += centroid
    return np.clip(xyzs_rot, 0, 1)

def check_and_fix_normals(pcd):
    """
    Check if the point cloud has valid normals.
    If missing or invalid (NaN/Inf or mostly zero), estimate normals using Open3D KDTree.
    Orient normals towards the camera location [0,0,0].
    """
    points = np.asarray(pcd.points)

    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
        pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
        return pcd

    normals = np.asarray(pcd.normals)

    if np.any(np.isinf(normals)) or np.any(np.isnan(normals)):
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
        pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
        return pcd


    norms = np.linalg.norm(normals, axis=1)
    if np.sum(norms < 1e-6) > len(points) * 0.5:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
        pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
        return pcd
        
    return pcd

def find_obj_in_dir(instance_path):
    """
    Find the OBJ file in a given directory or return the path if a file is provided.
    """
    if os.path.isfile(instance_path):
        return instance_path
    
    obj_path = os.path.join(instance_path, "model.obj")
    if os.path.exists(obj_path):
        return obj_path
        
    candidates = [f for f in os.listdir(instance_path) if f.endswith('.obj')]
    if len(candidates) > 0:
        return os.path.join(instance_path, candidates[0])
        
    return None

def load_geometry_only_safe(src_path, temp_clean_path):
    """
    Read an OBJ file safely by ignoring material references (mtl), which can cause parsing errors.
    Returns:
        mesh: Open3D TriangleMesh object or None if failed
    """
    try:
        with open(src_path, 'r', encoding='utf-8', errors='ignore') as f_in:
            with open(temp_clean_path, 'w', encoding='utf-8') as f_out:
                for line in f_in:
                    if line.strip().startswith(('mtllib', 'usemtl')):
                        continue
                    f_out.write(line)
        
        mesh = o3d.io.read_triangle_mesh(temp_clean_path)
        return mesh
    except Exception as e:
        print(f"[Sanitize Error] Failed to clean OBJ {src_path}: {e}")
        return None

def numpy_random_downsample(pcd, n_points):
    """
    Randomly downsample the point cloud to n_points if the point count exceeds n_points.
    Preserves normals if they exist.
    """
    points = np.asarray(pcd.points)
    num_points = points.shape[0]
    
    if num_points <= n_points:
        return pcd 
    
    choice_indices = np.random.choice(num_points, n_points, replace=False)
    
    new_points = points[choice_indices]
    new_pcd = o3d.geometry.PointCloud()
    
    try:
        new_pcd.points = o3d.utility.Vector3dVector(new_points)
    except AttributeError:
        new_pcd.points = o3d.Vector3dVector(new_points)
        
    if pcd.has_normals():
        normals = np.asarray(pcd.normals)
        new_normals = normals[choice_indices]
        try:
            new_pcd.normals = o3d.utility.Vector3dVector(new_normals)
        except AttributeError:
            new_pcd.normals = o3d.Vector3dVector(new_normals)
            
    return new_pcd

def process_and_save_files(instance_path, strategy, mesh_save_dir, pcd_save_dir, n_points):
    """
    Given a 3D model folder or OBJ file, process it according to a specified strategy 
    and save both mesh and point cloud (PCD) versions.
    
    Args:
        instance_path: Path to model folder or OBJ file
        strategy: 'montecarlo_sample' or 'subdivide'
        mesh_save_dir: Directory to save processed meshes
        pcd_save_dir: Directory to save point clouds
        n_points: Number of points to sample for the PCD
    Returns:
        pcd: Open3D point cloud object
        target_pcd_path: Path to saved PCD file
    """
    src_obj_path = find_obj_in_dir(instance_path)
    if src_obj_path is None:
        print(f"[Error] No .obj found in {instance_path}")
        return None, None

    file_name = os.path.basename(instance_path.rstrip('/')) 
    
    target_mesh_path = os.path.join(mesh_save_dir, f"{file_name}.obj")
    target_pcd_path = os.path.join(pcd_save_dir, f"{file_name}.ply")

    if os.path.exists(target_pcd_path) and os.path.getsize(target_pcd_path) > 1024:
        
        try:
            pcd = o3d.io.read_point_cloud(target_pcd_path)
            if len(pcd.points) > 0:
                return pcd, target_pcd_path
        except:
            pass 

    mesh = None
    if strategy == 'montecarlo_sample':
        cmd = f"{MANIFOLD_SCRIPT_PATH} {src_obj_path} {target_mesh_path} 20000 > /dev/null 2>&1"
        ret = os.system(cmd)
        if ret == 0 and os.path.exists(target_mesh_path):
            mesh = o3d.io.read_triangle_mesh(target_mesh_path)
        else:
            temp_clean_path = os.path.join(TEMP_DIR, f"clean_{file_name}.obj")
            mesh = load_geometry_only_safe(src_obj_path, temp_clean_path)
            if os.path.exists(temp_clean_path): os.remove(temp_clean_path)
            if mesh: o3d.io.write_triangle_mesh(target_mesh_path, mesh)
    else: 
        temp_clean_path = os.path.join(TEMP_DIR, f"clean_{file_name}.obj")
        mesh = load_geometry_only_safe(src_obj_path, temp_clean_path)
        
        if mesh is not None and mesh.has_triangles():
            mesh = mesh.subdivide_midpoint(number_of_iterations=2)
            o3d.io.write_triangle_mesh(target_mesh_path, mesh)

        if os.path.exists(temp_clean_path): os.remove(temp_clean_path)
    
    if mesh is None or not mesh.has_triangles():
        return None, None

    pcd = mesh.sample_points_uniformly(number_of_points=n_points)

    if len(pcd.points) > n_points:
        pcd = numpy_random_downsample(pcd, n_points)

    pcd = check_and_fix_normals(pcd)
    
    try:
        pcd.colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))
    except:
        pcd.colors = o3d.Vector3dVector(np.zeros((0, 3)))

    o3d.io.write_point_cloud(target_pcd_path, pcd, write_ascii=True)
    
    return pcd, target_pcd_path

def divide_cube(xyzs, normals, map_size, cube_size, min_num, augment=False):
    """
    Divide a point cloud into smaller 3D cubes for patch-based training.
    Optionally apply random rotation augmentation.
    
    Returns:
        output_points: dict mapping cube_id -> points+normals array
        meta_data: normalization metadata
    """
    xyzs_norm, meta_data = normalize_pcd_compatible(xyzs)
    
    if augment:
        xyzs_norm = random_rotate(xyzs_norm)

    map_xyzs = xyzs_norm * map_size
    grid_coords = np.floor(map_xyzs).astype(int)
    grid_coords = np.clip(grid_coords, 0, map_size - 1)
    
    chunk_indices = grid_coords // cube_size
    dim_cube_num = int(np.ceil(map_size / cube_size))
    
    cube_ids = chunk_indices[:, 0] * dim_cube_num * dim_cube_num + \
               chunk_indices[:, 1] * dim_cube_num + \
               chunk_indices[:, 2]
               
    output_points = {}
    unique_ids = np.unique(cube_ids)
    
    for uid in unique_ids:
        mask = (cube_ids == uid)
        pts_in_cube = map_xyzs[mask]
        normals_in_cube = normals[mask]
        
        if len(pts_in_cube) < min_num: continue
            
        data_block = np.concatenate([pts_in_cube, normals_in_cube], axis=-1)
        output_points[uid] = data_block.astype(np.float32)

    return output_points, meta_data

def get_balanced_dataset_split(data_root, num_per_class=100):
    """
    Returns:
        all_train, all_val, all_test: lists of instance paths
    """
    categories = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    
    all_train, all_val, all_test = [], [], []
    
    for cat in tqdm(categories, desc="Splitting categories"):
        cat_dir = os.path.join(data_root, cat)
        instances = [os.path.join(cat_dir, i) for i in os.listdir(cat_dir) if os.path.isdir(os.path.join(cat_dir, i))]
        
        if len(instances) >= num_per_class:
            selected = random.sample(instances, num_per_class)
        else:
            selected = instances
        
        if len(selected) == 0: continue

        if len(selected) < 3:
            train, val, test = selected, [], []
        else:
            train, temp = train_test_split(selected, train_size=0.8, random_state=42)
            if len(temp) == 1:
                val, test = [], temp
            else:
                val, test = train_test_split(temp, test_size=0.5, random_state=42)
            
        all_train.extend(train)
        all_val.extend(val)
        all_test.extend(test)
        
    return all_train, all_val, all_test

def assign_strategy(file_list, ratio=0.8):
    """
    Randomly assign a processing strategy for each file in the list.
    ratio: fraction of 'montecarlo_sample'; rest are 'subdivide'.
    Returns a list of tuples: (file_path, strategy)
    """
    if len(file_list) == 0: return []
    n_monte = int(len(file_list) * ratio)
    n_sub = len(file_list) - n_monte
    strategies = ['montecarlo_sample'] * n_monte + ['subdivide'] * n_sub
    random.shuffle(strategies)
    return list(zip(file_list, strategies))

def process_dataset(data_list, mode, args, mesh_root, pcd_root, txt_save_path):
    """
    Process a list of 3D model instances into point clouds, divide into cubes, and save as .pkl and txt.
    
    Args:
        data_list: List of tuples (file_path, strategy)
        mode: 'train', 'val', or 'test'
        args: command-line arguments object
        mesh_root: root folder to save processed meshes
        pcd_root: root folder to save point clouds
        txt_save_path: path to save text file listing processed PCDs
    """
    pkl_save_path = os.path.join(args.save_path, f'{args.dataset_name}_{mode}_cube{args.cube_size}.pkl')
    
    mode_mesh_dir = os.path.join(mesh_root, mode)
    mode_pcd_dir = os.path.join(pcd_root, mode)
    os.makedirs(mode_mesh_dir, exist_ok=True)
    os.makedirs(mode_pcd_dir, exist_ok=True)
    
    do_augment = (mode == 'train')
    print(f"\nProcessing {mode} set: {len(data_list)} models | Augmentation: {do_augment} | Points: {args.n_points}")
    
    data_dict = {}
    valid_count = 0
    total_patches = 0
    txt_records = []
    
    for idx, (path, strategy) in enumerate(tqdm(data_list)):
        try:
            pcd, saved_pcd_path = process_and_save_files(
                path, strategy, mode_mesh_dir, mode_pcd_dir, n_points=args.n_points
            )
            
            if pcd is None or len(pcd.points) == 0: continue

            abs_pcd_path = os.path.abspath(saved_pcd_path)
            txt_records.append(f"{abs_pcd_path} {strategy}\n")

            points = np.asarray(pcd.points)
            normals = np.asarray(pcd.normals)
            
            cubes, meta = divide_cube(
                points, normals, 
                map_size=args.map_size, 
                cube_size=args.cube_size, 
                min_num=args.min_num,
                augment=do_augment
            )
            
            if len(cubes) == 0: continue
                
            file_name = os.path.splitext(os.path.basename(saved_pcd_path))[0]
            
            data_dict[valid_count] = {
                'points': cubes,
                'meta_data': meta,
                'name': file_name,
                'file_path': saved_pcd_path,
                'strategy': strategy
            }
            
            total_patches += sum([len(v) for v in cubes.values()])
            valid_count += 1
            
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue

    print(f"[{mode}] Saved {valid_count} models, Total Patches: {total_patches}")
    
    with open(pkl_save_path, 'wb') as f:
        pkl.dump(data_dict, f, protocol=pkl.HIGHEST_PROTOCOL)
        
    with open(txt_save_path, 'w') as f:
        f.writelines(txt_records)
    print(f"[{mode}] Path list saved to {txt_save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='./dataBeforeProcessing/ShapeNetCore.v1')
    parser.add_argument('--save_path', default='./data/shapenet')
    parser.add_argument('--dataset_name', default='shapenet')
    parser.add_argument('--num_per_class', type=int, default=100)
    parser.add_argument('--map_size', type=int, default=128)
    parser.add_argument('--cube_size', type=int, default=32) 
    parser.add_argument('--min_num', type=int, default=1024)
    parser.add_argument('--n_points', type=int, default=100000)
    
    args = parser.parse_args()
    
    mesh_save_root = os.path.join(args.save_path, 'mesh')
    pcd_save_root = os.path.join(args.save_path, 'pcd')
    
    os.makedirs(args.save_path, exist_ok=True)
    # Split dataset into train, val, test sets
    train_files, val_files, test_files = get_balanced_dataset_split(
        args.data_root, args.num_per_class
    )
    # Assign processing strategies for each split
    train_list = assign_strategy(train_files, 0.8)
    val_list = assign_strategy(val_files, 0.8)
    test_list = assign_strategy(test_files, 0.8)
    # Shuffle lists for randomness
    random.shuffle(train_list) 
    random.shuffle(val_list)
    random.shuffle(test_list)
    # Process all splits and save
    process_dataset(train_list, 'train', args, mesh_save_root, pcd_save_root, os.path.join(args.save_path, 'train.txt'))
    process_dataset(val_list, 'val', args, mesh_save_root, pcd_save_root, os.path.join(args.save_path, 'val.txt'))
    process_dataset(test_list, 'test', args, mesh_save_root, pcd_save_root, os.path.join(args.save_path, 'test.txt'))

if __name__ == "__main__":
    main() 