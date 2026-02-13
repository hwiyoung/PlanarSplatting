import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'planarsplat'))
import argparse
import torch
import shutil
import numpy as np
import open3d as o3d
from pathlib import Path
from pyhocon import ConfigFactory
from pyhocon import ConfigTree
from utils_demo.run_metric3d import extract_mono_geo_demo
from utils_demo.run_vggt import run_vggt
from utils_demo.misc import is_video_file, save_frames_from_video
from utils_demo.run_planarSplatting import run_planarSplatting


def save_pointcloud_from_depth_data(data, out_path, stride=20):
    required_keys = ['depth', 'intrinsics', 'extrinsics']
    if not isinstance(data, dict) or any(k not in data for k in required_keys):
        return ''

    points_world_all = []
    colors_all = []
    color_list = data.get('color', None)

    for i, (depth_i, intrinsic_i, c2w_i) in enumerate(zip(data['depth'], data['intrinsics'], data['extrinsics'])):
        depth = np.asarray(depth_i, dtype=np.float32)
        if depth.ndim != 2:
            continue

        h, w = depth.shape
        ys, xs = np.indices((h, w))
        valid = depth > 0.0
        if stride > 1:
            valid = valid & ((ys % stride) == 0) & ((xs % stride) == 0)
        if not np.any(valid):
            continue

        K = np.asarray(intrinsic_i, dtype=np.float32)
        c2w = np.asarray(c2w_i, dtype=np.float32)
        fx = max(float(K[0, 0]), 1e-6)
        fy = max(float(K[1, 1]), 1e-6)
        cx = float(K[0, 2])
        cy = float(K[1, 2])

        z = depth[valid]
        x = (xs[valid].astype(np.float32) - cx) / fx * z
        y = (ys[valid].astype(np.float32) - cy) / fy * z
        points_cam = np.stack([x, y, z], axis=-1)

        R = c2w[:3, :3]
        t = c2w[:3, 3]
        points_world = points_cam @ R.T + t[None, :]
        points_world_all.append(points_world)

        if color_list is not None and i < len(color_list):
            color = np.asarray(color_list[i])
            if color.ndim == 3 and color.shape[0] == h and color.shape[1] == w:
                color_sel = color[valid]
                if color_sel.dtype == np.uint8:
                    color_sel = color_sel.astype(np.float32) / 255.0
                else:
                    color_sel = color_sel.astype(np.float32)
                colors_all.append(np.clip(color_sel, 0.0, 1.0))

    if not points_world_all:
        return ''

    points_world = np.concatenate(points_world_all, axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_world)
    if colors_all:
        colors = np.concatenate(colors_all, axis=0)
        if colors.shape[0] == points_world.shape[0]:
            pcd.colors = o3d.utility.Vector3dVector(colors)

    pcd_path = os.path.join(out_path, 'pcd_vggt_downsampled.ply')
    o3d.io.write_point_cloud(pcd_path, pcd)
    return pcd_path


def get_latest_run_dir(out_path, conf):
    expname = conf.get_string('train.expname')
    scan_id = conf.get_string('dataset.scan_id', default='-1')
    exp_folder = expname if scan_id == '-1' else f'{expname}_{scan_id}'
    exp_root = Path(out_path) / exp_folder
    if not exp_root.exists():
        return None
    candidates = [p for p in exp_root.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def save_run_input_snapshot(run_dir, data, pcd_path=''):
    if run_dir is None:
        return
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    data_snapshot_path = run_dir / 'input_data.pth'
    torch.save(data, data_snapshot_path)
    if pcd_path and os.path.exists(pcd_path):
        shutil.copy2(pcd_path, run_dir / 'input_pointcloud.ply')
    print(f"saved run input snapshot: {data_snapshot_path}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-d", "--data_path", "--data-path", dest="data_path", type=str, default='examples/living_room/images', help='path of input data')
    parser.add_argument("-o", "--out_path", type=str, default='planarSplat_ExpRes/demo', help='path of output dir')
    parser.add_argument("-s", "--frame_step", type=int, default=1, help='sampling step for video extraction and VGGT image subsampling')
    parser.add_argument("--plot_freq", type=int, default=500, help='visualization frequency during optimization (also affects TensorBoard image update frequency)')
    parser.add_argument("--depth_conf", type=float, default=2.0, help='depth confidence threshold of vggt')
    parser.add_argument("--conf_path", type=str, default='utils_demo/demo.conf', help='path of configure file')
    parser.add_argument('--use_precomputed_data', default=False, action="store_true", help='use processed data from input images')
    parser.add_argument('--overwrite_exp', default=False, action="store_true", help='remove previous experiment folder with same expname before running')
    args = parser.parse_args()

    data_path = args.data_path
    if not os.path.exists(data_path):
        raise ValueError(f'The input data path {data_path} does not exist.')
    
    image_path = None
    if os.path.isdir(data_path):
        image_path = data_path
    else:
        if is_video_file(data_path):
            absolute_video_path = os.path.abspath(data_path)
            current_dir = os.path.dirname(absolute_video_path)
            image_path = os.path.join(current_dir, 'images')
            save_frames_from_video(data_path, image_path, args.frame_step)
        else:
            raise ValueError(f'The input file {data_path} is not a video file.')
    assert image_path is not None, f"Can not find images or videos from {data_path}."

    out_path = args.out_path
    os.makedirs(out_path, exist_ok=True)
    precomputed_data_path = os.path.join(out_path, 'data.pth')
    use_precomputed_data = args.use_precomputed_data
    
    if use_precomputed_data and os.path.exists(precomputed_data_path):
        data = torch.load(precomputed_data_path)
        print(f"loading precomputed data from {precomputed_data_path}")
        if isinstance(data, dict) and 'vggt_pcd_path' not in data:
            pcd_guess = os.path.join(out_path, 'pcd_vggt_downsampled.ply')
            if os.path.exists(pcd_guess):
                data['vggt_pcd_path'] = pcd_guess
    else:
        # run vggt (subsample images by frame_step to reduce VRAM usage)
        data = run_vggt(image_path, out_path, step=args.frame_step, depth_conf_thresh=args.depth_conf)

        # run metric3dv2
        _, normal_maps_list = extract_mono_geo_demo(data['color'], data['intrinsics'])
        data['normal'] = normal_maps_list
        torch.save(data, precomputed_data_path)

    pcd_path = save_pointcloud_from_depth_data(data, out_path, stride=20)
    if pcd_path:
        data['vggt_pcd_path'] = pcd_path

    # run planarSplatting
    '''
        data = {
            'color': [...],
            'depth': [...],
            'normal': [...],
            'image_paths': [...],
            'extrinsics': [...],  # c2w
            'intrinsics': [...],
        }
    '''
    # load conf
    base_conf = ConfigFactory.parse_file('planarsplat/confs/base_conf_planarSplatCuda.conf')
    demo_conf = ConfigFactory.parse_file(args.conf_path)
    conf = ConfigTree.merge_configs(base_conf, demo_conf)
    conf.put('train.exps_folder_name', out_path)
    conf.put('train.plot_freq', args.plot_freq)
    img_res = [data['color'][0].shape[0], data['color'][0].shape[1]]
    conf.put('dataset.img_res', img_res)

    if args.overwrite_exp:
        expname = conf.get_string('train.expname')
        scan_id = conf.get_string('dataset.scan_id', default='-1')
        exp_folder = expname if scan_id == '-1' else f'{expname}_{scan_id}'
        exp_path = os.path.join(out_path, exp_folder)
        if os.path.exists(exp_path):
            print(f"overwrite_exp=True, removing old experiment folder: {exp_path}")
            shutil.rmtree(exp_path)

    run_planarSplatting(data=data, conf=conf)
    latest_run_dir = get_latest_run_dir(out_path, conf)
    save_run_input_snapshot(latest_run_dir, data, pcd_path=pcd_path)


    
