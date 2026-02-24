import os
import sys
from datetime import datetime
from tqdm import tqdm
import torch
import numpy as np
from random import randint
import math
from loguru import logger
import open3d as o3d
from PIL import Image
from .net_wrapper import PlanarRecWrapper

from utils.misc_util import setup_logging, get_train_param, save_config_files, prepare_folders, get_class
from utils.trainer_util import resume_model, calculate_plane_depth, plot_plane_img, save_checkpoints
from utils.mesh_util import get_coarse_mesh
from utils.merge_util import merge_plane
from utils.loss_util import normal_loss, metric_depth_loss, semantic_loss, normal_consistency_loss

import rerun as rr
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

class PlanarSplatTrainRunner():
    def __init__(self, **kwargs):
        torch.set_default_dtype(torch.float32)
        self.conf = kwargs['conf']
        self.expname, self.scan_id, self.timestamp, is_continue = get_train_param(kwargs, self.conf)
        self.expdir, self.plane_plots_dir, self.checkpoints_path, self.model_subdir = prepare_folders(kwargs, self.expname, self.timestamp)
        setup_logging(os.path.join(self.expdir, 'train.log'))
        
        logger.info('Shell command : {0}'.format(' '.join(sys.argv)))
        save_config_files(self.expdir, self.conf)

        # =======================================  loading dataset
        logger.info('Loading data...')
        if 'data' in kwargs:
            self.dataset = get_class(self.conf.get_string('train.dataset_class'))(kwargs['data'], **self.conf.get_config('dataset'))
        else:
            self.dataset = get_class(self.conf.get_string('train.dataset_class'))(**self.conf.get_config('dataset'))
        self.ds_len = self.dataset.n_images
        self.H = self.conf.dataset.img_res[0]
        self.W = self.conf.dataset.img_res[1]
        logger.info('Data loaded. Frame number = {0}'.format(self.ds_len))

        # =======================================  build plane model
        self.plane_model_conf = self.conf.get_config('plane_model')
        self.conf['dataset']['mesh_path'] = self.dataset.mono_mesh_dest

        net = PlanarRecWrapper(self.conf, self.plane_plots_dir)
        self.net = net.cuda()
        self.resumed = False
        self.start_iter = resume_model(self) if is_continue else 0
        self.iter_step = self.start_iter
        self.net.build_optimizer_and_LRscheduler()

        # ======================================= plot settings
        self.do_vis = kwargs['do_vis']
        self.plot_freq = self.conf.get_int('train.plot_freq')        
        
        # ======================================= loss settings
        loss_plane_conf = self.conf.get_config('plane_model.plane_loss')
        self.weight_plane_normal = loss_plane_conf.get_float('weight_mono_normal')
        self.weight_plane_depth = loss_plane_conf.get_float('weight_mono_depth')
        # Semantic losses (Phase 2-B)
        self.enable_semantic = self.conf.get_bool('plane_model.enable_semantic', default=False)
        self.lambda_sem = loss_plane_conf.get_float('lambda_sem', default=0.1)
        self.lambda_geo = loss_plane_conf.get_float('lambda_geo', default=0.0)

        # ======================================= training settings
        self.max_total_iters = self.conf.get_int('train.max_total_iters')
        self.process_plane_freq_ite = self.conf.get_int('train.process_plane_freq_ite')
        self.coarse_stage_ite = self.conf.get_int('train.coarse_stage_ite')
        self.split_start_ite = self.conf.get_int('train.split_start_ite')
        self.check_vis_freq_ite = self.conf.get_int('train.check_plane_vis_freq_ite')
        self.data_order = self.conf.get_string('train.data_order')
        self.log_freq = self.conf.get_int('train.log_freq', default=50)
        self.use_tensorboard = self.conf.get_bool('train.use_tensorboard', default=True)
        self.tb_log_mesh = self.conf.get_bool('train.tb_log_mesh', default=True)
        self.tb_log_text = self.conf.get_bool('train.tb_log_text', default=True)
        self.tb_log_input_geometry = self.conf.get_bool('train.tb_log_input_geometry', default=True)
        self.tb_camera_max_count = self.conf.get_int('train.tb_camera_max_count', default=64)
        self.tb_camera_frustum_ratio = self.conf.get_float('train.tb_camera_frustum_ratio', default=0.08)
        self.tb_trend_window = self.conf.get_int('train.tb_trend_window', default=200)
        self.tb_image_freq = self.conf.get_int('train.tb_image_freq', default=500)
        self.tb_iter_hist = []
        self.tb_total_hist = []
        self.tb_writer = self._build_tb_writer()

    def run(self):
        self._log_tb_input_geometry()
        self.train()
        self.merger()
        if self.tb_writer is not None:
            self.tb_writer.flush()
            self.tb_writer.close()
    
    def _build_tb_writer(self):
        if not self.use_tensorboard:
            return None
        if SummaryWriter is None:
            logger.warning("TensorBoard writer is unavailable. Install `tensorboard` to enable real-time monitor.")
            return None
        tb_logdir = os.path.join(self.expdir, 'tensorboard')
        logger.info(f'TensorBoard logging enabled at: {tb_logdir}')
        return SummaryWriter(log_dir=tb_logdir)

    def _log_tb_scalars(
        self,
        iter,
        plane_num,
        loss_depth,
        loss_normal_l1,
        loss_normal_cos,
        loss_plane,
        loss_total,
        trend_slope=None,
        trend_delta=None,
        trend_state=None,
    ):
        if self.tb_writer is None:
            return
        self.tb_writer.add_scalar('loss/depth', loss_depth, iter)
        self.tb_writer.add_scalar('loss/normal_l1', loss_normal_l1, iter)
        self.tb_writer.add_scalar('loss/normal_cos', loss_normal_cos, iter)
        self.tb_writer.add_scalar('loss/plane', loss_plane, iter)
        self.tb_writer.add_scalar('loss/total', loss_total, iter)
        self.tb_writer.add_scalar('model/plane_count', plane_num, iter)
        if trend_slope is not None:
            self.tb_writer.add_scalar('trend/total_slope', trend_slope, iter)
        if trend_delta is not None:
            self.tb_writer.add_scalar('trend/total_delta_percent', trend_delta, iter)
        if trend_state is not None:
            self.tb_writer.add_scalar('trend/state_code', trend_state, iter)

    def _estimate_total_trend(self):
        n = min(self.tb_trend_window, len(self.tb_total_hist))
        if n < 2:
            return 0.0, 0.0, 0, "stable"
        x = np.asarray(self.tb_iter_hist[-n:], dtype=np.float64)
        y = np.asarray(self.tb_total_hist[-n:], dtype=np.float64)
        slope = float(np.polyfit(x, y, deg=1)[0])
        y0 = float(y[0])
        y1 = float(y[-1])
        if abs(y0) < 1e-12:
            delta = 0.0
        else:
            delta = (y1 - y0) / abs(y0) * 100.0
        if slope < -1e-5 and delta <= -3.0:
            return slope, delta, 1, "improving"
        if slope > 1e-5 and delta >= 3.0:
            return slope, delta, -1, "degrading"
        return slope, delta, 0, "stable"

    def _log_tb_text(self, iter, text):
        if self.tb_writer is None or not self.tb_log_text:
            return
        self.tb_writer.add_text('status/latest', text.replace("\n", "  \n"), iter)

    def _log_tb_image(self, iter, vis_path):
        if self.tb_writer is None:
            return
        if not os.path.exists(vis_path):
            return
        vis_np = np.asarray(Image.open(vis_path).convert('RGB'))
        self.tb_writer.add_image('render/vis_compare', vis_np, iter, dataformats='HWC')

    def _log_tb_images_detailed(self, iter):
        """Log RGB/Depth/Normal images to TensorBoard at tb_image_freq.

        Tag naming: ``compare/N_type`` groups all comparisons together in
        TensorBoard's alphabetical sort.  Each image is a GT|Rendered
        side-by-side pair so the user can compare without scrolling.
        Individual ``gt/`` and ``render/`` channels are also logged for
        zoom-in inspection.
        """
        if self.tb_writer is None:
            return
        self.net.eval()
        try:
            view_info = self.dataset.view_info_list[0]
            raster_cam_w2c = view_info.raster_cam_w2c
            with torch.no_grad():
                rendered_rgb, allmap = self.net.planarSplat(view_info, iter, return_rgb=True)
                depth = allmap[0:1].squeeze()
                normal_local = allmap[2:5]
                normal_global = (normal_local.permute(1, 2, 0) @ (raster_cam_w2c[:3, :3].T)).reshape(-1, 3)
                normal_global = torch.nn.functional.normalize(normal_global, dim=-1)

            import matplotlib.cm as cm

            # --- RGB (NOTE: colors are random each forward pass, so RGB comparison is
            # NOT meaningful for tracking quality. Use depth/normal instead.) ---
            rgb_hw_c = rendered_rgb.permute(1, 2, 0).clamp(0, 1)
            if rgb_hw_c.shape[2] > 3:
                rgb_hw_c = rgb_hw_c[:, :, :3]
            rgb_np = (rgb_hw_c * 255).cpu().numpy().astype(np.uint8)
            gt_rgb_np = (view_info.rgb.reshape(self.H, self.W, 3) * 255).cpu().numpy().astype(np.uint8)
            rgb_compare = np.concatenate([gt_rgb_np, rgb_np], axis=1)
            self.tb_writer.add_image('compare/1_rgb', rgb_compare, iter, dataformats='HWC')
            self.tb_writer.add_image('gt/rgb', gt_rgb_np, iter, dataformats='HWC')
            self.tb_writer.add_image('render/rgb', rgb_np, iter, dataformats='HWC')

            # --- Depth (viridis colormap, shared scale) ---
            depth_np = depth.reshape(self.H, self.W).cpu().numpy()
            gt_depth_np = view_info.mono_depth.reshape(self.H, self.W).cpu().numpy()
            vmax = max(float(gt_depth_np[gt_depth_np > 0].max()) if (gt_depth_np > 0).any() else 1.0, 0.1)
            depth_color = (cm.viridis(np.clip(depth_np / vmax, 0, 1))[:, :, :3] * 255).astype(np.uint8)
            gt_depth_color = (cm.viridis(np.clip(gt_depth_np / vmax, 0, 1))[:, :, :3] * 255).astype(np.uint8)
            depth_compare = np.concatenate([gt_depth_color, depth_color], axis=1)
            self.tb_writer.add_image('compare/2_depth', depth_compare, iter, dataformats='HWC')
            self.tb_writer.add_image('gt/depth', gt_depth_color, iter, dataformats='HWC')
            self.tb_writer.add_image('render/depth', depth_color, iter, dataformats='HWC')

            # --- Normal (color = (n+1)/2) ---
            normal_np = normal_global.reshape(self.H, self.W, 3).cpu().numpy()
            normal_color = ((normal_np + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)
            gt_normal_np = torch.nn.functional.normalize(view_info.mono_normal_global, dim=-1).reshape(self.H, self.W, 3).cpu().numpy()
            gt_normal_color = ((gt_normal_np + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)
            normal_compare = np.concatenate([gt_normal_color, normal_color], axis=1)
            self.tb_writer.add_image('compare/3_normal', normal_compare, iter, dataformats='HWC')
            self.tb_writer.add_image('gt/normal', gt_normal_color, iter, dataformats='HWC')
            self.tb_writer.add_image('render/normal', normal_color, iter, dataformats='HWC')

            # --- Semantic (Phase 2-B) ---
            if self.enable_semantic:
                # Rendered semantic: argmax of alpha-blended features
                sem_pred = rendered_rgb.argmax(dim=0).cpu().numpy()  # (H, W)
                # Color map: bg=black, roof=red, wall=blue, ground=gray
                sem_color_map = np.array([
                    [0, 0, 0],       # 0: bg
                    [255, 0, 0],     # 1: roof
                    [0, 0, 255],     # 2: wall
                    [180, 180, 180], # 3: ground
                ], dtype=np.uint8)
                sem_color = sem_color_map[sem_pred.clip(0, 3)]
                self.tb_writer.add_image('render/semantic', sem_color, iter, dataformats='HWC')
                # GT semantic
                if view_info.seg_map is not None:
                    gt_seg = view_info.seg_map.reshape(self.H, self.W).cpu().numpy()
                    gt_seg_color = sem_color_map[gt_seg.clip(0, 3)]
                    sem_compare = np.concatenate([gt_seg_color, sem_color], axis=1)
                    self.tb_writer.add_image('compare/4_semantic', sem_compare, iter, dataformats='HWC')
                    self.tb_writer.add_image('gt/semantic', gt_seg_color, iter, dataformats='HWC')
        finally:
            self.net.train()

    def _log_tb_mesh(self, iter, mesh, tag):
        if self.tb_writer is None or not self.tb_log_mesh:
            return
        if mesh is None:
            return
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        colors = np.asarray(mesh.vertex_colors)
        if vertices.size == 0 or faces.size == 0:
            return
        vertices = torch.from_numpy(vertices.astype(np.float32)).unsqueeze(0)
        faces = torch.from_numpy(faces.astype(np.int64)).unsqueeze(0)
        if colors.size > 0:
            colors = torch.from_numpy((colors * 255.0).astype(np.uint8)).unsqueeze(0)
            self.tb_writer.add_mesh(tag, vertices=vertices, faces=faces, colors=colors, global_step=iter)
        else:
            self.tb_writer.add_mesh(tag, vertices=vertices, faces=faces, global_step=iter)

    def _log_tb_pointcloud(self, iter, pcd_path, tag):
        if self.tb_writer is None or not self.tb_log_input_geometry:
            return
        if not pcd_path or not os.path.exists(pcd_path):
            return
        pcd = o3d.io.read_point_cloud(pcd_path)
        points = np.asarray(pcd.points)
        if points.size == 0:
            return
        max_points = 200000
        sample_idx = None
        if points.shape[0] > max_points:
            sample_idx = np.random.choice(points.shape[0], size=max_points, replace=False)
            points = points[sample_idx]
        vertices = torch.from_numpy(points.astype(np.float32)).unsqueeze(0)
        colors = np.asarray(pcd.colors)
        if colors.size > 0:
            if sample_idx is not None:
                colors = colors[sample_idx]
            colors = torch.from_numpy((colors * 255.0).astype(np.uint8)).unsqueeze(0)
            self.tb_writer.add_mesh(tag, vertices=vertices, colors=colors, global_step=iter)
        else:
            self.tb_writer.add_mesh(tag, vertices=vertices, global_step=iter)

    def _build_camera_frustum_mesh(self):
        intrinsics_all = getattr(self.dataset, 'intrinsics_all', None)
        poses_all = getattr(self.dataset, 'poses_all', None)
        if intrinsics_all is None or poses_all is None:
            return None
        cam_count = min(len(intrinsics_all), len(poses_all))
        if cam_count == 0:
            return None

        keep_count = min(cam_count, max(self.tb_camera_max_count, 1))
        cam_indices = np.linspace(0, cam_count - 1, keep_count, dtype=np.int64)
        frustum_depth = max(
            1e-3,
            self.conf.get_float('dataset.scene_bounding_sphere', default=5.0) * self.tb_camera_frustum_ratio,
        )

        base_faces = np.array(
            [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1], [1, 2, 3], [1, 3, 4]],
            dtype=np.int64,
        )
        frustum_color = np.array([1.0, 0.8, 0.1], dtype=np.float32)

        vertices_all = []
        faces_all = []
        colors_all = []
        vertex_offset = 0

        for cam_idx in cam_indices:
            K = intrinsics_all[int(cam_idx)]
            c2w = poses_all[int(cam_idx)]
            if torch.is_tensor(K):
                K = K.detach().cpu().numpy()
            if torch.is_tensor(c2w):
                c2w = c2w.detach().cpu().numpy()

            fx = max(float(K[0, 0]), 1e-6)
            fy = max(float(K[1, 1]), 1e-6)
            cx = float(K[0, 2])
            cy = float(K[1, 2])
            H, W = self.H, self.W

            corners_cam = np.array(
                [
                    [(0.0 - cx) / fx * frustum_depth, (0.0 - cy) / fy * frustum_depth, frustum_depth],
                    [((W - 1.0) - cx) / fx * frustum_depth, (0.0 - cy) / fy * frustum_depth, frustum_depth],
                    [((W - 1.0) - cx) / fx * frustum_depth, ((H - 1.0) - cy) / fy * frustum_depth, frustum_depth],
                    [(0.0 - cx) / fx * frustum_depth, ((H - 1.0) - cy) / fy * frustum_depth, frustum_depth],
                ],
                dtype=np.float32,
            )

            R = c2w[:3, :3]
            t = c2w[:3, 3]
            corners_world = corners_cam @ R.T + t[None, :]
            origin_world = t[None, :]
            vertices = np.concatenate([origin_world, corners_world], axis=0)

            vertices_all.append(vertices)
            faces_all.append(base_faces + vertex_offset)
            colors_all.append(np.tile(frustum_color[None, :], (vertices.shape[0], 1)))
            vertex_offset += vertices.shape[0]

        if not vertices_all:
            return None

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(np.concatenate(vertices_all, axis=0))
        mesh.triangles = o3d.utility.Vector3iVector(np.concatenate(faces_all, axis=0))
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.concatenate(colors_all, axis=0))
        return mesh

    def _merge_mesh(self, mesh_a, mesh_b):
        if mesh_a is None:
            return mesh_b
        if mesh_b is None:
            return mesh_a

        va = np.asarray(mesh_a.vertices)
        fa = np.asarray(mesh_a.triangles)
        ca = np.asarray(mesh_a.vertex_colors)
        vb = np.asarray(mesh_b.vertices)
        fb = np.asarray(mesh_b.triangles)
        cb = np.asarray(mesh_b.vertex_colors)

        if va.size == 0 or fa.size == 0:
            return mesh_b
        if vb.size == 0 or fb.size == 0:
            return mesh_a

        if ca.size == 0:
            ca = np.tile(np.array([[0.7, 0.7, 0.7]], dtype=np.float32), (va.shape[0], 1))
        if cb.size == 0:
            cb = np.tile(np.array([[1.0, 0.8, 0.1]], dtype=np.float32), (vb.shape[0], 1))

        merged = o3d.geometry.TriangleMesh()
        merged.vertices = o3d.utility.Vector3dVector(np.concatenate([va, vb], axis=0))
        merged.triangles = o3d.utility.Vector3iVector(np.concatenate([fa, fb + va.shape[0]], axis=0))
        merged.vertex_colors = o3d.utility.Vector3dVector(np.concatenate([ca, cb], axis=0))
        return merged

    def _log_tb_input_geometry(self):
        if self.tb_writer is None or not self.tb_log_input_geometry:
            return

        camera_mesh = self._build_camera_frustum_mesh()
        if camera_mesh is not None:
            self._log_tb_mesh(iter=0, mesh=camera_mesh, tag='input/camera_frustums')
            self._log_tb_text(
                iter=0,
                text=f"input camera frustums: {min(self.ds_len, self.tb_camera_max_count)} / {self.ds_len}",
            )

        mono_mesh_path = getattr(self.dataset, 'mono_mesh_dest', '')
        if mono_mesh_path and os.path.exists(mono_mesh_path):
            mono_mesh = o3d.io.read_triangle_mesh(mono_mesh_path)
            self._log_tb_mesh(iter=0, mesh=mono_mesh, tag='input/mono_mesh')
            if camera_mesh is not None:
                mono_with_cam_mesh = self._merge_mesh(mono_mesh, camera_mesh)
                self._log_tb_mesh(iter=0, mesh=mono_with_cam_mesh, tag='input/mono_mesh_with_cameras')
            self._log_tb_text(iter=0, text=f"input mono mesh: {mono_mesh_path}")

        vggt_pcd_path = getattr(self.dataset, 'vggt_pcd_path', '')
        if vggt_pcd_path and os.path.exists(vggt_pcd_path):
            self._log_tb_pointcloud(iter=0, pcd_path=vggt_pcd_path, tag='input/vggt_pointcloud')
            self._log_tb_text(iter=0, text=f"input vggt point cloud: {vggt_pcd_path}")
    
    def train(self):
        logger.info("Training...")
        if self.start_iter >= self.max_total_iters:
            return
        weight_decay_list = []
        for i in tqdm(range(self.max_total_iters+1), desc="generating sampling idx list..."):
            weight_decay_list.append(max(math.exp(-i / self.max_total_iters), 0.1))
        logger.info('Start training at {:%Y_%m_%d_%H_%M_%S}'.format(datetime.now()))
        self.net.train()
        if self.iter_step == 0:
            self.check_plane_visibility_cuda()  

        view_info_list = None
        progress_bar = tqdm(range(self.start_iter, self.max_total_iters+1), desc="Training progress")
        calculate_plane_depth(self)
        for iter in range(self.start_iter, self.max_total_iters + 1):
            self.iter_step = iter
            # ======================================= process planes
            if iter > self.coarse_stage_ite and iter % self.process_plane_freq_ite==0:  
                self.net.regularize_plane_shape()
                self.net.prune_small_plane()
                if iter > self.split_start_ite and iter <= self.max_total_iters - 1000:
                    logger.info('splitting...')
                    ori_num = self.net.planarSplat.get_plane_num()
                    self.net.split_plane()
                    new_num = self.net.planarSplat.get_plane_num()
                    logger.info(f'plane num: {ori_num} ---> {new_num}')
            # ======================================= get view info
            if not view_info_list:
                view_info_list = self.dataset.view_info_list.copy()
            if self.data_order == 'rand':
                view_info = view_info_list.pop(randint(0, len(view_info_list)-1))
            else:
                view_info = view_info_list.pop(0)
            raster_cam_w2c = view_info.raster_cam_w2c
            # ======================================= zero grad
            self.net.optimizer.zero_grad()
            #  ======================================= plane forward
            rendered_features, allmap = self.net.planarSplat(view_info, iter, return_rgb=True)
            # ------------ get rendered maps
            depth = allmap[0:1].squeeze().view(-1)
            normal_local_ = allmap[2:5]
            normal_global = (normal_local_.permute(1,2,0) @ (raster_cam_w2c[:3,:3].T)).view(-1, 3)
            # ------------ get aux maps
            vis_weight = allmap[1:2].squeeze().view(-1)
            valid_ray_mask = vis_weight > 0.00001
            valid_normal_mask = view_info.mono_normal_global.abs().sum(dim=-1) > 0
            valid_depth_mask = view_info.mono_depth.abs() > 0
            valid_ray_mask = valid_ray_mask & valid_depth_mask & valid_normal_mask

            # ======================================= calculate losses
            loss_final = 0.
            decay = weight_decay_list[iter]
            # ------------ calculate plane loss
            loss_plane_normal_l1, loss_plane_normal_cos = normal_loss(normal_global, view_info.mono_normal_global, valid_ray_mask)
            loss_plane_depth = metric_depth_loss(depth, view_info.mono_depth, valid_ray_mask, max_depth=10.0)
            loss_plane = (loss_plane_depth * 1.0) * self.weight_plane_depth \
                        + (loss_plane_normal_l1 + loss_plane_normal_cos) * self.weight_plane_normal
            loss_final += loss_plane * decay

            # ------------ semantic losses (Phase 2-B)
            loss_sem_value = 0.
            loss_geo_value = 0.
            if self.enable_semantic:
                # L_sem: CrossEntropyLoss on rendered semantic features vs seg_map GT
                if view_info.seg_map is not None and self.lambda_sem > 0:
                    loss_sem = semantic_loss(rendered_features, view_info.seg_map, mask=valid_ray_mask)
                    loss_final += self.lambda_sem * loss_sem * decay
                    loss_sem_value = loss_sem.detach().item()

                # L_geo: normal consistency (rendered normal vs depth-derived normal)
                if self.lambda_geo > 0:
                    depth_2d = allmap[0:1].squeeze()  # (H, W)
                    normal_local_hw3 = normal_local_.permute(1, 2, 0)  # (H, W, 3) in camera frame
                    intrinsic = view_info.intrinsic
                    vis_mask_2d = (allmap[1:2].squeeze() > 0.00001)
                    loss_geo = normal_consistency_loss(depth_2d, normal_local_hw3, intrinsic, mask=vis_mask_2d)
                    loss_final += self.lambda_geo * loss_geo * decay
                    loss_geo_value = loss_geo.detach().item()

            # ======================================= backward & update plane denom & update learning rate
            loss_final.backward()
            self.net.optimizer.step()
            self.net.update_grad_stats()
            self.net.regularize_plane_shape(False)
            image_index = view_info.index
            self.dataset.view_info_list[image_index].plane_depth = depth.detach().clone()

            with torch.no_grad():
                # Progress bar
                plane_num = self.net.planarSplat.get_plane_num()
                if iter % self.log_freq == 0:
                    loss_depth_value = loss_plane_depth.detach().item()
                    loss_normal_l1_value = loss_plane_normal_l1.detach().item()
                    loss_normal_cos_value = loss_plane_normal_cos.detach().item()
                    loss_plane_value = loss_plane.detach().item()
                    loss_final_value = loss_final.detach().item()
                    self.tb_iter_hist.append(iter)
                    self.tb_total_hist.append(loss_final_value)
                    trend_slope, trend_delta, trend_state_code, trend_state_name = self._estimate_total_trend()
                    loss_dict = {
                        "Planes": f"{plane_num}",
                        "L_depth": f"{loss_depth_value:.4f}",
                        "L_n_l1": f"{loss_normal_l1_value:.4f}",
                        "L_n_cos": f"{loss_normal_cos_value:.4f}",
                        "L_total": f"{loss_final_value:.4f}",
                        "Trend": trend_state_name,
                    }
                    if self.enable_semantic:
                        loss_dict["L_sem"] = f"{loss_sem_value:.4f}"
                        loss_dict["L_geo"] = f"{loss_geo_value:.4f}"
                    progress_bar.set_postfix(loss_dict)
                    status_line = (
                        "iter={} planes={} depth_loss={:.6f} normal_l1={:.6f} normal_cos={:.6f} "
                        "plane_loss={:.6f} total_loss={:.6f} trend={} slope={:.3e} delta={:+.2f}%"
                    ).format(
                        iter,
                        plane_num,
                        loss_depth_value,
                        loss_normal_l1_value,
                        loss_normal_cos_value,
                        loss_plane_value,
                        loss_final_value,
                        trend_state_name,
                        trend_slope,
                        trend_delta,
                    )
                    logger.info(status_line)
                    self._log_tb_scalars(
                        iter=iter,
                        plane_num=plane_num,
                        loss_depth=loss_depth_value,
                        loss_normal_l1=loss_normal_l1_value,
                        loss_normal_cos=loss_normal_cos_value,
                        loss_plane=loss_plane_value,
                        loss_total=loss_final_value,
                        trend_slope=trend_slope,
                        trend_delta=trend_delta,
                        trend_state=trend_state_code,
                    )
                    # Semantic TensorBoard logging (Phase 2-B)
                    if self.enable_semantic and self.tb_writer is not None:
                        self.tb_writer.add_scalar('loss/semantic', loss_sem_value, iter)
                        self.tb_writer.add_scalar('loss/geo_nc', loss_geo_value, iter)
                        # Log class distribution every 100 iters
                        if iter % 100 == 0:
                            with torch.no_grad():
                                sem_feat = self.net.planarSplat._plane_semantic_features
                                class_pred = sem_feat.argmax(dim=-1)
                                for c_idx, c_name in enumerate(['bg', 'roof', 'wall', 'ground']):
                                    count = (class_pred == c_idx).sum().item()
                                    self.tb_writer.add_scalar(f'semantic/class_{c_name}_count', count, iter)
                    self._log_tb_text(iter=iter, text=status_line)
                if iter % 10 == 0:
                    progress_bar.update(10)
                if iter == self.max_total_iters:
                    progress_bar.close()
            
            # ======================================= TB image logging (lightweight, separate from plot)
            if iter > 0 and iter % self.tb_image_freq == 0:
                with torch.no_grad():
                    self._log_tb_images_detailed(iter)

            # ======================================= plot model outputs
            if self.do_vis and iter % self.plot_freq == 0:
                self.net.regularize_plane_shape()
                self.net.eval()
                mesh_n, mesh_p = self.net.planarSplat.draw_plane(epoch=iter)
                vis_info = plot_plane_img(self)
                self.net.train()
                if vis_info is not None:
                    self._log_tb_image(iter=iter, vis_path=vis_info['vis_path'])
                self._log_tb_mesh(iter=iter, mesh=mesh_p, tag='mesh/prim')
                self._log_tb_mesh(iter=iter, mesh=mesh_n, tag='mesh/normal')
            
            if iter > 0 and iter % self.check_vis_freq_ite == 0:
                self.check_plane_visibility_cuda()
        
        
        self.check_plane_visibility_cuda()
        save_checkpoints(self, iter=self.iter_step, only_latest=False)

    def merger(self, save_mesh=True):
        logger.info("Merging 3D planar primitives...")
        output_dir = self.conf.get_string('train.rec_folder_name', default='')
        if len(output_dir) == 0:
            output_dir = self.expdir
        self.net.eval()
        save_root = os.path.join(output_dir, f'{self.scan_id}')
        os.makedirs(save_root, exist_ok=True)

        ## prune planes whose maximum radii lower than the threshold
        self.net.prune_small_plane(min_radii=0.02 * self.net.planarSplat.pose_cfg.scale)
        logger.info("number of 3D planar primitives = %d"%(self.net.planarSplat.get_plane_num()))

        coarse_mesh = get_coarse_mesh(
            self.net, 
            self.dataset.view_info_list.copy(), 
            self.H, 
            self.W, 
            voxel_length=0.02, 
            sdf_trunc=0.08)
        
        merge_config_coarse = self.conf.get_config('merge_coarse', default=None)
        merge_config_fine = self.conf.get_config('merge_fine', default=None)
        if merge_config_coarse is not None:
            logger.info(f'mergeing (coarse)...')
            planarSplat_eval_mesh, plane_ins_id_new = merge_plane(
                self.net, 
                coarse_mesh, 
                plane_ins_id=None,
                **merge_config_coarse)
            if merge_config_fine is not None:
                logger.info(f'mergeing (fine)...')
                planarSplat_eval_mesh, plane_ins_id_new = merge_plane(
                    self.net, 
                    coarse_mesh, 
                    plane_ins_id=plane_ins_id_new,
                    **merge_config_fine)
        else:
            raise ValueError("No merge configuration found!")
        
        if save_mesh:
            save_path = os.path.join(save_root, f"{self.scan_id}_planar_mesh.ply")
            logger.info(f'saving final planar mesh to {save_path}')
            o3d.io.write_triangle_mesh(
                        save_path, 
                        planarSplat_eval_mesh)
        return planarSplat_eval_mesh

    def check_plane_visibility_cuda(self):   
        self.net.regularize_plane_shape(False)     
        logger.info('checking plane visibility...')
        self.net.eval()
        self.net.reset_plane_vis()
        view_info_list = self.dataset.view_info_list.copy()
        for iter in tqdm(range(self.ds_len)):
            # ========================= get view info
            view_info = view_info_list.pop(randint(0, len(view_info_list)-1))
            raster_cam_w2c = view_info.raster_cam_w2c
            # ----------- plane forward
            allmap = self.net.planarSplat(view_info, self.iter_step)
            # get rendered maps
            depth = allmap[0:1].view(-1)
            normal_local_ = allmap[2:5]
            normal_global = (normal_local_.permute(1,2,0) @ (raster_cam_w2c[:3,:3].T)).view(-1, 3)
            # get aux maps
            vis_weight = allmap[1:2].view(-1)
            valid_ray_mask = vis_weight > 0.00001

            loss_final = 0.
            # ======================================= calculate plane losses
            loss_mono_depth = metric_depth_loss(depth, view_info.mono_depth, valid_ray_mask)
            loss_normal_l1, loss_normal_cos = normal_loss(normal_global, view_info.mono_normal_global, valid_ray_mask)
            loss_final += loss_mono_depth + loss_normal_cos + loss_normal_l1

            loss_final.backward() 
            # update plane visibility
            self.net.update_plane_vis() 
            self.net.optimizer.zero_grad()

        self.net.optimizer.zero_grad()
        self.net.train()
        self.net.prune_invisible_plane()
        self.net.planarSplat.draw_plane(epoch=self.iter_step)
    
    
    
    
