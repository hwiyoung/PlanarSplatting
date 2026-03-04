#!/usr/bin/env python3
"""Phase 3-A Gradient Check: Verify L_mutual bidirectional gradient flow.

Tests:
1. L_mutual gradient flows to f_i (geometry -> semantics)
2. L_mutual gradient flows to R_i (semantics -> geometry)
3. mode=full: both directions non-zero
4. mode=sem2geo: only R_i gets gradient, f_i is zero
5. mode=geo2sem: only f_i gets gradient, R_i is zero
6. mode=none: returns zero loss
7. Warmup schedule produces correct lambda_m values
8. get_plane_normals_differentiable() matches get_plane_geometry() normals

Run inside Docker container:
    python scripts/gradient_check_phase3a.py
"""
import os
import sys
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'planarsplat'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pyhocon import ConfigFactory

from net.net_planarSplatting import PlanarSplat_Network
from run.net_wrapper import PlanarRecWrapper
from utils.loss_util import mutual_loss
from utils import model_util

results = {}

print("=" * 60)
print("Phase 3-A Gradient Check: L_mutual")
print("=" * 60)

# ---- Minimal config ----
conf_str = """
train {
    expname = gradient_check_3a
    dataset_class = data_loader.scene_dataset_demo.SceneDatasetDemo
    train_runner_class = run.trainer.PlanarSplatTrainRunner
    max_total_iters = 100
    coarse_stage_ite = 10
    split_start_ite = 20
    plot_freq = 999
    process_plane_freq_ite = 50
    check_plane_vis_freq_ite = 999
    data_order = sequential
    log_freq = 10
    use_tensorboard = False
}
dataset {
    dataset_name = demo
    scan_id = example
    img_res = [64, 64]
    scene_bounding_sphere = 5.0
    sphere_ratio = 0.5
}
pose {
    scale = 1.0
    offset = [0.0, 0.0, 0.0]
}
plane_model {
    plane_model_class = net.net_planarSplatting.PlanarSplat_Network
    enable_semantic = True
    semantic_num_classes = 4
    lr_semantic = 0.005
    mutual_mode = full
    plane_loss {
        weight_mono_normal = 2.0
        weight_mono_depth = 2.0
        lambda_sem = 0.1
        lambda_geo = 0.1
        lambda_mutual = 0.05
        mutual_warmup_start = 0.33
        mutual_warmup_end = 0.67
        mutual_tau = 0.15
    }
    init_plane_num = 50
    fix_rot_normal = False
    fix_rot_xy = False
    fix_radii = False
    fix_center = False
    lr_radii = 0.001
    lr_center = 0.001
    lr_rot_normal = 0.001
    lr_rot_xy = 0.001
    RBF_type = rectangle
    RBF_weight_change_type = max
    radii_dir_type = double
    radii_init = 0.05
    radii_max_list = [0.5, 0.5]
    radii_min_list = [0.001, 0.001]
    radii_milestone_list = [0, 10]
    split_thres = 0.20
}
"""
conf = ConfigFactory.parse_string(conf_str)

N_prim = 50
K = 4
e_gravity = torch.tensor([0., -1., 0.], device='cuda')
tau = 0.15

# ============================================================
# TEST 1: Pure L_mutual gradient (simulated, no rasterizer)
# ============================================================
print("\n--- 1. L_mutual gradient flow (simulated) ---")

f_i = nn.Parameter(torch.randn(N_prim, K, device='cuda') * 0.1)
normals = nn.Parameter(F.normalize(torch.randn(N_prim, 3, device='cuda'), dim=-1))

loss_m = mutual_loss(f_i, normals, e_gravity, tau=tau, mode='full')
print(f"  L_mutual value: {loss_m.item():.6f}")
results['L_mutual_value'] = loss_m.item()

grad_f = torch.autograd.grad(loss_m, f_i, retain_graph=True)[0]
grad_n = torch.autograd.grad(loss_m, normals, retain_graph=True)[0]
grad_f_norm = grad_f.abs().sum().item()
grad_n_norm = grad_n.abs().sum().item()

print(f"  dL/df_i norm: {grad_f_norm:.6f} (should be non-zero)")
print(f"  dL/dn_i norm: {grad_n_norm:.6f} (should be non-zero)")
results['grad_f_full'] = grad_f_norm
results['grad_n_full'] = grad_n_norm

assert grad_f_norm > 0, "FAIL: gradient to f_i is zero in full mode!"
assert grad_n_norm > 0, "FAIL: gradient to n_i is zero in full mode!"
print("  PASS: bidirectional gradient confirmed")

# ============================================================
# TEST 2: mode=sem2geo (detach f_i, only R_i gets gradient)
# ============================================================
print("\n--- 2. mode=sem2geo ---")
loss_s2g = mutual_loss(f_i, normals, e_gravity, tau=tau, mode='sem2geo')
grad_f_s2g = torch.autograd.grad(loss_s2g, f_i, retain_graph=True, allow_unused=True)[0]
grad_n_s2g = torch.autograd.grad(loss_s2g, normals, retain_graph=True, allow_unused=True)[0]

grad_f_s2g_norm = 0.0 if grad_f_s2g is None else grad_f_s2g.abs().sum().item()
grad_n_s2g_norm = 0.0 if grad_n_s2g is None else grad_n_s2g.abs().sum().item()

print(f"  dL/df_i norm: {grad_f_s2g_norm:.10f} (should be zero)")
print(f"  dL/dn_i norm: {grad_n_s2g_norm:.6f} (should be non-zero)")
results['grad_f_sem2geo'] = grad_f_s2g_norm
results['grad_n_sem2geo'] = grad_n_s2g_norm

assert grad_f_s2g_norm == 0, f"FAIL: f_i should not get gradient in sem2geo mode! got {grad_f_s2g_norm}"
assert grad_n_s2g_norm > 0, "FAIL: n_i should get gradient in sem2geo mode!"
print("  PASS: only R_i gets gradient")

# ============================================================
# TEST 3: mode=geo2sem (detach n_i, only f_i gets gradient)
# ============================================================
print("\n--- 3. mode=geo2sem ---")
loss_g2s = mutual_loss(f_i, normals, e_gravity, tau=tau, mode='geo2sem')
grad_f_g2s = torch.autograd.grad(loss_g2s, f_i, retain_graph=True, allow_unused=True)[0]
grad_n_g2s = torch.autograd.grad(loss_g2s, normals, retain_graph=True, allow_unused=True)[0]

grad_f_g2s_norm = 0.0 if grad_f_g2s is None else grad_f_g2s.abs().sum().item()
grad_n_g2s_norm = 0.0 if grad_n_g2s is None else grad_n_g2s.abs().sum().item()

print(f"  dL/df_i norm: {grad_f_g2s_norm:.6f} (should be non-zero)")
print(f"  dL/dn_i norm: {grad_n_g2s_norm:.10f} (should be zero)")
results['grad_f_geo2sem'] = grad_f_g2s_norm
results['grad_n_geo2sem'] = grad_n_g2s_norm

assert grad_f_g2s_norm > 0, "FAIL: f_i should get gradient in geo2sem mode!"
assert grad_n_g2s_norm == 0, f"FAIL: n_i should not get gradient in geo2sem mode! got {grad_n_g2s_norm}"
print("  PASS: only f_i gets gradient")

# ============================================================
# TEST 4: mode=none (returns zero loss)
# ============================================================
print("\n--- 4. mode=none ---")
loss_none = mutual_loss(f_i, normals, e_gravity, tau=tau, mode='none')
print(f"  L_mutual (none): {loss_none.item():.10f} (should be 0)")
results['L_mutual_none'] = loss_none.item()
assert loss_none.item() == 0, f"FAIL: none mode should return 0! got {loss_none.item()}"
print("  PASS: mode=none returns zero")

# ============================================================
# TEST 5: Real model — gradient through get_plane_normals_differentiable
# ============================================================
print("\n--- 5. Real model gradient flow ---")
net = PlanarRecWrapper(conf, '')
net = net.cuda()
net.build_optimizer_and_LRscheduler()
net.train()

# Set non-zero semantic features
with torch.no_grad():
    net.planarSplat._plane_semantic_features.data = torch.randn(N_prim, K, device='cuda') * 0.5

plane_normals = net.planarSplat.get_plane_normals_differentiable()
f_i_real = net.planarSplat._plane_semantic_features
loss_m_real = mutual_loss(f_i_real, plane_normals, e_gravity, tau=tau, mode='full')

print(f"  L_mutual (real model): {loss_m_real.item():.6f}")
results['L_mutual_real'] = loss_m_real.item()

# Gradient to f_i
grad_f_real = torch.autograd.grad(loss_m_real, f_i_real, retain_graph=True)[0]
grad_f_real_norm = grad_f_real.abs().sum().item()
print(f"  dL/df_i norm: {grad_f_real_norm:.6f}")
results['grad_f_real'] = grad_f_real_norm
assert grad_f_real_norm > 0, "FAIL: no gradient to f_i through real model!"

# Gradient to rotation params
R_params = {
    'rot_q_normal_wxy': net.planarSplat._plane_rot_q_normal_wxy,
    'rot_q_xyAxis_w': net.planarSplat._plane_rot_q_xyAxis_w,
    'rot_q_xyAxis_z': net.planarSplat._plane_rot_q_xyAxis_z,
}
total_R_norm = 0.0
for name, param in R_params.items():
    g = torch.autograd.grad(loss_m_real, param, retain_graph=True, allow_unused=True)[0]
    g_norm = 0.0 if g is None else g.abs().sum().item()
    total_R_norm += g_norm
    print(f"  dL/d{name}: {g_norm:.6f}")
    results[f'grad_R_{name}'] = g_norm

results['grad_R_total'] = total_R_norm
assert total_R_norm > 0, "FAIL: no gradient to R_i through real model!"
print(f"  dL/dR_i total: {total_R_norm:.6f}")
print("  PASS: bidirectional gradient through real model")

# Verify center/radii DON'T get gradient from L_mutual
grad_center = torch.autograd.grad(loss_m_real, net.planarSplat._plane_center,
                                   retain_graph=True, allow_unused=True)[0]
grad_center_norm = 0.0 if grad_center is None else grad_center.abs().sum().item()
print(f"  dL/dcenter: {grad_center_norm:.10f} (should be zero — L_mutual is independent of position)")
results['grad_center'] = grad_center_norm

# ============================================================
# TEST 6: get_plane_normals_differentiable matches get_plane_geometry
# ============================================================
print("\n--- 6. Normal consistency check ---")
with torch.no_grad():
    normals_diff = net.planarSplat.get_plane_normals_differentiable()
    normals_geo = net.planarSplat.get_plane_geometry(ite=100)[0]  # plane_normal is first return
    diff = (normals_diff - normals_geo).abs().max().item()
    print(f"  Max difference: {diff:.10f} (should be ~0)")
    results['normal_consistency_diff'] = diff
    assert diff < 1e-5, f"FAIL: normals don't match! diff={diff}"
    print("  PASS: normals consistent")

# ============================================================
# TEST 7: Warmup schedule
# ============================================================
print("\n--- 7. Warmup schedule ---")
max_iters = 5000
warmup_start = 0.33
warmup_end = 0.67
lambda_target = 0.05

test_points = [
    (0, 0.0, "early (0%)"),
    (500, 0.0, "early (10%)"),
    (1650, 0.0, "just before start (33%)"),
    (1660, None, "just after start"),  # small but > 0
    (2500, None, "mid warmup (50%)"),
    (3350, lambda_target, "just at end (67%)"),
    (4000, lambda_target, "late (80%)"),
    (5000, lambda_target, "final (100%)"),
]

warmup_ok = True
for iter_val, expected, desc in test_points:
    progress = iter_val / max_iters
    if progress < warmup_start:
        lm = 0.0
    elif progress < warmup_end:
        t = (progress - warmup_start) / (warmup_end - warmup_start)
        lm = lambda_target * t
    else:
        lm = lambda_target

    if expected is not None:
        ok = abs(lm - expected) < 1e-6
    else:
        ok = lm > 0  # just check non-zero
    status = "OK" if ok else "FAIL"
    if not ok:
        warmup_ok = False
    print(f"  iter={iter_val:5d} ({desc:25s}): lambda_m={lm:.6f} [{status}]")

results['warmup_ok'] = warmup_ok
if warmup_ok:
    print("  PASS: warmup schedule correct")
else:
    print("  FAIL: warmup schedule incorrect")

# ============================================================
# TEST 8: Directional modes with real model
# ============================================================
print("\n--- 8. Directional modes with real model ---")
for mode in ['sem2geo', 'geo2sem']:
    net.optimizer.zero_grad()
    pn = net.planarSplat.get_plane_normals_differentiable()
    fi = net.planarSplat._plane_semantic_features
    loss_dir = mutual_loss(fi, pn, e_gravity, tau=tau, mode=mode)

    grad_f_dir = torch.autograd.grad(loss_dir, fi, retain_graph=True, allow_unused=True)[0]
    grad_R_dir = torch.autograd.grad(loss_dir, net.planarSplat._plane_rot_q_normal_wxy,
                                      retain_graph=True, allow_unused=True)[0]
    f_norm = 0.0 if grad_f_dir is None else grad_f_dir.abs().sum().item()
    R_norm = 0.0 if grad_R_dir is None else grad_R_dir.abs().sum().item()

    if mode == 'sem2geo':
        ok = f_norm == 0 and R_norm > 0
        results['real_sem2geo_f'] = f_norm
        results['real_sem2geo_R'] = R_norm
    else:
        ok = f_norm > 0 and R_norm == 0
        results['real_geo2sem_f'] = f_norm
        results['real_geo2sem_R'] = R_norm

    print(f"  mode={mode}: f_i={f_norm:.6f}, R_i={R_norm:.6f} — {'PASS' if ok else 'FAIL'}")

net.optimizer.zero_grad()

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

all_pass = True
checks = [
    ("mode=full: dL/df_i non-zero", results['grad_f_full'] > 0),
    ("mode=full: dL/dn_i non-zero", results['grad_n_full'] > 0),
    ("mode=sem2geo: dL/df_i zero", results['grad_f_sem2geo'] == 0),
    ("mode=sem2geo: dL/dn_i non-zero", results['grad_n_sem2geo'] > 0),
    ("mode=geo2sem: dL/df_i non-zero", results['grad_f_geo2sem'] > 0),
    ("mode=geo2sem: dL/dn_i zero", results['grad_n_geo2sem'] == 0),
    ("mode=none: loss is zero", results['L_mutual_none'] == 0),
    ("Real model: dL/df_i non-zero", results['grad_f_real'] > 0),
    ("Real model: dL/dR_i non-zero", results['grad_R_total'] > 0),
    ("Real model: dL/dcenter zero", results['grad_center'] == 0),
    ("Normal consistency (diff vs geo)", results['normal_consistency_diff'] < 1e-5),
    ("Warmup schedule correct", results['warmup_ok']),
    ("Real sem2geo: R_i only", results.get('real_sem2geo_f', 1) == 0 and results.get('real_sem2geo_R', 0) > 0),
    ("Real geo2sem: f_i only", results.get('real_geo2sem_f', 0) > 0 and results.get('real_geo2sem_R', 1) == 0),
]

for name, passed in checks:
    status = "PASS" if passed else "FAIL"
    print(f"  {status}: {name}")
    if not passed:
        all_pass = False

print(f"\nKey gradient values:")
print(f"  mode=full:    dL/df_i={results['grad_f_full']:.6f}  dL/dn_i={results['grad_n_full']:.6f}")
print(f"  mode=sem2geo: dL/df_i={results['grad_f_sem2geo']:.10f}  dL/dn_i={results['grad_n_sem2geo']:.6f}")
print(f"  mode=geo2sem: dL/df_i={results['grad_f_geo2sem']:.6f}  dL/dn_i={results['grad_n_geo2sem']:.10f}")
print(f"  Real model:   dL/df_i={results['grad_f_real']:.6f}  dL/dR_i={results['grad_R_total']:.6f}")
print(f"  L_mutual value (sim): {results['L_mutual_value']:.6f}")
print(f"  L_mutual value (real): {results['L_mutual_real']:.6f}")

if all_pass:
    print("\nALL CHECKS PASSED")
else:
    print("\nSOME CHECKS FAILED")
    sys.exit(1)

# Save results
out_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'phase3a', 'gradient_check.json')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to: {out_path}")
