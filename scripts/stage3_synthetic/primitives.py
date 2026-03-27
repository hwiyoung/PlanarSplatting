#!/usr/bin/env python3
"""
Stage 3-Synthetic A: Generate primitives from GT building faces.

Each GT face → N primitives scattered across its surface.
Primitives have: center (3,), normal (3,), area (scalar), semantic_prob (4,).
"""

import numpy as np


def triangulate_polygon(vertices, face_indices):
    """Fan-triangulate a polygon. Returns list of (v0, v1, v2) arrays."""
    pts = vertices[face_indices]
    tris = []
    for i in range(1, len(pts) - 1):
        tris.append((pts[0], pts[i], pts[i + 1]))
    return tris


def polygon_area(vertices, face_indices):
    """Compute polygon area via fan triangulation."""
    pts = vertices[face_indices]
    area = 0.0
    for i in range(1, len(pts) - 1):
        area += np.linalg.norm(np.cross(pts[i] - pts[0], pts[i + 1] - pts[0])) / 2
    return area


def sample_point_on_polygon(vertices, face_indices, rng):
    """Sample a uniform random point on a polygon surface."""
    tris = triangulate_polygon(vertices, face_indices)
    # Weight by triangle area
    areas = [np.linalg.norm(np.cross(t[1] - t[0], t[2] - t[0])) / 2 for t in tris]
    total = sum(areas)
    if total < 1e-12:
        return vertices[face_indices[0]]
    probs = [a / total for a in areas]
    ti = rng.choice(len(tris), p=probs)
    v0, v1, v2 = tris[ti]
    u, v = rng.random(), rng.random()
    if u + v > 1:
        u, v = 1 - u, 1 - v
    return (1 - u - v) * v0 + u * v1 + v * v2


def generate_primitives_for_building(bldg, n_prims_per_face, seed=42):
    """
    Generate primitives for one building.

    Args:
        bldg: dict with vertices, faces, normals, labels
        n_prims_per_face: int, number of primitives per GT face
        seed: random seed

    Returns:
        dict with centers (M,3), normals (M,3), areas (M,),
        semantic_probs (M,4), face_ids (M,) mapping to GT face index
    """
    rng = np.random.RandomState(seed)
    V = bldg['vertices']

    all_centers = []
    all_normals = []
    all_areas = []
    all_probs = []
    all_face_ids = []

    for fi, face in enumerate(bldg['faces']):
        label = bldg['labels'][fi]
        normal = bldg['normals'][fi]
        face_area = polygon_area(V, face)
        prim_area = face_area / n_prims_per_face

        # One-hot semantic prob for GT
        prob = np.zeros(4)
        prob[label] = 1.0

        for _ in range(n_prims_per_face):
            center = sample_point_on_polygon(V, face, rng)
            # Tiny jitter for realism (within-face only)
            center += rng.randn(3) * 0.01

            all_centers.append(center)
            all_normals.append(normal.copy())
            all_areas.append(prim_area)
            all_probs.append(prob.copy())
            all_face_ids.append(fi)

    return {
        'centers': np.array(all_centers),
        'normals': np.array(all_normals),
        'areas': np.array(all_areas),
        'semantic_probs': np.array(all_probs),
        'face_ids': np.array(all_face_ids),
    }


def add_noise_normal(primitives, sigma_deg, rng):
    """
    Add angular noise to primitive normals via tangent-plane rotation.

    For each normal n:
      1. Build orthonormal tangent basis (t1, t2) on the plane perpendicular to n
      2. Sample rotation angle θ ~ Rayleigh(σ), azimuth φ ~ Uniform(0, 2π)
      3. Rotate n by θ around a random axis in the tangent plane

    This ensures the angular deviation follows a proper distribution on the
    sphere: σ_deg controls the RMS angular deviation regardless of n direction.
    """
    prims = {k: v.copy() for k, v in primitives.items()}
    sigma_rad = np.radians(sigma_deg)
    N = len(prims['normals'])
    normals = prims['normals'].copy()

    for i in range(N):
        n = normals[i]
        n = n / (np.linalg.norm(n) + 1e-12)

        # Build tangent basis
        if abs(n[0]) < 0.9:
            t1 = np.cross(n, np.array([1, 0, 0]))
        else:
            t1 = np.cross(n, np.array([0, 1, 0]))
        t1 /= np.linalg.norm(t1) + 1e-12
        t2 = np.cross(n, t1)

        # Sample angular deviation (2D Gaussian on tangent plane → Rayleigh angle)
        dx = rng.randn() * sigma_rad
        dy = rng.randn() * sigma_rad
        perturbed = n + dx * t1 + dy * t2
        perturbed /= np.linalg.norm(perturbed) + 1e-12
        normals[i] = perturbed

    prims['normals'] = normals
    return prims


def add_noise_position(primitives, sigma_m, mode, rng):
    """
    Add positional noise.
    mode: 'isotropic' (3-axis equal), 'vertical' (Y-only), 'horizontal' (XZ-only)
    """
    prims = {k: v.copy() for k, v in primitives.items()}
    N = len(prims['centers'])
    if mode == 'isotropic':
        prims['centers'] += rng.randn(N, 3) * sigma_m
    elif mode == 'vertical':
        prims['centers'][:, 1] += rng.randn(N) * sigma_m
    elif mode == 'horizontal':
        noise = rng.randn(N, 3) * sigma_m
        noise[:, 1] = 0
        prims['centers'] += noise
    return prims


def add_noise_classification(primitives, error_rate, rng):
    """Randomly misclassify a fraction of primitives (among roof=1, wall=2, ground=3)."""
    prims = {k: v.copy() for k, v in primitives.items()}
    N = len(prims['semantic_probs'])
    n_flip = int(N * error_rate)
    flip_ids = rng.choice(N, size=n_flip, replace=False)
    classes = [1, 2, 3]
    for idx in flip_ids:
        current = prims['semantic_probs'][idx].argmax()
        others = [c for c in classes if c != current]
        new_cls = rng.choice(others)
        prims['semantic_probs'][idx] = 0.0
        prims['semantic_probs'][idx, new_cls] = 1.0
    return prims


def add_noise_missing_faces(primitives, drop_rate, rng):
    """Randomly remove a fraction of individual primitives."""
    prims = {k: v.copy() for k, v in primitives.items()}
    N = len(prims['centers'])
    n_drop = int(N * drop_rate)
    if n_drop >= N - 4:
        n_drop = N - 4  # keep at least 4
    drop_ids = rng.choice(N, size=n_drop, replace=False)
    keep = np.ones(N, dtype=bool)
    keep[drop_ids] = False
    for k in ['centers', 'normals', 'areas', 'semantic_probs', 'face_ids']:
        prims[k] = prims[k][keep]
    return prims


def add_noise_outliers(primitives, outlier_rate, rng):
    """
    Add outlier primitives: large normal deviation + position shift + class error.
    """
    prims = {k: v.copy() for k, v in primitives.items()}
    N = len(prims['centers'])
    n_outlier = max(1, int(N * outlier_rate))
    outlier_ids = rng.choice(N, size=n_outlier, replace=False)

    for idx in outlier_ids:
        # Random normal (90°+ from original)
        random_n = rng.randn(3)
        random_n /= np.linalg.norm(random_n) + 1e-12
        orig_n = prims['normals'][idx]
        # Ensure > 90° deviation
        if np.dot(random_n, orig_n) > 0:
            random_n = -random_n
        prims['normals'][idx] = random_n

        # Large position shift (2-5m)
        prims['centers'][idx] += rng.randn(3) * rng.uniform(2, 5)

        # Random class
        new_cls = rng.choice([1, 2, 3])
        prims['semantic_probs'][idx] = 0.0
        prims['semantic_probs'][idx, new_cls] = 1.0

    return prims


def add_noise_area(primitives, sigma_pct, rng):
    """
    Scale each primitive's area by a log-normal factor.
    sigma_pct: standard deviation as percentage (e.g., 50 means ~×0.5~×2.0).
    Log-normal ensures area stays positive.
    """
    prims = {k: v.copy() for k, v in primitives.items()}
    N = len(prims['areas'])
    log_std = np.log(1.0 + sigma_pct / 100.0)
    factors = np.exp(rng.randn(N) * log_std)
    prims['areas'] = prims['areas'] * factors
    return prims
