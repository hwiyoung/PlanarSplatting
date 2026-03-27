#!/usr/bin/env python3
"""
Parse 3D BAG CityJSON tiles into building dicts compatible with the
Synthetic A experiment pipeline.

Each building: {id, name, type, vertices, faces, labels, normals, face_names}
Labels: 1=Roof, 2=Wall, 3=Ground

3D BAG coordinate system: EPSG:28992 (RD New), Z-up.
Our pipeline expects COLMAP convention (-Y up), so we convert:
  3D BAG (X, Y, Z-up) → COLMAP (X, -Z, -Y) approximately.
  Actually, since Stage 3 uses gravity=[0,-1,0] in COLMAP,
  we map Z-up to -Y: swap Y↔Z and negate Y.
"""

import glob
import json
import os
import sys

import numpy as np


def _polygon_area_3d(pts):
    """Compute area of a 3D polygon using cross products."""
    if len(pts) < 3:
        return 0.0
    area = 0.0
    for k in range(1, len(pts) - 1):
        e1 = pts[k] - pts[0]
        e2 = pts[k + 1] - pts[0]
        area += np.linalg.norm(np.cross(e1, e2)) / 2
    return area


def _face_normal_3d(pts):
    """Compute outward normal of a 3D polygon."""
    if len(pts) < 3:
        return np.array([0.0, 0.0, 1.0])
    e1 = pts[1] - pts[0]
    e2 = pts[2] - pts[0]
    n = np.cross(e1, e2)
    nrm = np.linalg.norm(n)
    return n / nrm if nrm > 1e-12 else np.array([0.0, 0.0, 1.0])


def _zup_to_colmap(vertices, normals):
    """Convert Z-up (3D BAG) to COLMAP convention (-Y up).

    3D BAG: X=east, Y=north, Z=up
    COLMAP: X=right, Y=down, Z=forward

    Mapping: X_colmap = X_bag, Y_colmap = -Z_bag, Z_colmap = Y_bag
    """
    V = vertices.copy()
    N = normals.copy()

    # Swap Y and Z, negate new Y (was Z, now -Z = down)
    new_y = -V[:, 2].copy()  # -Z_bag
    new_z = V[:, 1].copy()   # Y_bag
    V[:, 1] = new_y
    V[:, 2] = new_z

    new_ny = -N[:, 2].copy()
    new_nz = N[:, 1].copy()
    N[:, 1] = new_ny
    N[:, 2] = new_nz

    return V, N


def parse_tile(tile_path, min_area=10.0, max_faces=200):
    """Parse a 3D BAG CityJSON tile into building dicts.

    Args:
        tile_path: path to .city.json file
        min_area: minimum ground area to include (m²)
        max_faces: maximum faces per building (skip very complex buildings)

    Returns:
        list of building dicts
    """
    with open(tile_path) as f:
        data = json.load(f)

    transform = data.get('transform', {})
    scale = np.array(transform.get('scale', [1, 1, 1]))
    translate = np.array(transform.get('translate', [0, 0, 0]))
    raw_verts = np.array(data.get('vertices', []), dtype=np.float64)
    world_verts = raw_verts * scale + translate

    # Build parent→children mapping
    children_map = {}  # parent_name → [child_names]
    for obj_name, obj in data.get('CityObjects', {}).items():
        for parent in obj.get('parents', []):
            children_map.setdefault(parent, []).append(obj_name)

    buildings = []

    for obj_name, obj in data.get('CityObjects', {}).items():
        if obj.get('type') != 'Building':
            continue

        attrs = obj.get('attributes', {})
        roof_type = attrs.get('b3_dak_type', 'unknown')
        ground_area = attrs.get('b3_opp_grond', 0)

        if ground_area < min_area:
            continue

        # Get BuildingPart children with LOD2.2
        children = children_map.get(obj_name, [])

        all_faces = []  # list of (vertices_array, label, face_name)

        for child_name in children:
            child = data['CityObjects'].get(child_name)
            if not child:
                continue

            for geom in child.get('geometry', []):
                if str(geom.get('lod')) != '2.2':
                    continue
                if geom.get('type') != 'Solid':
                    continue

                sem = geom.get('semantics', {})
                surfaces = sem.get('surfaces', [])
                values = sem.get('values', [[]])[0]
                boundaries = geom.get('boundaries', [[]])[0]

                for fi, face_rings in enumerate(boundaries):
                    ring = face_rings[0]  # outer ring only
                    if len(ring) < 3:
                        continue

                    pts = world_verts[ring]

                    # Semantic type
                    stype = 'Unknown'
                    if (fi < len(values) and values[fi] is not None
                            and values[fi] < len(surfaces)):
                        stype = surfaces[values[fi]].get('type', 'Unknown')

                    label = 0
                    if 'Roof' in stype:
                        label = 1
                    elif 'Wall' in stype:
                        label = 2
                    elif 'Ground' in stype:
                        label = 3

                    if label == 0:
                        continue  # skip unknown faces

                    all_faces.append((pts, label, stype))

        if not all_faces:
            continue
        if len(all_faces) > max_faces:
            continue  # skip very complex buildings

        n_roof = sum(1 for _, l, _ in all_faces if l == 1)
        n_wall = sum(1 for _, l, _ in all_faces if l == 2)
        if n_roof == 0 or n_wall == 0:
            continue

        # Build vertices array and face index lists
        all_verts = []
        faces_list = []
        labels_list = []
        normals_list = []
        names_list = []
        vert_offset = 0

        for pts, label, stype in all_faces:
            n_pts = len(pts)
            all_verts.append(pts)
            faces_list.append(list(range(vert_offset, vert_offset + n_pts)))
            labels_list.append(label)
            normals_list.append(_face_normal_3d(pts))
            names_list.append(stype)
            vert_offset += n_pts

        V = np.vstack(all_verts)
        N = np.array(normals_list)

        # Convert to COLMAP convention
        V, N = _zup_to_colmap(V, N)

        # Center the building at origin (XZ)
        center_xz = V[:, [0, 2]].mean(axis=0)
        V[:, 0] -= center_xz[0]
        V[:, 2] -= center_xz[1]

        buildings.append({
            'id': len(buildings),
            'name': obj_name,
            'type': roof_type,
            'vertices': V,
            'faces': faces_list,
            'labels': np.array(labels_list),
            'normals': N,
            'face_names': names_list,
            'source_tile': os.path.basename(tile_path),
            'ground_area': ground_area,
        })

    return buildings


def load_scene(scene_dir, **kwargs):
    """Load all buildings from all tiles in a scene directory."""
    tile_files = sorted(glob.glob(os.path.join(scene_dir, '*.city.json')))
    all_buildings = []
    for tf in tile_files:
        buildings = parse_tile(tf, **kwargs)
        # Re-index
        for b in buildings:
            b['id'] = len(all_buildings)
            all_buildings.append(b)
    return all_buildings


def load_all_scenes(base_dir='results/stage3_synthetic_a/3dbag_raw', **kwargs):
    """Load buildings from all scenes."""
    scenes = {}
    for scene_name in sorted(os.listdir(base_dir)):
        scene_dir = os.path.join(base_dir, scene_name)
        if not os.path.isdir(scene_dir):
            continue
        buildings = load_scene(scene_dir, **kwargs)
        scenes[scene_name] = buildings
    return scenes


if __name__ == '__main__':
    scenes = load_all_scenes()
    for scene_name, buildings in scenes.items():
        roof_types = {}
        for b in buildings:
            roof_types[b['type']] = roof_types.get(b['type'], 0) + 1

        print(f'{scene_name}: {len(buildings)} buildings')
        print(f'  Roof types: {roof_types}')
        if buildings:
            face_counts = [len(b['faces']) for b in buildings]
            print(f'  Faces: min={min(face_counts)}, max={max(face_counts)}, '
                  f'mean={np.mean(face_counts):.0f}')
        print()
