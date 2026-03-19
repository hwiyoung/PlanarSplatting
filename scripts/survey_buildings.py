"""Survey all buildings to find well-formed ones for Phase 4-A pilot."""
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster

prims = np.load('results/phase4/part1/primitives_data.npz')
faces = np.load('results/phase4_prime/step3_buildings/building_faces.npz', allow_pickle=True)

gravity = np.array([0.1223, 0.3702, 0.9209])
gravity /= np.linalg.norm(gravity)

all_bids = np.unique(faces['building_ids'])
all_bids = all_bids[all_bids >= 0]

print(f"Total buildings: {len(all_bids)}")
print(f"{'BID':>4} {'R':>3} {'W':>3} {'Tot':>4} | {'R_h%':>4} {'W_v30%':>6} {'W_v50%':>6} | "
      f"{'H':>5} {'Sprd':>5} {'Asp':>4} | {'Wclus':>5} {'Rabove':>6} | Note")
print("-" * 100)

good = []

for bid in all_bids:
    bld_mask = faces['building_ids'] == bid
    unique_prims = np.unique(faces['face_prim_indices'][bld_mask])
    unique_prims = unique_prims[unique_prims >= 0]

    centers = prims['centers'][unique_prims]
    normals = prims['normals'][unique_prims]
    classes = prims['class_pred'][unique_prims]
    sprobs = prims['semantic_probs'][unique_prims]

    bmask = (classes == 1) | (classes == 2)
    if bmask.sum() < 5:
        continue

    bc, bn, bcl, bsp = centers[bmask], normals[bmask], classes[bmask], sprobs[bmask]
    nr, nw = int((bcl == 1).sum()), int((bcl == 2).sum())

    if nr == 0 or nw < 3:
        continue

    dots = np.abs(bn @ gravity)

    r_horiz = 100 * (dots[bcl == 1] > 0.7).mean()
    w_vert30 = 100 * (dots[bcl == 2] < 0.3).mean()
    w_vert50 = 100 * (dots[bcl == 2] < 0.5).mean()

    heights = bc @ (-gravity)
    hr = float(heights.max() - heights.min())
    horiz = bc - np.outer(bc @ gravity, gravity)
    hs = float(np.linalg.norm(horiz.max(0) - horiz.min(0)))
    asp = hs / max(hr, 1e-6)

    r_heights = heights[bcl == 1]
    w_heights = heights[bcl == 2]
    roof_above = r_heights.mean() > w_heights.mean()

    n_wall_clusters = 0
    if nw >= 2:
        wn = bn[bcl == 2]
        cdist = [max(0., 1. - float(np.dot(wn[i], wn[j])))
                 for i in range(len(wn)) for j in range(i + 1, len(wn))]
        if len(cdist) > 0:
            Z = linkage(np.array(cdist), method='complete')
            labels = fcluster(Z, t=0.30, criterion='distance')
            n_wall_clusters = len(np.unique(labels))

    note = ""
    if not roof_above:
        note += "Rlow "
    if asp < 1.5:
        note += "narrow "
    if asp > 10:
        note += "spread "
    if n_wall_clusters < 2:
        note += "1wdir "

    is_good = (r_horiz >= 80 and w_vert50 >= 60 and roof_above and
               n_wall_clusters >= 2 and 1.5 <= asp <= 10 and nr >= 3 and nw >= 8)

    marker = "*** " if is_good else ""
    print(f"{bid:4d} {nr:3d} {nw:3d} {nr+nw:4d} | {r_horiz:3.0f}% {w_vert30:5.0f}% {w_vert50:5.0f}% | "
          f"{hr:5.3f} {hs:5.3f} {asp:4.1f} | {n_wall_clusters:5d} {str(roof_above):>6} | {marker}{note}")

    if is_good:
        good.append((bid, nr, nw, w_vert50, n_wall_clusters, asp))

print(f"\n*** Good candidates: {len(good)}")
for bid, nr, nw, wv, nwc, asp in sorted(good, key=lambda x: -x[3]):
    print(f"  Building {bid}: R={nr} W={nw} W_vert50={wv:.0f}% wall_clusters={nwc} aspect={asp:.1f}")
