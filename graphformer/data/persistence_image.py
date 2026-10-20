"""
Persistence Image Generation Module

Functions for generating persistence images from SWC neuron morphology files
using Topological Morphology Descriptor (TMD) approach.
"""

import math
import numpy as np
from collections import defaultdict, deque
from typing import Tuple, Dict, List


def read_swc_robust(path: str):
    """Parse a SWC file -> list of tuples (id, type, x, y, z, r, parent)."""
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 7:
                continue
            try:
                nid = int(float(parts[0]))
                ntype = int(float(parts[1]))
                x = float(parts[2]); y = float(parts[3]); z = float(parts[4])
                r = float(parts[5]); parent = int(float(parts[6]))
                rows.append((nid, ntype, x, y, z, r, parent))
            except Exception:
                continue
    if not rows:
        raise ValueError(f"No valid SWC rows in {path}")
    return rows


def build_tree(rows) -> Tuple[int, Dict[int, List[int]], Dict[int, Tuple]]:
    """Return (root_id, children_map, id->row_dict)."""
    id2row = {int(r[0]): r for r in rows}
    children = defaultdict(list)
    for (nid, _t, _x, _y, _z, _r, parent) in rows:
        if int(parent) != -1:
            children[int(parent)].append(int(nid))
    roots = [int(r[0]) for r in rows if int(r[6]) == -1]
    root = roots[0] if roots else int(rows[0][0])
    return root, children, id2row


def enforce_monotone_radial(root: int, children: Dict[int, List[int]], id2row: Dict[int, Tuple]) -> Dict[int, float]:
    """Radial Euclidean distance to soma, enforced non-decreasing along root->leaf."""
    sx, sy, sz = id2row[root][2], id2row[root][3], id2row[root][4]
    raw = {}
    for nid, (_nid, _t, x, y, z, _r, _p) in id2row.items():
        raw[nid] = math.sqrt((x - sx) ** 2 + (y - sy) ** 2 + (z - sz) ** 2)

    fval = {root: raw[root]}
    q = deque([root])
    while q:
        u = q.popleft()
        for v in children[u]:
            fval[v] = max(fval[u], raw[v])
            q.append(v)
    return fval


def tmd_pairs_with_identities(root: int, children: Dict[int, List[int]], fval: Dict[int, float]):
    """TMD elder-rule pairing on a rooted tree for a monotone node function fval."""
    stack = [(root, 0)]
    post = []
    while stack:
        u, st = stack.pop()
        if st == 0:
            stack.append((u, 1))
            for v in children[u]:
                stack.append((v, 0))
        else:
            post.append(u)

    champion = {}
    pairs = []
    for u in post:
        if not children[u]:
            champion[u] = (u, fval[u])
        else:
            child_champs = [champion[v] for v in children[u]]
            best_idx = int(np.argmax([b for (_leaf, b) in child_champs]))
            best = child_champs[best_idx]
            for j, ch in enumerate(child_champs):
                if j == best_idx:
                    continue
                leaf_id, birth_val = ch
                pairs.append((float(birth_val), float(fval[u]), int(leaf_id), int(u)))
            champion[u] = best
    return pairs


def parent_map(rows) -> Dict[int, int]:
    return {int(r[0]): int(r[6]) for r in rows}


def path_nodes(parent: Dict[int, int], start: int, stop_exclusive: int) -> List[int]:
    """Nodes from 'start' up to but not including 'stop_exclusive'."""
    u = start
    nodes = [u]
    while u != stop_exclusive and u != -1:
        u = parent.get(u, -1)
        if u == -1:
            break
        nodes.append(u)
    return nodes


def compute_pair_features(pairs, parent, id2row):
    """Add persistence and mean radius along the leaf->death path."""
    feats = []
    for (birth, death, leaf, death_node) in pairs:
        nodes = path_nodes(parent, leaf, death_node)
        radii = [float(id2row[n][5]) for n in nodes if n in id2row]
        mean_rad = float(np.mean(radii)) if radii else 0.0
        feats.append({
            "birth": float(birth),
            "death": float(death),
            "persistence": float(birth - death),
            "mean_radius": mean_rad
        })
    return feats


def pi_weighted(pairs_feats: List[dict], H=256, W=256, sigma_px=16.0, weight_key=None, pdf=True,
                global_bounds=None) -> np.ndarray:
    """Build a 2D density on (birth, persistence).

    Args:
        pairs_feats: List of feature dictionaries
        H, W: Image dimensions
        sigma_px: Gaussian kernel width in pixels
        weight_key: Optional key for weighting points
        pdf: If True, normalize kernel as PDF
        global_bounds: Optional tuple (birth_min, birth_max, pers_min, pers_max)
    """
    if not pairs_feats:
        return np.zeros((H, W), dtype=np.float32)

    births = np.array([p["birth"] for p in pairs_feats], dtype=np.float32)
    pers = np.array([p["persistence"] for p in pairs_feats], dtype=np.float32)

    mask = pers > 1e-9
    births, pers = births[mask], pers[mask]

    if len(births) == 0:
        return np.zeros((H, W), dtype=np.float32)

    if global_bounds is not None:
        xmin, xmax, ymin, ymax = global_bounds
    else:
        xmin, xmax = births.min(), births.max()
        ymin, ymax = pers.min(), pers.max()

    eps = 1e-8
    births_clamped = np.clip(births, xmin, xmax)
    pers_clamped = np.clip(pers, ymin, ymax)

    u = (births_clamped - xmin) / (xmax - xmin + eps) * (W - 1)
    v = (pers_clamped - ymin) / (ymax - ymin + eps) * (H - 1)

    if weight_key is None:
        w = np.ones_like(u, dtype=np.float32)
    else:
        raw = np.array([p[weight_key] for p in pairs_feats], dtype=np.float32)[mask]
        lo, hi = np.percentile(raw, 1), np.percentile(raw, 99)
        w = np.clip((raw - lo) / (hi - lo + 1e-8), 0, 1).astype(np.float32)
        if np.all(w == 0):
            w = np.ones_like(u, dtype=np.float32)

    X, Y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    out = np.zeros((H, W), dtype=np.float32)
    s2 = 2.0 * (sigma_px ** 2)
    k_norm = (1.0 / (2.0 * math.pi * (sigma_px ** 2))) if pdf else 1.0

    for i in range(len(u)):
        cx, cy = u[i], v[i]
        D = (X - cx) ** 2 + (Y - cy) ** 2
        out += (w[i] * k_norm) * np.exp(-D / s2)

    return out


def make_rgb_pi(rows, H=256, W=256, sigma_px=16.0, global_bounds=(0, 300, 0, 100),
                persistence_aug=None, channel_mode='RGB') -> np.ndarray:
    """Compose R/G/B channels and return float32 image in [0,1].

    Args:
        rows: SWC rows from read_swc_robust
        H, W: Image dimensions
        sigma_px: Gaussian kernel width
        global_bounds: Tuple (birth_min, birth_max, pers_min, pers_max)
        persistence_aug: Optional CombinedPersistenceAugmentation instance
        channel_mode: 'RGB', 'R_only', 'G_only', 'B_only', etc.
    """
    root, children, id2row = build_tree(rows)
    fval = enforce_monotone_radial(root, children, id2row)
    pairs = tmd_pairs_with_identities(root, children, fval)
    pmap = parent_map(rows)
    feats = compute_pair_features(pairs, pmap, id2row)

    if persistence_aug is not None:
        feats = persistence_aug.augment_pairs_features(feats, global_bounds)
        sigma_px = persistence_aug.sample_sigma(sigma_px)

    R = pi_weighted(feats, H=H, W=W, sigma_px=sigma_px, weight_key=None, pdf=True, global_bounds=global_bounds)
    G = pi_weighted(feats, H=H, W=W, sigma_px=sigma_px, weight_key="persistence", pdf=True, global_bounds=global_bounds)
    B = pi_weighted(feats, H=H, W=W, sigma_px=sigma_px, weight_key="mean_radius", pdf=True, global_bounds=global_bounds)

    img = np.stack([R, G, B], axis=-1).astype(np.float32)
    max_val = img.max()
    min_val = img.min()
    img = (img - min_val) / (max_val - min_val + 1e-8)

    return img
