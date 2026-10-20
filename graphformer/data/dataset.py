from logging import raiseExceptions
import torch, dgl, os, pickle
import networkx as nx
from dgl.convert import from_networkx
from collections import namedtuple, defaultdict, deque
from nltk import tree
from tqdm import tqdm
import numpy as np
import re
import math
import colorsys
from typing import Tuple, Dict, List
from PIL import Image
from ..augmentations.persistence_augmentations import CombinedPersistenceAugmentation


# Image generation functions
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
                # Skip malformed lines
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
    """Radial Euclidean distance to soma, enforced non-decreasing along root→leaf."""
    sx, sy, sz = id2row[root][2], id2row[root][3], id2row[root][4]
    raw = {}
    for nid, (_nid, _t, x, y, z, _r, _p) in id2row.items():
        raw[nid] = math.sqrt((x - sx) ** 2 + (y - sy) ** 2 + (z - sz) ** 2)

    fval = {root: raw[root]}
    q = deque([root])
    while q:
        u = q.popleft()
        for v in children[u]:
            fval[v] = max(fval[u], raw[v])  # cumulative max
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
    """Add persistence and mean radius along the leaf→death path."""
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
        global_bounds: Optional tuple (birth_min, birth_max, pers_min, pers_max) for consistent normalization
    """
    if not pairs_feats:
        return np.zeros((H, W), dtype=np.float32)

    births = np.array([p["birth"] for p in pairs_feats], dtype=np.float32)
    pers   = np.array([p["persistence"] for p in pairs_feats], dtype=np.float32)

    mask = pers > 1e-9
    births, pers = births[mask], pers[mask]

    if len(births) == 0:
        return np.zeros((H, W), dtype=np.float32)

    # Use global bounds if provided, otherwise use local bounds
    if global_bounds is not None:
        xmin, xmax, ymin, ymax = global_bounds
    else:
        xmin, xmax = births.min(), births.max()
        ymin, ymax = pers.min(),   pers.max()

    eps = 1e-8
    # Clamp values to global bounds to handle outliers
    births_clamped = np.clip(births, xmin, xmax)
    pers_clamped = np.clip(pers, ymin, ymax)

    u = (births_clamped - xmin) / (xmax - xmin + eps) * (W - 1)
    v = (pers_clamped   - ymin) / (ymax - ymin + eps) * (H - 1)

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

def compute_global_bounds(all_rows):
    births, pers = [], []
    for rows in all_rows:
        root, children, id2row = build_tree(rows)
        fval = enforce_monotone_radial(root, children, id2row)
        pairs = tmd_pairs_with_identities(root, children, fval)
        feats = compute_pair_features(pairs, parent_map(rows), id2row)
        for p in feats:
            births.append(p["birth"])
            pers.append(p["persistence"])
    return np.min(births), np.max(births), np.min(pers), np.max(pers)

def make_rgb_pi(rows, H=256, W=256, sigma_px=16.0, global_bounds=(0, 300, 0, 100), persistence_aug=None, channel_mode='RGB') -> np.ndarray:
    """Compose R/G/B channels and return float32 image in [0,1].

    Args:
        rows: SWC rows from read_swc_robust
        H, W: Image dimensions
        sigma_px: Gaussian kernel width
        global_bounds: Optional tuple (birth_min, birth_max, pers_min, pers_max) for consistent normalization
                      Uses same birth/persistence bounds for all channels (x/y axis consistency)
        persistence_aug: Optional CombinedPersistenceAugmentation instance
        channel_mode: 'RGB' (default), 'R_only', 'G_only', 'B_only' - controls which channels to use
    """
    root, children, id2row = build_tree(rows)
    fval = enforce_monotone_radial(root, children, id2row)
    pairs = tmd_pairs_with_identities(root, children, fval)
    pmap = parent_map(rows)
    feats = compute_pair_features(pairs, pmap, id2row)

    # Apply persistence space augmentation if provided
    if persistence_aug is not None:
        feats = persistence_aug.augment_pairs_features(feats, global_bounds)
        sigma_px = persistence_aug.sample_sigma(sigma_px)

    # All channels use same birth/persistence bounds for spatial consistency
    # The difference is in the WEIGHTING (weight_key), not the coordinate system
    R = pi_weighted(feats, H=H, W=W, sigma_px=sigma_px, weight_key=None,         pdf=True, global_bounds=global_bounds)
    G = pi_weighted(feats, H=H, W=W, sigma_px=sigma_px, weight_key="persistence", pdf=True, global_bounds=global_bounds)
    B = pi_weighted(feats, H=H, W=W, sigma_px=sigma_px, weight_key="mean_radius", pdf=True, global_bounds=global_bounds)

    # Create image based on channel mode
    if channel_mode == 'R_only':
        # R channel replicated to all 3 channels (grayscale from R)
        img = np.stack([R,  np.zeros_like(R), np.zeros_like(R)], axis=-1).astype(np.float32)
    elif channel_mode == 'G_only':
        # G channel replicated to all 3 channels (grayscale from G)
        img = np.stack([G,  np.zeros_like(G), np.zeros_like(G)], axis=-1).astype(np.float32)
    elif channel_mode == 'B_only':
        # B channel replicated to all 3 channels (grayscale from B)
        img = np.stack([B, np.zeros_like(B), np.zeros_like(B)], axis=-1).astype(np.float32)
    elif channel_mode == 'RG':
        # R and G channels, B is zero
        img = np.stack([R, G, np.zeros_like(R)], axis=-1).astype(np.float32)
    elif channel_mode == 'RB':
        # R and B channels, G is zero
        img = np.stack([R, np.zeros_like(R), B], axis=-1).astype(np.float32)
    elif channel_mode == 'GB':
        # G and B channels, R is zero
        img = np.stack([np.zeros_like(R), G, B], axis=-1).astype(np.float32)
    else:  # 'RGB' or default
        # Standard RGB image
        img = np.stack([R, G, B], axis=-1).astype(np.float32)

    img = np.stack([R, G, B], axis=-1).astype(np.float32)
#     print(img.min(), img.max())
    max_val = img.max()
    min_val = img.min()
#     if max_val > 0:
    img = (img -min_val)/(max_val-min_val)
    
    return img

def make_hsl_pi(rows, H=256, W=256, sigma_px=16.0, global_bounds=(0, 300, 0, 100),
                persistence_aug=None, channel_mode='RGB') -> np.ndarray:
    """Compose HSL color mapping of topology filtrations and return RGB float32 image in [0,1].

    Args:
        rows: SWC rows from read_swc_robust
        H, W: Image dimensions
        sigma_px: Gaussian kernel width
        global_bounds: Optional tuple (birth_min, birth_max, pers_min, pers_max)
        persistence_aug: Optional CombinedPersistenceAugmentation instance
        channel_mode: 'RGB' (default) - other modes not supported for HSL
    """
    root, children, id2row = build_tree(rows)
    fval = enforce_monotone_radial(root, children, id2row)
    pairs = tmd_pairs_with_identities(root, children, fval)
    pmap = parent_map(rows)
    feats = compute_pair_features(pairs, pmap, id2row)

    # Apply augmentation if provided
    if persistence_aug is not None:
        feats = persistence_aug.augment_pairs_features(feats, global_bounds)
        sigma_px = persistence_aug.sample_sigma(sigma_px)

    # Compute persistence image maps
    birth_map = pi_weighted(feats, H=H, W=W, sigma_px=sigma_px,
                            weight_key="birth", pdf=True, )#global_bounds=global_bounds)
    pers_map = pi_weighted(feats, H=H, W=W, sigma_px=sigma_px,
                           weight_key="persistence", pdf=True, )#global_bounds=global_bounds)
    radius_map = pi_weighted(feats, H=H, W=W, sigma_px=sigma_px,
                             weight_key="mean_radius", pdf=True,)# global_bounds=global_bounds)

    # Normalize to [0,1]
    def norm01(x):
        xmin, xmax = x.min(), x.max()
        if xmax - xmin < 1e-8:
            return np.zeros_like(x)
        return (x - xmin) / (xmax - xmin)

    H_hue = np.clip(norm01(birth_map), 0.1, 1)
    S_sat = np.clip(norm01(pers_map), 0.1, 1)
    L_light = np.clip(norm01(radius_map), 0.3, 1)

    # Convert HSL -> RGB (vectorized for speed)
    # colorsys uses HLS order (Hue, Lightness, Saturation)
    h, l, s = H_hue, L_light, S_sat

    # Vectorized HSL to RGB conversion
    def hls_to_rgb_vectorized(h, l, s):
        # Based on colorsys.hls_to_rgb algorithm, vectorized
        # When s == 0, color is achromatic (grey)
        rgb = np.zeros((h.shape[0], h.shape[1], 3), dtype=np.float32)

        # Calculate chroma
        c = (1 - np.abs(2 * l - 1)) * s
        h_prime = h * 6.0  # Scale hue to [0, 6]
        x = c * (1 - np.abs(h_prime % 2 - 1))
        m = l - c / 2

        # Determine RGB based on hue sector
        mask0 = (h_prime >= 0) & (h_prime < 1)
        mask1 = (h_prime >= 1) & (h_prime < 2)
        mask2 = (h_prime >= 2) & (h_prime < 3)
        mask3 = (h_prime >= 3) & (h_prime < 4)
        mask4 = (h_prime >= 4) & (h_prime < 5)
        mask5 = (h_prime >= 5) & (h_prime < 6)

        rgb[:, :, 0] = np.where(mask0, c, np.where(mask1, x, np.where(mask2, 0, np.where(mask3, 0, np.where(mask4, x, c)))))
        rgb[:, :, 1] = np.where(mask0, x, np.where(mask1, c, np.where(mask2, c, np.where(mask3, x, np.where(mask4, 0, 0)))))
        rgb[:, :, 2] = np.where(mask0, 0, np.where(mask1, 0, np.where(mask2, x, np.where(mask3, c, np.where(mask4, c, x)))))

        rgb += m[:, :, np.newaxis]
        return np.clip(rgb, 0, 1)

    rgb = hls_to_rgb_vectorized(h, l, s)
#     rgb = np.zeros((H, W, 3), dtype=np.float32)
#     for i in range(H):
#         for j in range(W):
#             rgb[i, j] = colorsys.hls_to_rgb(H_hue[i, j], L_light[i, j], S_sat[i, j])
#     print(rgb.max())
    # Normalize
    max_val = rgb.max()
    if max_val > 0:
        rgb /= max_val

    return rgb


def rgb_to_hsi(rgb_image):
    """
    Convert RGB image to HSI (Hue, Saturation, Intensity) color space.

    Args:
        rgb_image: numpy array of shape (H, W, 3) with values in [0, 1]

    Returns:
        hsi_image: numpy array of shape (H, W, 3) with H in [0, 2π], S in [0, 1], I in [0, 1]
    """
    rgb = rgb_image.astype(np.float32)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    # Intensity
    I = (R + G + B) / 3.0

    # Saturation
    min_rgb = np.minimum(np.minimum(R, G), B)
    S = np.where(I > 1e-6, 1 - (min_rgb / (I + 1e-6)), 0)

    # Hue
    numerator = 0.5 * ((R - G) + (R - B))
    denominator = np.sqrt((R - G)**2 + (R - B) * (G - B) + 1e-6)
    theta = np.arccos(np.clip(numerator / denominator, -1, 1))
    H = np.where(B <= G, theta, 2 * np.pi - theta)

    # Normalize H to [0, 1] for consistency
    H = H / (2 * np.pi)

    hsi_image = np.stack([H, S, I], axis=-1)
    return hsi_image


class NeuronBatch:
    def __init__(self, graph, feats, label, offset):
        self.graph = graph
        self.feats = feats
        self.label = label
        self.offset = offset
    
    def to(self, device):
        return NeuronBatch(
            graph=self.graph.to(device),
            feats=self.feats.to(device),
            label=self.label.to(device),
            offset=self.offset.to(device)
        )

class NeuronBatchSingle:
    def __init__(self, graph, feats, offset):
        self.graph = graph
        self.feats = feats
        self.offset = offset
    
    def to(self, device):
        return NeuronBatchSingle(
            graph=self.graph.to(device),
            feats=self.feats.to(device),
            offset=self.offset.to(device)
        )

class NeuronBatchTwoViews:
    def __init__(self, view1, view2):
        self.view1 = view1
        self.view2 = view2
    
    def to(self, device):
        return NeuronBatchTwoViews(
            view1=self.view1.to(device),
            view2=self.view2.to(device)
        )

# Multimodal batch classes
class NeuronBatchMultiModal:
    def __init__(self, graph, feats, label, offset, images, persistence_coords=None, pixel_coords=None):
        self.graph = graph
        self.feats = feats
        self.label = label
        self.offset = offset
        self.images = images
        self.persistence_coords = persistence_coords
        self.pixel_coords = pixel_coords

    def to(self, device):
        return NeuronBatchMultiModal(
            graph=self.graph.to(device),
            feats=self.feats.to(device),
            label=self.label.to(device),
            offset=self.offset.to(device),
            images=self.images.to(device),
            persistence_coords=self.persistence_coords.to(device) if self.persistence_coords is not None else None,
            pixel_coords=self.pixel_coords.to(device) if self.pixel_coords is not None else None
        )

class NeuronBatchSingleMultiModal:
    def __init__(self, graph, feats, offset, images, persistence_coords=None, pixel_coords=None):
        self.graph = graph
        self.feats = feats
        self.offset = offset
        self.images = images
        self.persistence_coords = persistence_coords
        self.pixel_coords = pixel_coords

    def to(self, device):
        return NeuronBatchSingleMultiModal(
            graph=self.graph.to(device),
            feats=self.feats.to(device),
            offset=self.offset.to(device),
            images=self.images.to(device),
            persistence_coords=self.persistence_coords.to(device) if self.persistence_coords is not None else None,
            pixel_coords=self.pixel_coords.to(device) if self.pixel_coords is not None else None
        )

class NeuronBatchTwoViewsMultiModal:
    def __init__(self, view1, view2):
        self.view1 = view1
        self.view2 = view2
    
    def to(self, device):
        return NeuronBatchTwoViewsMultiModal(
            view1=self.view1.to(device),
            view2=self.view2.to(device)
        )

all_wo_others = {
    "VPM": 0,
    "Isocortex_layer23": 1,
    "Isocortex_layer4": 2,
    "PRE": 3,
    "SUB": 4,
    "CP": 5,
    "VPL": 6,
    "Isocortex_layer6": 7,
    "MG": 8,
    "Isocortex_layer5": 9,
}

Glia_classes = {
    "mouse": 0,
    "rat": 1,
    "human": 2,
    "hippopotamus": 3,
    "monkey": 4,
    "sloth": 5,
    "potto": 6,
    "spotted-sandpiper": 7,
    "bonobo": 8,
    "lemur": 9,
    "guinea-pig": 10,
    "porpoise": 11,
    "gorilla": 12,
    "baboon": 13,
    "bat": 14,
    "minke-whale": 15,
    "red-panda": 16,
    "orangutan": 17,
    "wallaby": 18,
    "semipalmated-sandpiper": 19,
    "opossum": 20,
    "western-tarsier": 21,
    "giant-anteater": 22,
    "ruddy-turnstone": 23,
    "chimpanzee": 24,
    "rabbit": 25,
    "collared-plover": 26,
    "sheep": 27,
    "semipalmated-plover": 28,
    "elephant": 29
}

# Filtered Glia classes with >1000 samples
Glia_filtered_1000 = {
    "mouse": 0,
    "rat": 1,
    "semipalmated-sandpiper": 2,
    "semipalmated-plover": 3,
}

BIL_6_classes = {
    "CP": 0,
    "Isocortex_layer23": 1,
    "Isocortex_layer4": 2,
    "Isocortex_layer5": 3,
    "Isocortex_layer6": 4,
    "VPM": 5,
}

JM_classes = {
    "Isocortex_layer23": 0,
    "Isocortex_layer5": 1,
    "Isocortex_layer6": 2,
    "VPM": 3,
}

ACT_classes = {
    "Isocortex_layer23": 0,
    "Isocortex_layer4": 1,
    "Isocortex_layer5": 2,
    "Isocortex_layer6": 3,
}

neuron7_classes = {
    "amacrine": 0,
    "aspiny": 1,
    "basket": 2,
    "bipolar": 3,
    "pyramidal": 4,
    "spiny": 5,
    "stellate": 6,
}

# Combined dataset: BIL + ACT + JM + neuron7 (10 + 7 = 17 classes)
all_with_neuron7 = {
    # Original all_wo_others classes (0-9)
    "VPM": 0,
    "Isocortex_layer23": 1,
    "Isocortex_layer4": 2,
    "PRE": 3,
    "SUB": 4,
    "CP": 5,
    "VPL": 6,
    "Isocortex_layer6": 7,
    "MG": 8,
    "Isocortex_layer5": 9,
    # neuron7 classes (10-16)
    "amacrine": 10,
    "aspiny": 11,
    "basket": 12,
    "bipolar": 13,
    "pyramidal": 14,
    "spiny": 15,
    "stellate": 16,
}

m1_exc_cell = {
    "other": 0,
    "tufted": 1,
    "untufted": 2,
}

m1_exc_region = {
    "CT": 0,
    "IT": 1,
    "PT": 2,
}

# Combined dataset: BIL + ACT + JM + M1_EXC + neuron7 (10 + 3 + 3 + 7 = 23 classes)
all_wo_n7_m1 = {
    # Original all_wo_others classes (0-9)
    "VPM": 0,
    "Isocortex_layer23": 1,
    "Isocortex_layer4": 2,
    "PRE": 3,
    "SUB": 4,
    "CP": 5,
    "VPL": 6,
    "Isocortex_layer6": 7,
    "MG": 8,
    "Isocortex_layer5": 9,
    # M1_EXC_region classes (10-12)
    "CT": 10,
    "IT": 11,
    "PT": 12,
    # M1_EXC_cell classes (13-15)
    "other": 13,
    "tufted": 14,
    "untufted": 15,
    # neuron7 classes (16-22)
    "amacrine": 16,
    "aspiny": 17,
    "basket": 18,
    "bipolar": 19,
    "pyramidal": 20,
    "spiny": 21,
    "stellate": 22,
}

# Combined dataset: BIL + ACT + JM + M1_EXC + neuron7 + swc_glia (23 + 30 = 53 classes)
all_with_swc_glia = {
    # Original all_wo_others classes (0-9)
    "VPM": 0,
    "Isocortex_layer23": 1,
    "Isocortex_layer4": 2,
    "PRE": 3,
    "SUB": 4,
    "CP": 5,
    "VPL": 6,
    "Isocortex_layer6": 7,
    "MG": 8,
    "Isocortex_layer5": 9,
    # M1_EXC_region classes (10-12)
    "CT": 10,
    "IT": 11,
    "PT": 12,
    # M1_EXC_cell classes (13-15)
    "other": 13,
    "tufted": 14,
    "untufted": 15,
    # neuron7 classes (16-22)
    "amacrine": 16,
    "aspiny": 17,
    "basket": 18,
    "bipolar": 19,
    "pyramidal": 20,
    "spiny": 21,
    "stellate": 22,
    # swc_glia classes (23-52)
    "mouse": 23,
    "rat": 24,
    "human": 25,
    "hippopotamus": 26,
    "monkey": 27,
    "sloth": 28,
    "potto": 29,
    "spotted-sandpiper": 30,
    "bonobo": 31,
    "lemur": 32,
    "guinea-pig": 33,
    "porpoise": 34,
    "gorilla": 35,
    "baboon": 36,
    "bat": 37,
    "minke-whale": 38,
    "red-panda": 39,
    "orangutan": 40,
    "wallaby": 41,
    "semipalmated-sandpiper": 42,
    "opossum": 43,
    "western-tarsier": 44,
    "giant-anteater": 45,
    "ruddy-turnstone": 46,
    "chimpanzee": 47,
    "rabbit": 48,
    "collared-plover": 49,
    "sheep": 50,
    "semipalmated-plover": 51,
    "elephant": 52
}

LABEL_DICT = {
    "all_wo_others": all_wo_others,
    "all_with_neuron7": all_with_neuron7,  # BIL+ACT+JM+neuron7
    "all_wo_n7_m1": all_wo_n7_m1,  # BIL+ACT+JM+M1_EXC+neuron7 (all datasets combined)
    "all_with_swc_glia": all_with_swc_glia,  # BIL+ACT+JM+M1_EXC+neuron7+swc_glia (all datasets + glia)
    "bil_6_classes": BIL_6_classes,
    "JM": JM_classes,           # Keep for backward compatibility
    "JM_4_classes": JM_classes,
    "ACT": ACT_classes,         # Keep for backward compatibility
    "ACT_4_classes": ACT_classes,
    "neuron7": neuron7_classes,
    "m1_exc_cell": m1_exc_cell,
    "m1_exc_region": m1_exc_region,
    "ssl_combined": None,       # No labels for SSL pretraining
    "glia": Glia_classes,
    "swc_glia": Glia_classes,   # Alias for glia dataset
    "swc_glia_filtered_1000": Glia_filtered_1000,  # Filtered glia with >1000 samples
}

    
def get_collate_fn(device, use_two_views=False, use_images=False):
    def collate_fn(batch):
        if use_images:
            trees, offsets, labels, images = [], [0], [], []
            all_persistence_coords, all_pixel_coords = [], []
            cnt = 0
            for b in batch:
                trees.append(b[0])
                cnt += b[1]
                offsets.append(cnt)
                labels.append(b[2])
                # Handle both numpy arrays and tensors
                img = b[3]
                if isinstance(img, np.ndarray):
                    # If numpy array with shape (H, W, C), convert to (C, H, W) tensor
                    if img.ndim == 3 and img.shape[2] == 3:
                        img = torch.from_numpy(img).permute(2, 0, 1)
                    else:
                        img = torch.from_numpy(img)
                elif torch.is_tensor(img):
                    # Already a tensor (from transforms), should already be (C, H, W)
                    pass
                images.append(img)

                # Collect persistence coordinates
                persistence_coords = b[4] if len(b) > 4 else np.zeros((0, 3), dtype=np.float32)
                pixel_coords = b[5] if len(b) > 5 else np.zeros((0, 2), dtype=np.float32)
                all_persistence_coords.append(torch.from_numpy(persistence_coords))
                all_pixel_coords.append(torch.from_numpy(pixel_coords))

            offsets.pop(-1)
            batch_trees = dgl.batch(trees)
            batch_images = torch.stack(images)

            # Pad persistence coordinates to same length for batching
            max_coords = max(pc.shape[0] for pc in all_persistence_coords) if all_persistence_coords else 0
            if max_coords > 0:
                padded_pers_coords = []
                padded_pix_coords = []
                for pc, pix in zip(all_persistence_coords, all_pixel_coords):
                    if pc.shape[0] < max_coords:
                        # Pad with zeros
                        pad_pers = torch.zeros(max_coords - pc.shape[0], 3)
                        pad_pix = torch.zeros(max_coords - pix.shape[0], 2)
                        pc = torch.cat([pc, pad_pers], dim=0)
                        pix = torch.cat([pix, pad_pix], dim=0)
                    padded_pers_coords.append(pc)
                    padded_pix_coords.append(pix)
                batch_pers_coords = torch.stack(padded_pers_coords)
                batch_pix_coords = torch.stack(padded_pix_coords)
            else:
                batch_pers_coords = None
                batch_pix_coords = None

            return NeuronBatchMultiModal(
                graph=batch_trees,
                feats=batch_trees.ndata["feats"].float(),
                label=torch.as_tensor(labels),
                offset=torch.IntTensor(offsets),
                images=batch_images.float(),
                persistence_coords=batch_pers_coords,
                pixel_coords=batch_pix_coords,
            )
        else:
            trees, offsets, labels = [], [0], []
            cnt = 0
            for b in batch:
                trees.append(b[0])
                cnt += b[1]
                offsets.append(cnt)
                labels.append(b[2])
            offsets.pop(-1)
            batch_trees = dgl.batch(trees)
            return NeuronBatch(
                graph=batch_trees,
                feats=batch_trees.ndata["feats"].float(),
                label=torch.as_tensor(labels),
                offset=torch.IntTensor(offsets),
            )

    def collate_fn_two_views(batch):
        if use_images:
            trees1, offsets1, trees2, offsets2, images1, images2 = [], [0], [], [0], [], []
            cnt1, cnt2 = 0, 0
            for b in batch:
                trees1.append(b[0])
                trees2.append(b[2])
                cnt1 += b[1]
                cnt2 += b[3]
                offsets1.append(cnt1)
                offsets2.append(cnt2)
                # Use different images for two views (b[4] for view1, b[5] for view2)
                img1 = b[4]
                img2 = b[5]
                if isinstance(img1, np.ndarray):
                    # If numpy array with shape (H, W, C), convert to (C, H, W) tensor
                    if img1.ndim == 3 and img1.shape[2] == 3:
                        img1 = torch.from_numpy(img1).permute(2, 0, 1)
                    else:
                        img1 = torch.from_numpy(img1)
                elif torch.is_tensor(img1):
                    # Already a tensor (from transforms), should already be (C, H, W)
                    pass
                if isinstance(img2, np.ndarray):
                    # If numpy array with shape (H, W, C), convert to (C, H, W) tensor
                    if img2.ndim == 3 and img2.shape[2] == 3:
                        img2 = torch.from_numpy(img2).permute(2, 0, 1)
                    else:
                        img2 = torch.from_numpy(img2)
                elif torch.is_tensor(img2):
                    # Already a tensor (from transforms), should already be (C, H, W)
                    pass
                images1.append(img1)
                images2.append(img2)

            offsets1.pop(-1)
            offsets2.pop(-1)
            # Debug: Check feature dimensions before batching
            for i, tree in enumerate(trees1):
                if i > 0 and tree.ndata['feats'].shape[1] != trees1[0].ndata['feats'].shape[1]:
                    raise ValueError(f"Feature dimension mismatch in trees1: tree 0 has {trees1[0].ndata['feats'].shape[1]} features, tree {i} has {tree.ndata['feats'].shape[1]} features")
            for i, tree in enumerate(trees2):
                if i > 0 and tree.ndata['feats'].shape[1] != trees2[0].ndata['feats'].shape[1]:
                    raise ValueError(f"Feature dimension mismatch in trees2: tree 0 has {trees2[0].ndata['feats'].shape[1]} features, tree {i} has {tree.ndata['feats'].shape[1]} features")
            batch_trees1, batch_trees2 = dgl.batch(trees1), dgl.batch(trees2)
            batch_images1 = torch.stack(images1)
            batch_images2 = torch.stack(images2)

            return NeuronBatchTwoViewsMultiModal(
                view1=NeuronBatchSingleMultiModal(
                    graph=batch_trees1,
                    feats=batch_trees1.ndata["feats"].float(),
                    offset=torch.IntTensor(offsets1),
                    images=batch_images1.float(),
                ),
                view2=NeuronBatchSingleMultiModal(
                    graph=batch_trees2,
                    feats=batch_trees2.ndata["feats"].float(),
                    offset=torch.IntTensor(offsets2),
                    images=batch_images2.float(),
                ),
            )
        else:
            trees1, offsets1, trees2, offsets2 = [], [0], [], [0]
            cnt1, cnt2 = 0, 0
            for b in batch:
                trees1.append(b[0])
                trees2.append(b[2])
                cnt1 += b[1]
                cnt2 += b[3]
                offsets1.append(cnt1)
                offsets2.append(cnt2)
            offsets1.pop(-1)
            offsets2.pop(-1)
            batch_trees1, batch_trees2 = dgl.batch(trees1), dgl.batch(trees2)
            return NeuronBatchTwoViews(
                view1=NeuronBatchSingle(
                    graph=batch_trees1,
                    feats=batch_trees1.ndata["feats"].float(),
                    offset=torch.IntTensor(offsets1),
                ),
                view2=NeuronBatchSingle(
                    graph=batch_trees2,
                    feats=batch_trees2.ndata["feats"].float(),
                    offset=torch.IntTensor(offsets2),
                ),
            )

    return collate_fn_two_views if use_two_views else collate_fn


def get_tree_len(tree):
    tree_len = 1
    for child in tree:
        tree_len += get_tree_len(child)
    return tree_len


class NeuronTreeDataset(object):
    def __init__(
        self,
        phase="train",
        input_features=[2, 3, 4, 12, 13],
        topology_transformations=None,
        attribute_transformations=None,
        dataset="bil_6_classes",
        data_dir=None,
        labels=None,
        label_dict=None,
        balance=True,
        reload=False,
        use_images=False,
        image_size=256,
        cache_images=True,
        image_transform=None,
        persistence_augmentation=None,
        imagenet_normalize=True,
        use_hsi=False,
        channel_mode='RGB',
        sigma_px=16.0,
    ):
        assert phase in ["train", "test", "full"]
        self.phase = phase
        self.input_features = input_features
        self.topology_transformations = topology_transformations
        self.attribute_transformations = attribute_transformations
        self.dataset = dataset
        self.data_dir = data_dir

        self.balance = balance
        self.use_images = use_images
        self.image_size = image_size
        self.cache_images = cache_images
        self.image_transform = image_transform
        self.persistence_augmentation = persistence_augmentation
        self.imagenet_normalize = imagenet_normalize
        self.use_hsi = use_hsi
        self.channel_mode = channel_mode
        self.sigma_px = sigma_px

        # Image cache
        self.image_cache = {} if cache_images else None
        self.full_lines_cache = {}  # Cache for full SWC data
        self.global_bounds = None# (0, 500, 0, 200)  # Will be computed if use_images=True
        if label_dict is None and labels is not None:
            self.classes = labels
            self.label_dict = {_class: i for i, _class in enumerate(self.classes)}
            # print("it is here", labels)
        elif label_dict is not None:
            self.label_dict = label_dict
            self.classes = label_dict.keys()
        else:
            raise ValueError("One of label_dict and labels should be given")
        self.label_idx_to_class_names = {v: k for k, v in self.label_dict.items()}
        self.reload = reload
        self.load_filecache()
        # print stats
        print(
            self.dataset,
            phase,
            "distribution:",
            np.unique(self.targets, return_counts=True),
        )
        print(self.dataset, phase, self.label_dict)

        # faster caching for no augmentation dataset
        if topology_transformations is None and attribute_transformations is None:
            self.neuron_trees, self.tree_lens, self.tree_heights = self.process()
        else:
            self.neuron_trees, self.tree_lens, self.tree_heights = None, None, None

    def process(self):
        neurons = []
        tree_heights = []
        for i, line in enumerate(self.lines.copy()):
            root = self.lines2trees(line, self.i2ps[i])
            neurons.append(root)
            tree_heights.append(root.height())
        neuron_trees = []
        tree_lens = []
        for neuron in neurons:
            _tree, tree_len = self._build_tree(neuron)
            neuron_trees.append(_tree)
            tree_lens.append(tree_len)
        return neuron_trees, tree_lens, tree_heights

    def load_filecache(self):
        print(f"load_fileCache function {self.dataset}")
        if self.dataset != "rebuttal":
            if not os.path.exists("data/dendrite/processed_datasets/"):
                os.makedirs("data/dendrite/processed_datasets/")
            saved_cache = (
                f"data/dendrite/processed_datasets/{self.dataset}_{self.phase}.pkl"
            )
            print("loading from", saved_cache)
            if os.path.exists(saved_cache) and not self.reload:
                # print("test")
                with open(saved_cache, "rb") as f:
                    saved_cache = pickle.load(f)
                lines, i2ps, targets = (
                    saved_cache["lines"],
                    saved_cache["i2ps"],
                    saved_cache["targets"],
                )
                file_list = saved_cache["file_list"]
            else:

                lines, i2ps, targets, file_list = self.load_and_cache(saved_cache)
                # print("here", lines)
            self.lines = [x[:, self.input_features] for x in lines]
            self.i2ps, self.targets = i2ps, targets
            self.file_list = file_list
        else:
            self.lines = []
            self.i2ps, self.targets, self.file_list = [], [], []
            for dataset in ["bil_6_classes", "JM", "ACT"]:
                saved_cache = f"data/dendrite/processed_datasets/{dataset}_train.pkl"
                print("loading from", saved_cache)
                with open(saved_cache, "rb") as f:
                    saved_cache = pickle.load(f)
                # print(dataset, saved_cache.keys())
                lines, i2ps, targets = (
                    saved_cache["lines"],
                    saved_cache["i2ps"],
                    saved_cache["targets"],
                )
                # lines, i2ps, targets, file_list = self.load_and_cache(saved_cache)
                file_list = saved_cache["file_list"]
                lines = [x[:, self.input_features] for x in lines]
                self.lines.extend(lines)
                self.i2ps.extend(i2ps)
                self.targets.extend(targets)
                self.file_list.extend(file_list)
            idxs = np.arange(len(self.lines))
            np.random.shuffle(idxs)
            # print(idxs)
            self.lines = np.array(self.lines)[idxs]
            self.i2ps = np.array(self.i2ps)[idxs]
            self.targets = np.array(self.targets)[idxs]
            self.file_list = np.array(self.file_list)[idxs]
        print(len(self.lines))
        print("dataset loaded")

    def load_and_cache(self, saved_cache):
        _file_list = self.list_file()
        # print(_file_list)
        cached = {"file_cache": {}}
        file_list, lines, i2ps, targets = [], [], [], []
        for sub_list in _file_list:
            for filename in sub_list:
                if filename not in cached["file_cache"]:
                    _lines = self.readswc(filename)
                    cached["file_cache"][filename] = _lines
                else:
                    _lines = cached["file_cache"][filename]
                file_list.append(filename)
                lines.append(cached["file_cache"][filename])
                i2ps.append(dict([[int(p[0]), int(p[6])] for p in _lines]))
                # try:
                try:
                    # print("printing labelsdict and dataset", self.label_dict, self.dataset, filename)
                    targets.append(self.label_dict[filename.split("/")[-3]])
                except:
                    # for others
                    targets.append(self.label_dict[filename.split("/")[-4]])
        cached["lines"], cached["i2ps"], cached["targets"] = lines, i2ps, targets
        cached["file_list"] = file_list
        with open(saved_cache, "wb") as f:
            pickle.dump(cached, f)
        print("saving to", saved_cache)
        return lines, i2ps, targets, file_list

    def lines2trees(self, lines, i2p):
        p2t = dict()
        for i in i2p:
            if int(i2p[i]) == -1:
                root = tree.Tree(lines[int(i - 1)], [])
                p2t[i] = root
                break

        def add_node(i, line):
            temp = tree.Tree(line, [])
            if int(i2p[i]) not in p2t:
                add_node(int(i2p[i]), lines[int(i2p[i])])
            p2t[int(i2p[i])].append(temp)
            p2t[i] = temp

        for i, line in enumerate(lines):
            try:
                if int(i2p[i + 1]) == -1:
                    continue
            except KeyError:
                print(lines)
            if i + 1 not in p2t:
                add_node(i + 1, line)
        return root

    def _build_tree(self, root):
        g = nx.DiGraph()
        expected_feat_dim = None

        def _rec_build(nid, node):
            nonlocal expected_feat_dim
            for child in node:
                cid = g.number_of_nodes()
                feats = child.label()
                # Validate feature dimensions
                if expected_feat_dim is None:
                    expected_feat_dim = len(feats) if hasattr(feats, '__len__') else 1
                elif hasattr(feats, '__len__') and len(feats) != expected_feat_dim:
                    raise ValueError(f"Feature dimension mismatch in tree building: expected {expected_feat_dim}, got {len(feats)}. Features: {feats}")
                g.add_node(cid, feats=feats)
                if len(child) != 0:
                    _rec_build(cid, child)
                g.add_edge(cid, nid)

        # add root
        root_feats = root.label()
        if hasattr(root_feats, '__len__'):
            expected_feat_dim = len(root_feats)
        g.add_node(0, feats=root_feats)
        _rec_build(0, root)
        ret = from_networkx(g, node_attrs=["feats"])
        return ret, ret.number_of_nodes()
        
    
    def __getitem__(self, idx):
        if (
            self.attribute_transformations is None
            and self.topology_transformations is None
        ):
            tree_data = (self.neuron_trees[idx], self.tree_lens[idx], self.targets[idx])
        else:
            lines = self.lines[idx].copy()

            if self.attribute_transformations is not None:
                lines = self.attribute_transformations(lines)

            neuron = self.lines2trees(lines, self.i2ps[idx])

            if self.topology_transformations is not None:
                neuron = self.topology_transformations(neuron)

            neuron_tree, tree_len = self._build_tree(neuron)
            label = self.targets[idx]
            tree_data = (neuron_tree, tree_len, label)

        if self.use_images:
            image = self.generate_persistent_image(idx, apply_transform=True)

            # Extract persistence coordinates
            filename = self.file_list[idx]
            if filename in self.full_lines_cache:
                full_rows = self.full_lines_cache[filename]
            else:
                try:
                    full_rows = read_swc_robust(filename)
                    self.full_lines_cache[filename] = full_rows
                except Exception as e:
                    print(f"Error reading SWC for persistence coords {filename}: {e}")
                    # Return empty coordinates on error
                    persistence_coords = np.zeros((0, 3), dtype=np.float32)
                    pixel_coords = np.zeros((0, 2), dtype=np.float32)
                    return tree_data + (image, persistence_coords, pixel_coords)

            persistence_coords, pixel_coords = self.extract_persistence_coordinates(
                full_rows,
                global_bounds=self.global_bounds
            )

            return tree_data + (image, persistence_coords, pixel_coords)
        else:
            return tree_data

    def __len__(self):
        r"""Number of graphs in the dataset."""
        return len(self.targets)

    def list_file(self):
        assert self.data_dir is not None, "To list_file, you have to specifiy data_dir"
        self.data_dir = self.data_dir[self.phase]
        ls = [[] for d in self.data_dir]
        file_list = [[] for d in self.data_dir]

        for i in range(len(ls)):
            self.recur_listdir(self.data_dir[i], ls[i])

        print("reading data list ... ")
        all_lines = []
        for i, d in enumerate(ls):
            for filename in tqdm(d):
                if filename[-3:] != "swc":
                    continue
                with open(filename, "r") as f:
                    lines = f.read().strip().split("\n")

                j = 0
                while lines[j].startswith("#"):
                    j += 1
                lines = lines[j:]

                if len(lines) >= 10:
                    all_lines.append(len(lines))
                    file_list[i].append(filename)
        print(np.max(all_lines), np.min(all_lines), np.mean(all_lines))
        if self.balance:
            if self.phase == "train":
                m = max([len(fi) for fi in file_list])
                for fi in file_list:
                    if len(fi) < m:
                        try:
                            extra = np.random.choice(fi, size=m - len(fi))
                        except:
                            print(m, len(fi), m - len(fi))
                            continue
                        fi.extend(extra)

        print(self.dataset, ":", self.phase, [len(fi) for fi in file_list])

        return file_list

    def recur_listdir(self, path, dir_list):
        for f in os.listdir(path):
            if os.path.isdir(path + "/" + f):
                self.recur_listdir(path + "/" + f, dir_list)
            else:
                dir_list.append(path + "/" + f)

    def readswc(self, filename):
        with open(filename, "r") as f:
            lines = f.read().strip().split("\n")
        i = 0
        while lines[i].startswith("#"):
            i += 1
        lines = lines[i:]

        for i in range(len(lines)):
            lines[i] = re.split(r"[\[\],\s]", lines[i])
            while "" in lines[i]:
                lines[i].remove("")
        lines = np.array(lines).astype(float)
        # if self.filter_type:
        # lines = self.filter_lines(lines)
        return lines

    def extract_persistence_coordinates(
        self,
        rows,
        global_bounds=None
    ):
        """
        Extract (birth, death, persistence) coordinates for each topological feature.

        Args:
            rows: SWC rows from read_swc_robust
            global_bounds: (birth_min, birth_max, pers_min, pers_max) for normalization

        Returns:
            persistence_coords: (N_features, 3) array of (birth, death, persistence)
            pixel_coords: (N_features, 2) array of (x, y) pixel locations in image
        """
        # Build tree and compute persistence pairs
        root, children, id2row = build_tree(rows)
        fval = enforce_monotone_radial(root, children, id2row)
        pairs = tmd_pairs_with_identities(root, children, fval)
        pmap = parent_map(rows)
        feats = compute_pair_features(pairs, pmap, id2row)

        if not feats:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

        # Extract persistence space coordinates
        births = np.array([p["birth"] for p in feats], dtype=np.float32)
        deaths = np.array([p["death"] for p in feats], dtype=np.float32)
        persistence = np.array([p["persistence"] for p in feats], dtype=np.float32)

        # Filter out near-zero persistence
        mask = persistence > 1e-9
        births = births[mask]
        deaths = deaths[mask]
        persistence = persistence[mask]

        if len(births) == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

        # Normalize using global bounds if provided
        if global_bounds is not None:
            xmin, xmax, ymin, ymax = global_bounds
        else:
            xmin, xmax = births.min(), births.max()
            ymin, ymax = persistence.min(), persistence.max()

        eps = 1e-8
        # Map to pixel coordinates (same as in pi_weighted)
        u = (births - xmin) / (xmax - xmin + eps) * (self.image_size - 1)
        v = (persistence - ymin) / (ymax - ymin + eps) * (self.image_size - 1)

        pixel_coords = np.stack([u, v], axis=-1).astype(np.float32)

        # Normalize persistence values to [0, 1] for encoding
        births_norm = (births - xmin) / (xmax - xmin + eps)
        deaths_norm = (deaths - xmin) / (xmax - xmin + eps)  # Use same scale as births
        pers_norm = (persistence - ymin) / (ymax - ymin + eps)

        persistence_coords = np.stack([births_norm, deaths_norm, pers_norm], axis=-1).astype(np.float32)

        return persistence_coords, pixel_coords

    def generate_persistent_image(self, idx, apply_transform=False):
        """Generate persistent image for a neuron."""
        # Step 1: Get or generate base persistent image (always cache this if enabled)
        if self.cache_images and idx in self.image_cache:
            # Use cached base image
            base_image = self.image_cache[idx]
        else:
            # Generate base image
            filename = self.file_list[idx]

            if filename in self.full_lines_cache:
                full_rows = self.full_lines_cache[filename]
            else:
                try:
                    # Use robust SWC reader instead of the old one
                    full_rows = read_swc_robust(filename)
                    self.full_lines_cache[filename] = full_rows
                except Exception as e:
                    print(f"Error reading SWC file {filename}: {e}")
                    # Return zero image on error
                    base_image = np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)
                    if self.cache_images:
                        self.image_cache[idx] = base_image
                    return base_image

            # Generate persistent image
            # Note: For caching, we only cache the base image WITHOUT augmentation
            # Augmentation is applied later if transforms are requested
            try:
                if self.use_hsi:
                    # Generate using HSL->RGB conversion
                    base_image = make_hsl_pi(
                        full_rows,
                        H=self.image_size,
                        W=self.image_size,
                        sigma_px=self.sigma_px,
                        global_bounds=self.global_bounds,
                        persistence_aug=None,  # Don't apply augmentation to cached base image
                        channel_mode=self.channel_mode
                    )
                else:
                    # Generate standard RGB persistent image
                    base_image = make_rgb_pi(
                        full_rows,
                        H=self.image_size,
                        W=self.image_size,
                        sigma_px=self.sigma_px,
                        global_bounds=self.global_bounds,
                        persistence_aug=None,  # Don't apply augmentation to cached base image
                        channel_mode=self.channel_mode  # Use specified channel mode
                    )

                # Check if image is empty (all zeros) and add small noise to avoid issues
                if base_image.max() == 0:
                    print(f"Warning: Generated empty image for {filename}, adding small variation")
                    # Add very small random noise to avoid completely black images
                    base_image = np.random.uniform(0, 0.001, base_image.shape).astype(np.float32)

            except Exception as e:
                print(f"Error generating image for {filename}: {e}")
                # Return zero image on error
                base_image = np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)

            # Cache the base image if caching is enabled
            if self.cache_images:
                self.image_cache[idx] = base_image.copy()

        # Step 2: Apply persistence augmentation if specified (regenerate image with augmentation)
        if apply_transform and self.persistence_augmentation is not None:
            # For persistence augmentation, we need to regenerate the image from scratch
            # Get the full SWC data again
            filename = self.file_list[idx]

            if filename in self.full_lines_cache:
                full_rows = self.full_lines_cache[filename]
            else:
                try:
                    full_rows = read_swc_robust(filename)
                    self.full_lines_cache[filename] = full_rows
                except Exception as e:
                    print(f"Error reading SWC file {filename}: {e}")
                    # Return base image on error
                    return base_image

            # Regenerate image WITH persistence augmentation
            try:
                if self.use_hsi:
                    # Generate using HSL->RGB conversion with augmentation
                    image = make_hsl_pi(
                        full_rows,
                        H=self.image_size,
                        W=self.image_size,
                        global_bounds=self.global_bounds,
                        persistence_aug=self.persistence_augmentation,  # Apply augmentation
                        channel_mode=self.channel_mode
                    )
                else:
                    # Generate standard RGB with augmentation
                    image = make_rgb_pi(
                        full_rows,
                        H=self.image_size,
                        W=self.image_size,
                        global_bounds=self.global_bounds,
                        persistence_aug=self.persistence_augmentation,  # Apply augmentation
                        channel_mode=self.channel_mode  # Use specified channel mode
                    )
            except Exception as e:
                print(f"Error generating augmented image for {filename}: {e}")
                image = base_image.copy()
        else:
            # No persistence augmentation, use base image
            image = base_image

        # Step 3: Apply PIL/torchvision transforms if specified (for legacy compatibility)
        if apply_transform and self.image_transform is not None:
            # Make a copy to avoid modifying cached image
            image = image.copy()

            # Convert to PIL Image (expected format for torchvision transforms)
            # Image is in [0, 1] float32, convert to uint8 for PIL
            image_uint8 = (image * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_uint8)

            # Apply the transform (returns tensor in [C, H, W] format)
            image = self.image_transform(pil_image)

            # Return as tensor (collate function expects this)
            return image
        else:
            # No PIL transforms - convert to tensor and apply ImageNet normalization if needed
            # Image is in [H, W, C] format with values in [0, 1]
            # Note: If use_hsi=True, image was already generated via HSL->RGB conversion

            # Convert to tensor [C, H, W]
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()

            # Apply ImageNet normalization if enabled
#             if self.imagenet_normalize:
#                 mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
#                 std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
#                 image_tensor = (image_tensor - mean) / std

            return image_tensor


    def filter_lines(self, datum):
        # unused
        datum = datum[~np.isin(datum[:, 1], self.filter_type), :]
        return datum


class NeuronTreeDatasetTwoViews(NeuronTreeDataset):
    def __init__(
        self,
        phase="train",
        input_features=[2, 3, 4, 12, 13],
        topology_transformations=None,
        attribute_transformations=None,
        dataset="data_bil_3class",
        data_dir=None,
        labels=None,
        label_dict=None,
        use_images=False,
        image_size=256,
        cache_images=True,
        image_transform=None,
        persistence_augmentation=None,
        imagenet_normalize=True,
        use_hsi=False,
        channel_mode='RGB',
        reload=False,
    ):
        assert phase in ["train", "test", "full"]
        super(NeuronTreeDatasetTwoViews, self).__init__(
            phase=phase,
            input_features=input_features,
            topology_transformations=topology_transformations,
            attribute_transformations=attribute_transformations,
            dataset=dataset,
            data_dir=data_dir,
            labels=labels,
            label_dict=label_dict,
            use_images=use_images,
            image_size=image_size,
            cache_images=cache_images,
            image_transform=image_transform,
            persistence_augmentation=persistence_augmentation,
            imagenet_normalize=imagenet_normalize,
            use_hsi=use_hsi,
            channel_mode=channel_mode,
            reload=reload,
        )

    def __getitem__(self, idx):
        if (
            self.attribute_transformations is None
            and self.topology_transformations is None
        ):
            neuron_tree1 = self.neuron_trees[idx]
            tree_len1 = self.tree_lens[idx]
            neuron_tree2 = self.neuron_trees[idx]
            tree_len2 = self.tree_lens[idx]
            tree_data = (neuron_tree1, tree_len1, neuron_tree2, tree_len2)
        else:
            lines = self.lines[idx].copy()
            if self.attribute_transformations is not None:
                lines1 = self.attribute_transformations(lines.copy())
                lines2 = self.attribute_transformations(lines.copy())
            else:
                lines1 = lines2 = lines.copy()
            neuron1 = self.lines2trees(lines1, self.i2ps[idx])
            neuron2 = self.lines2trees(lines2, self.i2ps[idx])
            if self.topology_transformations is not None:
                neuron1 = self.topology_transformations(neuron1)
                neuron2 = self.topology_transformations(neuron2)
            neuron_tree1, tree_len1 = self._build_tree(neuron1)
            neuron_tree2, tree_len2 = self._build_tree(neuron2)
            tree_data = (neuron_tree1, tree_len1, neuron_tree2, tree_len2)

        if self.use_images:
            # Generate two different views with transforms applied independently
            image1 = self.generate_persistent_image(idx, apply_transform=True)
            image2 = self.generate_persistent_image(idx, apply_transform=True)
            return tree_data + (image1, image2)
        else:
            return tree_data



def process_diff_datasets():
    root = "data/dendrite/all_eswc_soma0_ssl"
    bil_classes = [
        "CP",
        "Isocortex_layer23",
        "Isocortex_layer4",
        "Isocortex_layer5",
        "Isocortex_layer6",
        "VPM",
    ]
    bil_train, bil_test, bil_full = [], [], []
    for label in bil_classes:
        bil_train += [f"{root}/{label}/bil-{i}" for i in range(8)]
        bil_test += [f"{root}/{label}/bil-{i}" for i in range(8, 10)]
        bil_full += [f"{root}/{label}/bil-{i}" for i in range(10)]
    bil_6_classes = {"train": bil_train, "test": bil_test, "full": bil_full}

    JM_classes = ["Isocortex_layer23", "Isocortex_layer5", "Isocortex_layer6", "VPM"]
    JMclasses_label_dict = {
        "Isocortex_layer23": 1,
        "Isocortex_layer5": 3,
        "Isocortex_layer6": 4,
        "VPM": 5,
    }
    ACT_classes = [
        "Isocortex_layer23",
        "Isocortex_layer5",
        "Isocortex_layer6",
        "Isocortex_layer4",
    ]
    ACT_classes_label_dict = {
        "Isocortex_layer23": 1,
        "Isocortex_layer5": 3,
        "Isocortex_layer6": 4,
        "Isocortex_layer4": 2,
    }
    JM_test = []
    for label in JM_classes:
        JM_test += [f"{root}/{label}/janelia_mouselight-{i}" for i in range(10)]
    JM_4_classes = {"test": JM_test}

    ACT_test = []
    for label in ACT_classes:
        ACT_test += [f"{root}/{label}/allen_cell_type-{i}" for i in range(10)]

    ACT_4_classes = {"test": ACT_test}

    labels = [bil_classes, JM_classes, ACT_classes]
    label_dicts = [None, JMclasses_label_dict, ACT_classes_label_dict]
    dataset_names = ["bil_6_classes", "JM_4_classes", "ACT_4_classes"]
    for ix, data_dir in enumerate([bil_6_classes, JM_4_classes, ACT_4_classes]):
        for phase in ["train", "test", "full"]:
            if phase not in data_dir:
                continue
            ds = NeuronTreeDataset(
                phase=phase,
                input_features=[2, 3, 4, 12, 13],
                topology_transformations=None,
                attribute_transformations=None,
                dataset=dataset_names[ix],
                labels=labels[ix],
                label_dict=label_dicts[ix],
                data_dir=data_dir,
                reload=True,
                use_images = True,
            )


def get_all_datasets():
    root = "data/dendrite/all_eswc_soma0_ssl"
    classes = os.listdir(root)
    full_wt_others, full_wo_others = [], []

    # Neuron7 class names to exclude from all_wo_others
    neuron7_class_names = set(neuron7_classes.keys())
    # M1_EXC class names to exclude from all_wo_others (these are processed separately)
    m1_exc_class_names = set(list(m1_exc_cell.keys()) + list(m1_exc_region.keys()))
    # Glia class names to exclude from all_wo_others (these are processed separately)
    glia_class_names = set(Glia_classes.keys())

    for label in classes:
        # Skip neuron7, M1_EXC, and glia class folders entirely when building all_wo_others
        if label in neuron7_class_names or label in m1_exc_class_names or label in glia_class_names:
            continue

        candidate_folds = os.listdir(f"{root}/{label}")
        # Filter out neuron7, M1_EXC, and swc_glia folders from all_wo_others dataset
        non_excluded_folds = [x for x in candidate_folds
                             if not x.startswith("neuron7-")
                             and not x.startswith("M1_EXC_region-")
                             and not x.startswith("M1_EXC_cell_type-")
                             and not x.startswith("swc_glia-")]

        if label == "other":
            # "other" is in the m1_exc_cell classes, skip it entirely
            continue
        else:
            full_wt_others += [f"{root}/{label}/{x}" for x in non_excluded_folds]
            full_wo_others += [f"{root}/{label}/{x}" for x in non_excluded_folds]
    dataset_wt_others = {"full": full_wt_others}
    dataset_wo_others = {"full": full_wo_others}

    # Neuron7 dataset
    neuron7_full = []
    for label in neuron7_classes.keys():
        candidate_folds = os.listdir(f"{root}/{label}")
        neuron7_folds = [x for x in candidate_folds if x.startswith("neuron7-")]
        neuron7_full += [f"{root}/{label}/{x}" for x in neuron7_folds]
    dataset_neuron7 = {"full": neuron7_full}

    # Combined dataset: all_wo_others + neuron7 (BIL+ACT+JM+neuron7)
    dataset_all_with_neuron7 = {"full": full_wo_others + neuron7_full}

    datasets = [dataset_wo_others, dataset_neuron7, dataset_all_with_neuron7]

    label_dicts = [all_wo_others, neuron7_classes, all_with_neuron7]
    dataset_names = ["all_wo_others", "neuron7", "all_with_neuron7"]
    for ix, data_dir in enumerate(datasets):
        for phase in ["full"]:
            ds = NeuronTreeDataset(
                phase=phase,
                input_features=[2, 3, 4, 12, 13],
                topology_transformations=None,
                attribute_transformations=None,
                dataset=dataset_names[ix],
                label_dict=label_dicts[ix],
                data_dir=data_dir,
                reload=True,
                use_images = True,
            )
            print(len(ds.file_list))


def split_BIL_JM_ACT():
    root = "data/dendrite/all_eswc_soma0_ssl"

    bil_train, bil_test, bil_full = [], [], []
    for label in BIL_6_classes.keys():
        bil_train += [f"{root}/{label}/bil-{i}" for i in range(7)]
        bil_test += [f"{root}/{label}/bil-{i}" for i in range(7, 10)]
        bil_full += [f"{root}/{label}/bil-{i}" for i in range(10)]
    bil_6_classes = {"train": bil_train, "test": bil_test, "full": bil_full}

    JM_train, JM_test, JM_full = [], [], []
    for label in JM_classes.keys():
        JM_train += [f"{root}/{label}/janelia_mouselight-{i}" for i in range(7)]
        JM_test += [f"{root}/{label}/janelia_mouselight-{i}" for i in range(7, 10)]
        JM_full += [f"{root}/{label}/janelia_mouselight-{i}" for i in range(10)]
    JM_4_classes = {"train": JM_train, "test": JM_test, "full": JM_full}

    ACT_train, ACT_test, ACT_full = [], [], []
    for label in ACT_classes.keys():
        ACT_train += [f"{root}/{label}/allen_cell_type-{i}" for i in range(7)]
        ACT_test += [f"{root}/{label}/allen_cell_type-{i}" for i in range(7, 10)]
        ACT_full += [f"{root}/{label}/allen_cell_type-{i}" for i in range(10)]
    ACT_4_classes = {"train": ACT_train, "test": ACT_test, "full": ACT_full}

    N7_train, N7_test, N7_full = [], [], []
    for label in neuron7_classes.keys():
        N7_train += [f"{root}/{label}/neuron7-{i}" for i in range(7)]
        N7_test += [f"{root}/{label}/neuron7-{i}" for i in range(7, 10)]
        N7_full += [f"{root}/{label}/neuron7-{i}" for i in range(10)]
    N7_7_classes = {"train": N7_train, "test": N7_test, "full": N7_full}

    # Combined BIL+ACT+JM+neuron7
    all_with_n7_train = bil_train + JM_train + ACT_train + N7_train
    all_with_n7_test = bil_test + JM_test + ACT_test + N7_test
    all_with_n7_full = bil_full + JM_full + ACT_full + N7_full
    all_with_n7_dict = {"train": all_with_n7_train, "test": all_with_n7_test, "full": all_with_n7_full}

    label_dicts = [BIL_6_classes, JM_classes, ACT_classes, neuron7_classes, all_with_neuron7]
    dataset_names = ["bil_6_classes", "JM", "ACT", "neuron7", "all_with_neuron7"]
    for ix, data_dir in enumerate([bil_6_classes, JM_4_classes, ACT_4_classes, N7_7_classes, all_with_n7_dict]):
        for phase in ["full", "train", "test"]:
            if phase not in data_dir:
                continue
            ds = NeuronTreeDataset(
                phase=phase,
                input_features=[2, 3, 4, 12, 13],
                topology_transformations=None,
                attribute_transformations=None,
                dataset=dataset_names[ix],
                label_dict=label_dicts[ix],
                data_dir=data_dir,
                reload=True,
                balance=False,
                use_images = True,
            )


def split_M1_EXC_datasets():
    """Process M1_EXC cell type and region datasets."""
    root = "data/dendrite/all_eswc_soma0_ssl"

    # M1_EXC_cell_type dataset (uses "other", "tufted", "untufted")
    m1_cell_train, m1_cell_test, m1_cell_full = [], [], []
    for label in m1_exc_cell.keys():
        m1_cell_train += [f"{root}/{label}/M1_EXC_cell_type-{i}" for i in range(7)]
        m1_cell_test += [f"{root}/{label}/M1_EXC_cell_type-{i}" for i in range(7, 10)]
        m1_cell_full += [f"{root}/{label}/M1_EXC_cell_type-{i}" for i in range(10)]
    m1_cell_dict = {"train": m1_cell_train, "test": m1_cell_test, "full": m1_cell_full}

    # M1_EXC_region dataset (uses "CT", "IT", "PT")
    m1_region_train, m1_region_test, m1_region_full = [], [], []
    for label in m1_exc_region.keys():
        m1_region_train += [f"{root}/{label}/M1_EXC_region-{i}" for i in range(7)]
        m1_region_test += [f"{root}/{label}/M1_EXC_region-{i}" for i in range(7, 10)]
        m1_region_full += [f"{root}/{label}/M1_EXC_region-{i}" for i in range(10)]
    m1_region_dict = {"train": m1_region_train, "test": m1_region_test, "full": m1_region_full}

    label_dicts = [m1_exc_cell, m1_exc_region]
    dataset_names = ["m1_exc_cell", "m1_exc_region"]
    for ix, data_dir in enumerate([m1_cell_dict, m1_region_dict]):
        for phase in ["full", "train", "test"]:
            if phase not in data_dir:
                continue
            ds = NeuronTreeDataset(
                phase=phase,
                input_features=[2, 3, 4, 12, 13],
                topology_transformations=None,
                attribute_transformations=None,
                dataset=dataset_names[ix],
                label_dict=label_dicts[ix],
                data_dir=data_dir,
                reload=True,
                balance=False,
                use_images=True,
            )


def split_all_wo_n7_m1():
    """Process combined dataset: BIL+ACT+JM+M1_EXC+neuron7 (all datasets)."""
    root = "data/dendrite/all_eswc_soma0_ssl"

    # BIL dataset
    bil_train, bil_test, bil_full = [], [], []
    for label in BIL_6_classes.keys():
        bil_train += [f"{root}/{label}/bil-{i}" for i in range(7)]
        bil_test += [f"{root}/{label}/bil-{i}" for i in range(7, 10)]
        bil_full += [f"{root}/{label}/bil-{i}" for i in range(10)]

    # JM dataset
    JM_train, JM_test, JM_full = [], [], []
    for label in JM_classes.keys():
        JM_train += [f"{root}/{label}/janelia_mouselight-{i}" for i in range(7)]
        JM_test += [f"{root}/{label}/janelia_mouselight-{i}" for i in range(7, 10)]
        JM_full += [f"{root}/{label}/janelia_mouselight-{i}" for i in range(10)]

    # ACT dataset
    ACT_train, ACT_test, ACT_full = [], [], []
    for label in ACT_classes.keys():
        ACT_train += [f"{root}/{label}/allen_cell_type-{i}" for i in range(7)]
        ACT_test += [f"{root}/{label}/allen_cell_type-{i}" for i in range(7, 10)]
        ACT_full += [f"{root}/{label}/allen_cell_type-{i}" for i in range(10)]

    # M1_EXC_cell dataset
    m1_cell_train, m1_cell_test, m1_cell_full = [], [], []
    for label in m1_exc_cell.keys():
        m1_cell_train += [f"{root}/{label}/M1_EXC_cell_type-{i}" for i in range(7)]
        m1_cell_test += [f"{root}/{label}/M1_EXC_cell_type-{i}" for i in range(7, 10)]
        m1_cell_full += [f"{root}/{label}/M1_EXC_cell_type-{i}" for i in range(10)]

    # M1_EXC_region dataset
    m1_region_train, m1_region_test, m1_region_full = [], [], []
    for label in m1_exc_region.keys():
        m1_region_train += [f"{root}/{label}/M1_EXC_region-{i}" for i in range(7)]
        m1_region_test += [f"{root}/{label}/M1_EXC_region-{i}" for i in range(7, 10)]
        m1_region_full += [f"{root}/{label}/M1_EXC_region-{i}" for i in range(10)]

    # neuron7 dataset
    N7_train, N7_test, N7_full = [], [], []
    for label in neuron7_classes.keys():
        N7_train += [f"{root}/{label}/neuron7-{i}" for i in range(7)]
        N7_test += [f"{root}/{label}/neuron7-{i}" for i in range(7, 10)]
        N7_full += [f"{root}/{label}/neuron7-{i}" for i in range(10)]

    # Combined dataset (BIL + JM + ACT + M1_cell + M1_region + neuron7)
    all_wo_n7_m1_train = bil_train + JM_train + ACT_train + m1_cell_train + m1_region_train + N7_train
    all_wo_n7_m1_test = bil_test + JM_test + ACT_test + m1_cell_test + m1_region_test + N7_test
    all_wo_n7_m1_full = bil_full + JM_full + ACT_full + m1_cell_full + m1_region_full + N7_full
    all_wo_n7_m1_dict = {"train": all_wo_n7_m1_train, "test": all_wo_n7_m1_test, "full": all_wo_n7_m1_full}

    for phase in ["full", "train", "test"]:
        if phase not in all_wo_n7_m1_dict:
            continue
        ds = NeuronTreeDataset(
            phase=phase,
            input_features=[2, 3, 4, 12, 13],
            topology_transformations=None,
            attribute_transformations=None,
            dataset="all_wo_n7_m1",
            label_dict=all_wo_n7_m1,
            data_dir=all_wo_n7_m1_dict,
            reload=True,
            balance=False,
            use_images=True,
        )


def split_swc_glia():
    """Process swc_glia dataset with train/test splits."""
    root = "data/dendrite/all_eswc_soma0_ssl"

    # swc_glia dataset
    glia_train, glia_test, glia_full = [], [], []
    for label in Glia_classes.keys():
        glia_train += [f"{root}/{label}/swc_glia-{i}" for i in range(7)]
        glia_test += [f"{root}/{label}/swc_glia-{i}" for i in range(7, 10)]
        glia_full += [f"{root}/{label}/swc_glia-{i}" for i in range(10)]
    glia_dict = {"train": glia_train, "test": glia_test, "full": glia_full}

    for phase in ["full", "train", "test"]:
        if phase not in glia_dict:
            continue
        ds = NeuronTreeDataset(
            phase=phase,
            input_features=[2, 3, 4, 12, 13],
            topology_transformations=None,
            attribute_transformations=None,
            dataset="swc_glia",
            label_dict=Glia_classes,
            data_dir=glia_dict,
            reload=True,
            balance=False,
            use_images=True,
        )


def split_swc_glia_filtered_1000():
    """Process filtered swc_glia dataset with only classes having >1000 samples."""
    root = "data/dendrite/all_eswc_soma0_ssl"

    # Filtered swc_glia dataset (only 4 classes: mouse, rat, semipalmated-sandpiper, semipalmated-plover)
    glia_train, glia_test, glia_full = [], [], []
    for label in Glia_filtered_1000.keys():
        glia_train += [f"{root}/{label}/swc_glia-{i}" for i in range(7)]
        glia_test += [f"{root}/{label}/swc_glia-{i}" for i in range(7, 10)]
        glia_full += [f"{root}/{label}/swc_glia-{i}" for i in range(10)]
    glia_dict = {"train": glia_train, "test": glia_test, "full": glia_full}

    for phase in ["full", "train", "test"]:
        if phase not in glia_dict:
            continue
        ds = NeuronTreeDataset(
            phase=phase,
            input_features=[2, 3, 4, 12, 13],
            topology_transformations=None,
            attribute_transformations=None,
            dataset="swc_glia_filtered_1000",
            label_dict=Glia_filtered_1000,
            data_dir=glia_dict,
            reload=True,
            balance=False,
            use_images=True,
        )


def split_all_with_swc_glia():
    """Process combined dataset: BIL+ACT+JM+M1_EXC+neuron7+swc_glia (all datasets)."""
    root = "data/dendrite/all_eswc_soma0_ssl"

    # BIL dataset
    bil_train, bil_test, bil_full = [], [], []
    for label in BIL_6_classes.keys():
        bil_train += [f"{root}/{label}/bil-{i}" for i in range(7)]
        bil_test += [f"{root}/{label}/bil-{i}" for i in range(7, 10)]
        bil_full += [f"{root}/{label}/bil-{i}" for i in range(10)]

    # JM dataset
    JM_train, JM_test, JM_full = [], [], []
    for label in JM_classes.keys():
        JM_train += [f"{root}/{label}/janelia_mouselight-{i}" for i in range(7)]
        JM_test += [f"{root}/{label}/janelia_mouselight-{i}" for i in range(7, 10)]
        JM_full += [f"{root}/{label}/janelia_mouselight-{i}" for i in range(10)]

    # ACT dataset
    ACT_train, ACT_test, ACT_full = [], [], []
    for label in ACT_classes.keys():
        ACT_train += [f"{root}/{label}/allen_cell_type-{i}" for i in range(7)]
        ACT_test += [f"{root}/{label}/allen_cell_type-{i}" for i in range(7, 10)]
        ACT_full += [f"{root}/{label}/allen_cell_type-{i}" for i in range(10)]

    # M1_EXC_cell dataset
    m1_cell_train, m1_cell_test, m1_cell_full = [], [], []
    for label in m1_exc_cell.keys():
        m1_cell_train += [f"{root}/{label}/M1_EXC_cell_type-{i}" for i in range(7)]
        m1_cell_test += [f"{root}/{label}/M1_EXC_cell_type-{i}" for i in range(7, 10)]
        m1_cell_full += [f"{root}/{label}/M1_EXC_cell_type-{i}" for i in range(10)]

    # M1_EXC_region dataset
    m1_region_train, m1_region_test, m1_region_full = [], [], []
    for label in m1_exc_region.keys():
        m1_region_train += [f"{root}/{label}/M1_EXC_region-{i}" for i in range(7)]
        m1_region_test += [f"{root}/{label}/M1_EXC_region-{i}" for i in range(7, 10)]
        m1_region_full += [f"{root}/{label}/M1_EXC_region-{i}" for i in range(10)]

    # neuron7 dataset
    N7_train, N7_test, N7_full = [], [], []
    for label in neuron7_classes.keys():
        N7_train += [f"{root}/{label}/neuron7-{i}" for i in range(7)]
        N7_test += [f"{root}/{label}/neuron7-{i}" for i in range(7, 10)]
        N7_full += [f"{root}/{label}/neuron7-{i}" for i in range(10)]

    # swc_glia dataset
    glia_train, glia_test, glia_full = [], [], []
    for label in Glia_classes.keys():
        glia_train += [f"{root}/{label}/swc_glia-{i}" for i in range(7)]
        glia_test += [f"{root}/{label}/swc_glia-{i}" for i in range(7, 10)]
        glia_full += [f"{root}/{label}/swc_glia-{i}" for i in range(10)]

    # Combined dataset (BIL + JM + ACT + M1_cell + M1_region + neuron7 + swc_glia)
    all_with_glia_train = bil_train + JM_train + ACT_train + m1_cell_train + m1_region_train + N7_train + glia_train
    all_with_glia_test = bil_test + JM_test + ACT_test + m1_cell_test + m1_region_test + N7_test + glia_test
    all_with_glia_full = bil_full + JM_full + ACT_full + m1_cell_full + m1_region_full + N7_full + glia_full
    all_with_glia_dict = {"train": all_with_glia_train, "test": all_with_glia_test, "full": all_with_glia_full}

    for phase in ["full", "train", "test"]:
        if phase not in all_with_glia_dict:
            continue
        ds = NeuronTreeDataset(
            phase=phase,
            input_features=[2, 3, 4, 12, 13],
            topology_transformations=None,
            attribute_transformations=None,
            dataset="all_with_swc_glia",
            label_dict=all_with_swc_glia,
            data_dir=all_with_glia_dict,
            reload=True,
            balance=False,
            use_images=True,
        )


if __name__ == "__main__":
    split_BIL_JM_ACT()
    get_all_datasets()
    split_M1_EXC_datasets()
    split_all_wo_n7_m1()
    split_swc_glia()
    split_swc_glia_filtered_1000()
    split_all_with_swc_glia()
