"""Radial tree persistence and three-channel Gaussian persistence images.

Coordinates are distances from the soma. At each branch the longest surviving
leaf continues toward the soma; other branches die at the merge. Channels encode
unweighted, persistence-weighted, and mean branch radius-weighted densities.
"""

import numpy as np
import torch


def compute_persistence_pairs(lines):
    """Return persistence features from SWC rows, allowing non-contiguous IDs."""
    lines = np.asarray(lines, dtype=float)
    if lines.ndim != 2 or lines.shape[1] < 7 or not len(lines):
        raise ValueError("Expected nonempty SWC rows with at least seven columns")
    ids = lines[:, 0].astype(int)
    if len(set(ids)) != len(ids):
        raise ValueError("SWC node IDs must be unique")
    lookup = {node_id: i for i, node_id in enumerate(ids)}
    roots = np.flatnonzero(lines[:, 6] == -1)
    if len(roots) != 1:
        raise ValueError("Expected exactly one soma/root (parent -1)")
    root = int(roots[0])
    children = [[] for _ in lines]
    for i, parent in enumerate(lines[:, 6].astype(int)):
        if parent != -1:
            if parent not in lookup:
                raise ValueError(f"Missing SWC parent {parent}")
            children[lookup[parent]].append(i)
    radial = np.linalg.norm(lines[:, 2:5] - lines[root, 2:5], axis=1)
    # Iterative postorder also detects disconnected cycles without recursion limits.
    order, stack = [], [root]
    seen = set()
    while stack:
        node = stack.pop()
        if node in seen:
            raise ValueError("Cycle in SWC tree")
        seen.add(node)
        order.append(node)
        stack.extend(children[node])
    if len(order) != len(lines):
        raise ValueError("SWC contains disconnected nodes or a cycle")
    active, pairs = {}, []

    def emit(branch, death):
        birth, radii = branch
        persistence = max(float(birth - death), 0.0)
        if persistence > 0:
            pairs.append(dict(birth=float(birth), death=float(death),
                              persistence=persistence, mean_radius=float(np.mean(radii))))

    for node in reversed(order):
        if not children[node]:
            active[node] = (radial[node], [lines[node, 5]])
            continue
        branches = [active.pop(child) for child in children[node]]
        survivor = max(range(len(branches)), key=lambda i: branches[i][0])
        for i, branch in enumerate(branches):
            if i != survivor:
                emit(branch, radial[node])
        birth, radii = branches[survivor]
        active[node] = (max(birth, radial[node]), radii + [lines[node, 5]])
    emit(active[root], 0.0)
    return pairs


def persistence_image(pairs_feats, image_size=224, sigma_px=16.0, global_bounds=None):
    """Render features to a finite float32 CHW tensor with channels in [0, 1].

    Bounds are (birth_min, birth_max, persistence_min, persistence_max).
    Without explicit bounds each diagram is scaled to its own extent.
    """
    if image_size <= 0 or sigma_px <= 0:
        raise ValueError("image_size and sigma_px must be positive")
    result = np.zeros((3, image_size, image_size), dtype=np.float32)
    if not pairs_feats:
        return torch.from_numpy(result)
    births = np.array([p['birth'] for p in pairs_feats])
    pers = np.array([p['persistence'] for p in pairs_feats])
    radii = np.array([p['mean_radius'] for p in pairs_feats])
    if not np.isfinite(np.concatenate([births, pers, radii])).all():
        raise ValueError("Persistence features must be finite")
    if global_bounds is None:
        global_bounds = (0, max(float(births.max()), 1e-6), 0, max(float(pers.max()), 1e-6))
    bmin, bmax, pmin, pmax = global_bounds
    if bmax <= bmin or pmax <= pmin:
        raise ValueError("Persistence bounds must have positive extent")
    grid = np.arange(image_size, dtype=np.float32)
    for birth, persistence, radius in zip(births, pers, radii):
        x = (birth - bmin) / (bmax - bmin) * (image_size - 1)
        y = (persistence - pmin) / (pmax - pmin) * (image_size - 1)
        kernel = np.exp(-((grid[None, :] - x)**2 + (grid[:, None] - y)**2) / (2 * sigma_px**2))
        result += np.array([1, max(persistence, 0), max(radius, 0)], dtype=np.float32)[:, None, None] * kernel
    maxima = result.max(axis=(1, 2), keepdims=True)
    np.divide(result, maxima, out=result, where=maxima > 0)
    return torch.from_numpy(result)


def compute_persistence_image(lines, image_size=224, sigma_px=16.0,
                              augmentation=None, global_bounds=None):
    """Compute an image, optionally augmenting features before rendering."""
    pairs = compute_persistence_pairs(lines)
    if augmentation is not None:
        pairs = augmentation.augment_pairs_features(pairs, global_bounds)
        sigma_px = augmentation.sample_sigma(sigma_px)
    return persistence_image(pairs, image_size, sigma_px, global_bounds)


# PNG-backed loaders for persistence images that have already been generated.
from collections import namedtuple
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

ImageBatch = namedtuple('ImageBatch', ['images', 'labels'])
ImageBatchTwoViews = namedtuple('ImageBatchTwoViews', ['view1', 'view2'])

IMAGE_LABEL_DICT = {
    'BIL': dict(CP=0, Isocortex_layer23=1, Isocortex_layer4=2,
                Isocortex_layer5=3, Isocortex_layer6=4, VPM=5),
    'JML': dict(Isocortex_layer23=0, Isocortex_layer5=1, Isocortex_layer6=2, VPM=3),
    'ACT': dict(Isocortex_layer23=0, Isocortex_layer4=1,
                Isocortex_layer5=2, Isocortex_layer6=3),
    'ALL': dict(VPM=0, Isocortex_layer23=1, Isocortex_layer4=2, PRE=3,
                SUB=4, CP=5, VPL=6, Isocortex_layer6=7, MG=8, Isocortex_layer5=9),
}


# Preserve the mapping name used by standalone PNG-loading callers.
LABEL_DICT = IMAGE_LABEL_DICT


class PersistenceImageDataset(Dataset):
    """Load RGB PNGs under class/source_split directories.

    Dataset-specific source names prevent mixing BIL, JML, and ACT samples
    when their split directories share a class folder. ALL includes all sources.
    Corrupt images raise an error rather than silently introducing blank samples.
    """
    def __init__(self, data_root, dataset, splits, transform=None, phase='train'):
        if dataset not in IMAGE_LABEL_DICT:
            raise ValueError(f'Dataset must be one of {list(IMAGE_LABEL_DICT)}')
        self.data_root, self.dataset = Path(data_root), dataset
        self.splits, self.transform, self.phase = list(splits), transform, phase
        self.label_dict = IMAGE_LABEL_DICT[dataset]
        self.classes = list(self.label_dict)
        self.num_classes = len(self.classes)
        self.samples = self._build_samples()
        self.targets = [label for _, label in self.samples]
        if not self.samples:
            raise ValueError(f'No PNG samples in {self.data_root} for {dataset}, splits {self.splits}')

    def _build_samples(self):
        import re
        prefixes = {'BIL': {'bil'}, 'JML': {'janelia_mouselight', 'jml', 'jm'},
                    'ACT': {'allen_cell_type', 'act'}}
        samples = []
        for name, label in self.label_dict.items():
            class_dir = self.data_root / name
            if not class_dir.is_dir():
                continue
            for folder in sorted(class_dir.iterdir()):
                if not folder.is_dir():
                    continue
                match = re.fullmatch(r'(?:(.+)[_-])?(\d+)', folder.name)
                if not match or int(match.group(2)) not in self.splits:
                    continue
                source = match.group(1)
                if source is not None and self.dataset != 'ALL' and source.lower() not in prefixes[self.dataset]:
                    continue
                samples.extend((str(path), label) for path in sorted(folder.iterdir())
                               if path.is_file() and path.suffix.lower() == '.png')
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as source:
            image = source.convert('RGB')
        return (self.transform(image) if self.transform is not None else image), label


class PersistenceImageDatasetTwoViews(PersistenceImageDataset):
    def __getitem__(self, idx):
        path, _ = self.samples[idx]
        with Image.open(path) as source:
            image = source.convert('RGB')
        transform = self.transform
        if transform is None:
            from torchvision.transforms import ToTensor
            transform = ToTensor()
        return transform(image.copy()), transform(image.copy())


def get_collate_fn(device, use_two_views=False):
    """Collate PNG datasets; graph datasets use dataset.get_collate_fn."""
    def collate(batch):
        first, second = zip(*batch)
        first = torch.stack(first).to(device)
        if use_two_views:
            return ImageBatchTwoViews(first, torch.stack(second).to(device))
        return ImageBatch(first, torch.tensor(second, dtype=torch.long, device=device))
    return collate


def get_transforms(image_size=224, augment=True, is_training=True):
    """Return RGB transforms with ImageNet normalization for DINOv2."""
    from torchvision import transforms
    if is_training and augment:
        operations = [transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
                      transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                      transforms.RandomRotation(15),
                      transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                      transforms.RandomGrayscale(p=0.2)]
    else:
        operations = [transforms.Resize(image_size), transforms.CenterCrop(image_size)]
    return transforms.Compose(operations + [transforms.ToTensor(), transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
