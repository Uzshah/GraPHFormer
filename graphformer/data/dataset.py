"""SWC neuron datasets and graph/image batch collation."""

from collections import namedtuple
from pathlib import Path
import pickle

import dgl
import networkx as nx
import numpy as np
import torch
from nltk.tree import Tree
from torch.utils.data import Dataset

from .persistence_image import compute_persistence_image

all_wo_others = dict(VPM=0, Isocortex_layer23=1, Isocortex_layer4=2,
                    PRE=3, SUB=4, CP=5, VPL=6, Isocortex_layer6=7,
                    MG=8, Isocortex_layer5=9)
BIL_6_classes = dict(CP=0, Isocortex_layer23=1, Isocortex_layer4=2,
                     Isocortex_layer5=3, Isocortex_layer6=4, VPM=5)
JM_classes = dict(Isocortex_layer23=0, Isocortex_layer5=1, Isocortex_layer6=2, VPM=3)
ACT_classes = dict(Isocortex_layer23=0, Isocortex_layer4=1,
                   Isocortex_layer5=2, Isocortex_layer6=3)
LABEL_DICT = {'all_wo_others': all_wo_others, 'bil_6_classes': BIL_6_classes,
              'JM': JM_classes, 'ACT': ACT_classes}


class _DeviceBatch:
    def to(self, device):
        return type(self)(*(value.to(device) if hasattr(value, 'to') else value for value in self))


class NeuronBatch(_DeviceBatch, namedtuple('NeuronBatch', 'graph feats label offset images', defaults=[None])):
    __slots__ = ()


class NeuronBatchSingle(_DeviceBatch, namedtuple('NeuronBatchSingle', 'graph feats offset images', defaults=[None])):
    __slots__ = ()


class NeuronBatchTwoViews(_DeviceBatch, namedtuple('NeuronBatchTwoViews', 'view1 view2')):
    __slots__ = ()


def get_collate_fn(device, use_two_views=False, use_images=False):
    def single(batch, labeled=True):
        graphs = dgl.batch([sample[0] for sample in batch])
        counts = [sample[1] for sample in batch]
        offsets = torch.tensor([0] + list(np.cumsum(counts)[:-1]), dtype=torch.int32)
        images = torch.stack([sample[3 if labeled else 2] for sample in batch]) if use_images else None
        if labeled:
            result = NeuronBatch(graphs, graphs.ndata['feats'].float(),
                                 torch.tensor([sample[2] for sample in batch], dtype=torch.long), offsets, images)
        else:
            result = NeuronBatchSingle(graphs, graphs.ndata['feats'].float(), offsets, images)
        return result.to(device)

    def collate(batch):
        if not use_two_views:
            return single(batch)
        stride = 3 if use_images else 2
        return NeuronBatchTwoViews(single([s[:stride] for s in batch], False),
                                   single([s[stride:] for s in batch], False))
    return collate


def get_tree_len(tree):
    return sum(1 for _ in tree.subtrees())


class NeuronTreeDataset(Dataset):
    def __init__(self, phase='train', input_features=(2, 3, 4, 12, 13),
                 topology_transformations=None, attribute_transformations=None,
                 dataset='bil_6_classes', data_dir=None, labels=None, label_dict=None,
                 balance=True, reload=False, use_images=False, image_size=224,
                 cache_images=False, persistence_augmentation=None, sigma_px=16.0):
        if phase not in ('train', 'test', 'full'):
            raise ValueError('phase must be train, test, or full')
        self.phase, self.dataset = phase, dataset
        self.input_features = list(input_features)
        self.topology_transformations = topology_transformations
        self.attribute_transformations = attribute_transformations
        self.label_dict = dict(label_dict if label_dict is not None else
                               ({name: i for i, name in enumerate(labels)} if labels is not None else LABEL_DICT[dataset]))
        self.classes = list(self.label_dict)
        self.num_classes = len(self.classes)
        self.label_idx_to_class_names = {v: k for k, v in self.label_dict.items()}
        self.use_images, self.image_size = use_images, image_size
        self.cache_images, self.sigma_px = cache_images, sigma_px
        self.persistence_augmentation = persistence_augmentation
        self._image_cache = {}
        cache = Path('data/dendrite/processed_datasets') / f'{dataset}_{phase}.pkl'
        if cache.exists() and not reload:
            with cache.open('rb') as handle:
                stored = pickle.load(handle)
            self.raw_lines = list(stored['lines'])
            self.targets = list(stored['targets'])
            self.file_list = list(stored['file_list'])
        else:
            root = Path('data/dendrite/all_eswc_soma0_ssl') if data_dir is None else data_dir
            self.file_list = self.list_file(root)
            self.raw_lines = [self.readswc(path) for path in self.file_list]
            self.targets = [self._label(path) for path in self.file_list]
        if not self.raw_lines:
            raise ValueError(f'No SWC samples found for {dataset}/{phase}; provide data_dir or {cache}')
        if balance and phase == 'train':
            groups = [[i for i, target in enumerate(self.targets) if target == label]
                      for label in self.label_dict.values()]
            maximum = max(map(len, groups))
            extra = [int(i) for group in groups if group
                     for i in np.random.choice(group, maximum - len(group))]
            for i in extra:
                self.raw_lines.append(self.raw_lines[i])
                self.targets.append(self.targets[i])
                self.file_list.append(self.file_list[i])
        self.lines = [rows[:, self.input_features] for rows in self.raw_lines]
        self.i2ps = [dict(zip(rows[:, 0].astype(int), rows[:, 6].astype(int))) for rows in self.raw_lines]
        self.neuron_trees = self.tree_lens = self.tree_heights = None
        if topology_transformations is None and attribute_transformations is None:
            self.neuron_trees, self.tree_lens, self.tree_heights = self.process()

    def _label(self, path):
        for part in reversed(Path(path).parts[:-1]):
            if part in self.label_dict:
                return self.label_dict[part]
        raise ValueError(f'No known class folder in {path}')

    def list_file(self, data_dir):
        if isinstance(data_dir, dict):
            roots = data_dir[self.phase]
            filter_splits = False
        else:
            roots = [data_dir]
            filter_splits = True
        if isinstance(roots, (str, Path)):
            roots = [roots]
        prefixes = {'bil_6_classes': ('bil',), 'JM': ('janelia_mouselight',), 'ACT': ('allen_cell_type',)}
        paths = []
        for root in roots:
            for path in sorted(Path(root).rglob('*.swc')):
                if not any(part in self.label_dict for part in path.parts[:-1]):
                    continue
                if filter_splits:
                    import re
                    match = re.fullmatch(r'(.+)[_-](\d+)', path.parent.name)
                    if match:
                        source, split = match.group(1), int(match.group(2))
                        if self.dataset in prefixes and source not in prefixes[self.dataset]:
                            continue
                        if self.phase == 'train' and split >= 8:
                            continue
                        if self.phase == 'test' and split < 8:
                            continue
                paths.append(str(path))
        return paths

    @staticmethod
    def readswc(filename):
        rows = np.loadtxt(filename, comments='#', ndmin=2)
        if not rows.size or rows.shape[1] < 7 or not np.isfinite(rows).all():
            raise ValueError(f'Invalid SWC data: {filename}')
        return rows

    @staticmethod
    def lines2trees(lines, i2p):
        if len(lines) != len(i2p):
            raise ValueError('SWC IDs must be unique and match the feature rows')
        nodes = {node_id: Tree(row, []) for node_id, row in zip(i2p, lines)}
        roots = []
        for node_id, parent in i2p.items():
            if parent == -1:
                roots.append(nodes[node_id])
            elif parent in nodes:
                nodes[parent].append(nodes[node_id])
            else:
                raise ValueError(f'Missing SWC parent {parent}')
        if len(roots) != 1:
            raise ValueError('Expected one SWC root')
        seen, stack = set(), [roots[0]]
        while stack:
            node = stack.pop()
            if id(node) in seen:
                raise ValueError('Cycle in SWC tree')
            seen.add(id(node))
            stack.extend(node)
        if len(seen) != len(nodes):
            raise ValueError('Disconnected SWC nodes or cycle')
        return roots[0]

    @staticmethod
    def _build_tree(root):
        graph = nx.DiGraph()
        graph.add_node(0, feats=np.asarray(root.label(), dtype=np.float32))
        stack = [(0, root)]
        while stack:
            parent_id, node = stack.pop()
            for child in node:
                child_id = len(graph)
                graph.add_node(child_id, feats=np.asarray(child.label(), dtype=np.float32))
                graph.add_edge(child_id, parent_id)
                stack.append((child_id, child))
        graph = dgl.from_networkx(graph, node_attrs=['feats'])
        return graph, graph.num_nodes()

    def process(self):
        graphs, lengths, heights = [], [], []
        for lines, parents in zip(self.lines, self.i2ps):
            root = self.lines2trees(lines, parents)
            graph, length = self._build_tree(root)
            graphs.append(graph)
            lengths.append(length)
            heights.append(root.height())
        return graphs, lengths, heights

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if self.neuron_trees is not None:
            graph, length = self.neuron_trees[idx], self.tree_lens[idx]
        else:
            lines = self.lines[idx].copy()
            if self.attribute_transformations is not None:
                lines = self.attribute_transformations(lines)
            root = self.lines2trees(lines, self.i2ps[idx])
            if self.topology_transformations is not None:
                root = self.topology_transformations(root)
            graph, length = self._build_tree(root)
        result = (graph, length, self.targets[idx])
        if self.use_images:
            # Cache only deterministic images; stochastic features are redrawn each access.
            if idx in self._image_cache:
                image = self._image_cache[idx]
            else:
                image = compute_persistence_image(self.raw_lines[idx], self.image_size,
                                                  self.sigma_px, self.persistence_augmentation)
                if self.cache_images and self.persistence_augmentation is None:
                    self._image_cache[idx] = image
            result += (image,)
        return result


class NeuronTreeDatasetTwoViews(NeuronTreeDataset):
    def __getitem__(self, idx):
        first, second = super().__getitem__(idx), super().__getitem__(idx)
        if self.use_images:
            return first[:2] + (first[3],) + second[:2] + (second[3],)
        return first[:2] + second[:2]
