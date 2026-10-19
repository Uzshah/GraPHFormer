"""Tests independent of DGL so rendering can be checked on CPU-only setups."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

spec = importlib.util.spec_from_file_location(
    'persistence_image', Path(__file__).parents[1] / 'graphformer/data/persistence_image.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_branch_pairs_and_rendering():
    # The shorter branch dies at radial distance 1; the longer survives to 0.
    rows = np.array([[10, 1, 0, 0, 0, 1, -1], [20, 3, 1, 0, 0, 2, 10],
                     [30, 3, 3, 0, 0, 3, 20], [40, 3, 2, 0, 0, 4, 20]])
    pairs = module.compute_persistence_pairs(rows)
    assert sorted((p['birth'], p['death']) for p in pairs) == [(2, 1), (3, 0)]
    image = module.compute_persistence_image(rows, image_size=32, sigma_px=2)
    assert image.shape == (3, 32, 32)
    assert image.dtype == torch.float32
    assert torch.isfinite(image).all()
    assert image.min() >= 0 and image.max() <= 1
    assert not torch.equal(image[0], image[1])


def test_empty_and_invalid_diagrams():
    assert not module.persistence_image([], 16).any()
    with pytest.raises(ValueError, match='parent'):
        module.compute_persistence_pairs([[1, 1, 0, 0, 0, 1, -1], [2, 3, 1, 0, 0, 1, 9]])
    with pytest.raises(ValueError, match='positive'):
        module.persistence_image([], sigma_px=0)


def test_png_source_selection_and_all_sources(tmp_path):
    for folder in ['bil-0', 'bil_0', 'janelia_mouselight-0', 'bil-1']:
        directory = tmp_path / 'Isocortex_layer23' / folder
        directory.mkdir(parents=True)
        Image.new('RGB', (8, 8)).save(directory / 'sample.png')
    dataset = module.PersistenceImageDataset(tmp_path, 'BIL', [0])
    assert len(dataset) == 2
    assert dataset[0][0].mode == 'RGB'
    assert len(module.PersistenceImageDataset(tmp_path, 'ALL', [0])) == 3
    with pytest.raises(ValueError, match='No PNG'):
        module.PersistenceImageDataset(tmp_path, 'BIL', [9])


def test_two_views_apply_transform_independently(tmp_path):
    folder = tmp_path / 'CP' / 'bil-0'
    folder.mkdir(parents=True)
    Image.new('RGB', (8, 8)).save(folder / 'sample.png')
    calls = []
    def transform(image):
        calls.append(image)
        return torch.full((3, 8, 8), len(calls))
    dataset = module.PersistenceImageDatasetTwoViews(tmp_path, 'BIL', [0], transform)
    first, second = dataset[0]
    assert len(calls) == 2 and not torch.equal(first, second)
    batch = module.get_collate_fn('cpu', use_two_views=True)([(first, second)])
    assert batch.view1.shape == (1, 3, 8, 8)
