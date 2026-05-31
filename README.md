# GraPHFormer: A Multimodal Graph Persistent Homology Transformer for the Analysis of Neuroscience Morphologies

**CVPR 2026 (Main Track)**

Uzair Shah, Marco Agus, Mahmoud Gamal, Mahmood Alzubaidi, Corrado Cali, Pierre J. Magistretti, Abdesselam Bouzerdoum, Mowafa Househ

[[Paper]](https://arxiv.org/abs/2603.20970)

---

## Overview

GraPHFormer is a multimodal self-supervised framework for neuronal morphology analysis that jointly models **topological** and **structural** information from neuron reconstructions. It combines:

- **Vision branch**: A three-channel persistence image (unweighted, persistence-weighted, and radius-weighted topological densities) processed by a frozen DINOv2-ViT-S backbone.
- **Graph branch**: A TreeLSTM encoder that captures geometric and radial attributes from the morphological skeleton graph.

The two branches are aligned in a shared embedding space using CLIP-style contrastive learning with a symmetric InfoNCE loss. Persistence-space augmentations are used during training to maintain topological meaning across views.

GraPHFormer achieves state-of-the-art performance on five of six neuronal morphology benchmarks spanning both self-supervised and supervised settings.

## Repository Structure

```
GraPHFormer/
├── train.py                          # Self-supervised pretraining
├── finetune.py                       # Supervised fine-tuning
├── setup.py
├── scripts/
│   └── prepare_data.py               # Data preprocessing pipeline
└── graphformer/
    ├── models/
    │   ├── clip_model.py             # CLIP-style dual-branch model
    │   ├── image_encoder.py          # DINOv2 / ResNet image encoders
    │   ├── tree_encoder.py           # TreeLSTM graph encoder
    │   ├── fusion.py                 # Multimodal fusion heads
    │   └── finetune_model.py         # Fine-tuning wrapper
    ├── data/
    │   ├── dataset.py                # NeuronTreeDataset
    │   └── persistence_image.py      # Persistence image computation
    ├── losses/
    │   ├── infonce.py
    │   └── contrastive.py
    └── augmentations/
        ├── tree_augmentations.py     # Graph-space augmentations
        └── persistence_augmentations.py  # Topology-preserving image augmentations
```

## Installation

```bash
pip install -e .
```

**Dependencies**: Python >= 3.8, PyTorch >= 1.10, torchvision >= 0.11, DGL >= 0.8, scikit-learn, networkx, nltk, Pillow, tqdm, numpy.

## Data Preparation

Dataset downloading and preprocessing follow the [TreeMoCo](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9f989633ffbd47a83caddacad0f0261f-Abstract-Conference.html) paper (NeurIPS 2022). Please refer to the [TreeMoCo GitHub repository](https://github.com/TencentAILabHealthcare/NeuronRepresentation) for instructions on downloading the three datasets: BIL (Brain Image Library), ACT (Allen Cell Types), and JML (Janelia MouseLight).

Once the raw SWC files are in place, run the preprocessing script:

```bash
python scripts/prepare_data.py
```

This normalizes soma position/orientation/scale, removes axon compartments, computes branch-level features, and creates 10-fold cross-validation splits.

## Training

### Self-Supervised Pretraining

```bash
python train.py \
    --exp_name my_experiment \
    --dataset all_wo_others \
    --image_encoder dinov2_vits14 \
    --tree_model double \
    --embed_dim 128 \
    --batch_size 128 \
    --epochs 100 \
    --lr 3e-4 \
    --use_knn_eval \
    --eval_jm --eval_act
```

Key options:

| Argument | Default | Description |
|---|---|---|
| `--image_encoder` | `resnet18` | `dinov2_vits14`, `resnet18`, `resnet50`, `persistencevit` |
| `--tree_model` | `double` | TreeLSTM variant: `ori`, `v2`, `double` |
| `--embed_dim` | `128` | Shared embedding dimension |
| `--loss_type` | `clip` | `clip`, `infonce`, `ntxent`, `triplet` |
| `--use_persistence_aug` | off | Enable persistence-space augmentations |
| `--knn_fusion` | `concat` | How to combine modalities for KNN eval |

Tree augmentations: `--aug_rotate`, `--aug_flip`, `--aug_jitter_coords`, `--aug_drop_tree`, `--aug_skip_parent_node`, `--aug_swap_sibling_subtrees`

### Fine-Tuning

```bash
python finetune.py \
    --exp_name my_finetune \
    --pretrained_checkpoint work_dir/my_experiment/best_BIL.pth \
    --dataset bil_6_classes \
    --mode multimodal \
    --fusion_mode concat \
    --epochs 50 \
    --lr 1e-4
```

Fine-tuning modes: `multimodal`, `image_only`, `tree_only`. Fusion modes: `concat`, `add`, `cross_attention`, `gated`, `cmf`, `mhcma`.

Two-stage training (linear probe then full fine-tune):

```bash
python finetune.py \
    --exp_name my_finetune \
    --pretrained_checkpoint work_dir/my_experiment/best_BIL.pth \
    --dataset bil_6_classes \
    --linear_probe_epochs 10 \
    --epochs 50
```

## Checkpoints

We provide all trained model checkpoints on Hugging Face for reproducibility and easy access:

🔗 Hugging Face Model Repository:  
https://huggingface.co/uzsh31989/GraPHFormer

📁 Checkpoints directory:  
https://huggingface.co/uzsh31989/GraPHFormer/tree/main/checkpoints

To download programmatically:

```python
from huggingface_hub import hf_hub_download

file_path = hf_hub_download(
    repo_id="uzsh31989/GraPHFormer",
    filename="checkpoints/N7_best_weights.pth"
)

## Benchmarks

Evaluated on six datasets:

| Dataset | Task | Classes |
|---|---|---|
| BIL-6 | Brain region classification | 6 |
| ACT-4 | Cortical layer classification | 4 |
| JML-4 | Brain region classification | 4 |
| N7 | Neuron type classification | 7 |
| M1-Cell | Cell type classification | 3 |
| M1-REG | Cortical region classification | 3 |

GraPHFormer achieves state-of-the-art on 5/6 benchmarks, outperforming topology-only, graph-only, and morphometrics baselines.

## Citation

```bibtex
@inproceedings{shah2026graphformer,
  title     = {GraPHFormer: A Multimodal Graph Persistent Homology Transformer for the Analysis of Neuroscience Morphologies},
  author    = {Shah, Uzair and Agus, Marco and Gamal, Mahmoud and Alzubaidi, Mahmood and Cali, Corrado and Magistretti, Pierre J. and Bouzerdoum, Abdesselam and Househ, Mowafa},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (Main Track)},
  year      = {2026}
}
```
