"""
GraPHFormer: Graph-Persistence Hybrid Transformer for Neuron Morphology

A CLIP-style contrastive learning framework for neuron representation learning
that combines tree-structured morphology with persistence images.
"""

from .models import CLIPModel, FineTuneModel, CLIPLoss
from .losses import InfoNCELoss, NTXentLoss, TripletLoss

__version__ = "1.0.0"
__all__ = ["CLIPModel", "FineTuneModel", "CLIPLoss", "InfoNCELoss", "NTXentLoss", "TripletLoss"]
