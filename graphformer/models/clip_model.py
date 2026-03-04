"""
CLIP-style Contrastive Model for Neuron Morphology

Aligns tree structure representations with persistence images using
contrastive learning with separate encoders.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tree_encoder import TreeLSTM, TreeLSTMv2, TreeLSTMDouble
from .image_encoder import ImageEncoder


class CLIPLoss(nn.Module):
    """CLIP-style symmetric contrastive loss"""
    def __init__(self, temperature=0.07):
        super(CLIPLoss, self).__init__()
        self.temperature = temperature

    def forward(self, tree_features, image_features):
        """
        Args:
            tree_features: (B, dim) - normalized tree embeddings
            image_features: (B, dim) - normalized image embeddings
        Returns:
            loss: scalar contrastive loss
        """
        tree_features = F.normalize(tree_features, dim=-1)
        image_features = F.normalize(image_features, dim=-1)

        logits = torch.matmul(tree_features, image_features.T) / self.temperature

        batch_size = tree_features.shape[0]
        labels = torch.arange(batch_size, device=tree_features.device)

        loss_tree_to_image = F.cross_entropy(logits, labels)
        loss_image_to_tree = F.cross_entropy(logits.T, labels)

        loss = (loss_tree_to_image + loss_image_to_tree) / 2

        return loss


class CLIPModel(nn.Module):
    """CLIP-style model with separate tree and image encoders"""
    def __init__(self, args):
        super(CLIPModel, self).__init__()

        self.tree_encoder_type = args.tree_model

        # Tree encoder
        if args.tree_model == "ori":
            self.tree_encoder = TreeLSTM(
                x_size=len(args.input_features),
                h_size=args.h_size,
                num_classes=0,
                fc=False,
                bn=args.bn,
                mode=args.child_mode,
            )
        elif args.tree_model == "v2":
            self.tree_encoder = TreeLSTMv2(
                x_size=len(args.input_features),
                h_size=args.h_size,
                num_classes=0,
                fc=False,
                bn=args.bn,
                mode=args.child_mode,
            )
        elif args.tree_model == "double":
            self.tree_encoder = TreeLSTMDouble(
                x_size=len(args.input_features),
                h_size=args.h_size,
                num_classes=0,
                fc=False,
                bn=args.bn,
                mode=args.child_mode,
            )
        else:
            raise ValueError(f"Unknown tree model: {args.tree_model}")

        # Image encoder
        self.image_encoder = ImageEncoder(
            output_dim=args.h_size,
            model_type=args.image_encoder,
            image_size=args.image_size,
            freeze_backbone=args.freeze_image_backbone,
        )

        # Get image encoder output dimension
        if hasattr(self.image_encoder.encoder if hasattr(self.image_encoder, 'encoder') else self.image_encoder, 'feat_dim'):
            image_feat_dim = self.image_encoder.encoder.feat_dim if hasattr(self.image_encoder, 'encoder') else self.image_encoder.feat_dim
        else:
            image_feat_dim = args.h_size

        # Projection heads
        if getattr(args, 'single_linear_proj', False):
            self.tree_projection = nn.Linear(args.h_size, args.embed_dim)
            self.image_projection = nn.Linear(image_feat_dim, args.embed_dim)
        else:
            self.tree_projection = nn.Sequential(
                nn.Linear(args.h_size, args.h_size),
                nn.ReLU(),
                nn.Linear(args.h_size, args.embed_dim),
            )
            self.image_projection = nn.Sequential(
                nn.Linear(image_feat_dim, args.h_size),
                nn.ReLU(),
                nn.Linear(args.h_size, args.embed_dim),
            )

        self.loss_type = args.loss_type

        # Loss function
        if args.loss_type == 'infonce':
            from ..losses import SymmetricInfoNCELoss
            self.criterion = SymmetricInfoNCELoss(temperature=args.temperature)
        elif args.loss_type == 'ntxent':
            from ..losses import NTXentLoss
            self.criterion = NTXentLoss(temperature=args.temperature)
        elif args.loss_type == 'triplet':
            from ..losses import TripletLoss
            self.criterion = TripletLoss(
                margin=args.triplet_margin,
                distance_metric=args.triplet_distance,
                mining=args.triplet_mining
            )
        else:  # 'clip' (default)
            self.criterion = CLIPLoss(temperature=args.temperature)

    def encode_tree(self, batch):
        """Encode tree data"""
        tree_feats = self.tree_encoder(batch)
        tree_embed = self.tree_projection(tree_feats)
        return tree_embed

    def encode_image(self, images):
        """Encode image data"""
        image_feats = self.image_encoder(images)
        image_embed = self.image_projection(image_feats)
        return image_embed

    def forward(self, batch, return_recon=False):
        """
        Forward pass computing CLIP loss

        Args:
            batch: contains batch.graph, batch.feats, batch.images
            return_recon: unused, kept for compatibility
        Returns:
            loss: Contrastive loss
        """
        tree_embed = self.encode_tree(batch)
        images = batch.images.cuda() if not batch.images.is_cuda else batch.images

        image_feats = self.image_encoder(images)
        image_embed = self.image_projection(image_feats)

        # Contrastive loss
        if self.loss_type == 'triplet':
            embeddings = torch.cat([tree_embed, image_embed], dim=0)
            labels_gpu = batch.label.cuda() if not batch.label.is_cuda else batch.label
            labels = torch.cat([labels_gpu, labels_gpu], dim=0)
            clip_loss = self.criterion(embeddings, labels)
        else:
            clip_loss = self.criterion(tree_embed, image_embed)

        return clip_loss
