"""
Fine-tuning Model for GraPHFormer

Supports three modes: image_only, tree_only, multimodal
Loads pretrained weights from CLIP-style training checkpoints
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fusion import (
    CrossAttentionFusion, BiDirectionalCrossAttention,
    GatedFusion, CMF, MultiHeadCrossModalAttention
)


class ArcMarginProduct(nn.Module):
    """ArcFace: Additive Angular Margin Loss"""
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.easy_margin = easy_margin

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = np.cos(m)
        self.sin_m = np.sin(m)
        self.th = np.cos(np.pi - m)
        self.mm = np.sin(np.pi - m) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class FineTuneModel(nn.Module):
    """Fine-tuning model with classification head"""
    def __init__(
        self,
        pretrained_model,
        num_classes,
        mode='multimodal',
        freeze_encoders=False,
        fusion_mode='concat',
        dropout=0.5,
        label_smoothing=0.0,
        use_projection=False,
        use_arcface=False,
        arcface_s=30.0,
        arcface_m=0.50,
        freeze_image_only=False
    ):
        """
        Args:
            pretrained_model: Pretrained CLIPModel
            num_classes: Number of classes for classification
            mode: 'image_only', 'tree_only', or 'multimodal'
            freeze_encoders: If True, freeze encoder weights
            fusion_mode: For multimodal - 'concat', 'add', 'cross_attention', 'bi_attention', 'gated', 'cmf', 'mhcma'
            dropout: Dropout rate for first layer
            label_smoothing: Label smoothing factor
            use_projection: If True, use projection heads from pretrained model
            use_arcface: If True, use ArcFace loss instead of CrossEntropy
            arcface_s: ArcFace scale parameter
            arcface_m: ArcFace margin parameter
            freeze_image_only: If True, freeze only image encoder
        """
        super(FineTuneModel, self).__init__()

        self.mode = mode
        self.fusion_mode = fusion_mode
        self.use_projection = use_projection
        self.use_arcface = use_arcface
        self.tree_encoder_type = pretrained_model.tree_encoder_type

        # Copy encoders from pretrained model
        if mode in ['tree_only', 'multimodal']:
            self.tree_encoder = pretrained_model.tree_encoder
            if use_projection:
                self.tree_projection = pretrained_model.tree_projection

        if mode in ['image_only', 'multimodal']:
            self.image_encoder = pretrained_model.image_encoder
            if use_projection:
                self.image_projection = pretrained_model.image_projection

        # Freeze encoders if requested
        if freeze_encoders:
            if mode in ['tree_only', 'multimodal']:
                for param in self.tree_encoder.parameters():
                    param.requires_grad = False
                if use_projection:
                    for param in self.tree_projection.parameters():
                        param.requires_grad = False

            if mode in ['image_only', 'multimodal']:
                for param in self.image_encoder.parameters():
                    param.requires_grad = False
                if use_projection:
                    for param in self.image_projection.parameters():
                        param.requires_grad = False

        # Freeze only image encoder
        if freeze_image_only and mode == 'multimodal':
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            if use_projection:
                for param in self.image_projection.parameters():
                    param.requires_grad = False

        # Get embedding dimensions
        if use_projection:
            tree_embed_dim = pretrained_model.tree_projection[-1].out_features
            image_embed_dim = pretrained_model.image_projection[-1].out_features
        else:
            if mode in ['tree_only', 'multimodal']:
                tree_embed_dim = pretrained_model.tree_encoder.h_size
            else:
                tree_embed_dim = 0

            if mode in ['image_only', 'multimodal']:
                if hasattr(pretrained_model.image_encoder, 'encoder'):
                    if hasattr(pretrained_model.image_encoder.encoder, 'feat_dim'):
                        image_embed_dim = pretrained_model.image_encoder.encoder.feat_dim
                    else:
                        image_embed_dim = pretrained_model.image_encoder.encoder[-1].in_features
                elif hasattr(pretrained_model.image_encoder, 'feat_dim'):
                    image_embed_dim = pretrained_model.image_encoder.feat_dim
                else:
                    image_embed_dim = pretrained_model.tree_encoder.h_size
            else:
                image_embed_dim = 0

        # Setup fusion for multimodal
        if mode == 'multimodal':
            if fusion_mode == 'concat':
                fusion_dim = tree_embed_dim + image_embed_dim
            elif fusion_mode == 'add':
                fusion_dim = min(tree_embed_dim, image_embed_dim)
                if tree_embed_dim != image_embed_dim:
                    self.tree_dim_match = nn.Linear(tree_embed_dim, fusion_dim) if tree_embed_dim != fusion_dim else nn.Identity()
                    self.image_dim_match = nn.Linear(image_embed_dim, fusion_dim) if image_embed_dim != fusion_dim else nn.Identity()
            elif fusion_mode == 'cross_attention':
                fusion_dim = min(tree_embed_dim, image_embed_dim)
                if tree_embed_dim != image_embed_dim:
                    self.tree_dim_match = nn.Linear(tree_embed_dim, fusion_dim) if tree_embed_dim != fusion_dim else nn.Identity()
                    self.image_dim_match = nn.Linear(image_embed_dim, fusion_dim) if image_embed_dim != fusion_dim else nn.Identity()
                self.fusion_layer = CrossAttentionFusion(fusion_dim, num_heads=4)
            elif fusion_mode == 'bi_attention':
                fusion_dim = min(tree_embed_dim, image_embed_dim)
                if tree_embed_dim != image_embed_dim:
                    self.tree_dim_match = nn.Linear(tree_embed_dim, fusion_dim) if tree_embed_dim != fusion_dim else nn.Identity()
                    self.image_dim_match = nn.Linear(image_embed_dim, fusion_dim) if image_embed_dim != fusion_dim else nn.Identity()
                self.fusion_layer = BiDirectionalCrossAttention(fusion_dim, num_heads=4)
            elif fusion_mode == 'gated':
                fusion_dim = min(tree_embed_dim, image_embed_dim)
                if tree_embed_dim != image_embed_dim:
                    self.tree_dim_match = nn.Linear(tree_embed_dim, fusion_dim) if tree_embed_dim != fusion_dim else nn.Identity()
                    self.image_dim_match = nn.Linear(image_embed_dim, fusion_dim) if image_embed_dim != fusion_dim else nn.Identity()
                self.fusion_layer = GatedFusion(fusion_dim)
            elif fusion_mode == 'cmf':
                fusion_dim = min(tree_embed_dim, image_embed_dim)
                if tree_embed_dim != image_embed_dim:
                    self.tree_dim_match = nn.Linear(tree_embed_dim, fusion_dim) if tree_embed_dim != fusion_dim else nn.Identity()
                    self.image_dim_match = nn.Linear(image_embed_dim, fusion_dim) if image_embed_dim != fusion_dim else nn.Identity()
                self.fusion_layer = CMF(fusion_dim)
            elif fusion_mode == 'mhcma':
                fusion_dim = min(tree_embed_dim, image_embed_dim)
                if tree_embed_dim != image_embed_dim:
                    self.tree_dim_match = nn.Linear(tree_embed_dim, fusion_dim) if tree_embed_dim != fusion_dim else nn.Identity()
                    self.image_dim_match = nn.Linear(image_embed_dim, fusion_dim) if image_embed_dim != fusion_dim else nn.Identity()
                self.fusion_layer = MultiHeadCrossModalAttention(fusion_dim, num_heads=8)
            else:
                raise ValueError(f"Unknown fusion_mode: {fusion_mode}")
            input_dim = fusion_dim
        else:
            if mode == 'tree_only':
                input_dim = tree_embed_dim
            else:
                input_dim = image_embed_dim

        # Classification head
        if use_arcface:
            self.feature_extractor = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(input_dim, input_dim // 2),
                nn.BatchNorm1d(input_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout * 0.7),
            )
            self.arcface = ArcMarginProduct(input_dim // 2, num_classes, s=arcface_s, m=arcface_m)
            self.classifier = None
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(input_dim, input_dim // 2),
                nn.BatchNorm1d(input_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout * 0.7),
                nn.Linear(input_dim // 2, num_classes)
            )
            self.feature_extractor = None
            self.arcface = None

        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def encode_tree(self, batch):
        """Encode tree data"""
        tree_feats = self.tree_encoder(batch)

        if self.use_projection:
            tree_embed = self.tree_projection(tree_feats)
            return tree_embed
        else:
            return tree_feats

    def encode_image(self, images):
        """Encode image data"""
        image_feats = self.image_encoder(images)

        if self.use_projection:
            image_embed = self.image_projection(image_feats)
            return image_embed
        else:
            return image_feats

    def forward(self, batch, return_features=False):
        """
        Forward pass

        Args:
            batch: contains batch.graph, batch.feats, batch.images, batch.label
            return_features: if True, return embeddings along with logits
        Returns:
            loss: classification loss
            logits: (B, num_classes)
            features: (optional) embeddings
        """
        images = batch.images.cuda() if not batch.images.is_cuda else batch.images
        labels = batch.label.cuda() if not batch.label.is_cuda else batch.label

        if self.mode == 'tree_only':
            tree_embed = self.encode_tree(batch)
            tree_embed = F.normalize(tree_embed, dim=-1)
            features = tree_embed

        elif self.mode == 'image_only':
            image_embed = self.encode_image(images)
            image_embed = F.normalize(image_embed, dim=-1)
            features = image_embed

        else:  # multimodal
            tree_embed = self.encode_tree(batch)
            image_embed = self.encode_image(images)

            tree_embed = F.normalize(tree_embed, dim=-1)
            image_embed = F.normalize(image_embed, dim=-1)

            if self.fusion_mode == 'concat':
                features = torch.cat([tree_embed, image_embed], dim=1)
            elif self.fusion_mode == 'add':
                if hasattr(self, 'tree_dim_match'):
                    tree_embed = self.tree_dim_match(tree_embed)
                    image_embed = self.image_dim_match(image_embed)
                features = tree_embed + image_embed
            elif self.fusion_mode in ['gated', 'cmf', 'cross_attention', 'bi_attention', 'mhcma']:
                if hasattr(self, 'tree_dim_match'):
                    tree_embed = self.tree_dim_match(tree_embed)
                    image_embed = self.image_dim_match(image_embed)
                features = self.fusion_layer(tree_embed, image_embed)
            else:
                raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")

        # Classification
        if self.use_arcface:
            extracted_features = self.feature_extractor(features)
            logits = self.arcface(extracted_features, labels)
            loss = self.criterion(logits, labels)
        else:
            logits = self.classifier(features)
            loss = self.criterion(logits, labels)

        if return_features:
            return loss, logits, features
        return loss, logits

    def unfreeze_encoders(self):
        """Unfreeze encoder weights for full fine-tuning"""
        if self.mode in ['tree_only', 'multimodal'] and hasattr(self, 'tree_encoder'):
            for param in self.tree_encoder.parameters():
                param.requires_grad = True
            if self.use_projection and hasattr(self, 'tree_projection'):
                for param in self.tree_projection.parameters():
                    param.requires_grad = True

        if self.mode in ['image_only', 'multimodal'] and hasattr(self, 'image_encoder'):
            for param in self.image_encoder.parameters():
                param.requires_grad = True
            if self.use_projection and hasattr(self, 'image_projection'):
                for param in self.image_projection.parameters():
                    param.requires_grad = True
