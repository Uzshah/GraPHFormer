"""
Image Encoder Components for Persistence Images

This module contains various image encoders optimized for persistence images:
- SimpleCNN: Lightweight CNN
- SmallViT: Compact Vision Transformer
- PersistenceViT: Topologically-aware Vision Transformer with persistence-weighted positional encoding
- ResNet18/ResNet50: Standard ResNet encoders
- DINOv2: Self-supervised visual encoders
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet50_Weights, ResNet101_Weights
from typing import Tuple, Optional


# ============================================================================
# Simple CNN Encoder
# ============================================================================

class SimpleCNN(nn.Module):
    """Lightweight CNN for persistence images"""
    def __init__(self, output_dim=128):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ============================================================================
# Vision Transformer Components
# ============================================================================

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention for Vision Transformer"""
    def __init__(self, dim, heads, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and MLP"""
    def __init__(self, dim, heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(dim, heads, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SmallViT(nn.Module):
    """Lightweight Vision Transformer for persistence images"""
    def __init__(
        self,
        image_size=224,
        patch_size=14,
        output_dim=128,
        dim=128,
        depth=4,
        heads=4,
        mlp_dim=256,
        channels=3,
        dropout=0.2,
        emb_dropout=0.2
    ):
        super().__init__()
        assert image_size % patch_size == 0
        num_patches = (image_size // patch_size) ** 2

        self.patch_embedding = nn.Sequential(
            nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_dim, dropout)
            for _ in range(depth)
        ])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, output_dim)
        )

    def forward(self, img):
        x = self.patch_embedding(img).transpose(1, 2)
        b, n, _ = x.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        for transformer_block in self.transformer:
            x = transformer_block(x)

        return self.mlp_head(x[:, 0])


# ============================================================================
# Topologically-Aware Vision Transformer (PersistenceViT)
# ============================================================================

class TopologicalPatchEmbedding(nn.Module):
    """Topological-Aware Patch Embedding with importance weighting"""
    def __init__(self, in_channels, dim, patch_size, image_size):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)

        # Importance estimator for high-persistence regions
        self.importance_estimator = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=patch_size, stride=patch_size),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        patches = self.proj(x).flatten(2).transpose(1, 2)  # (B, N, dim)
        importance = self.importance_estimator(x).flatten(2).transpose(1, 2)  # (B, N, 1)
        weighted_patches = patches * (1 + importance)
        return weighted_patches, importance.squeeze(-1)


class BirthDeathAttention(nn.Module):
    """Attention mechanism with persistence-based modulation"""
    def __init__(self, dim, heads, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        self.persistence_bias = nn.Parameter(torch.zeros(1, heads, 1, 1))

    def forward(self, x, importance_weights=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if importance_weights is not None:
            importance = importance_weights.unsqueeze(1).unsqueeze(-1)
            attn = attn + self.persistence_bias + importance * 0.1

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class BirthDeathTransformerBlock(nn.Module):
    """Transformer block with topological awareness"""
    def __init__(self, dim, heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.attention = BirthDeathAttention(dim, heads, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, importance_weights=None):
        x = x + self.attention(self.norm1(x), importance_weights)
        x = x + self.mlp(self.norm2(x))
        return x


class TopologicalFeatureAggregation(nn.Module):
    """Aggregates CLS token with homology-specific tokens"""
    def __init__(self, dim, num_homology_dims):
        super().__init__()
        self.num_homology = num_homology_dims
        self.cross_attn = nn.MultiheadAttention(dim, num_heads=4, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.aggregation_weights = nn.Parameter(torch.ones(1, num_homology_dims + 1, 1))

    def forward(self, tokens):
        B = tokens.shape[0]
        cls_token = tokens[:, 0:1]
        homology_tokens = tokens[:, 1:]

        aggregated, _ = self.cross_attn(cls_token, homology_tokens, homology_tokens)
        aggregated = self.norm(aggregated)

        weights = F.softmax(self.aggregation_weights, dim=1)
        weighted_tokens = tokens * weights
        final = aggregated + weighted_tokens.sum(dim=1, keepdim=True)

        return final.squeeze(1)


class PersistenceWeightedPositionalEncoding(nn.Module):
    """
    NOVEL CONTRIBUTION: Positional encoding based on topological persistence
    rather than just spatial grid location.

    Key insight: Patches corresponding to high-persistence features should have
    similar positional encodings regardless of their spatial location, because
    they represent similar topological significance.
    """
    def __init__(self, dim, image_size=224, patch_size=14):
        super().__init__()
        self.dim = dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # Spatial positional encoding (standard grid-based)
        self.spatial_pos = nn.Parameter(torch.randn(1, self.num_patches, dim // 2))

        # Persistence-based encoding
        self.birth_encoder = nn.Sequential(
            nn.Linear(1, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim // 4)
        )

        self.persistence_encoder = nn.Sequential(
            nn.Linear(1, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim // 4)
        )

        # Fusion of spatial and topological encodings
        self.fusion = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Tanh()
        )

    def encode_patch_persistence(
        self,
        persistence_coords: torch.Tensor,  # (N_features, 3)
        pixel_coords: torch.Tensor         # (N_features, 2)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Assign persistence values to patches based on which features fall in each patch.

        Returns:
            patch_birth: (num_patches, 1) - average birth for each patch
            patch_persistence: (num_patches, 1) - average persistence for each patch
        """
        device = persistence_coords.device
        num_patches_h = self.image_size // self.patch_size
        num_patches_w = self.image_size // self.patch_size

        # Initialize patch statistics
        patch_birth_sum = torch.zeros(num_patches_h, num_patches_w, device=device)
        patch_pers_sum = torch.zeros(num_patches_h, num_patches_w, device=device)
        patch_count = torch.zeros(num_patches_h, num_patches_w, device=device)

        # Assign features to patches
        for i in range(len(pixel_coords)):
            # Skip padded entries (all zeros)
            if torch.all(pixel_coords[i] == 0) and torch.all(persistence_coords[i] == 0):
                continue

            x, y = pixel_coords[i]
            patch_x = int(torch.clamp(x / self.patch_size, 0, num_patches_w - 1))
            patch_y = int(torch.clamp(y / self.patch_size, 0, num_patches_h - 1))

            birth = persistence_coords[i, 0]
            pers = persistence_coords[i, 2]

            patch_birth_sum[patch_y, patch_x] += birth
            patch_pers_sum[patch_y, patch_x] += pers
            patch_count[patch_y, patch_x] += 1

        # Average persistence values per patch
        mask = patch_count > 0
        patch_birth_avg = torch.zeros_like(patch_birth_sum)
        patch_pers_avg = torch.zeros_like(patch_pers_sum)

        patch_birth_avg[mask] = patch_birth_sum[mask] / patch_count[mask]
        patch_pers_avg[mask] = patch_pers_sum[mask] / patch_count[mask]

        # Flatten to (num_patches, 1)
        patch_birth_flat = patch_birth_avg.flatten().unsqueeze(-1)  # (num_patches, 1)
        patch_pers_flat = patch_pers_avg.flatten().unsqueeze(-1)    # (num_patches, 1)

        return patch_birth_flat, patch_pers_flat

    def forward(
        self,
        batch_size: int,
        persistence_coords: Optional[torch.Tensor] = None,
        pixel_coords: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate positional encoding combining spatial and persistence information.

        Args:
            batch_size: Batch size
            persistence_coords: (B, N_features, 3) - optional persistence coordinates
            pixel_coords: (B, N_features, 2) - optional pixel coordinates

        Returns:
            pos_encoding: (B, num_patches, dim)
        """
        # Start with spatial encoding
        spatial_enc = self.spatial_pos.expand(batch_size, -1, -1)  # (B, num_patches, dim//2)

        if persistence_coords is None or pixel_coords is None:
            # Fall back to spatial-only encoding
            zero_enc = torch.zeros(batch_size, self.num_patches, self.dim // 2,
                                  device=spatial_enc.device)
            return torch.cat([spatial_enc, zero_enc], dim=-1)

        # Encode persistence information for each sample in batch
        batch_persistence_enc = []

        for b in range(batch_size):
            pers_coords_b = persistence_coords[b]  # (N_features, 3)
            pix_coords_b = pixel_coords[b]         # (N_features, 2)

            # Get per-patch persistence statistics
            patch_birth, patch_pers = self.encode_patch_persistence(pers_coords_b, pix_coords_b)

            # Encode through MLPs
            birth_enc = self.birth_encoder(patch_birth)        # (num_patches, dim//4)
            pers_enc = self.persistence_encoder(patch_pers)    # (num_patches, dim//4)

            # Combine
            persistence_enc = torch.cat([birth_enc, pers_enc], dim=-1)  # (num_patches, dim//2)
            batch_persistence_enc.append(persistence_enc)

        persistence_enc = torch.stack(batch_persistence_enc, dim=0)  # (B, num_patches, dim//2)

        # Combine spatial and persistence encodings
        combined = torch.cat([spatial_enc, persistence_enc], dim=-1)  # (B, num_patches, dim)

        # Fuse through learned transformation
        pos_encoding = self.fusion(combined)

        return pos_encoding


class PersistenceViT(nn.Module):
    """
    Vision Transformer for Persistence Images with topological inductive biases.

    Features:
    - Topological-aware patch embedding with importance weighting
    - Birth-death attention mechanism
    - Multi-scale persistence encoding via homology tokens
    - Topological feature aggregation
    - Persistence-weighted positional encoding
    """
    def __init__(
        self,
        image_size=256,
        patch_size=16,
        output_dim=128,
        dim=128,
        depth=4,
        heads=4,
        mlp_dim=256,
        channels=3,
        dropout=0.2,
        homology_dims=3,
    ):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2

        # Topological-aware patch embedding
        self.patch_embedding = TopologicalPatchEmbedding(
            channels, dim, patch_size, image_size
        )

        # Multi-scale persistence tokens
        self.homology_tokens = nn.Parameter(torch.randn(1, homology_dims, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # Persistence-weighted positional encoding
        self.pos_encoder = PersistenceWeightedPositionalEncoding(
            dim=dim,
            image_size=image_size,
            patch_size=patch_size
        )
        self.dropout = nn.Dropout(dropout)

        # Birth-death transformer blocks
        self.transformer_blocks = nn.ModuleList([
            BirthDeathTransformerBlock(dim, heads, mlp_dim, dropout)
            for _ in range(depth)
        ])

        # Topological aggregation
        self.topological_aggregation = TopologicalFeatureAggregation(dim, homology_dims)

        # Output head
        self.norm = nn.LayerNorm(dim)
        self.mlp_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, output_dim)
        )

    def forward(self, persistence_img, persistence_coords=None, pixel_coords=None):
        """
        Forward pass with optional persistence coordinate information.

        Args:
            persistence_img: (B, C, H, W) - persistence image
            persistence_coords: (B, N_features, 3) - optional normalized (birth, death, persistence)
            pixel_coords: (B, N_features, 2) - optional (x, y) pixel locations

        Returns:
            output: (B, output_dim) - encoded representation
        """
        B = persistence_img.shape[0]

        # Topological-aware patch embedding
        x, importance_weights = self.patch_embedding(persistence_img)  # (B, N, dim)

        # Generate persistence-weighted positional encoding
        pos_encoding = self.pos_encoder(
            batch_size=B,
            persistence_coords=persistence_coords,
            pixel_coords=pixel_coords
        )  # (B, num_patches, dim)

        # Add positional encoding
        x = x + pos_encoding

        # Add special tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        homology_tokens = self.homology_tokens.expand(B, -1, -1)
        x = torch.cat([cls_tokens, homology_tokens, x], dim=1)

        # Pad importance_weights for special tokens (cls + homology)
        num_special_tokens = 1 + self.homology_tokens.shape[1]
        importance_padding = torch.zeros(B, num_special_tokens, device=importance_weights.device)
        importance_weights_padded = torch.cat([importance_padding, importance_weights], dim=1)

        x = self.dropout(x)

        # Birth-death transformer blocks
        for block in self.transformer_blocks:
            x = block(x, importance_weights_padded)

        x = self.norm(x)

        # Topological aggregation
        x = self.topological_aggregation(x[:, :1+self.homology_tokens.shape[1]])

        return self.mlp_head(x)


# ============================================================================
# Unified Image Encoder Interface
# ============================================================================

class ImageEncoder(nn.Module):
    """
    Unified image encoder interface supporting multiple architectures:
    - SimpleCNN: Lightweight CNN
    - SmallViT: Compact Vision Transformer
    - PersistenceViT: Topologically-aware ViT
    - ResNet18/ResNet50/ResNet101: Standard CNNs with ImageNet pretraining
    - DINOv2: Self-supervised vision encoders (ViT-S/B/L/g)
    - ConvNeXt-Small: Modern CNN architecture
    - HybridResNetViT: ResNet conv layers + PersistenceViT attention
    """
    def __init__(self, output_dim=128, model_type='resnet18', image_size=224, freeze_backbone=False):
        super(ImageEncoder, self).__init__()

        if model_type == 'hybrid_resnet18_vit':
            from .hybrid_resnet_persistencevit import create_hybrid_encoder
            self.encoder = create_hybrid_encoder(
                output_dim=output_dim,
                image_size=image_size,
                freeze_backbone=freeze_backbone,
                resnet_type='resnet18',
                resnet_layers=2,
                dim=256,
                depth=4,
                heads=8,
                mlp_dim=512,
                dropout=0.2,
            )
            self.use_simple_model = True

        elif model_type == 'hybrid_resnet50_vit':
            from .hybrid_resnet_persistencevit import create_hybrid_encoder
            self.encoder = create_hybrid_encoder(
                output_dim=output_dim,
                image_size=image_size,
                freeze_backbone=freeze_backbone,
                resnet_type='resnet50',
                resnet_layers=2,
                dim=256,
                depth=4,
                heads=8,
                mlp_dim=512,
                dropout=0.2,
            )
            self.use_simple_model = True
            
        elif model_type == 'simplecnn':
            self.encoder = SimpleCNN(output_dim=output_dim)
            self.use_simple_model = True
        elif model_type == 'smallvit':
            self.encoder = SmallViT(image_size=image_size, patch_size=16, output_dim=output_dim,
                                   dim=128, depth=6, heads=4, mlp_dim=256, channels=3,
                                   dropout=0.2, emb_dropout=0.2)
            self.use_simple_model = True
        elif model_type == 'persistencevit':
            self.encoder = PersistenceViT(image_size=image_size, patch_size=16, output_dim=output_dim,
                                         dim=128, depth=6, heads=4, mlp_dim=256, channels=3,
                                         dropout=0.2, homology_dims=3)
            self.use_simple_model = True
        elif model_type == 'resnet18':
            resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            feature_dim = 512
            self.encoder = nn.Sequential(*list(resnet.children())[:-1])
            self.fc = nn.Sequential(
                 nn.Linear(feature_dim, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Linear(256, output_dim),
            )
            self.use_simple_model = False
        elif model_type == 'resnet50':
            resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            feature_dim = 2048
            self.encoder = nn.Sequential(*list(resnet.children())[:-1])
            self.fc = nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Linear(256, output_dim),
            )
            self.use_simple_model = False
        elif model_type == 'resnet101':
            resnet = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
            feature_dim = 2048
            self.encoder = nn.Sequential(*list(resnet.children())[:-1])
            self.fc = nn.Sequential(
                nn.Linear(feature_dim, 256)
            )
            self.use_simple_model = False
        elif model_type.startswith('dinov2'):
            self.encoder = DINOv2ImageEncoder(output_dim=output_dim, freeze_backbone=False, model_variant=model_type)
        
            self.use_simple_model = True
        elif model_type == 'convnext_small':
            from torchvision.models import convnext_small, ConvNeXt_Small_Weights
            convnext = convnext_small(weights=ConvNeXt_Small_Weights.IMAGENET1K_V1)
            feature_dim = 768
            self.encoder = nn.Sequential(*list(convnext.children())[:-1])
            if freeze_backbone:
                for param in self.encoder.parameters():
                    param.requires_grad = False
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(feature_dim, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Linear(256, output_dim),
            )
            self.use_simple_model = False
        else:
            raise ValueError(f"model_type must be 'simplecnn', 'smallvit', 'persistencevit', 'resnet18', 'resnet50', 'resnet101', 'dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14', 'convnext_small', or 'hybrid_resnet_vit', got {model_type}")

        self.model_type = model_type

    def forward(self, images, persistence_coords=None, pixel_coords=None):
        """
        Args:
            images: (B, 3, H, W) or (B, 1, H, W) tensor of persistence images
            persistence_coords: (B, N_features, 3) - optional for PersistenceViT
            pixel_coords: (B, N_features, 2) - optional for PersistenceViT
        Returns:
            features: (B, output_dim) tensor of image features
        """
        if self.use_simple_model:
            # PersistenceViT can use the coordinates
            if self.model_type == 'persistencevit':
                return self.encoder(images, persistence_coords, pixel_coords)
            else:
                return self.encoder(images)
        else:
            features = self.encoder(images)  # (B, feature_dim, 1, 1)
            features = features.view(features.size(0), -1)  # (B, feature_dim)
            features = self.fc(features)  # (B, output_dim)
            return features


class DINOv2ImageEncoder(nn.Module):
    """
    DINOv2-based image encoder for persistence images.
    Supports multiple DINOv2 variants: ViT-S/14, ViT-B/14, ViT-L/14, ViT-g/14.
    Returns raw features (384-dim for vits14) without projection head.
    """
    def __init__(self, output_dim=128, freeze_backbone=True, model_variant='dinov2_vits14'):
        super(DINOv2ImageEncoder, self).__init__()

        # Map model variant to feature dimension
        variant_dims = {
            'dinov2_vits14': 384,
            'dinov2_vitb14': 768,
            'dinov2_vitl14': 1024,
            'dinov2_vitg14': 1536
        }

        if model_variant not in variant_dims:
            raise ValueError(f"Unknown DINOv2 variant: {model_variant}")

        self.feat_dim = variant_dims[model_variant]

        # Load DINOv2 model
        self.backbone = torch.hub.load('facebookresearch/dinov2', model_variant)

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, images, persistence_coords=None, pixel_coords=None):
        """
        Args:
            images: (B, 3, H, W) tensor
            persistence_coords: ignored (for interface compatibility)
            pixel_coords: ignored (for interface compatibility)
        Returns:
            features: (B, feat_dim) raw DINOv2 features (384 for vits14)
        """
        # DINOv2 forward - return raw features
        features = self.backbone(images)  # (B, feat_dim)

        return features
