"""
Hybrid ResNet-PersistenceViT Architecture

Combines:
- ResNet's early convolutional layers (conv1, bn1, relu, maxpool, layer1, layer2)
- PersistenceViT's topological attention mechanism (replacing layer3, layer4)

This hybrid leverages:
1. ResNet's proven feature extraction in lower layers
2. PersistenceViT's topological awareness in higher layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet50_Weights


# Import PersistenceViT components
from .image_encoder import (
    BirthDeathAttention,
    BirthDeathTransformerBlock,
    TopologicalFeatureAggregation,
)


class ResNetFeatureExtractor(nn.Module):
    """Extract features from early ResNet layers"""
    def __init__(self, model_type='resnet18', num_layers=2):
        """
        Args:
            model_type: 'resnet18' or 'resnet50'
            num_layers: How many ResNet layers to keep (1, 2, or 3)
                       1 = conv1 + layer1 (64 channels, /4 spatial)
                       2 = conv1 + layer1 + layer2 (128/512 channels, /8 spatial)
                       3 = conv1 + layer1 + layer2 + layer3 (256/1024 channels, /16 spatial)
        """
        super().__init__()

        if model_type == 'resnet18':
            resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.layer_dims = [64, 128, 256, 512]
        elif model_type == 'resnet50':
            resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self.layer_dims = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"model_type must be 'resnet18' or 'resnet50'")

        self.model_type = model_type
        self.num_layers = num_layers

        # Initial conv layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # ResNet layers
        self.layer1 = resnet.layer1
        if num_layers >= 2:
            self.layer2 = resnet.layer2
        if num_layers >= 3:
            self.layer3 = resnet.layer3

        # Output channels
        self.out_channels = self.layer_dims[num_layers - 1]

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            features: (B, out_channels, H', W') where H' = H / (4 * 2^num_layers)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        if self.num_layers >= 2:
            x = self.layer2(x)
        if self.num_layers >= 3:
            x = self.layer3(x)

        return x


class CNNToTransformerAdapter(nn.Module):
    """Convert CNN feature maps to transformer tokens"""
    def __init__(self, in_channels, dim, patch_size=2):
        """
        Args:
            in_channels: Number of input channels from CNN
            dim: Transformer hidden dimension
            patch_size: Size of patches to group CNN features (default: 2x2)
        """
        super().__init__()
        self.patch_size = patch_size

        # 1x1 conv to reduce channels, then adaptive pooling
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

        # Importance estimator (similar to PersistenceViT)
        self.importance_estimator = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=patch_size, stride=patch_size),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: (B, in_channels, H, W)
        Returns:
            tokens: (B, N, dim) where N = (H/patch_size) * (W/patch_size)
            importance: (B, N)
        """
        B, C, H, W = x.shape

        # Project to embedding dimension
        tokens = self.proj(x)  # (B, dim, H', W')
        tokens = tokens.flatten(2).transpose(1, 2)  # (B, N, dim)

        # Compute importance weights
        importance = self.importance_estimator(x)  # (B, 1, H', W')
        importance = importance.flatten(2).transpose(1, 2).squeeze(-1)  # (B, N)

        # Weight tokens by importance
        weighted_tokens = tokens * (1 + importance.unsqueeze(-1))

        return weighted_tokens, importance


class HybridResNetPersistenceViT(nn.Module):
    """
    Hybrid architecture combining ResNet and PersistenceViT

    Architecture:
    1. ResNet early layers (conv1 + layer1 + layer2) - proven feature extraction
    2. CNN-to-Transformer adapter - convert feature maps to tokens
    3. PersistenceViT attention blocks - topological reasoning
    4. Classification head
    """
    def __init__(
        self,
        output_dim=128,
        image_size=224,
        resnet_type='resnet18',
        resnet_layers=2,
        dim=256,
        depth=4,
        heads=8,
        mlp_dim=512,
        dropout=0.2,
        homology_dims=3,
        freeze_resnet=False,
    ):
        """
        Args:
            output_dim: Output feature dimension
            image_size: Input image size
            resnet_type: 'resnet18' or 'resnet50'
            resnet_layers: Number of ResNet layers to keep (1, 2, or 3)
            dim: Transformer hidden dimension
            depth: Number of transformer blocks
            heads: Number of attention heads
            mlp_dim: MLP hidden dimension
            dropout: Dropout rate
            homology_dims: Number of homology dimension tokens
            freeze_resnet: Freeze ResNet layers
        """
        super().__init__()

        self.image_size = image_size
        self.dim = dim

        # 1. ResNet feature extractor
        self.resnet_extractor = ResNetFeatureExtractor(
            model_type=resnet_type,
            num_layers=resnet_layers
        )

        if freeze_resnet:
            for param in self.resnet_extractor.parameters():
                param.requires_grad = False

        # 2. CNN-to-Transformer adapter
        # After resnet_layers=2: spatial size is image_size / 8
        # We further reduce by patch_size=2, so final is image_size / 16
        self.adapter = CNNToTransformerAdapter(
            in_channels=self.resnet_extractor.out_channels,
            dim=dim,
            patch_size=2
        )

        # Calculate number of patches
        spatial_reduction = 8 * 2  # ResNet (layer2) + adapter patch_size
        self.num_patches = (image_size // spatial_reduction) ** 2

        # 3. Special tokens (like PersistenceViT)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.homology_tokens = nn.Parameter(torch.randn(1, homology_dims, dim))

        # 4. Positional encoding
        num_special_tokens = 1 + homology_dims
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_special_tokens + self.num_patches, dim)
        )
        self.dropout = nn.Dropout(dropout)

        # 5. PersistenceViT transformer blocks
        self.transformer_blocks = nn.ModuleList([
            BirthDeathTransformerBlock(dim, heads, mlp_dim, dropout)
            for _ in range(depth)
        ])

        # 6. Topological aggregation
        self.topological_aggregation = TopologicalFeatureAggregation(
            dim, homology_dims
        )

        # 7. Output head
        self.norm = nn.LayerNorm(dim)
        self.mlp_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, output_dim)
        )

    def forward(self, images):
        """
        Args:
            images: (B, 3, H, W)
        Returns:
            features: (B, output_dim)
        """
        B = images.shape[0]

        # 1. Extract features with ResNet
        cnn_features = self.resnet_extractor(images)  # (B, C, H', W')

        # 2. Convert to transformer tokens
        tokens, importance_weights = self.adapter(cnn_features)  # (B, N, dim)

        # 3. Add special tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        homology_tokens = self.homology_tokens.expand(B, -1, -1)
        x = torch.cat([cls_tokens, homology_tokens, tokens], dim=1)  # (B, 1+H+N, dim)

        # 4. Add positional encoding
        x = x + self.pos_embedding[:, :(x.shape[1])]
        x = self.dropout(x)

        # 5. Pad importance weights for special tokens
        num_special_tokens = 1 + self.homology_tokens.shape[1]
        importance_padding = torch.zeros(B, num_special_tokens, device=importance_weights.device)
        importance_weights_padded = torch.cat([importance_padding, importance_weights], dim=1)

        # 6. Apply transformer blocks with topological attention
        for block in self.transformer_blocks:
            x = block(x, importance_weights_padded)

        x = self.norm(x)

        # 7. Topological aggregation (use cls + homology tokens)
        x = self.topological_aggregation(x[:, :num_special_tokens])

        # 8. Output projection
        output = self.mlp_head(x)

        return output


class HybridImageEncoder(nn.Module):
    """
    Wrapper to integrate HybridResNetPersistenceViT into existing codebase
    Compatible with ImageEncoder interface
    """
    def __init__(
        self,
        output_dim=128,
        image_size=224,
        resnet_type='resnet18',
        resnet_layers=2,
        dim=256,
        depth=4,
        heads=8,
        mlp_dim=512,
        dropout=0.2,
        homology_dims=3,
        freeze_resnet=False,
    ):
        super().__init__()

        self.encoder = HybridResNetPersistenceViT(
            output_dim=output_dim,
            image_size=image_size,
            resnet_type=resnet_type,
            resnet_layers=resnet_layers,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            homology_dims=homology_dims,
            freeze_resnet=freeze_resnet,
        )

    def forward(self, images, persistence_coords=None, pixel_coords=None):
        """
        Args:
            images: (B, 3, H, W)
            persistence_coords: ignored (for interface compatibility)
            pixel_coords: ignored (for interface compatibility)
        Returns:
            features: (B, output_dim)
        """
        return self.encoder(images)


# ============================================================================
# Integration into existing ImageEncoder
# ============================================================================

def create_hybrid_encoder(output_dim=128, image_size=224, freeze_backbone=False, **kwargs):
    """
    Factory function to create hybrid encoder

    Args:
        output_dim: Output dimension
        image_size: Input image size
        freeze_backbone: Freeze ResNet layers
        **kwargs: Additional arguments
            - resnet_type: 'resnet18' or 'resnet50' (default: 'resnet50')
            - resnet_layers: Number of ResNet layers (1, 2, or 3) (default: 2)
            - dim: Transformer dimension (default: 256)
            - depth: Number of transformer blocks (default: 4)
            - heads: Number of attention heads (default: 8)
    """
    resnet_type = kwargs.get('resnet_type', 'resnet50')
    resnet_layers = kwargs.get('resnet_layers', 2)
    dim = kwargs.get('dim', 256)
    depth = kwargs.get('depth', 4)
    heads = kwargs.get('heads', 8)
    mlp_dim = kwargs.get('mlp_dim', 512)
    dropout = kwargs.get('dropout', 0.2)
    homology_dims = kwargs.get('homology_dims', 3)

    return HybridImageEncoder(
        output_dim=output_dim,
        image_size=image_size,
        resnet_type=resnet_type,
        resnet_layers=resnet_layers,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        dropout=dropout,
        homology_dims=homology_dims,
        freeze_resnet=freeze_backbone,
    )


if __name__ == "__main__":
    # Test the hybrid model
    print("Testing HybridResNetPersistenceViT...")

    # Create model
    model = HybridResNetPersistenceViT(
        output_dim=256,
        image_size=224,
        resnet_type='resnet18',
        resnet_layers=2,
        dim=256,
        depth=4,
        heads=8,
        mlp_dim=512,
        dropout=0.2,
        homology_dims=3,
        freeze_resnet=False,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)

    output = model(images)
    print(f"Input shape: {images.shape}")
    print(f"Output shape: {output.shape}")

    # Component breakdown
    resnet_params = sum(p.numel() for p in model.resnet_extractor.parameters())
    adapter_params = sum(p.numel() for p in model.adapter.parameters())
    transformer_params = sum(p.numel() for p in model.transformer_blocks.parameters())

    print(f"\nComponent breakdown:")
    print(f"  ResNet layers: {resnet_params:,}")
    print(f"  CNN-to-Transformer adapter: {adapter_params:,}")
    print(f"  Transformer blocks: {transformer_params:,}")

    print("\nHybrid model test passed!")
