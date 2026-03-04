"""
Fusion Mechanisms for Multimodal Learning

This module contains various fusion strategies to combine tree and image features:
- CrossAttentionFusion: Cross-modal attention
- CMF: Cross-Modal Fusion
- BiDirectionalCrossAttention: Bidirectional cross-attention
- GatedFusion: Gated fusion with learnable gates
- MultiHeadCrossModalAttention: Multi-head cross-modal attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion for tree and image features"""
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super(CrossAttentionFusion, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads

        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        # Query, Key, Value projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, tree_feat, image_feat):
        """
        Args:
            tree_feat: (B, dim) tree features
            image_feat: (B, dim) image features
        Returns:
            fused: (B, dim) fused features
        """
        B = tree_feat.shape[0]

        # Add sequence dimension: (B, 1, dim)
        tree_feat = tree_feat.unsqueeze(1)
        image_feat = image_feat.unsqueeze(1)

        # Tree attends to image (tree as query, image as key/value)
        Q = self.q_proj(tree_feat).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(image_feat).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(image_feat).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = (attn @ V).transpose(1, 2).contiguous().view(B, 1, self.dim)

        # Project and add residual
        out = self.out_proj(out.squeeze(1))
        fused = out + tree_feat.squeeze(1)

        return fused


class CMF(nn.Module):
    """Cross-Modal Fusion with attention mechanism"""
    def __init__(self, dim, dropout=0.1):
        super(CMF, self).__init__()
        self.dim = dim

        # Feature-level attention
        self.tree_attn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1)
        )
        self.image_attn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1)
        )

        # Cross-modal interaction
        self.cross_proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )

    def forward(self, tree_feat, image_feat):
        """
        Args:
            tree_feat: (B, dim)
            image_feat: (B, dim)
        Returns:
            fused: (B, dim)
        """
        # Compute attention weights
        tree_weight = torch.sigmoid(self.tree_attn(tree_feat))
        image_weight = torch.sigmoid(self.image_attn(image_feat))

        # Normalize weights
        total_weight = tree_weight + image_weight + 1e-8
        tree_weight = tree_weight / total_weight
        image_weight = image_weight / total_weight

        # Weighted combination
        weighted_tree = tree_feat * tree_weight
        weighted_image = image_feat * image_weight

        # Cross-modal projection
        combined = torch.cat([weighted_tree, weighted_image], dim=1)
        fused = self.cross_proj(combined)

        return fused


class BiDirectionalCrossAttention(nn.Module):
    """Bidirectional cross-attention: tree→image and image→tree"""
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super(BiDirectionalCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads

        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        # Tree → Image attention
        self.tree2img_q = nn.Linear(dim, dim)
        self.tree2img_k = nn.Linear(dim, dim)
        self.tree2img_v = nn.Linear(dim, dim)
        self.tree2img_out = nn.Linear(dim, dim)

        # Image → Tree attention
        self.img2tree_q = nn.Linear(dim, dim)
        self.img2tree_k = nn.Linear(dim, dim)
        self.img2tree_v = nn.Linear(dim, dim)
        self.img2tree_out = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

        # Layer norm
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def _compute_attention(self, q_proj, k_proj, v_proj, query, key_value):
        """Helper function to compute cross-attention"""
        B = query.shape[0]

        # Add sequence dimension
        query = query.unsqueeze(1)  # (B, 1, dim)
        key_value = key_value.unsqueeze(1)  # (B, 1, dim)

        # Project
        Q = q_proj(query).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = k_proj(key_value).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        V = v_proj(key_value).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply to values
        out = (attn @ V).transpose(1, 2).contiguous().view(B, 1, self.dim)
        return out.squeeze(1)

    def forward(self, tree_feat, image_feat):
        """
        Args:
            tree_feat: (B, dim)
            image_feat: (B, dim)
        Returns:
            tree_enhanced: (B, dim)
            image_enhanced: (B, dim)
        """
        # Tree attends to image
        tree_enhanced = self._compute_attention(
            self.tree2img_q, self.tree2img_k, self.tree2img_v,
            tree_feat, image_feat
        )
        tree_enhanced = self.tree2img_out(tree_enhanced)
        tree_enhanced = self.norm1(tree_feat + tree_enhanced)

        # Image attends to tree
        image_enhanced = self._compute_attention(
            self.img2tree_q, self.img2tree_k, self.img2tree_v,
            image_feat, tree_feat
        )
        image_enhanced = self.img2tree_out(image_enhanced)
        image_enhanced = self.norm2(image_feat + image_enhanced)

        # Concatenate both enhanced features
        fused = torch.cat([tree_enhanced, image_enhanced], dim=1)

        return fused


class GatedFusion(nn.Module):
    """Gated fusion with learnable gates for tree and image modalities"""
    def __init__(self, dim, dropout=0.1):
        super(GatedFusion, self).__init__()
        self.dim = dim

        # Gating mechanism
        self.gate_tree = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        self.gate_image = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

        # Feature transformation
        self.tree_transform = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.image_transform = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Output projection
        self.output = nn.Linear(dim, dim)

    def forward(self, tree_feat, image_feat):
        """
        Args:
            tree_feat: (B, dim)
            image_feat: (B, dim)
        Returns:
            fused: (B, dim)
        """
        # Concatenate features for gating
        combined = torch.cat([tree_feat, image_feat], dim=1)

        # Compute gates
        gate_t = self.gate_tree(combined)
        gate_i = self.gate_image(combined)

        # Transform features
        tree_transformed = self.tree_transform(tree_feat)
        image_transformed = self.image_transform(image_feat)

        # Apply gates
        gated_tree = gate_t * tree_transformed
        gated_image = gate_i * image_transformed

        # Combine
        fused = gated_tree + gated_image
        fused = self.output(fused)

        return fused


class MultiHeadCrossModalAttention(nn.Module):
    """Multi-head cross-modal attention for flexible fusion"""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super(MultiHeadCrossModalAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads

        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        # Projections
        self.q_tree = nn.Linear(dim, dim)
        self.k_tree = nn.Linear(dim, dim)
        self.v_tree = nn.Linear(dim, dim)

        self.q_image = nn.Linear(dim, dim)
        self.k_image = nn.Linear(dim, dim)
        self.v_image = nn.Linear(dim, dim)

        self.out_proj = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

        self.norm = nn.LayerNorm(dim)

    def forward(self, tree_feat, image_feat):
        """
        Args:
            tree_feat: (B, dim)
            image_feat: (B, dim)
        Returns:
            fused: (B, dim)
        """
        B = tree_feat.shape[0]

        # Add sequence dimension
        tree_feat = tree_feat.unsqueeze(1)  # (B, 1, dim)
        image_feat = image_feat.unsqueeze(1)  # (B, 1, dim)

        # Project tree features
        Q_t = self.q_tree(tree_feat).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K_t = self.k_tree(tree_feat).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        V_t = self.v_tree(tree_feat).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)

        # Project image features
        Q_i = self.q_image(image_feat).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K_i = self.k_image(image_feat).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        V_i = self.v_image(image_feat).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)

        # Tree self-attention with image context
        attn_t = (Q_t @ K_i.transpose(-2, -1)) * self.scale
        attn_t = torch.softmax(attn_t, dim=-1)
        attn_t = self.dropout(attn_t)
        out_t = (attn_t @ V_i).transpose(1, 2).contiguous().view(B, 1, self.dim)

        # Image self-attention with tree context
        attn_i = (Q_i @ K_t.transpose(-2, -1)) * self.scale
        attn_i = torch.softmax(attn_i, dim=-1)
        attn_i = self.dropout(attn_i)
        out_i = (attn_i @ V_t).transpose(1, 2).contiguous().view(B, 1, self.dim)

        # Concatenate and project
        combined = torch.cat([out_t.squeeze(1), out_i.squeeze(1)], dim=1)
        fused = self.out_proj(combined)
        fused = self.norm(fused)

        return fused
