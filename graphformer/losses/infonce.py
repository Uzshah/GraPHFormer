"""
InfoNCE Loss Implementation

InfoNCE (Information Noise-Contrastive Estimation) loss from:
"Representation Learning with Contrastive Predictive Coding" (van den Oord et al., 2018)

Used in many contrastive learning methods like MoCo, SimCLR, etc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    InfoNCE loss for contrastive learning

    The loss encourages positive pairs to have high similarity while
    negative pairs have low similarity using a contrastive objective.
    """
    def __init__(self, temperature=0.07, reduction='mean'):
        """
        Args:
            temperature: Temperature parameter for scaling similarities
            reduction: 'mean' or 'sum' for loss reduction
        """
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, anchor, positive, negatives=None):
        """
        Compute InfoNCE loss

        Args:
            anchor: (B, D) anchor embeddings
            positive: (B, D) positive embeddings (paired with anchor)
            negatives: (B, N, D) or None - negative embeddings
                      If None, uses all other samples in batch as negatives (in-batch negatives)

        Returns:
            loss: scalar InfoNCE loss
        """
        # Normalize embeddings
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)

        B = anchor.shape[0]

        if negatives is None:
            # Use in-batch negatives (like SimCLR)
            # Each sample uses all other samples as negatives

            # Positive similarity: (B,)
            pos_sim = torch.sum(anchor * positive, dim=-1) / self.temperature

            # Negative similarities: (B, B)
            # anchor vs all positives (including itself)
            neg_sim = torch.matmul(anchor, positive.T) / self.temperature

            # Create labels (diagonal elements are positive pairs)
            labels = torch.arange(B, device=anchor.device)

            # InfoNCE loss using cross-entropy
            loss = F.cross_entropy(neg_sim, labels, reduction=self.reduction)

        else:
            # Explicit negatives provided
            negatives = F.normalize(negatives, dim=-1)

            # Positive similarity: (B,)
            pos_sim = torch.sum(anchor * positive, dim=-1, keepdim=True) / self.temperature

            # Negative similarities: (B, N)
            neg_sim = torch.matmul(anchor.unsqueeze(1), negatives.transpose(1, 2)).squeeze(1) / self.temperature

            # Concatenate positive and negative similarities
            logits = torch.cat([pos_sim, neg_sim], dim=1)  # (B, 1+N)

            # Labels: first position (index 0) is the positive
            labels = torch.zeros(B, dtype=torch.long, device=anchor.device)

            # InfoNCE loss
            loss = F.cross_entropy(logits, labels, reduction=self.reduction)

        return loss


class SymmetricInfoNCELoss(nn.Module):
    """
    Symmetric InfoNCE loss (like CLIP)

    Computes InfoNCE in both directions:
    - anchor -> positive (image -> text in CLIP)
    - positive -> anchor (text -> image in CLIP)

    Then averages the two losses.
    """
    def __init__(self, temperature=0.07):
        super(SymmetricInfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, embedding_a, embedding_b):
        """
        Compute symmetric InfoNCE loss between two sets of embeddings

        Args:
            embedding_a: (B, D) - first modality (e.g., tree embeddings)
            embedding_b: (B, D) - second modality (e.g., image embeddings)

        Returns:
            loss: scalar symmetric InfoNCE loss
        """
        # Normalize
        embedding_a = F.normalize(embedding_a, dim=-1)
        embedding_b = F.normalize(embedding_b, dim=-1)

        B = embedding_a.shape[0]

        # Compute similarity matrix: (B, B)
        similarity = torch.matmul(embedding_a, embedding_b.T) / self.temperature

        # Labels: diagonal elements are positive pairs
        labels = torch.arange(B, device=embedding_a.device)

        # Loss in both directions
        loss_a2b = F.cross_entropy(similarity, labels)
        loss_b2a = F.cross_entropy(similarity.T, labels)

        # Average
        loss = (loss_a2b + loss_b2a) / 2

        return loss


class HardNegativeInfoNCELoss(nn.Module):
    """
    InfoNCE with hard negative mining

    Selects the hardest negatives (highest similarity) for each anchor
    to make training more challenging and effective.
    """
    def __init__(self, temperature=0.07, num_hard_negatives=10):
        """
        Args:
            temperature: Temperature parameter
            num_hard_negatives: Number of hard negatives to mine per anchor
        """
        super(HardNegativeInfoNCELoss, self).__init__()
        self.temperature = temperature
        self.num_hard_negatives = num_hard_negatives

    def forward(self, anchor, positive, negative_pool):
        """
        Compute InfoNCE with hard negative mining

        Args:
            anchor: (B, D) anchor embeddings
            positive: (B, D) positive embeddings
            negative_pool: (M, D) pool of negative embeddings (M >> B)

        Returns:
            loss: scalar InfoNCE loss with hard negatives
        """
        # Normalize
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negative_pool = F.normalize(negative_pool, dim=-1)

        B = anchor.shape[0]

        # Positive similarity: (B,)
        pos_sim = torch.sum(anchor * positive, dim=-1, keepdim=True) / self.temperature

        # Compute similarity to all negatives: (B, M)
        all_neg_sim = torch.matmul(anchor, negative_pool.T) / self.temperature

        # Select top-k hard negatives (highest similarity = hardest)
        hard_neg_sim, _ = torch.topk(all_neg_sim, k=self.num_hard_negatives, dim=1)

        # Concatenate positive and hard negative similarities
        logits = torch.cat([pos_sim, hard_neg_sim], dim=1)  # (B, 1+K)

        # Labels: first position is positive
        labels = torch.zeros(B, dtype=torch.long, device=anchor.device)

        # InfoNCE loss
        loss = F.cross_entropy(logits, labels)

        return loss


class MultiModalInfoNCELoss(nn.Module):
    """
    Multi-modal InfoNCE for learning joint embeddings across multiple modalities

    Example: tree structure + persistence image + graph features
    """
    def __init__(self, temperature=0.07, weight_modalities=None):
        """
        Args:
            temperature: Temperature parameter
            weight_modalities: List of weights for each modality pair loss
        """
        super(MultiModalInfoNCELoss, self).__init__()
        self.temperature = temperature
        self.weight_modalities = weight_modalities

    def forward(self, embeddings):
        """
        Compute InfoNCE across all modality pairs

        Args:
            embeddings: List of (B, D) embeddings for each modality

        Returns:
            loss: scalar multi-modal InfoNCE loss
        """
        num_modalities = len(embeddings)

        # Normalize all embeddings
        embeddings = [F.normalize(emb, dim=-1) for emb in embeddings]

        B = embeddings[0].shape[0]
        labels = torch.arange(B, device=embeddings[0].device)

        # Compute loss for all pairs of modalities
        total_loss = 0
        num_pairs = 0

        for i in range(num_modalities):
            for j in range(i + 1, num_modalities):
                # Similarity matrix
                similarity = torch.matmul(embeddings[i], embeddings[j].T) / self.temperature

                # Symmetric loss
                loss_ij = F.cross_entropy(similarity, labels)
                loss_ji = F.cross_entropy(similarity.T, labels)
                pair_loss = (loss_ij + loss_ji) / 2

                # Apply weight if provided
                if self.weight_modalities is not None:
                    weight = self.weight_modalities[num_pairs]
                    pair_loss = weight * pair_loss

                total_loss += pair_loss
                num_pairs += 1

        # Average over all pairs
        loss = total_loss / num_pairs

        return loss


if __name__ == "__main__":
    print("Testing InfoNCE Loss implementations...")

    B, D = 32, 128

    # Test 1: Basic InfoNCE with in-batch negatives
    print("\n1. Basic InfoNCE Loss:")
    loss_fn = InfoNCELoss(temperature=0.07)

    anchor = torch.randn(B, D)
    positive = torch.randn(B, D)

    loss = loss_fn(anchor, positive)
    print(f"   Loss: {loss.item():.4f}")

    # Test 2: Symmetric InfoNCE (CLIP-style)
    print("\n2. Symmetric InfoNCE Loss (CLIP-style):")
    symmetric_loss = SymmetricInfoNCELoss(temperature=0.07)

    tree_embeddings = torch.randn(B, D)
    image_embeddings = torch.randn(B, D)

    loss = symmetric_loss(tree_embeddings, image_embeddings)
    print(f"   Loss: {loss.item():.4f}")

    # Test 3: Hard Negative Mining
    print("\n3. InfoNCE with Hard Negative Mining:")
    hard_neg_loss = HardNegativeInfoNCELoss(temperature=0.07, num_hard_negatives=10)

    anchor = torch.randn(B, D)
    positive = torch.randn(B, D)
    negative_pool = torch.randn(500, D)  # Large pool of negatives

    loss = hard_neg_loss(anchor, positive, negative_pool)
    print(f"   Loss: {loss.item():.4f}")

    # Test 4: Multi-modal InfoNCE
    print("\n4. Multi-modal InfoNCE:")
    multimodal_loss = MultiModalInfoNCELoss(temperature=0.07)

    tree_emb = torch.randn(B, D)
    image_emb = torch.randn(B, D)
    graph_emb = torch.randn(B, D)

    loss = multimodal_loss([tree_emb, image_emb, graph_emb])
    print(f"   Loss: {loss.item():.4f}")

    print("\nAll tests passed!")
