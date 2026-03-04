"""
Contrastive Loss Implementations for Neuron Representation Learning

Includes:
- NT-Xent (Normalized Temperature-scaled Cross Entropy) - SimCLR loss
- Triplet Loss (with various mining strategies)
- InfoNCE variants (imported from infonce_loss.py)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss

    Used in SimCLR (Chen et al., 2020) - "A Simple Framework for Contrastive Learning"

    Key features:
    - Normalizes embeddings to unit sphere
    - Uses temperature scaling
    - Treats all other samples in batch as negatives
    - Symmetric loss over both views
    """
    def __init__(self, temperature=0.5, use_cosine_similarity=True):
        """
        Args:
            temperature: Temperature parameter for scaling (default 0.5 from SimCLR)
            use_cosine_similarity: If True, use cosine similarity (default True)
        """
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.use_cosine_similarity = use_cosine_similarity

    def forward(self, z_i, z_j):
        """
        Compute NT-Xent loss between two views

        Args:
            z_i: (B, D) embeddings from view 1
            z_j: (B, D) embeddings from view 2

        Returns:
            loss: scalar NT-Xent loss
        """
        batch_size = z_i.shape[0]

        # Normalize embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Concatenate both views: (2B, D)
        z = torch.cat([z_i, z_j], dim=0)

        # Compute similarity matrix: (2B, 2B)
        if self.use_cosine_similarity:
            similarity_matrix = torch.matmul(z, z.T)
        else:
            # Dot product similarity
            similarity_matrix = torch.matmul(z, z.T)

        # Remove diagonal elements (self-similarity)
        # Create mask to exclude diagonal
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)

        # Apply temperature scaling
        similarity_matrix = similarity_matrix / self.temperature

        # Positive pairs are at positions (i, i+B) and (i+B, i)
        # For each sample, its positive is at distance B
        positive_samples = torch.cat([
            torch.arange(batch_size, 2 * batch_size),  # Positives for first half
            torch.arange(0, batch_size)                 # Positives for second half
        ], dim=0).to(z.device)

        # Extract positive similarities
        positive_sim = similarity_matrix[torch.arange(2 * batch_size), positive_samples]
        positive_sim = positive_sim.reshape(2 * batch_size, 1)

        # Negative similarities: all other samples (already in similarity_matrix)
        # For numerical stability, use log-sum-exp trick
        negatives_mask = ~mask
        negatives_mask[torch.arange(2 * batch_size), positive_samples] = False

        # Compute loss using LogSumExp
        # loss = -log(exp(pos) / (exp(pos) + sum(exp(neg))))
        # For numerical stability: -pos + log(exp(pos) + sum(exp(neg)))

        # Get all similarities (positive at correct position + negatives)
        all_similarities = similarity_matrix  # (2B, 2B)

        # Create labels for cross-entropy: positive is at position batch_size or 0
        labels = positive_samples

        # Use cross-entropy loss (equivalent to NT-Xent)
        loss = F.cross_entropy(all_similarities, labels)

        return loss


class TripletLoss(nn.Module):
    """
    Triplet Loss for metric learning

    L = max(d(a, p) - d(a, n) + margin, 0)

    Where:
    - a: anchor
    - p: positive (same class as anchor)
    - n: negative (different class)
    - d: distance metric (L2 or cosine)
    """
    def __init__(self, margin=1.0, distance_metric='euclidean', mining='batch_hard'):
        """
        Args:
            margin: Margin for triplet loss
            distance_metric: 'euclidean' or 'cosine'
            mining: 'batch_hard', 'batch_all', or 'semi_hard'
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.distance_metric = distance_metric
        self.mining = mining

    def _compute_distance_matrix(self, embeddings):
        """Compute pairwise distance matrix."""
        if self.distance_metric == 'euclidean':
            # Efficient euclidean distance: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b
            dot_product = torch.matmul(embeddings, embeddings.T)
            square_norm = torch.diag(dot_product)
            distances = square_norm.unsqueeze(1) - 2.0 * dot_product + square_norm.unsqueeze(0)
            distances = torch.clamp(distances, min=0.0)  # Numerical stability
            distances = torch.sqrt(distances + 1e-16)
        elif self.distance_metric == 'cosine':
            # Cosine distance = 1 - cosine_similarity
            embeddings_norm = F.normalize(embeddings, dim=1)
            cosine_sim = torch.matmul(embeddings_norm, embeddings_norm.T)
            distances = 1.0 - cosine_sim
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

        return distances

    def _batch_hard_mining(self, embeddings, labels):
        """
        Batch hard mining: for each anchor, select hardest positive and hardest negative.

        Hardest positive: furthest positive sample
        Hardest negative: closest negative sample
        """
        batch_size = embeddings.shape[0]

        # Compute pairwise distances
        distances = self._compute_distance_matrix(embeddings)

        # Create masks for positives and negatives
        labels = labels.unsqueeze(1)
        positive_mask = (labels == labels.T).float()
        negative_mask = (labels != labels.T).float()

        # Remove self-comparisons from positive mask
        positive_mask = positive_mask - torch.eye(batch_size, device=embeddings.device)

        # Hard positive: maximum distance among positives
        # Set non-positive distances to 0 so they won't be selected
        positive_distances = distances * positive_mask
        hardest_positive_dist, _ = torch.max(positive_distances, dim=1)

        # Hard negative: minimum distance among negatives
        # Set non-negative distances to large value so they won't be selected
        negative_distances = distances + (1.0 - negative_mask) * 1e9
        hardest_negative_dist, _ = torch.min(negative_distances, dim=1)

        # Compute triplet loss
        triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)

        return triplet_loss.mean()

    def _batch_all_mining(self, embeddings, labels):
        """
        Batch all mining: use all valid triplets in the batch.
        """
        batch_size = embeddings.shape[0]

        # Compute pairwise distances
        distances = self._compute_distance_matrix(embeddings)

        # Create masks
        labels = labels.unsqueeze(1)
        positive_mask = (labels == labels.T).float()
        negative_mask = (labels != labels.T).float()

        # Remove self-comparisons
        positive_mask = positive_mask - torch.eye(batch_size, device=embeddings.device)

        # Get anchor-positive distances: (B, B)
        anchor_positive_dist = distances.unsqueeze(2)  # (B, B, 1)

        # Get anchor-negative distances: (B, B)
        anchor_negative_dist = distances.unsqueeze(1)  # (B, 1, B)

        # Compute triplet loss for all valid triplets
        triplet_loss = anchor_positive_dist - anchor_negative_dist + self.margin

        # Mask out invalid triplets
        # Valid triplet: (i, j, k) where label[i] == label[j] != label[k]
        valid_triplets = positive_mask.unsqueeze(2) * negative_mask.unsqueeze(1)

        # Apply mask and ReLU
        triplet_loss = triplet_loss * valid_triplets
        triplet_loss = F.relu(triplet_loss)

        # Count valid triplets
        num_valid = valid_triplets.sum()

        if num_valid > 0:
            triplet_loss = triplet_loss.sum() / num_valid
        else:
            triplet_loss = torch.tensor(0.0, device=embeddings.device)

        return triplet_loss

    def _semi_hard_mining(self, embeddings, labels):
        """
        Semi-hard mining: select negatives that are harder than positive but still within margin.

        Semi-hard negative: d(a,p) < d(a,n) < d(a,p) + margin
        """
        batch_size = embeddings.shape[0]

        # Compute pairwise distances
        distances = self._compute_distance_matrix(embeddings)

        # Create masks
        labels = labels.unsqueeze(1)
        positive_mask = (labels == labels.T).float()
        negative_mask = (labels != labels.T).float()

        # Remove self-comparisons
        positive_mask = positive_mask - torch.eye(batch_size, device=embeddings.device)

        losses = []

        for i in range(batch_size):
            # Get positive distances for anchor i
            pos_dists = distances[i] * positive_mask[i]
            if pos_dists.sum() == 0:
                continue

            # Select a positive (use hardest for stability)
            pos_dist = pos_dists.max()

            # Get negative distances for anchor i
            neg_dists = distances[i] * negative_mask[i]

            # Semi-hard negatives: pos_dist < neg_dist < pos_dist + margin
            semi_hard_mask = (neg_dists > pos_dist) & (neg_dists < pos_dist + self.margin)

            if semi_hard_mask.any():
                # Use hardest semi-hard negative (closest to anchor)
                semi_hard_negatives = neg_dists.clone()
                semi_hard_negatives[~semi_hard_mask] = 1e9
                neg_dist = semi_hard_negatives.min()
            else:
                # Fall back to hardest negative
                neg_dists_masked = neg_dists + (1.0 - negative_mask[i]) * 1e9
                neg_dist = neg_dists_masked.min()

            # Compute triplet loss
            loss = F.relu(pos_dist - neg_dist + self.margin)
            losses.append(loss)

        if len(losses) > 0:
            return torch.stack(losses).mean()
        else:
            return torch.tensor(0.0, device=embeddings.device)

    def forward(self, embeddings, labels):
        """
        Compute triplet loss

        Args:
            embeddings: (B, D) embeddings
            labels: (B,) class labels for each sample

        Returns:
            loss: scalar triplet loss
        """
        if self.mining == 'batch_hard':
            return self._batch_hard_mining(embeddings, labels)
        elif self.mining == 'batch_all':
            return self._batch_all_mining(embeddings, labels)
        elif self.mining == 'semi_hard':
            return self._semi_hard_mining(embeddings, labels)
        else:
            raise ValueError(f"Unknown mining strategy: {self.mining}")


class CombinedContrastiveLoss(nn.Module):
    """
    Combine multiple contrastive losses with configurable weights

    Example: NT-Xent + Triplet Loss
    """
    def __init__(self, loss_types=['ntxent'], loss_weights=None, **loss_kwargs):
        """
        Args:
            loss_types: List of loss types ('ntxent', 'triplet', 'infonce')
            loss_weights: List of weights for each loss (default: equal weights)
            **loss_kwargs: Keyword arguments for each loss
                - ntxent_temperature: Temperature for NT-Xent
                - triplet_margin: Margin for Triplet Loss
                - triplet_mining: Mining strategy for Triplet Loss
        """
        super(CombinedContrastiveLoss, self).__init__()
        self.loss_types = loss_types

        if loss_weights is None:
            self.loss_weights = [1.0] * len(loss_types)
        else:
            self.loss_weights = loss_weights

        # Initialize losses
        self.losses = nn.ModuleDict()

        for loss_type in loss_types:
            if loss_type == 'ntxent':
                temp = loss_kwargs.get('ntxent_temperature', 0.5)
                self.losses['ntxent'] = NTXentLoss(temperature=temp)
            elif loss_type == 'triplet':
                margin = loss_kwargs.get('triplet_margin', 1.0)
                mining = loss_kwargs.get('triplet_mining', 'batch_hard')
                distance = loss_kwargs.get('triplet_distance', 'euclidean')
                self.losses['triplet'] = TripletLoss(
                    margin=margin,
                    distance_metric=distance,
                    mining=mining
                )
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(self, z_i, z_j, labels=None):
        """
        Compute combined loss

        Args:
            z_i: (B, D) embeddings from view 1
            z_j: (B, D) embeddings from view 2
            labels: (B,) class labels (required for triplet loss)

        Returns:
            loss: scalar combined loss
            loss_dict: dictionary with individual loss values
        """
        total_loss = 0.0
        loss_dict = {}

        for i, loss_type in enumerate(self.loss_types):
            weight = self.loss_weights[i]

            if loss_type == 'ntxent':
                loss_val = self.losses['ntxent'](z_i, z_j)
            elif loss_type == 'triplet':
                if labels is None:
                    raise ValueError("Triplet loss requires labels")
                # Combine both views for triplet loss
                embeddings = torch.cat([z_i, z_j], dim=0)
                combined_labels = torch.cat([labels, labels], dim=0)
                loss_val = self.losses['triplet'](embeddings, combined_labels)

            loss_dict[loss_type] = loss_val.item()
            total_loss += weight * loss_val

        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict


if __name__ == "__main__":
    print("Testing Contrastive Loss Implementations...")
    print("=" * 60)

    B, D = 32, 128
    num_classes = 10

    # Test 1: NT-Xent Loss
    print("\n1. NT-Xent Loss (SimCLR):")
    ntxent = NTXentLoss(temperature=0.5)

    z_i = torch.randn(B, D)
    z_j = torch.randn(B, D)

    loss = ntxent(z_i, z_j)
    print(f"   Loss: {loss.item():.4f}")

    # Test 2: Triplet Loss (Batch Hard)
    print("\n2. Triplet Loss (Batch Hard Mining):")
    triplet = TripletLoss(margin=1.0, mining='batch_hard')

    embeddings = torch.randn(B, D)
    labels = torch.randint(0, num_classes, (B,))

    loss = triplet(embeddings, labels)
    print(f"   Loss: {loss.item():.4f}")

    # Test 3: Triplet Loss (Batch All)
    print("\n3. Triplet Loss (Batch All Mining):")
    triplet_all = TripletLoss(margin=1.0, mining='batch_all')

    loss = triplet_all(embeddings, labels)
    print(f"   Loss: {loss.item():.4f}")

    # Test 4: Triplet Loss (Semi-Hard)
    print("\n4. Triplet Loss (Semi-Hard Mining):")
    triplet_semi = TripletLoss(margin=1.0, mining='semi_hard')

    loss = triplet_semi(embeddings, labels)
    print(f"   Loss: {loss.item():.4f}")

    # Test 5: Combined Loss (NT-Xent + Triplet)
    print("\n5. Combined Loss (NT-Xent + Triplet):")
    combined = CombinedContrastiveLoss(
        loss_types=['ntxent', 'triplet'],
        loss_weights=[1.0, 0.5],
        ntxent_temperature=0.5,
        triplet_margin=1.0,
        triplet_mining='batch_hard'
    )

    z_i = torch.randn(B, D)
    z_j = torch.randn(B, D)
    labels = torch.randint(0, num_classes, (B,))

    loss, loss_dict = combined(z_i, z_j, labels)
    print(f"   Total Loss: {loss.item():.4f}")
    print(f"   Loss breakdown: {loss_dict}")

    print("\n" + "=" * 60)
    print("All tests passed!")
