"""
Persistence Image Augmentation Module

This module provides augmentation strategies specifically designed for persistence images
generated from neuron morphology data. Unlike standard image augmentations (ColorJitter, etc.),
these augmentations work in the persistence space (birth, persistence) to create meaningful
variations while preserving the underlying topological structure.

Augmentation Strategies:
1. Translation in birth/persistence space - shifts points in the diagram
2. Gaussian noise addition - adds small random perturbations to point positions
3. Sigma variation - varies the Gaussian kernel width during image generation
"""

import numpy as np
import torch
from typing import Tuple, Optional
import random


class PersistenceSpaceAugmentation:
    """
    Augmentation that operates in persistence space before image generation.

    This augmentation modifies the birth/persistence coordinates of points
    in the persistence diagram, then regenerates the image with these modified coordinates.
    """

    def __init__(
        self,
        translation_scale: float = 0.05,
        noise_scale: float = 0.02,
        translation_prob: float = 0.5,
        noise_prob: float = 0.5,
        persistence_scale_prob: float = 0.0,
        persistence_scale_min: float = 0.9,
        persistence_scale_max: float = 1.1,
        radius_perturb_prob: float = 0.0,
        radius_perturb_min: float = 0.85,
        radius_perturb_max: float = 1.15,
    ):
        """
        Args:
            translation_scale: Scale of random translation (relative to data range)
            noise_scale: Scale of Gaussian noise (relative to data range)
            translation_prob: Probability of applying translation
            noise_prob: Probability of applying Gaussian noise
            persistence_scale_prob: Probability of applying persistence scaling
            persistence_scale_min: Minimum scaling factor for persistence
            persistence_scale_max: Maximum scaling factor for persistence
            radius_perturb_prob: Probability of applying radius perturbation
            radius_perturb_min: Minimum scaling factor for radius
            radius_perturb_max: Maximum scaling factor for radius
        """
        self.translation_scale = translation_scale
        self.noise_scale = noise_scale
        self.translation_prob = translation_prob
        self.noise_prob = noise_prob
        self.persistence_scale_prob = persistence_scale_prob
        self.persistence_scale_min = persistence_scale_min
        self.persistence_scale_max = persistence_scale_max
        self.radius_perturb_prob = radius_perturb_prob
        self.radius_perturb_min = radius_perturb_min
        self.radius_perturb_max = radius_perturb_max

    def augment_pairs_features(self, pairs_feats, global_bounds=None):
        """
        Augment the pairs features in persistence space.

        Args:
            pairs_feats: List of dictionaries with 'birth', 'death', 'persistence', 'mean_radius'
            global_bounds: Tuple of (birth_min, birth_max, pers_min, pers_max)

        Returns:
            Augmented pairs_feats
        """
        if not pairs_feats or len(pairs_feats) == 0:
            return pairs_feats

        # Create a copy to avoid modifying original
        augmented_feats = [f.copy() for f in pairs_feats]

        # Extract births and persistence values
        births = np.array([f['birth'] for f in augmented_feats])
        pers = np.array([f['persistence'] for f in augmented_feats])

        # Determine data range for scaling
        if global_bounds is not None:
            birth_range = global_bounds[1] - global_bounds[0]
            pers_range = global_bounds[3] - global_bounds[2]
        else:
            birth_range = births.max() - births.min() + 1e-6
            pers_range = pers.max() - pers.min() + 1e-6

        # Apply random translation in birth/persistence space
        if random.random() < self.translation_prob:
            birth_shift = np.random.uniform(-self.translation_scale, self.translation_scale) * birth_range
            pers_shift = np.random.uniform(-self.translation_scale, self.translation_scale) * pers_range

            births += birth_shift
            pers += pers_shift

        # Apply Gaussian noise
        if random.random() < self.noise_prob:
            birth_noise = np.random.normal(0, self.noise_scale * birth_range, size=births.shape)
            pers_noise = np.random.normal(0, self.noise_scale * pers_range, size=pers.shape)

            births += birth_noise
            pers += pers_noise

        # Apply persistence scaling
        if random.random() < self.persistence_scale_prob:
            alpha = np.random.uniform(self.persistence_scale_min, self.persistence_scale_max)
            pers *= alpha

        # Apply radius perturbation
        if random.random() < self.radius_perturb_prob:
            beta = np.random.uniform(self.radius_perturb_min, self.radius_perturb_max)
            for f in augmented_feats:
                if 'mean_radius' in f:
                    f['mean_radius'] *= beta

        # Ensure persistence values remain positive
        pers = np.maximum(pers, 1e-9)

        # Update the augmented features
        for i, f in enumerate(augmented_feats):
            f['birth'] = float(births[i])
            f['persistence'] = float(pers[i])
            # Note: death = birth - persistence (in TMD convention where persistence is negative)
            # Actually in this code: persistence = birth - death, so death = birth - persistence
            f['death'] = float(births[i] - pers[i])

        return augmented_feats


class SigmaVariationAugmentation:
    """
    Augmentation that varies the sigma parameter during Gaussian kernel application.

    This creates different "blur" levels in the persistence image, which can help
    the model learn features at multiple scales.
    """

    def __init__(
        self,
        sigma_min: float = 12.0,
        sigma_max: float = 20.0,
        prob: float = 1.0,
    ):
        """
        Args:
            sigma_min: Minimum sigma value
            sigma_max: Maximum sigma value
            prob: Probability of varying sigma (1.0 means always vary)
        """
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.prob = prob

    def sample_sigma(self, base_sigma: float = 16.0) -> float:
        """
        Sample a sigma value.

        Args:
            base_sigma: Base sigma value (unused when prob=1.0, kept for compatibility)

        Returns:
            Sampled sigma value
        """
        if random.random() < self.prob:
            return np.random.uniform(self.sigma_min, self.sigma_max)
        else:
            return base_sigma


class CombinedPersistenceAugmentation:
    """
    Combines multiple persistence space augmentations.

    This is the main augmentation class that should be used for training.
    """

    def __init__(
        self,
        translation_scale: float = 0.05,
        noise_scale: float = 0.02,
        sigma_min: float = 12.0,
        sigma_max: float = 20.0,
        translation_prob: float = 0.5,
        noise_prob: float = 0.5,
        sigma_variation_prob: float = 1.0,
        persistence_scale_prob: float = 0.0,
        persistence_scale_min: float = 0.9,
        persistence_scale_max: float = 1.1,
        radius_perturb_prob: float = 0.0,
        radius_perturb_min: float = 0.85,
        radius_perturb_max: float = 1.15,
    ):
        """
        Args:
            translation_scale: Scale of random translation in birth/persistence space
            noise_scale: Scale of Gaussian noise
            sigma_min: Minimum sigma for Gaussian kernel
            sigma_max: Maximum sigma for Gaussian kernel
            translation_prob: Probability of applying translation
            noise_prob: Probability of applying noise
            sigma_variation_prob: Probability of varying sigma
            persistence_scale_prob: Probability of applying persistence scaling
            persistence_scale_min: Minimum scaling factor for persistence
            persistence_scale_max: Maximum scaling factor for persistence
            radius_perturb_prob: Probability of applying radius perturbation
            radius_perturb_min: Minimum scaling factor for radius
            radius_perturb_max: Maximum scaling factor for radius
        """
        self.space_aug = PersistenceSpaceAugmentation(
            translation_scale=translation_scale,
            noise_scale=noise_scale,
            translation_prob=translation_prob,
            noise_prob=noise_prob,
            persistence_scale_prob=persistence_scale_prob,
            persistence_scale_min=persistence_scale_min,
            persistence_scale_max=persistence_scale_max,
            radius_perturb_prob=radius_perturb_prob,
            radius_perturb_min=radius_perturb_min,
            radius_perturb_max=radius_perturb_max,
        )
        self.sigma_aug = SigmaVariationAugmentation(
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            prob=sigma_variation_prob,
        )

    def augment_pairs_features(self, pairs_feats, global_bounds=None):
        """Augment pairs features in persistence space."""
        return self.space_aug.augment_pairs_features(pairs_feats, global_bounds)

    def sample_sigma(self, base_sigma: float = 16.0) -> float:
        """Sample a sigma value for image generation."""
        return self.sigma_aug.sample_sigma(base_sigma)


def get_default_augmentation(mode='train'):
    """
    Get default augmentation configuration.

    Args:
        mode: 'train' or 'test'

    Returns:
        CombinedPersistenceAugmentation instance
    """
    if mode == 'train':
        return CombinedPersistenceAugmentation(
            translation_scale=0.05,  # 5% of range
            noise_scale=0.02,         # 2% of range
            sigma_min=12.0,
            sigma_max=20.0,
            translation_prob=0.5,
            noise_prob=0.5,
            sigma_variation_prob=1.0,
        )
    else:
        # No augmentation for test
        return CombinedPersistenceAugmentation(
            translation_scale=0.0,
            noise_scale=0.0,
            sigma_min=16.0,
            sigma_max=16.0,
            translation_prob=0.0,
            noise_prob=0.0,
            sigma_variation_prob=0.0,
        )


# Example usage
if __name__ == "__main__":
    # Create augmentation
    aug = get_default_augmentation('train')

    # Example pairs features
    pairs_feats = [
        {'birth': 100.0, 'death': 50.0, 'persistence': 50.0, 'mean_radius': 2.5},
        {'birth': 150.0, 'death': 80.0, 'persistence': 70.0, 'mean_radius': 3.0},
        {'birth': 200.0, 'death': 120.0, 'persistence': 80.0, 'mean_radius': 2.8},
    ]

    # Augment
    global_bounds = (0, 300, 0, 100)
    augmented = aug.augment_pairs_features(pairs_feats, global_bounds)

    print("Original pairs:")
    for f in pairs_feats[:2]:
        print(f"  birth={f['birth']:.2f}, persistence={f['persistence']:.2f}")

    print("\nAugmented pairs:")
    for f in augmented[:2]:
        print(f"  birth={f['birth']:.2f}, persistence={f['persistence']:.2f}")

    # Sample sigma values
    print("\nSampled sigma values:")
    for _ in range(5):
        sigma = aug.sample_sigma()
        print(f"  sigma={sigma:.2f}")
