#!/usr/bin/env python3
"""
Gym observation wrapper - Thin wrapper around the ImageProcessor class defined in image_processor.py
Applies image processing pipeline to observations in Gym environments.
"""
# ============================================================================
# Import Required Libraries and Modules
# ============================================================================

import gym
import numpy as np
import sys
import os

# Add current directory and parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)

from image_processor import ImageProcessor

# ============================================================================
# Image Processing Wrapper Class
# ============================================================================

class ImageProcessingWrapper(gym.ObservationWrapper):
    """Gym observation wrapper, which applies image processing pipeline to observations in Gym environments."""

    def __init__(self, 
                 env,
                 roi_crop_top,
                 target_shape,
                 augmentation_probability
                 ):
        """
        Initialize the image processing wrapper
        
        Args:
            env: Base Gym environment to wrap
            roi_crop_top: Ratio of top image to crop
            target_shape: Target image dimensions (height, width)
            augmentation_probability: Probability of applying data augmentations
        """
        super().__init__(env)

        print("🔄🔄🔄🔄🔄 CALLFLOW: Entering image_processing_wrapper.py - Gym Observation Wrapper 🔄🔄🔄🔄🔄")

        # Initialize image processor instance
        self.image_processor = ImageProcessor(
            roi_crop_top=roi_crop_top,
            target_shape=target_shape,
            augmentation_probability=augmentation_probability
        )
        
        # Update observation space based on applied processing
        self._update_observation_space()

    def _update_observation_space(self):
        """Update observation space"""
        original_shape = self.env.observation_space.shape
        
        if len(original_shape) == 3 and original_shape[2] == 3: # RGB images - HWC format
            # Create dummy image with original dimensions
            dummy_image = np.zeros(original_shape, dtype=np.uint8)
            processed_dummy_image = self.image_processor.process_image(dummy_image)
            final_shape = processed_dummy_image.shape

            # Update observation space with processed dimensions
            self.observation_space = gym.spaces.Box(
                low=0, high=255,
                shape=final_shape,
                dtype=np.uint8
            )
        else:
            raise ValueError(f"❌ Unsupported observation shape: {original_shape}")

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Apply image processing pipeline to transform observation."""
        return self.image_processor.process_image(obs)
