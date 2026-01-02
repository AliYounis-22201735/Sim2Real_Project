#!/usr/bin/env python3
"""
Model detection module - detects model type (image vs. vector) and configuration (observation shape and stack size)
Employed by RLPilot class and performance evaluation scripts to load and detect models
"""
# ============================================================================
# Import Libraries
# ============================================================================

import numpy as np
from typing import Tuple
from stable_baselines3 import PPO

# ============================================================================
# Utility Functions
# ============================================================================

def _calculate_vector_base_dim():
    """Calculate base dimension of vector representations (in case of vector models)"""
    try:    
        from image_vectorizer import vectorize_image # Only import here to avoid unnecessary dependency in case of image models
        # Create a dummy test image and use it to get the vector base dimension
        dummy_img = np.zeros((120, 160, 3), dtype=np.uint8)  # Donkeycar standard image size (height=120, width=160, channels=3) in both the simulator and Jetson Nano
        dummy_vector = vectorize_image(dummy_img)
        return dummy_vector.shape[0]
    except Exception as e:
        print(f"⚠️ Could not detect vector base dimension: {e}")
        return None

# ============================================================================
# Model Detection Function
# ============================================================================

def detect_model(model_path: str) -> Tuple[str, Tuple, int]:
    """
    Auto-detect model type and configuration.
    
    Args:
        model_path (str): Path to saved trained model

    Returns:
        tuple: (model_type, obs_shape, stack_size)
    """
    print("🔄🔄🔄🔄🔄 CALLFLOW: Entering model_detector.py - Model Detection 🔄🔄🔄🔄🔄")

    try:
        # Load PPO model for detection
        model = PPO.load(model_path)
        obs_shape = model.observation_space.shape

        # Model detection logic
        if len(obs_shape) == 1: # Vector-based models - 1D observation space
            # Get vector base dimension
            vector_base_dim = _calculate_vector_base_dim()
            # Calculate stack size (if configured) based on vector base dimension
            stack_size = obs_shape[0] // vector_base_dim
            # Print detailed information about the detected model
            print(f"🔢 PPO vector model detected:")
            print(f"    ℹ️   Model observation dimension: {obs_shape[0]}")
            print(f"    ℹ️   Vector base dimension: {vector_base_dim}")
            print(f"    ℹ️   Stack size: {stack_size}")
            return "vector", obs_shape, stack_size

        elif len(obs_shape) == 3: # Image-based model - 3D observation space
            channels, _, _ = obs_shape # Since Pytorch models are trained on CHW image format   
            # Calculate stack size (if configured)
            stack_size = channels // 3 # 3 channels for RGB images
            # Print detailed information about the detected model
            print(f"📸 PPO image model detected:")
            print(f"    ℹ️   Model observation shape: {obs_shape}")
            print(f"    ℹ️   Stack size: {stack_size}")
            return "image", obs_shape, stack_size

        else:
            print(f"⚠️ Unknown observation space shape: {obs_shape}")
            print(f"    Expected 1D (vector) or 3D (image) observation space")
            return "unknown", obs_shape, None
            
    except Exception as e:
        print(f"❌ Model detection failed: {e}")
        return "unknown", None, None