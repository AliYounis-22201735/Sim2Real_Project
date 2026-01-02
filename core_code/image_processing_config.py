#!/usr/bin/env python3
"""
Centralized Image Processing Configuration - Single source of truth for all image processing parameters.
Used by training, evaluation, and image vectorization modules and scripts.
"""
# ============================================================================
# Import Required Modules and Libraries
# ============================================================================

from typing import Dict, Any

# ============================================================================
# Training Mode Management
# ============================================================================

# Global training mode - select between training and evaluation modes
training_mode = True # Default to True

def set_training_mode(training: bool) -> None:
    """
    Select between training and evaluation modes.

    Args:
        training: Boolean, True for training, False for evaluation
    """
    global training_mode
    training_mode = bool(training)

# ============================================================================
# Core Image Processing Parameters
# ============================================================================

# ROI cropping and resizing shape
ROI_CROP_TOP = 0.25 # Ratio of top image to crop
TARGET_SHAPE = (84, 84) # (height, width)

# Training vs Evaluation augmentation control
AUGMENTATION_PROBABILITY_TRAINING = 1.0  # Used during training
AUGMENTATION_PROBABILITY_EVALUATION = 0.0  # Used during evaluation, where no augmentation is applied

# Image saving parameters
IMAGE_SAVING_FREQ = 1000  # Save images every 1000 frames (during training or evaluation)

# ============================================================================
# t-SNE Analysis Parameters
# ============================================================================

# t-SNE sampling rate
TSNE_SAMPLE_RATE = 10  # Sample every 10 frames for t-SNE analysis

# ============================================================================
# Augmentation Probabilities
# ============================================================================

BRIGHTNESS_CONTRAST_PROB = 0.4
COLOR_JITTER_PROB = 0.25
COLOR_SHIFT_PROB = 0.25
ELASTIC_TRANSFORM_PROB = 0.2
GRID_DISTORTION_PROB = 0.2
HORIZONTAL_TRANSLATION_PROB = 0.3
LENS_DISTORTION_PROB = 0.15
MOTION_BLUR_PROB = 0.4
NOISE_PROB = 0.3
ROTATION_PROB = 0.2
SHADOW_CAST_PROB = 0.1

# ============================================================================
# Augmentation Hyperparameters
# ============================================================================

# Brightness and contrast settings
BRIGHTNESS_MIN = -0.15
BRIGHTNESS_MAX = 0.15
CONTRAST_MIN = -0.1
CONTRAST_MAX = 0.1


# Color jitter settings
BRIGHTNESS = 0.2
CONTRAST = 0.2
SATURATION = 0.2
HUE = 0.0

# Color shift settings
COLOR_SHIFT_MIN = -10
COLOR_SHIFT_MAX = 10

# Elastic transform settings (simulates track mat wrinkles)
ELASTIC_ALPHA = 5
ELASTIC_SIGMA = 3
ELASTIC_ALPHA_AFFINE = 0

# Grid distortion settings (simulates surface irregularities)
GRID_NUM_STEPS = 5
GRID_DISTORT_LIMIT = 0.10

# Horizontal translation settings
TRANSLATION_X_MIN = -0.08
TRANSLATION_X_MAX = 0.08

# Lens distortion settings (simulates barrel distortion from IMX219)
LENS_DISTORTION_MIN = -0.05
LENS_DISTORTION_MAX = 0.05
LENS_DISTORTION_SHIFT = 0.01

# Motion blur settings (simulates camera motion during driving)
MOTION_BLUR_LIMIT_MIN = 3            # Minimum blur kernel size
MOTION_BLUR_LIMIT_MAX = 9            # Maximum blur kernel size

# Gaussian Noise settings
NOISE_VAR_MIN = 5
NOISE_VAR_MAX = 25

# Rotation settings
ROTATION_ANGLE_MIN = -5
ROTATION_ANGLE_MAX = 5

# Shadow casting settings
SHADOW_ROI = (0, 0.7, 1, 1)
NUMBER_OF_SHADOWS_MIN = 1
NUMBER_OF_SHADOWS_MAX = 2
SHADOW_DIMENSION = 4

# ============================================================================
# Gaussian Blurring and Shadow/Highlight Correction Hyperparameters
# ============================================================================

# Gaussian blur
GAUSSIAN_KERNEL_SIZE = 3
GAUSSIAN_SIGMA_X = 0.8

# Shadow/highlight correction
SHADOW_AMOUNT = 0.4
SHADOW_TONE = 0.4
SHADOW_RADIUS = 5
HIGHLIGHT_AMOUNT = 0.2
HIGHLIGHT_TONE = 0.2
HIGHLIGHT_RADIUS = 5

# ============================================================================
# Image Vectorization Hyperparameters
# ============================================================================

# Steering edge detection hyperparameters
STEERING_CONFIG: Dict[str, Any] = {
    "image_processing_args": (ROI_CROP_TOP, TARGET_SHAPE, AUGMENTATION_PROBABILITY_TRAINING), # Defaults to training mode, subsequently training augmentation prob is set
    "num_lines": 19,
    "radius_from": 15,
    "skip_lines": 1,
    "edge_strategy": "roll",
    "distance": 1.5,
    "sqr_derivs": True,
    "use_global": False,
    "basis": "channels", 
    "roll_window": 3,
    "scale_roll": True,
    "combine_lines": "max",
}

# Throttle edge detection hyperparameters
THROTTLE_CONFIG: Dict[str, Any] = {
    "image_processing_args": (ROI_CROP_TOP, TARGET_SHAPE, AUGMENTATION_PROBABILITY_TRAINING), # Default to training mode
    "num_lines": 14,
    "radius_from": 15,
    "skip_lines": 1,
    "edge_strategy": "roll",
    "distance": 1.5,
    "sqr_derivs": True,
    "use_global": False,
    "basis": "channels",
    "roll_window": 3,
    "scale_roll": True,
    "combine_lines": "max",
}

# ============================================================================
# Configuration Retrieval Utility Functions
# ============================================================================

def _get_base_image_processor_config():
    """Helper function to get the base configuration of image processor"""
    return {
        'roi_crop_top': ROI_CROP_TOP,
        'target_shape': TARGET_SHAPE,
        'color_jitter_prob': COLOR_JITTER_PROB,
        'noise_prob': NOISE_PROB,
        'horizontal_translation_prob': HORIZONTAL_TRANSLATION_PROB,
        'shadow_cast_prob': SHADOW_CAST_PROB,
        'brightness_contrast_prob': BRIGHTNESS_CONTRAST_PROB,
        'rotation_prob': ROTATION_PROB,
        'color_shift_prob': COLOR_SHIFT_PROB,
        'brightness': BRIGHTNESS,
        'contrast': CONTRAST,
        'saturation': SATURATION,
        'hue': HUE,
        'noise_var_min': NOISE_VAR_MIN,
        'noise_var_max': NOISE_VAR_MAX,
        'translation_x_min': TRANSLATION_X_MIN,
        'translation_x_max': TRANSLATION_X_MAX,
        'brightness_min': BRIGHTNESS_MIN,
        'brightness_max': BRIGHTNESS_MAX,
        'contrast_min': CONTRAST_MIN,
        'contrast_max': CONTRAST_MAX,
        'rotation_angle_min': ROTATION_ANGLE_MIN,
        'rotation_angle_max': ROTATION_ANGLE_MAX,
        'color_shift_min': COLOR_SHIFT_MIN,
        'color_shift_max': COLOR_SHIFT_MAX,
        'motion_blur_prob': MOTION_BLUR_PROB,
        'motion_blur_limit_min': MOTION_BLUR_LIMIT_MIN,
        'motion_blur_limit_max': MOTION_BLUR_LIMIT_MAX,
        'lens_distortion_prob': LENS_DISTORTION_PROB,
        'lens_distortion_min': LENS_DISTORTION_MIN,
        'lens_distortion_max': LENS_DISTORTION_MAX,
        'lens_distortion_shift': LENS_DISTORTION_SHIFT,
        'gaussian_kernel_size': GAUSSIAN_KERNEL_SIZE,
        'gaussian_sigma_x': GAUSSIAN_SIGMA_X,
        'shadow_amount': SHADOW_AMOUNT,
        'shadow_tone': SHADOW_TONE,
        'shadow_radius': SHADOW_RADIUS,
        'highlight_amount': HIGHLIGHT_AMOUNT,
        'highlight_tone': HIGHLIGHT_TONE,
        'highlight_radius': HIGHLIGHT_RADIUS,
        'image_saving_freq': IMAGE_SAVING_FREQ,
        'shadow_roi': SHADOW_ROI,
        'number_of_shadows_min': NUMBER_OF_SHADOWS_MIN,
        'number_of_shadows_max': NUMBER_OF_SHADOWS_MAX,
        'shadow_dimension': SHADOW_DIMENSION,
        'elastic_transform_prob': ELASTIC_TRANSFORM_PROB,
        'elastic_alpha': ELASTIC_ALPHA,
        'elastic_sigma': ELASTIC_SIGMA,
        'elastic_alpha_affine': ELASTIC_ALPHA_AFFINE,
        'grid_distortion_prob': GRID_DISTORTION_PROB,
        'grid_num_steps': GRID_NUM_STEPS,
        'grid_distort_limit': GRID_DISTORT_LIMIT,
    }

def get_image_processor_training_config():
    """Get image processor configuration for training mode (with data augmentation)"""
    training_config = _get_base_image_processor_config()
    training_config['augmentation_probability'] = AUGMENTATION_PROBABILITY_TRAINING
    return training_config

def get_image_processor_evaluation_config():
    """Get image processor configuration for evaluation mode (no data augmentation)"""
    eval_config = _get_base_image_processor_config()
    eval_config['augmentation_probability'] = AUGMENTATION_PROBABILITY_EVALUATION
    return eval_config

def get_image_vectorizer_config(training=None):
    """
    Get steering and throttle configurations for image vectorization pipeline.
    
    Args:
        training: Optional; True for training, False for evaluation. If None, uses current session mode.
    """
    if training is None:
        training = training_mode # Global training mode

    augmentation_prob = AUGMENTATION_PROBABILITY_TRAINING if training else AUGMENTATION_PROBABILITY_EVALUATION
    
    steering_config = STEERING_CONFIG.copy()
    throttle_config = THROTTLE_CONFIG.copy()
    
    # Update augmentation probability based on the selected mode
    steering_config["image_processing_args"] = (ROI_CROP_TOP, TARGET_SHAPE, augmentation_prob)
    throttle_config["image_processing_args"] = (ROI_CROP_TOP, TARGET_SHAPE, augmentation_prob)
    
    return steering_config, throttle_config
