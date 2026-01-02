#!/usr/bin/env python3
"""
Image processing pipeline. Applies the following processing steps on raw images:
    - Vertical cropping from the top
    - Shape resizing (downsizing)
    - Data augmentation using Albumentations transforms, only during model training:
        * Geometric: horizontal translation, rotation, elastic transform, grid distortion
        * Lighting: shadow casting
        * Blur: motion blur, lens distortion
        * Color: brightness/contrast, color jitter, color shift
        * Texture: Gaussian noise
    - Gaussian Blurring
    - Shadow/highlight correction
"""
# ============================================================================
# Import Required Modules and Libraries
# ============================================================================

import os
import sys
from datetime import datetime
import cv2
import numpy as np
import random
import albumentations as A

# Add current directory and parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)

from image_processing_config import get_image_processor_training_config

# ================================================================================
# Shadow/Highlight Correction Function (A copy from Ben's correction.py code file)
# ================================================================================

# Taken from https://gist.github.com/HViktorTsoi/8e8b0468a9fb07842669aa368382a7df

t = np.arange(256)

_lut_cache = dict()

def get_LUT(shadow_gain: float, highlight_gain: float):

    # Tone LUT
    key = f"{shadow_gain}, {highlight_gain}"

    if key not in _lut_cache:
        LUT_shadow = (1 - np.power(1 - t / 255, shadow_gain)) * 255
        LUT_shadow = np.maximum(0, np.minimum(255, np.int_(LUT_shadow + 0.5)))
        LUT_highlight = np.power(t / 255, highlight_gain) * 255
        LUT_highlight = np.maximum(0, np.minimum(255, np.int_(LUT_highlight + 0.5)))
        _lut_cache[key] = (LUT_shadow, LUT_highlight)
    return _lut_cache[key]


def correction(
    img: np.ndarray,
    shadow_amount_percent: float,
    shadow_tone_percent: float,
    shadow_radius: int,
    highlight_amount_percent: float,
    highlight_tone_percent: float,
    highlight_radius: int,
):
    """
    Image Shadow / Highlight Correction. The same function as in Photoshop / GIMP

    Args:
        img (np.ndarray): input RGB image numpy array of shape (height, width, 3)
        shadow_amount_percent (float)[0.0 ~ 1.0]: Controls (separately for the highlight and shadow values in the image) how much of a correction to make.
        shadow_tone_percent (float)[0.0 ~ 1.0]: Controls the range of tones in the shadows or highlights that are modified.
        shadow_radius (int)[>0]: Controls the size of the local neighborhood around each pixel
        highlight_amount_percent (float)[0.0 ~ 1.0]: Controls (separately for the highlight and shadow values in the image) how much of a correction to make.
        highlight_tone_percent (float)[0.0 ~ 1.0]: Controls the range of tones in the shadows or highlights that are modified.
        highlight_radius (int)[>0]: Controls the size of the local neighborhood around each pixel

    Returns:
        np.ndarray: colour corrected image
    """
    shadow_tone = (shadow_tone_percent + 1e-6) * 255
    highlight_tone = 255 - highlight_tone_percent * 255

    shadow_gain = 1 + shadow_amount_percent * 6
    highlight_gain = 1 + highlight_amount_percent * 6

    # extract height, width
    height, width = img.shape[:2]

    # The entire correction process is carried out in YUV space,
    # adjust highlights/shadows in Y space, and adjust colors in UV space
    # convert to Y channel (grey intensity) and UV channel (color)
    # img_YUV = cv2.cvtColor(img.astype("float32"), cv2.COLOR_BGR2YUV)
    img_YUV = cv2.cvtColor(img.astype("float32"), cv2.COLOR_RGB2YUV)
    img_Y, img_U, img_V = (img_YUV[..., x].reshape(-1) for x in range(3))

    # extract shadow / highlight
    shadow_map = 255 - (img_Y * 255) / shadow_tone
    shadow_map[np.where(img_Y >= shadow_tone)] = 0

    highlight_map = 255 - (255 * (255 - img_Y)) / (255 - highlight_tone)
    highlight_map[np.where(img_Y <= highlight_tone)] = 0

    # Gaussian blur on tone map, for smoother transition
    if shadow_amount_percent * shadow_radius > 0:
        shadow_map = cv2.GaussianBlur(
            shadow_map.reshape(height, width), (shadow_radius, shadow_radius), sigmaX=0
        ).reshape(-1)

    if highlight_amount_percent * highlight_radius > 0:
        highlight_map = cv2.GaussianBlur(
            highlight_map.reshape(height, width),
            (highlight_radius, highlight_radius),
            0,
        ).reshape(-1)

    # Tone LUT
    LUT_shadow, LUT_highlight = get_LUT(shadow_gain, highlight_gain)

    # adjust tone
    shadow_map /= 255
    highlight_map /= 255

    iH = (1 - shadow_map) * img_Y + shadow_map * LUT_shadow[np.int_(img_Y)]
    img_Y = (1 - highlight_map) * iH + highlight_map * LUT_highlight[np.int_(iH)]

    # re convert to RGB channel
    img_YUV = (
        np.row_stack([img_Y, img_U, img_V]).T.reshape(height, width, 3).astype("float32")
    )
    # output = cv2.cvtColor(img_YUV, cv2.COLOR_YUV2BGR)
    output = cv2.cvtColor(img_YUV, cv2.COLOR_YUV2RGB)
    output = np.maximum(0, np.minimum(output, 255)).astype(np.uint8)
    return output

# ============================================================================
# Image Processor Class
# ============================================================================

class ImageProcessor:
    """
    Accepts core image processing parameters: roi_crop_top, target_shape, augmentation_probability
    Other parameters are loaded from the centralized image_processing_config module.
    """

    def __init__(self, roi_crop_top, target_shape, augmentation_probability):
        """
        Initialize image processor instance.

        Args:
            roi_crop_top: Ratio of top image to crop
            target_shape: Image shape after resizing(height, width)
            augmentation_probability: Probability of applying data augmentations (0 during evaluation)
        """
        print("🔄🔄🔄🔄🔄 CALLFLOW: Entering image_processor.py - Image Processing Pipeline 🔄🔄🔄🔄🔄")

        # Load the image processing configuration
        config = get_image_processor_training_config() # Defaults to training config
        
        # Set all image processing hyperparameters from the loaded configuration
        for key, value in config.items():
            setattr(self, key, value)

        # Override the core image processing hyperparameters, in the case of passing different values
        self.roi_crop_top = roi_crop_top
        self.target_shape = target_shape
        self.augmentation_probability = augmentation_probability
        
        # Handle target shape validation similar to 'ResizeObservation' Gym wrapper
        if self.target_shape is not None:
            if isinstance(self.target_shape, int):
                self.target_shape = (self.target_shape, self.target_shape)
            assert all(x > 0 for x in self.target_shape), self.target_shape
            self.target_shape = tuple(self.target_shape)

        # Image saving Counter
        self.image_saving_counter = 0
        
        # Create image saving directory with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.image_saving_dir = f"saved_images/run_{timestamp}"
        
        # Create subdirectories for raw and processed images
        self.raw_dir = os.path.join(self.image_saving_dir, "raw_images")
        self.processed_dir = os.path.join(self.image_saving_dir, "processed_images")
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

        print(f"💾 Raw and processed images will be saved to {self.image_saving_dir} every {self.image_saving_freq} frames.")
        
        # Initialize Albumentations transform pipeline
        self._init_augmentation_pipeline()
    
    def _init_augmentation_pipeline(self):
        """Initialize Albumentations augmentation pipeline with all transforms"""
        transform_list = []

        # Motion Blur (simulates camera motion during driving)
        if self.motion_blur_prob > 0:
            transform_list.append(
                A.MotionBlur(
                    blur_limit=(self.motion_blur_limit_min, self.motion_blur_limit_max),
                    allow_shifted=True,
                    p=self.motion_blur_prob
                )
            )
        
        # Lens Distortion (Models barrel distortion from IMX219 wide-angle lens)
        if self.lens_distortion_prob > 0:
            transform_list.append(
                A.OpticalDistortion(
                    distort_limit=(self.lens_distortion_min, self.lens_distortion_max),
                    shift_limit=self.lens_distortion_shift,
                    border_mode=cv2.BORDER_REPLICATE,
                    p=self.lens_distortion_prob
                )
            )
        
        # Horizontal Translation (simulates lateral (left-right) camera shifts)
        if self.horizontal_translation_prob > 0:
            transform_list.append(
                A.ShiftScaleRotate(
                    shift_limit_x=(self.translation_x_min, self.translation_x_max),
                    shift_limit_y=0,
                    scale_limit=0,
                    rotate_limit=0,
                    border_mode=cv2.BORDER_REPLICATE,
                    p=self.horizontal_translation_prob
                )
            )
        
        # Rotation (simulates slight camera rotations during driving)
        if self.rotation_prob > 0:
            transform_list.append(
                A.Rotate(
                    limit=(self.rotation_angle_min, self.rotation_angle_max),
                    border_mode=cv2.BORDER_REPLICATE,
                    p=self.rotation_prob
                )
            )
        
        # Shadow Casting (simulates shadows from surrounding objects or car parts)
        if self.shadow_cast_prob > 0:
            transform_list.append(
                A.RandomShadow(
                    shadow_roi=self.shadow_roi,
                    num_shadows_lower=self.number_of_shadows_min,
                    num_shadows_upper=self.number_of_shadows_max,
                    shadow_dimension=self.shadow_dimension,
                    p=self.shadow_cast_prob
                )
            )
        
        # Color Augmentations - Use OneOf to prevent stacking multiple color augmentations
        # This ensures only one color augmentation applies at a time, avoiding extreme distortions
        color_augmentations = []
        
        # Brightness/Contrast Adjustment
        if self.brightness_contrast_prob > 0:
            color_augmentations.append(
                A.RandomBrightnessContrast(
                    brightness_limit=(self.brightness_min, self.brightness_max),
                    contrast_limit=(self.contrast_min, self.contrast_max),
                    p=1.0  # Probability handled by OneOf
                )
            )
        
        # Color Shift
        if self.color_shift_prob > 0:
            color_augmentations.append(
                A.RGBShift(
                    r_shift_limit=(self.color_shift_min, self.color_shift_max),
                    g_shift_limit=(self.color_shift_min, self.color_shift_max),
                    b_shift_limit=(self.color_shift_min, self.color_shift_max),
                    p=1.0  # Probability handled by OneOf
                )
            )
        
        # Color Jitter
        if self.color_jitter_prob > 0:
            color_augmentations.append(
                A.ColorJitter(
                    brightness=self.brightness,
                    contrast=self.contrast,
                    saturation=self.saturation,
                    hue=self.hue,
                    p=1.0  # Probability handled by OneOf
                )
            )
        
        # Apply one color augmentation with 40% probability
        if len(color_augmentations) > 0:
            transform_list.append(
                A.OneOf(color_augmentations, p=0.4)
            )
        
        # Gaussian Noise (simulates camera sensor noise)
        if self.noise_prob > 0:
            transform_list.append(
                A.GaussNoise(
                    var_limit=(self.noise_var_min, self.noise_var_max),
                    per_channel=True,
                    p=self.noise_prob
                )
            )
        
        # Elastic Transform (simulates track mat wrinkles)
        if self.elastic_transform_prob > 0:
            transform_list.append(
                A.ElasticTransform(
                    alpha=self.elastic_alpha,
                    sigma=self.elastic_sigma,
                    alpha_affine=self.elastic_alpha_affine,
                    approximate=True,
                    border_mode=cv2.BORDER_REPLICATE,
                    p=self.elastic_transform_prob
                )
            )
        
        # Grid Distortion (simulates surface irregularities)
        if self.grid_distortion_prob > 0:
            transform_list.append(
                A.GridDistortion(
                    num_steps=self.grid_num_steps,
                    distort_limit=self.grid_distort_limit,
                    border_mode=cv2.BORDER_REPLICATE,
                    p=self.grid_distortion_prob
                )
            )
        
        # Compose all Albumentations transforms in a single pipeline
        self.augmentation_transform = A.Compose(transform_list)
        
    def _apply_vertical_crop(self, image: np.ndarray) -> np.ndarray:
        """Apply vertical cropping to image"""
        image_height = image.shape[0] # HWC format
        cropped_pixels = int(image_height * self.roi_crop_top)
        cropped_image = image[cropped_pixels:]
        return cropped_image

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target shape (matches the functionality of 'ResizeObservation' Gym wrapper)"""
        if self.target_shape is None:
            return image
        
        resized_image = cv2.resize(image, self.target_shape[::-1], interpolation=cv2.INTER_AREA)
        return resized_image

    def _apply_augmentations(self, image: np.ndarray) -> np.ndarray:
        """Apply data augmentations"""
        
        if random.random() >= self.augmentation_probability:
            return image  # No augmentation applied
        
        # Apply Albumentations transform pipeline
        augmented = self.augmentation_transform(image=image)
        return augmented['image']

    def _apply_gaussian_blurring(self, image: np.ndarray) -> np.ndarray:
        """Apply gaussian blurring"""
        return cv2.GaussianBlur(
            image,
            (self.gaussian_kernel_size, self.gaussian_kernel_size),
            self.gaussian_sigma_x
        )

    def _apply_shadow_highlight_correction(self, image: np.ndarray) -> np.ndarray:
        """Apply shadow and highlight correction"""
        return correction(
            img=image,
            shadow_amount_percent=self.shadow_amount,
            shadow_tone_percent=self.shadow_tone,
            shadow_radius=self.shadow_radius,
            highlight_amount_percent=self.highlight_amount,
            highlight_tone_percent=self.highlight_tone,
            highlight_radius=self.highlight_radius
        )
    
    def _save_images(self, raw_image: np.ndarray, processed_image: np.ndarray):
        """Save raw and processed images to disk."""
        try:
            # Prepare image filenames
            raw_filename = f"raw_frame_{self.image_saving_counter:08d}.png"
            processed_filename = f"processed_frame_{self.image_saving_counter:08d}.png"
            
            raw_path = os.path.join(self.raw_dir, raw_filename)
            processed_path = os.path.join(self.processed_dir, processed_filename)
            
            # Convert from RGB to BGR for cv2.imwrite
            raw_bgr_image = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)
            processed_bgr_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
            
            # Save images
            cv2.imwrite(raw_path, raw_bgr_image)
            cv2.imwrite(processed_path, processed_bgr_image)
            
            print(f"📸 Saved raw and processed images at frame {self.image_saving_counter}")
            
        except Exception as e:
            print(f"⚠️ Error saving images at frame {self.image_saving_counter}: {e}")

    def process_image(self, image: np.ndarray) -> np.ndarray:
        """
        Process images through the image processing pipeline.
        Args:
            image: Raw input image (HWC format, RGB)
        
        Returns:
            Processed image after applying all processing steps
        """
        # Validate image
        if image is None:
            return image
        
        # Ensure image is in numpy array format
        image = np.asarray(image)

        # Ensure uint8 format
        if image.dtype != np.uint8:
            print(f"⚠️ Converting image from {image.dtype} to uint8")
            image = np.clip(image, 0, 255).astype(np.uint8)
            
        processed_image = image.copy()

        # 1. Vertical cropping
        if self.roi_crop_top > 0:
            processed_image = self._apply_vertical_crop(processed_image)

        # 2. Resizing
        if self.target_shape is not None:
            processed_image = self._resize_image(processed_image)

        # 3. Augmentation
        if self.augmentation_probability > 0:
            processed_image = self._apply_augmentations(processed_image)

        # 4. Gaussian blurring
        processed_image = self._apply_gaussian_blurring(processed_image)

        # 5. Shadow/highlight correction
        processed_image = self._apply_shadow_highlight_correction(processed_image)

        # 6. Save images (raw and processed):
        self.image_saving_counter += 1 # Increment image saving counter
        if self.image_saving_counter % self.image_saving_freq == 0:
            self._save_images(image, processed_image)

        return processed_image

