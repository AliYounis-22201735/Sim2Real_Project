#!/usr/bin/env python3
"""
RLPilot Class - Integrates into the DonkeyCar vehicle's driving loop
Enable support for PyTorch-trained DRL models
Includes the functionalities of model detection, loading, and inference logic.
"""
# ============================================================================
# Import Required Libraries and Modules
# ============================================================================

import os
import sys
import time
import numpy as np
from typing import Tuple
from collections import deque
from stable_baselines3 import PPO

# Add current directory and parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)

from model_detector import detect_model
from image_processor import ImageProcessor
from image_processing_config import get_image_processor_evaluation_config, set_training_mode, TSNE_SAMPLE_RATE

# ============================================================================
# RLPilot Class - Model Detection, Loading, and Inference
# ============================================================================

class RLPilot:
    """
    RLPilot class, includes functionalities of model detection, loading, and inference logic, besides, collecting data for t-SNE analysis.
    """

    def __init__(self, model_path: str):
        """
        Initialize RLPilot instance.
        
        Args:
            model_path (str): Path to the saved PyTorch-trained model
        """
        print("🔄🔄🔄🔄🔄 CALLFLOW: Entering rl_pilot.py - DonkeyCar's DRL Pilot Part 🔄🔄🔄🔄🔄")

        # Set evaluation mode (training mode off)
        set_training_mode(training=False)

        # Initialize instance variables
        self.model_path = model_path
        self.model = None
        self.model_type = None
        self.obs_shape = None
        self.stack_size = None
        
        # Extract model filename for logging
        self.model_filename = os.path.splitext(os.path.basename(model_path))[0]
        self.inference_log_file = f"{self.model_filename}_realworld_inference.txt"
        
        # Frame stacking control
        self.vector_frame_stack = None
        self.image_frame_stack = None

        # Initialize image vectorization function
        self.vectorize_image = None
        
        # t-SNE data collection
        self.tsne_sample_rate = TSNE_SAMPLE_RATE
        self.tsne_collected_samples = []
        self.tsne_counter = 0
        self.tsne_data_file = f"{self.model_filename}_realworld_tsne_data.npz"
        self.tsne_file_saving_interval = 100  # Save t-SNE data every 100 collected samples
        
        # Episode step tracking
        self.episode_step_counter = 0

        # Initialize image processor instance
        image_processor_config = get_image_processor_evaluation_config()
        self.image_processor = ImageProcessor(
            roi_crop_top=image_processor_config['roi_crop_top'],
            target_shape=image_processor_config['target_shape'],
            augmentation_probability=image_processor_config['augmentation_probability']
        )

        # Load the Pytorch-trained model
        self._load_model()
    
    def _load_model(self):
        """Load the pretrained model and auto-detect its type and configuration"""

        # First, check if the model file exists
        if not os.path.exists(self.model_path):
            print(f"❌ Model file not found: {self.model_path}")
            return
        
        # Load the model and detect its type and configuration
        try:
            self.model = PPO.load(self.model_path)
            self.model_type, self.obs_shape, self.stack_size = detect_model(self.model_path)
        except Exception as e:
            print(f"❌ Model loading and detection failed: {e}")
            self.model, self.model_type, self.obs_shape, self.stack_size = None, None, None, None
            return

        # Lazily load image vectorization function (only in case of vector models)
        if self.model_type == "vector":
            from image_vectorizer import vectorize_image
            self.vectorize_image = vectorize_image
        
        # Initialize frame stacking if detected
        if self.model_type == "vector" and self.stack_size > 1:
            self.vector_frame_stack = deque(maxlen=self.stack_size)
        elif self.model_type == "image" and self.stack_size > 1:
            self.image_frame_stack = deque(maxlen=self.stack_size)

    def _prepare_vector_observation(self, img_arr: np.ndarray) -> np.ndarray:
        """Prepare vector observation for model inference."""
        # Convert raw images to feature vectors
        feature_vector = self.vectorize_image(img_arr)

        # If model was trained on stacked frames, then use deque for vector frame stacking
        if self.vector_frame_stack is not None:
            # Add feature vector to stack
            self.vector_frame_stack.append(feature_vector.copy())
            # If not enough frames yet (the case of first frame), then fill the stack with the first frame
            if len(self.vector_frame_stack) < self.stack_size:
                while len(self.vector_frame_stack) < self.stack_size:
                    self.vector_frame_stack.append(feature_vector.copy()) 
            # Stack vector frames by concatenation
            obs = np.concatenate(list(self.vector_frame_stack))
        else:
            # Single vector observation in case of no stacking
            obs = feature_vector
        
        # Add batch dimension: (N,) -> (1, N)
        obs = np.expand_dims(obs, axis=0)
        
        return obs

    def _prepare_image_observation(self, img_arr: np.ndarray) -> np.ndarray:
        """Prepare image observation for model inference."""

        # Process raw images using ImageProcessor instance
        processed_img = self.image_processor.process_image(img_arr)
        processed_img = np.transpose(processed_img, (2, 0, 1)) # HWC to CHW conversion

        # If model was trained on stacked frames, use deque for image frame stacking
        if self.image_frame_stack is not None:
            # Add processed image frame to stack
            self.image_frame_stack.append(processed_img.copy())
            # If not enough frames yet (the case of the first image), then fill with current frame
            if len(self.image_frame_stack) < self.stack_size:
                while len(self.image_frame_stack) < self.stack_size:
                    self.image_frame_stack.append(processed_img.copy())
            
            # Stack image frames along channel dimension
            obs = np.concatenate(list(self.image_frame_stack), axis=0)
        else:
            # Single image observation in case of no stacking
            obs = processed_img
        
        # Add batch dimension: (C, H, W) -> (1, C, H, W)
        obs = np.expand_dims(obs, axis=0)
        
        return obs

    def run(self, img_arr: np.ndarray) -> Tuple[float, float, float]:
        """
        Steering and throttle actions inference from images perceived by jetson nano camera.

        Args:
            img_arr: Input image array (H, W, C) – Raw image from jetson nano camera

        Returns:
            tuple: (steering, throttle, inference_time_ms)
        """        
        # Check if model loaded successfully
        if self.model is None:
            print("❌ Model not loaded. Cannot perform inference.")
            return None, None, None
        
        # Increment tracking counters
        self.tsne_counter += 1
        self.episode_step_counter += 1
        
        start_time = time.time()
        
        try:
            if self.model_type == "vector":
                obs = self._prepare_vector_observation(img_arr)
            elif self.model_type == "image":
                obs = self._prepare_image_observation(img_arr)
            else:
                print(f"❌ Unknown model type: {self.model_type}")
                return None, None, None # Indicating inference failure
            
            # Run inference
            action, _ = self.model.predict(obs, deterministic=True)

            # Collect t-SNE samples
            if self.tsne_counter % self.tsne_sample_rate == 0:
                self.tsne_collected_samples.append(obs.flatten().copy())
                    
                # Periodic save to prevent data loss in case Jetson Nano crash
                if len(self.tsne_collected_samples) % self.tsne_file_saving_interval == 0:
                    self.save_tsne_data()
                
            # Extract steering and throttle actions
            # Handle both (2,) and (1, 2) shapes
            if action.ndim == 1:
                steering, throttle = float(action[0]), float(action[1])
            else:
                steering, throttle = float(action[0, 0]), float(action[0, 1])
                
            inference_time_ms = (time.time() - start_time) * 1000 # Inference time in milliseconds
                
            # Log observation and action details to the log file
            with open(self.inference_log_file, "a") as f:
                f.write(f"[INFERENCE FRAME {self.episode_step_counter}] Observation Type:{type(obs).__name__}, Observation Dtype: {obs.dtype}, Observation Shape: {obs.shape}, Steering: {steering:.3f}, Throttle: {throttle:.3f}, Inference Time: {inference_time_ms:.3f} ms\n")

            print(f"Inference Frame {self.episode_step_counter}: Steering={steering:.3f}, Throttle={throttle:.3f}, Inference Time={inference_time_ms:.3f} ms")

            return steering, throttle, inference_time_ms

        except Exception as e:
            print(f"❌ Inference error: {e}")
            return None, None, None
        
    def reset(self):
        """Reset frame stacks for the current episode"""
        if self.vector_frame_stack is not None:
            self.vector_frame_stack.clear()
        if self.image_frame_stack is not None:
            self.image_frame_stack.clear()
        
        # Reset inference counter
        self.episode_step_counter = 0

    def save_tsne_data(self):
        """Save collected t-SNE samples to disk."""
        if self.tsne_collected_samples:
            try:
                np.savez(
                    self.tsne_data_file,
                    samples=np.array(self.tsne_collected_samples)
                )
                print(f"💾 Saved {len(self.tsne_collected_samples)} sample observations for t-SNE analysis to: {self.tsne_data_file}")
                return self.tsne_data_file
            except Exception as e:
                print(f"❌ Error saving t-SNE data: {e}")
                return None
        else:
            print("⚠️  No t-SNE samples collected yet")
            return None
    
    def shutdown(self):
        """Clean up resources."""
        # Final save of t-SNE data before shutdown
        print("🔄 Performing final t-SNE data save...")
        self.save_tsne_data()
        
        # Cleanup vectorizer drawer manager if using vector model
        if self.model_type == "vector" and self.vectorize_image is not None:
            try:
                from image_vectorizer import cleanup_drawer
                cleanup_drawer()
                print("🧹 Cleaned up vectorizer drawer manager")
            except Exception as e:
                print(f"⚠️  Error cleaning up drawer: {e}")
        
        # Clear instance variables
        self.model = None
        self.vectorize_image = None
        self.vector_frame_stack = None
        self.image_frame_stack = None
        self.image_processor = None
        self.tsne_collected_samples = None
        print("🛑 RLPilot shutdown complete")