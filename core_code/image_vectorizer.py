#!/usr/bin/env python3
"""
Image vectorization pipeline for DonkeyCar vector models - convert raw camera images into feature vector representations (steering & throttle edges and additional features).
"""
# ============================================================================
# Import Required Libraries and Modules
# ============================================================================

import os
import sys
import cv2
import numpy as np
import gym
from gym import spaces
from datetime import datetime

# Add current directory and parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)

from image_processing_config import get_image_vectorizer_config, IMAGE_SAVING_FREQ  # Centralized image processing config
from edge_detection_framework.line_drawer import LineDrawerParams, LineDrawer, ParallelLineDrawer

# ============================================================================
# Utility Functions
# ============================================================================

def max_distance_from_bottom_mid_point(img_width, img_height):
    """
    Calculate the maximum possible distance from bottom-mid (source) point to any point in the image - used for normalizing distance features.

    Args:
        img_width: Image width
        img_height: Image height
        
    Returns:
        float: Maximum possible distance from bottom-mid (source) point
    """
    bottom_mid_x, bottom_mid_y = img_width // 2, img_height - 1
    image_corners = [(0, 0), (img_width - 1, 0), (0, img_height - 1), (img_width - 1, img_height - 1)]
    return max(
        np.sqrt((bottom_mid_x - corner_x)**2 + (bottom_mid_y - corner_y)**2)
        for corner_x, corner_y in image_corners
    )

def normalize_edges(edges, img_width, img_height):
    """
    Normalize returned (x, y, distance) tuples by image dimensions.

    Args:
        edges: List of (x, y, distance) tuples # (pixel coordinates and distance from the source point)
        img_width: Image width
        img_height: Image height

    Returns:
        np.ndarray: Normalized edges in the form: [norm_x, norm_y, norm_dist, ...]
    """
    # Handle the edge case of empty edges returned
    if len(edges) == 0:
        return np.array([], dtype=np.float32) # Return empty array
    
    normalized_edges = [] # Initialize a list to hold normalized values
    max_dist_from_bottom_mid_point = max_distance_from_bottom_mid_point(img_width, img_height)
    
    for x, y, dist in edges:
        norm_x = x / (img_width - 1)    # Normalize x coordinates to [0, 1] range
        norm_y = y / (img_height - 1)  # Normalize y coordinates to [0, 1] range
        # Normalize distances
        norm_dist = dist / max_dist_from_bottom_mid_point
        normalized_edges.extend([norm_x, norm_y, norm_dist]) # Append normalized values to list

    return np.array(normalized_edges, dtype=np.float32)


def identical_configs(steering_config: LineDrawerParams, throttle_config: LineDrawerParams) -> bool:
    """Check if the two drawer configurations are identical"""
    return steering_config.keys() == throttle_config.keys() and all(steering_config.get(key) == throttle_config.get(key) for key in steering_config.keys())

def _get_target_shape():
    """
    Derive target image shape from the imported image vectorizer configuration.

    Returns:
        Tuple[int, int]: (img_height, img_width)
    """
    steering_config, _ = get_image_vectorizer_config()
    _, target_shape, _ = steering_config["image_processing_args"]
    return target_shape

# ============================================================================
# Additional Features
# ============================================================================

def calculate_additional_features(steering_edges=None, throttle_edges=None, img_width=None, img_height=None):
    """
    Derive additional features from the detected edges.
    
    Args:
        steering_edges: List of (x, y, distance) tuples for steering edges
        throttle_edges: List of (x, y, distance) tuples for throttle edges
        img_width: Image width
        img_height: Image height

    Returns:
        np.ndarray: Combined steering and throttle additional features [steering_features + throttle_features]
        Seven features in total:
        - Steering balance factor
        - Steering symmetry score  
        - Steering x positions dispersion
        - Throttle y positions average
        - Throttle y positions dispersion 
        - Throttle distances average
        - Throttle distances dispersion 
    """

    ####################################
    ### Additional Steering Features ###
    ####################################   
    if steering_edges is None or len(steering_edges) == 0:
        # No steering edges detected - return nan to indicate missing data
        print("⚠️  Warning: No steering edges detected")
        steering_features = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    else:
        center_x =  img_width // 2 # Source point x-coordinate
        steering_x_positions = np.array([x for x, _, _ in steering_edges], dtype=np.float32)
        
        # Retrieve left snd right edges, with respect to the source point
        left_edges = np.any(steering_x_positions < center_x)
        right_edges = np.any(steering_x_positions > center_x)

        if left_edges and right_edges:
            # Edges are spread on both sides of center_x
            # Calculate balance factor based on edges' spatial positions
            left_edges = steering_x_positions[steering_x_positions < center_x] # Edges to the left of the source point
            right_edges = steering_x_positions[steering_x_positions > center_x] # Edges to the right of the source point
            left_centroid = np.mean(left_edges)
            right_centroid = np.mean(right_edges)
            
            # Calculate centroid distances from center_x
            left_centroid_distance = center_x - left_centroid
            right_centroid_distance = right_centroid - center_x
            
            # Calculate balance factor, normalized by center_x
            balance_factor = (right_centroid_distance - left_centroid_distance) / center_x # range [-1, 1]
        else:
            # All edges on one side of the source point
            # Calculate balance factor based on how far from the source point the edges are
            overall_centroid = np.mean(steering_x_positions)
            distance_from_center = overall_centroid - center_x
            balance_factor = distance_from_center / center_x

        # Symmetry score: higher values indicate better left-right balance
        symmetry_score = 1.0 - abs(balance_factor)

        # Steering edges dispersion
        steering_x_positions_dispersion = np.std(steering_x_positions) / (img_width - 1)

        # Combined steering features
        steering_features = np.array([balance_factor, symmetry_score, steering_x_positions_dispersion], dtype=np.float32)

    ####################################
    ### Additional Throttle Features ###
    ####################################

    if throttle_edges is None or len(throttle_edges) == 0:
        # No throttle edges - use nan to indicate missing data
        print("⚠️  Warning: No throttle edges detected")
        throttle_features = np.array([np.nan, np.nan, np.nan, np.nan], dtype=np.float32)
    else:
        throttle_y_positions = np.array([y for _, y, _ in throttle_edges], dtype=np.float32)
        throttle_distances = np.array([d for _, _, d in throttle_edges], dtype=np.float32)
        
        # Y-position features
        throttle_y_average = np.mean(throttle_y_positions) / (img_height - 1)
        throttle_y_dispersion = np.std(throttle_y_positions) / (img_height - 1)
        
        # Distance features
        max_possible_distance = max_distance_from_bottom_mid_point(img_width, img_height)
        throttle_distances_average = np.mean(throttle_distances) / max_possible_distance
        throttle_distances_dispersion = np.std(throttle_distances) / max_possible_distance

        # Combined throttle features
        throttle_features = np.array([throttle_y_average, throttle_y_dispersion, throttle_distances_average, throttle_distances_dispersion], dtype=np.float32)

    # Concatenate steering and throttle additional features
    additional_features = np.concatenate([steering_features, throttle_features])
    
    return additional_features

# ============================================================================
# Line Drawer Manager
# ============================================================================

class LineDrawerManager:
    """
    Manages line drawer instances based on steering and throttle configurations.:
    - Uses single line drawer if steering & throttle configs are identical
    - Uses parallel drawers when steering & throttle configs differ (with a fallback possibility to sequential drawer)
    """
    def __init__(self, steering_config: LineDrawerParams, throttle_config: LineDrawerParams):
        self.steering_config = steering_config
        self.throttle_config = throttle_config
        self._drawer = None
        self._configs_identical = identical_configs(steering_config, throttle_config)

    def get_drawer(self):
        """Get the cached line drawer instance or initialize it lazily in case not available"""
        if self._drawer is None:
            self._initialize_drawer()
        return self._drawer
    
    def _initialize_drawer(self):
        """Select and initialize the most appropriate drawer type based on the steering & throttle configurations"""
        if self._configs_identical:
            # Configs are the same --> Use single drawer, duplicate results
            self._drawer = LineDrawer(drawer_1_kwargs=self.steering_config)
            print("✅   Identical steering/throttle configurations - LineDrawer initialized successfully")
        else:
            # Configs differ --> Use parallel drawers
            try:
                self._drawer = ParallelLineDrawer(
                    steering_kwargs=self.steering_config,
                    throttle_kwargs=self.throttle_config
                )
                print("✅   Different steering/throttle configurations - ParallelLineDrawer initialized successfully")
            except Exception as e:
                # Fallback to sequential drawer
                print(f"⚠️  ParallelLineDrawer initialization failed, falling back to sequential LineDrawer: {e}")
                self._drawer = LineDrawer(
                    drawer_1_kwargs=self.steering_config,
                    drawer_2_kwargs=self.throttle_config
                )
                print("✅   Sequential LineDrawer initialized successfully")
    
    def cleanup(self):
        """Clean up drawer resources."""
        if self._drawer and hasattr(self._drawer, 'cleanup'):
            self._drawer.cleanup()
        self._drawer = None

# ============================================================================
# Drawer Manager Instance Handling
# ============================================================================

# Lazy initialization of the drawer manager
_drawer_manager = None

def _get_drawer_manager() -> LineDrawerManager:
    """
    Get cached drawer manager instance or initialize it lazily in case needed.
    
    Returns:
        LineDrawerManager: Cached or newly created drawer manager instance
    """
    global _drawer_manager
    
    if _drawer_manager is None:
        # Create drawer manager
        steering_config, throttle_config = get_image_vectorizer_config()
        _drawer_manager = LineDrawerManager(steering_config, throttle_config)   
    return _drawer_manager

def cleanup_drawer():
    """Reset cached drawer manager."""
    global _drawer_manager
    
    if _drawer_manager:
        _drawer_manager.cleanup()
        _drawer_manager = None

# ============================================================================
# Image Vectorization Function
# ============================================================================

# Image saving parameters
_image_saving_counter = 0
_image_saving_dir = None

def vectorize_image(img: np.ndarray) -> np.ndarray:
    """
    Convert camera images to feature vectors
    
    Args:
        img: Image array # HWC Format
        
    Returns:
        Combined feature vector representation [steering edges + throttle edges + additional features] as 1D array
    """
    global _image_saving_counter, _image_saving_dir

    # Get drawer manager
    drawer_manager = _get_drawer_manager()
    drawer = drawer_manager.get_drawer()
    configs_identical = drawer_manager._configs_identical

    # Get image dimensions for normalization (from config)
    img_height, img_width = _get_target_shape()
    
    if configs_identical:
        # Same config for both steering and throttle --> run single drawer
        steering_edges, _ = drawer.get_edges(img)
        processed_img = drawer.get_processed_image()  # Retrieve the already-processed image
        steering_vec = normalize_edges(steering_edges, img_width, img_height)
        additional_features = calculate_additional_features(steering_edges, img_width=img_width, img_height=img_height)
        feature_vector = np.concatenate([steering_vec, additional_features])
        # feature_vector = steering_vec
        throttle_edges = None  # For image saving conditioning
    else:
        # Different configs --> process both steering and throttle edges
        steering_edges, throttle_edges = drawer.get_edges(img)
        processed_img = drawer.get_processed_image()  # Retrieve the already-processed image
        steering_vec = normalize_edges(steering_edges, img_width, img_height)
        throttle_vec = normalize_edges(throttle_edges, img_width, img_height)
        additional_features = calculate_additional_features(steering_edges, throttle_edges, img_width=img_width, img_height=img_height)
        feature_vector = np.concatenate([steering_vec, throttle_vec, additional_features])
        # feature_vector = np.concatenate([steering_vec, throttle_vec])
    
    # Save images with edges (if enabled)
    if IMAGE_SAVING_FREQ > 0:
        _image_saving_counter += 1
        if _image_saving_counter % IMAGE_SAVING_FREQ == 0:
            # Initialize save directory if not already done
            if _image_saving_dir is None:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                _image_saving_dir = f"saved_images/vec_run_{timestamp}"
                os.makedirs(_image_saving_dir, exist_ok=True)
                print(f"📁 Saving images to {_image_saving_dir} every {IMAGE_SAVING_FREQ} frames")
            
            # Draw edges on the already-processed image
            edge_vis_img = processed_img.copy()

            for x, y, _ in steering_edges:
                cv2.circle(edge_vis_img, (int(x), int(y)), 1, (0, 0, 255), -1)  # Red
            if throttle_edges:
                for x, y, _ in throttle_edges:
                    cv2.circle(edge_vis_img, (int(x), int(y)), 1, (255, 0, 0), -1)  # Blue
            
            # Save image with edges
            image_filename = os.path.join(_image_saving_dir, f"vec_frame_{_image_saving_counter:08d}.png")
            cv2.imwrite(image_filename, cv2.cvtColor(edge_vis_img, cv2.COLOR_RGB2BGR))

    return feature_vector

# ============================================================================
# Vectorized DonkeyCar Environment
# ============================================================================

class VectorizedDonkeyEnv(gym.Env):
    """
    Vectorized DonkeyCar environment - converts camera raw images into feature vectors.
    """

    def __init__(self, env_id: str, conf=None):
        """
        Args:
            env_id (str): DonkeyCar environment identifier
            conf (dict): Configuration dictionary for the DonkeyCar simulator
        """
        print("🔄🔄🔄🔄🔄 CALLFLOW: Entering image_vectorizer.py - DonkeyCar Environment Vectorization 🔄🔄🔄🔄🔄")
        self.env_id = env_id
        self.conf = conf
        
        # Create the base DonkeyCar environment
        self.base_env = gym.make(env_id, conf=conf)
        
        # Set up observation and action spaces
        self._setup_spaces()
    
    def _setup_spaces(self):
        """Setup Gym spaces"""
        # Get a sample environment image to determine the dimensions of the vectorized observation
        sample_img = self.base_env.reset()
        sample_vector = vectorize_image(sample_img)
        vector_dim = sample_vector.shape[0]

        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(vector_dim,),
            dtype=np.float32
        )

        # Standard action space
        self.action_space = self.base_env.action_space

    def reset(self):
        """Reset environment method"""
        obs = self.base_env.reset() # Image observation
        
        # Convert image observation to a feature vector
        feature_vector = vectorize_image(obs)
        
        return feature_vector
    
    def step(self, action):
        """Step method"""
        # Execute action in base environment
        obs, reward, done, info = self.base_env.step(action)

        # Convert image observation to a feature vector
        feature_vector = vectorize_image(obs)
        
        return feature_vector, reward, done, info
    
    def render(self, mode: str = "human"):
        """Render environment method"""
        return self.base_env.render(mode)
    
    def close(self):
        """Environment close method"""
        cleanup_drawer()
        self.base_env.close()