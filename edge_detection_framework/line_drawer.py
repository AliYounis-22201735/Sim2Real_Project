#!/usr/bin/env python3
"""
Part of the edge_detection_framework - detects track lane edges in captured images for the control of steering and throttle actions 
Based on Ben's line_drawer.py implementation.
Analyzes pixel intensity gradients along pre-calculated paths using a variety of statistical methods for outlier detection (InterQuartile Range, Standard Deviation, and Rolling Standard Deviation)

Returns (x, y, distance) tuples for each detected edge, where:
- x, y: Pixel coordinates of detected edge points
- distance: Corresponding Euclidean distance from the mid-bottom source point
"""
# ============================================================================
# Import Required Libraries and Modules
# ============================================================================

import os
import sys
import cv2
import atexit # cleanup functions
import numpy as np
import multiprocessing as mp
from typing import List, Tuple, Optional, Union
from typing_extensions import TypedDict # Backward compatible with Python versions < 3.8

# Add current directory and parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)

from paths import PathCalculator
from edge_detector import IQRDetector, StdDetector, RollStdDetector
from core_code.image_processor import ImageProcessor

# ============================================================================
# LineDrawerParams TypedDict Class
# ============================================================================

class LineDrawerParams(TypedDict):
    image_processing_args: Tuple[float, Tuple[int, int], float] # (roi_crop_top, target_shape, augmentation_probability)
    num_lines: int
    radius_from: int
    skip_lines: int
    edge_strategy: str
    distance: float
    sqr_derivs: bool
    use_global: bool
    basis: str
    roll_window: int
    scale_roll: bool
    combine_lines: str

# ============================================================================
# SoloLineDrawer Base Class
# ============================================================================

class SoloLineDrawer:
    """A base class to manage image processing, path calculation, and edge detection"""
    def __init__(self, **kwargs: LineDrawerParams) -> None:
        print("🔄🔄🔄🔄🔄 CALLFLOW: Entering line_drawer.py - Line Drawer(s) Setup 🔄🔄🔄🔄🔄")
        
        # Store image processing parameters as instance variables
        self.image_processing_args = kwargs['image_processing_args']

        # Store path calculation parameters as instance variables
        self.num_lines = kwargs['num_lines']
        self.radius_from = kwargs['radius_from']
        self.skip_lines = kwargs['skip_lines']

        # Store edge detection parameters as instance variables
        self.edge_strategy = kwargs['edge_strategy']
        self.distance = kwargs['distance']
        self.sqr_derivs = kwargs['sqr_derivs']
        self.use_global = kwargs['use_global']
        self.basis = kwargs['basis']
        self.roll_window = kwargs['roll_window']
        self.scale_roll = kwargs['scale_roll']
        self.combine_lines = kwargs['combine_lines']

        # Initialize image dimensions - will be auto-detected from the first processed image
        self.image_width = None
        self.image_height = None

        # Initialize a path calculator instance - will be created upon the reception of the first image
        self.path_calculator = None

        # Paths will be directly calculated from images
        self.paths = None # Will hold the list of unique_points along the casted ray lines
        self.distances = None # Will hold the list of distances corresponding to each unique point
        self.target_names = None # Will hold the list of target point names corresponding to each line

        # Initialize edge detector instance - will be created after paths calculation
        self.edge_detector = None

        # Store last processed image
        self.processed_img = None

        # Initialize an ImageProcessor instance
        if self.image_processing_args is not None:
            roi_crop_top, target_shape, augmentation_probability = self.image_processing_args
            self.image_processor = ImageProcessor(
                roi_crop_top,
                target_shape,
                augmentation_probability
            )
        else:
            raise ValueError("❌ image_processing_args are required but not provided")
        
    def _process_image(self, img: np.ndarray):
        """
        Process raw images through the image processing pipeline
            
        Args:
            img: Input image array

        Returns:
            Processed image
        """
        return self.image_processor.process_image(img)
    
    def _create_path_calculator(self, img: np.ndarray):
        """
        Create a PathCalculator instance based on the detected image dimensions
        
        Initialized under two conditions:
        1. The first time an image is processed
        2. If image dimensions change (a rare edge case)
        """
        height, width = img.shape[:2] # HWC image format 
        
        if (self.path_calculator is None or 
            self.image_width != width or 
            self.image_height != height):

            # Update stored image dimensions
            self.image_width = width
            self.image_height = height  
            # Initialize path calculator
            self.path_calculator = PathCalculator(width, height)

            # Get paths (unique points' x, y coordinates), distances, and target names
            self.paths, self.distances, self.target_names = self.path_calculator.get_paths(
                num_lines=self.num_lines, 
                radius_from=self.radius_from,
                skip_lines=self.skip_lines
            )
            # Initialize edge detector
            self._create_edge_detector()

    def _create_edge_detector(self):
        """Create edge detector instance with initialized paths"""

        # Wait for paths to be initialized
        if self.paths is None:
            return

        # Initialize edge detector based on selected edge_detection strategy    
        if self.edge_strategy == "iqr":
            self.edge_detector = IQRDetector(
                self.paths,
                self.distances,
                distance=self.distance,
                sqr_derivs=self.sqr_derivs
            )
        elif self.edge_strategy == "std":
            self.edge_detector = StdDetector(
                self.paths,
                self.distances,
                distance=self.distance,
                use_global=self.use_global,
                basis=self.basis,
                sqr_derivs=self.sqr_derivs,
            )
        elif self.edge_strategy == "roll":
            self.edge_detector = RollStdDetector(
                self.paths,
                self.distances,
                distance=self.distance,
                use_global=self.use_global,
                roll_window=self.roll_window,
                scale_roll=self.scale_roll,
                combine_lines=self.combine_lines,
                sqr_derivs=self.sqr_derivs,
            )
        else:
            raise ValueError("❌ Must specify a valid edge strategy from ['iqr', 'std', 'roll']")

    def get_edges(self, img: np.ndarray) -> List[Tuple[int, int, float]]: # Return x, y coordinates and distance (from the source point) of detected edges
        # First process the image
        self.processed_img = self._process_image(img)
        # Initialize PathCalculator (after image processing, as image dimensions will change due to the cropping and resizing operations)
        self._create_path_calculator(self.processed_img)
        # Return detected edges
        return self.edge_detector.get_edges(self.processed_img)

    def get_dists(self, img: np.ndarray) -> list:
        # Get distances only of the detected edges
        return [x[2] for x in self.get_edges(img)]

    def get_processed_image(self) -> np.ndarray:
        """Return the last processed image"""
        return self.processed_img

# ============================================================================
# LineDrawer Class
# ============================================================================
    
class LineDrawer:
    """A higher-level wrapper to manage one or two instances (based on configuration) of the SoloLineDrawer class"""

    def __init__(
            self,
            drawer_1_kwargs: LineDrawerParams,
            drawer_2_kwargs: Optional[LineDrawerParams] = None, # Optional, only if a second drawer is needed
            ) -> None:
        
        self.drawer1 = SoloLineDrawer(**drawer_1_kwargs) # 1st SoloLineDrawer instance
        if drawer_2_kwargs is not None:
            self.solo_drawer = False
            self.drawer2 = SoloLineDrawer(**drawer_2_kwargs) # 2nd SoloLineDrawer instance
        else:
            self.solo_drawer = True

    # Return detected edges
    def get_edges(self, img: Union[str, np.ndarray]): # Accepts either an image file or an image array
        image = cv2.imread(img) if isinstance(img, str) else img
        steering_edges = self.drawer1.get_edges(image)
        if self.solo_drawer:
            throttle_edges = steering_edges
        else:
            throttle_edges = self.drawer2.get_edges(image)
        return steering_edges, throttle_edges

    # Return only distances of detected edges
    def get_dists(self, img: Union[str, np.ndarray]): # Accepts either an image file or an image array
        image = cv2.imread(img) if isinstance(img, str) else img
        steering_distances = self.drawer1.get_dists(image)
        if self.solo_drawer:
            throttle_distances = steering_distances
        else:
            throttle_distances = self.drawer2.get_dists(image)
        return steering_distances, throttle_distances

    # Return the last processed image from drawer1 (steering) as it is the same as the image processed by drawer2 (throttle) if it exists
    def get_processed_image(self) -> np.ndarray:
        """Return the last processed image from drawer1 (steering)"""
        return self.drawer1.get_processed_image()

# ============================================================================
# SubLineDrawer Class
# ============================================================================

class SubLineDrawer(mp.Process):
    """A multiprocessing wrapper to manage one instance of the SoloLineDrawer class"""
    def __init__(
        self,
        img_queue: mp.Queue, # A multiprocessing queue for input images
        result_queue: mp.Queue, # A multiprocessing queue for output results
        drawer_kwargs: LineDrawerParams):

        super().__init__()
        self.img_queue = img_queue
        self.result_queue = result_queue
        self.drawer_kwargs = drawer_kwargs
        self.drawer = None  # Lazy initialization - will be created in run() method

    def run(self):
        """Run method with lazy initialization of SoloLineDrawer instances to prevent process hanging"""

        self.drawer = SoloLineDrawer(**self.drawer_kwargs)
            
        while True: # Keep running until no new request is received
            request = self.img_queue.get() # receive requests at the input queue
            if request is None:
                break # shutdown if no request is received
            command, img = request # Unpack received requests into command and image pairs, except for the get_processed_image command as no image is passed in the request
            if command == "get_edges":
                result = self.drawer.get_edges(img)
            elif command == "get_dists":
                result = self.drawer.get_dists(img)
            elif command == "get_processed_image":
                result = self.drawer.get_processed_image()
            else:
                raise ValueError(f"❌ Invalid command: {command}! command should be one of ['get_edges', 'get_dists', 'get_processed_image']")
            self.result_queue.put(result) # Send the result to the result queue

# ============================================================================
# ParallelLineDrawer Class
# ============================================================================
"""A class to manage parallel LineDrawer processes"""
class ParallelLineDrawer:
    def __init__(
            self,
            steering_kwargs: LineDrawerParams,
            throttle_kwargs: LineDrawerParams,
            ) -> None:

        # Initialize steering LineDrawer process
        self.steering_img_queue = mp.Queue()
        self.steering_result_queue = mp.Queue()
        self.steering_process = SubLineDrawer(self.steering_img_queue, self.steering_result_queue, steering_kwargs)

        # Initialize throttle LineDrawer process
        self.throttle_img_queue = mp.Queue()
        self.throttle_result_queue = mp.Queue()
        self.throttle_process = SubLineDrawer(self.throttle_img_queue, self.throttle_result_queue, throttle_kwargs)

        # Start steering and throttle processes
        self.steering_process.start()
        self.throttle_process.start()

        # Cleanup at exit
        atexit.register(self.cleanup)

    # Return detected edges
    def get_edges(self, img: Union[str, np.ndarray]):
        # Read image if a file path is provided, otherwise treat as a numpy array image
        image = cv2.imread(img) if isinstance(img, str) else img

        # Pass command and image pairs to the respective queues
        self.steering_img_queue.put(("get_edges", image))
        self.throttle_img_queue.put(("get_edges", image))

        # Send results of detected edges to the result queues
        steering_edges = self.steering_result_queue.get()
        throttle_edges = self.throttle_result_queue.get()
        return steering_edges, throttle_edges # Return list of tupples in the form [(x_coord, y_coord, distance), ...] for both of steering and throttle controls

    # Return only distances of detected edges
    def get_dists(self, img: Union[str, np.ndarray]):
        # Read image if a file path is provided, otherwise treat as a numpy array image
        image = cv2.imread(img) if isinstance(img, str) else img

        # Pass command and image pairs to the respective queues
        self.steering_img_queue.put(("get_dists", image))
        self.throttle_img_queue.put(("get_dists", image))

        # Send results to the result queues
        steering_distances = self.steering_result_queue.get()
        throttle_distances = self.throttle_result_queue.get()
        return steering_distances, throttle_distances # Return distances lists for both of steering and throttle controls

    # Return the last processed image from the steering process as it is the same as the image in the throttle process
    def get_processed_image(self) -> np.ndarray:
        """Return the last processed image (requests from steering process)"""
        self.steering_img_queue.put(("get_processed_image", None)) # No image is needed for this command
        # Send result to the steering result queue
        return self.steering_result_queue.get()

    # Cleanup
    def cleanup(self):
        self.steering_img_queue.put(None, timeout=2.0)
        self.throttle_img_queue.put(None, timeout=2.0)
        # Wait for the processes to finish
        self.steering_process.join(timeout=5.0)
        self.throttle_process.join(timeout=5.0)