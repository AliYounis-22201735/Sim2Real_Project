"""
Path calculation module, part of the edge_detection framework
Based on Ben's paths.py implementation. Refactored to support flexible image dimensions, replacing the hardcoded dimensions
Generates evenly-spaced radial sampling lines from the image bottom-center to boundary intersections, providing pixel coordinates and Euclidean distances for edge analysis.
"""
# ============================================================================
# Import Required Libraries
# ============================================================================

import numpy as np

# ============================================================================
# Generic PathCalculator Class
# ============================================================================

class PathCalculator:
    """Generic path calculator class"""
    
    def __init__(self, image_width, image_height):
        """
        Initialize a PathCalculator instance
        
        Args:
            image_width: Width of the image where paths will be calculated
            image_height: Height of the image where paths will be calculated
        """
        print("🔄🔄🔄🔄🔄 CALLFLOW: Entering paths.py - Generic Path Calculator 🔄🔄🔄🔄🔄")
        self.width = image_width
        self.height = image_height

        # Calculate the coordinates of bottom center point
        self.center_x = image_width // 2
        self.center_y = image_height - 1
        
        print(f"✅ Path calculator initialized for {image_width}x{image_height} images")
        print(f"📍  Source point: ({self.center_x}, {self.center_y})")
    
    def _left_intersection(self, angle):
        """Calculate intersection point with the left edge of the image"""
        return (0, self.center_y - np.tan(angle) * self.center_x)

    def _top_intersection(self, angle):
        """Calculate intersection point with top edge of the image"""
        return (self.center_x - self.center_y / np.tan(angle), 0)

    def _right_intersection(self, angle):
        """Calculate intersection point with the right edge of the image"""
        return (self.width - 1, self.center_y + np.tan(angle) * (self.width - 1 - self.center_x))

    def _get_targets(self, angle_step, skip_lines=0):
        """
        Get target (intersection) point for each line
        
        Args:
            angle_step: Angle step between lines in degrees
            skip_lines: Number of lines to skip from edges, defaults to 0 (no lines skipped)
        """
        # Calculate angle boundaries
        left_end = np.pi / 2 - np.arctan(self.center_x / self.center_y)
        top_end = np.pi - left_end
        
        def get_intercept(a):
            """Get intersection point for angle 'a'"""
            if a < left_end:
                return self._left_intersection(a)
            elif a < top_end:
                return self._top_intersection(a)
            else:
                return self._right_intersection(a)
        
        step_rad = angle_step * (np.pi / 180)  # Convert degrees into radians

        # Return the intersection points for each angle
        return [
            get_intercept(a)
            for a in np.arange(step_rad + step_rad * skip_lines, 
                             np.pi - step_rad * skip_lines, 
                             step_rad) # The last value is excluded
        ]

    def _line_points(self, source, target):
        """Calculate evenly-spaced integer points along the casted lines between the source point and targets"""
        length = int(np.hypot(target[0] - source[0], target[1] - source[1])) # Calculate the Euclidean distance between the source point and targets, truncated to integer
        return np.linspace(source, target, length, dtype=int)
    
    def _get_distance(self, source, target):
        """Calculate Euclidean distance between the source point and target points"""
        return np.hypot(target[0] - source[0], target[1] - source[1])

    def _get_unique_line_points(self, source, target):
        """Get unique points along the ray lines drawn between source and targets"""
        line_points = self._line_points(source, target)
        _, idx = np.unique(line_points, axis=0, return_index=True) # Filter out the unique points and retrieve their indices
        return line_points[np.sort(idx)].T # Sort unique points by their indices and transpose the array, returning x and y coordinates as separate arrays
  
    def get_paths(self, num_lines, radius_from, skip_lines):
        """
        Get unique points' coordinates, their distances from the source point, and names/tags of ray lines
        
        Args:
            num_lines: Number of lines to draw from the source point to image boundaries. Must result in a step angle that divides 180 evenly.
            radius_from: The radius from which to start collecting unique points (i.e., skip first points from the source).
            skip_lines: Number of lines to skip from both ends.
        
        Returns:
            (tuple[list[np.ndarray], list[np.ndarray], list[str]]):
            paths: NumPy arrays with X, Y coordinates of unique points along each line
            distances: NumPy arrays with the Euclidean distances from the source point at each corresponding coordinate
            names: Names of ray lines based on their target coordinates
        """
        if num_lines <= 1:
            raise ValueError("❌ There must be more than one line")
        if skip_lines >= num_lines // 2:
            raise ValueError("❌ Cannot skip all lines")

        # Use the default source point (bottom midpoint)
        source = (self.center_x, self.center_y)

        # Get target intersection points
        targets = self._get_targets(180 / (num_lines + 1), skip_lines)
        
        # Calculate unique points, distances, and target names
        paths = [self._get_unique_line_points(source, t)[..., radius_from:] for t in targets] # Get coordinates of unique points between the source point and image-boundary targets
        distances = [self._get_distance(source, unique_point) for unique_point in paths] # Get distances from the source point for each unique point calculated along the casted ray lines
        target_names = [f"({t[0]:3.0f}, {t[1]:3.0f})" for t in targets] # Use the coordinates of target points as lines' names

        return paths, distances, target_names
