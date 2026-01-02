#!/usr/bin/env python3
"""
Car Path Visualizer with CTE-Based Color Gradient
Visualizes car's path during evaluation within the DonkeyCar simulator
Colors the path based on absolute CTE values.

Usage:
    python car_path_visualizer.py <log_file_path>
"""
# ====================================================================================
# Import Required Libraries
# ====================================================================================

import re
import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.collections import LineCollection

# ====================================================================================
# LLNCS / LaTeX-Compatible Font Configuration
# ====================================================================================

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

# ====================================================================================
# CTE Threshold Configuration
# ====================================================================================

CTE_THRESHOLD = 2.0

# ====================================================================================
# Car Path Visualizer Class
# ====================================================================================

class CarPathVisualizer:
    """
    Visualizes car path colored by car CTE magnitude.
    """
    
    def __init__(self, log_file: str):
        self.log_file = log_file # Log file path
        self.positions = []      # (x, z) tuples
        self.cte_values = []     # CTE values
        self.output_filename = ""  # Will be set by main()
        
    def set_output_filename(self, filename: str):
        """Set the output filename for the plot"""
        self.output_filename = filename
        
    def has_data(self) -> bool:
        """Check if data was extracted from the log file"""
        return len(self.positions) > 0 and len(self.cte_values) > 0
        
    def extract_data(self) -> None:
        """Extract car position (X, Z) and CTE data from the log file"""
        print("Extracting car positions and CTE data...")
        
        pos_pattern = re.compile(r'pos:\s*\(([^,]+),\s*([^,]+),\s*([^)]+)\)')
        cte_pattern = re.compile(r'cte:\s*([-\d.]+)')
        
        try:
            with open(self.log_file, 'r') as f:
                content = f.read()
            
            # Extract positions - (x, y, z)
            pos_matches = pos_pattern.findall(content)
            for match in pos_matches:
                x = float(match[0].strip())
                y = float(match[1].strip())
                z = float(match[2].strip())
                self.positions.append((x, z))
            
            # Extract CTE values
            cte_matches = cte_pattern.findall(content)
            for match in cte_matches:
                self.cte_values.append(float(match))
            
            print(f"✅ Extracted {len(self.positions)} position points and {len(self.cte_values)} CTE values")
            
        except IOError as e:
            raise Exception(f"Could not read log file: {e}")
        except Exception as e:
            raise Exception(f"Error parsing data: {e}")
    
    def create_cte_colormap(self, cte_abs: np.ndarray):
        """
        Create a colormap for CTE visualization.
        Args:
            cte_abs (np.ndarray): Absolute CTE values
        Returns:
            tuple: (colormap, norm, vmax) for visualization
        """
        colors = [
            '#1a9850',  # Dark green (CTE = 0)
            '#91cf60',  # Light green (approaching CTE threshold)
            '#ffee00',  # Bright yellow (At CTE threshold)
            '#ff6600',  # Vivid orange (Exceeding CTE threshold)
            '#cc0000',  # Bright red (High CTE)
        ]
        
        # Create a perceptually uniform colormap
        cmap = LinearSegmentedColormap.from_list('cte_cmap', colors, N=256)
        
        # Fixed vmax for consistent cross-model comparison
        vmin = 0
        vmax = CTE_THRESHOLD * 2.0
        
        norm = TwoSlopeNorm(vmin=vmin, vcenter=CTE_THRESHOLD, vmax=vmax)
        
        # Clip extreme outliers to vmax (they'll anyway be coloured in red)
        cte_abs_clipped = np.clip(cte_abs, vmin, vmax)
        
        return cmap, norm, vmax
    
    def plot_car_path(self) -> None:
        """Plot car path with CTE-based color gradient"""
        if not self.has_data():
            print("❌ No data available for plotting")
            return
        
        positions = np.array(self.positions)
        cte_values = np.array(self.cte_values)
        cte_abs = np.abs(cte_values)
        
        # Calculate distances between consecutive points
        distances = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
        
        # Adaptive threshold: Q3 + 1.5*IQR (standard outlier detection)
        q1, q2, q3 = np.percentile(distances, [25, 50, 75])
        iqr = q3 - q1
        distance_threshold = q3 + 1.5 * iqr
        valid_segments = distances < distance_threshold

        print(f"Calculated {len(distances)} segments between position points.")
        print(f"📊 Stats of distances between points: Q1={q1:.2f}, Q2={q2:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}")
        print(f"🎯 Outlier threshold: {distance_threshold:.2f}")
        print(f"🔍 Filtered {(~valid_segments).sum()} discontinuous segments out of {len(distances)}")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create line segments for colored path (only for valid segments)
        points = positions.reshape(-1, 1, 2)
        all_segments = np.concatenate([points[:-1], points[1:]], axis=1)
        segments = all_segments[valid_segments]  # Filter out discontinuous segments resulting from episode resets
        valid_cte = cte_abs[:-1][valid_segments]  # Corresponding CTE values
        
        # Create custom colormap with fixed scale for cross-model comparison
        cmap, norm, vmax = self.create_cte_colormap(cte_abs)
        
        # Clip CTE values to vmax for consistent visualization across models
        valid_cte_clipped = np.clip(valid_cte, 0, vmax)
        num_clipped = (valid_cte > vmax).sum()
        print(f'🔴 Clipped CTE values (|CTE| > 2*CTE_Threshold): {num_clipped} out of {len(cte_values)}')

        
        # Create LineCollection with color mapping
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=1, alpha=0.8)
        lc.set_array(valid_cte_clipped)  # Use filtered and clipped CTE values
        
        # Add to plot
        line = ax.add_collection(lc)
        
        # Add colorbar (horizontal, below the plot)
        cbar = plt.colorbar(line, ax=ax, label='|CTE|', orientation='horizontal', pad=0.005, aspect=30)
        cbar.set_label('|CTE|', fontsize=16)
        
        # Set colorbar ticks
        tick_positions = [0, CTE_THRESHOLD * 0.5, CTE_THRESHOLD,
                          CTE_THRESHOLD * 1.5, CTE_THRESHOLD * 2.0]
        cbar.set_ticks(tick_positions)
        actual_max = np.max(cte_abs)
        tick_labels = [f'{t:.1f}' for t in tick_positions]
        if actual_max > vmax:
            tick_labels[-1] = f'{tick_labels[-1]}+' # Mark maximum as "vmax+" to indicate clipped values
        cbar.set_ticklabels(tick_labels)
        cbar.ax.tick_params(labelsize=12)
        
        # Mark start and end points
        ax.scatter(positions[0, 0], positions[0, 1], color='blue', s=150, 
                  marker='o', edgecolor='white', linewidth=2, zorder=5, label='Start Position')
        ax.scatter(positions[-1, 0], positions[-1, 1], color='black', s=150, 
                  marker='s', edgecolor='white', linewidth=2, zorder=5, label='End Position')
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                      markersize=8, markeredgecolor='white', label='Start'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='black', 
                      markersize=8, markeredgecolor='white', label='End')        ]
        ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9)
        
        # Set equal aspect ratio and add frame box
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
            spine.set_edgecolor('black')
        
        # Statistics
        avg_cte = np.mean(cte_abs)
        max_cte = np.max(cte_abs)
        exceeding_pct = 100 * np.sum(cte_abs > CTE_THRESHOLD) / len(cte_abs)
        
        # # Add statistics text box
        # stats_text = (
        #     f'Statistics:\n'
        #     f'Avg |CTE|: {avg_cte:.2f}\n'
        #     f'Max |CTE|: {max_cte:.2f}\n'
        #     f'Exceeding CTE %: {exceeding_pct:.1f}%'
        # )
        # ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
        #        verticalalignment='top', bbox=dict(boxstyle='round', 
        #        facecolor='white', alpha=0.8), fontsize=9)
        plt.tight_layout()
        plt.savefig(self.output_filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"💾 Plot saved as {self.output_filename}")
        print(f"📊 Statistics:")
        print(f"   Average |CTE|: {avg_cte:.2f}")
        print(f"   Maximum |CTE|: {max_cte:.2f}")
        print(f"   Exceeding threshold: {exceeding_pct:.1f}% of path")
        plt.close(fig)

# ====================================================================================
# Main Function with Command Line Argument Support
# ====================================================================================

def main():
    """Main function with command line argument support"""
    
    parser = argparse.ArgumentParser(
        description="Visualize car path colored by CTE magnitude",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "log_file",
        type=str,
        help="Path to the evaluation log file"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output filename (auto-generated if not specified)"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.log_file):
        print(f"❌ Error: Log file '{args.log_file}' does not exist!")
        sys.exit(1)
    
    # Auto-generate output filename if not specified
    if args.output is None:
        log_basename = os.path.splitext(os.path.basename(args.log_file))[0]
        model_identifier = log_basename.replace('evaluation_log_', '')
        output_filename = f"car_path_cte_{model_identifier}.png"
    else:
        output_filename = args.output
    
    print(f"📖 Loading log file: {args.log_file}")
    print(f"💾 Output image: {output_filename}")
    print(f"🎯 CTE threshold: {CTE_THRESHOLD}")
    
    try:
        # Create visualizer and process data
        visualizer = CarPathVisualizer(args.log_file)
        visualizer.set_output_filename(output_filename)
        visualizer.extract_data()
        
        if visualizer.has_data():
            visualizer.plot_car_path()
        else:
            print("❌ No valid data found in the log file!")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error processing log file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()