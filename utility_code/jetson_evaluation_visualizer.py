#!/usr/bin/env python3
"""
A script to visualize the real-world evaluation runs on Jetson Nano
Extract data (steering and throttle controls and inference times) from Jetson Nano inference log files and produces both time-series and distribution plots, along with summary statistics
"""
# ============================================================================
# Import Required Libraries and Modules
# ============================================================================

import re
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import PercentFormatter
from pathlib import Path

# ====================================================================================
# LLNCS / LaTeX-Compatible Font Configuration
# ====================================================================================
mpl.rcParams.update({
    # Serif fonts
    "font.family": "serif",
    "font.serif": [
        "Latin Modern Roman",
        "Computer Modern Roman",
        "Times New Roman",
        "DejaVu Serif"
    ],

    # Math font
    "mathtext.fontset": "cm",

    # Font sizes 
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,

    # Image saving resolution
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

# ====================================================================================
# Colour Configuration
# ====================================================================================

STEERING_COLOR = "#DE8F05"
THROTTLE_COLOR = "#0173B2"
MEAN_COLOR = "#D55E00"
MEDIAN_COLOR = "#0072B2"

# ============================================================================
# Data Extraction Function
# ============================================================================

def data_extraction(log_file):
    """
    Extract steering, throttle, and inference time values from Jetson Nano inference log files
    
    Args:
        log_file: Path to Jetson Nano inference log file
        
    Returns:
        tuple: (steering_values, throttle_values, inference_times)
    """
    steering_values = []
    throttle_values = []
    inference_times = []
    
    with open(log_file, 'r') as f:
        for line in f:
            # Extract steering, throttle, and inference time
            match = re.search(r'Steering:\s*([-+]?\d*\.?\d+),\s*Throttle:\s*([-+]?\d*\.?\d+)(?:.*?Inference Time:\s*([-+]?\d*\.?\d+)\s*ms)?', line)
            
            if match:
                try:
                    steering_values.append(float(match.group(1)))
                    throttle_values.append(float(match.group(2)))
                    inference_times.append(float(match.group(3)))
                except ValueError:
                    continue
    
    return (np.array(steering_values), 
            np.array(throttle_values), 
            np.array(inference_times))

# ============================================================================
# Visualization Functions: Time-Series
# ============================================================================

def plot_timeseries(steering, throttle, output_file='jetson_actions_timeseries.png'):
    """
    Plot time-series graphs for steering and throttle controls.
    
    Args:
        steering: Array of steering values
        throttle: Array of throttle values
        output_file: Output filename for the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot steering time-series
    ax1.plot(steering, linewidth=2, color=STEERING_COLOR, alpha=0.7)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Steering')
    ax1.set_ylim(-1.1, 1.1)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.axhline(y=np.mean(steering), color=MEAN_COLOR, linestyle='--', alpha=0.7, 
                label=f'Mean: {np.mean(steering):.3f}')
    ax1.axhline(y=np.median(steering), color=MEDIAN_COLOR, linestyle='--', alpha=0.7, 
                label=f'Median: {np.median(steering):.3f}')
    ax1.legend()
    
    # Plot throttle time-series
    ax2.plot(throttle, linewidth=2, color=THROTTLE_COLOR, alpha=0.7)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Throttle')
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.axhline(y=np.mean(throttle), color=MEAN_COLOR, linestyle='--', alpha=0.7, 
                label=f'Mean: {np.mean(throttle):.3f}')
    ax2.axhline(y=np.median(throttle), color=MEDIAN_COLOR, linestyle='--', alpha=0.7, 
                label=f'Median: {np.median(throttle):.3f}')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    print(f"✅ Time-series plot saved as '{output_file}'")
    plt.close()
    
    # Save individual plots
    output_path = Path(output_file)
    base_name = output_path.stem
    output_dir = output_path.parent
    
    # Steering timeseries plot
    fig_steering, ax = plt.subplots(figsize=(7, 5))
    ax.plot(steering, linewidth=2, color=STEERING_COLOR, alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Steering')
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.axhline(y=np.mean(steering), color=MEAN_COLOR, linestyle='--', alpha=0.7, 
               label=f'Mean: {np.mean(steering):.3f}')
    ax.axhline(y=np.median(steering), color=MEDIAN_COLOR, linestyle='--', alpha=0.7, 
               label=f'Median: {np.median(steering):.3f}')
    ax.legend()
    plt.tight_layout()
    steering_file = output_dir / f"{base_name}_steering.png"
    plt.savefig(steering_file, bbox_inches='tight')
    print(f"✅ Steering time-series saved as '{steering_file}'")
    plt.close()
    
    # Throttle timeseries plot
    fig_throttle, ax = plt.subplots(figsize=(7, 5))
    ax.plot(throttle, linewidth=2, color=THROTTLE_COLOR, alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Throttle')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.axhline(y=np.mean(throttle), color=MEAN_COLOR, linestyle='--', alpha=0.7, 
               label=f'Mean: {np.mean(throttle):.3f}')
    ax.axhline(y=np.median(throttle), color=MEDIAN_COLOR, linestyle='--', alpha=0.7, 
               label=f'Median: {np.median(throttle):.3f}')
    ax.legend()
    plt.tight_layout()
    throttle_file = output_dir / f"{base_name}_throttle.png"
    plt.savefig(throttle_file, bbox_inches='tight')
    print(f"✅ Throttle time-series saved as '{throttle_file}'")
    plt.close()

# ============================================================================
# Visualization Functions: Distribution
# ============================================================================

def plot_distributions(steering, throttle, output_file='jetson_actions_distributions.png'):
    """
    Plot steering and throttle action distributions.
    
    Args:
        steering: Array of steering values
        throttle: Array of throttle values
        output_file: Output filename for the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Steering distribution
    weights_steering = np.ones_like(steering) / len(steering) * 100
    ax1.hist(steering, bins=50, range=(-1, 1), weights=weights_steering, 
             color=STEERING_COLOR, alpha=0.7, edgecolor='black')
    ax1.axvline(np.mean(steering), color=MEAN_COLOR, linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(steering):.3f}')
    ax1.axvline(np.median(steering), color=MEDIAN_COLOR, linestyle='--', linewidth=2, 
                label=f'Median: {np.median(steering):.3f}')
    ax1.set_xlabel('Steering')
    ax1.set_ylabel('Percentage')
    ax1.set_xlim(-1.1, 1.1)
    ax1.set_ylim(-5, 105)
    ax1.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Throttle distribution
    weights_throttle = np.ones_like(throttle) / len(throttle) * 100
    ax2.hist(throttle, bins=50, range=(0, 1), weights=weights_throttle, 
             color=THROTTLE_COLOR, alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(throttle), color=MEAN_COLOR, linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(throttle):.3f}')
    ax2.axvline(np.median(throttle), color=MEDIAN_COLOR, linestyle='--', linewidth=2, 
                label=f'Median: {np.median(throttle):.3f}')
    ax2.set_xlabel('Throttle')
    ax2.set_ylabel('Percentage')
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-5, 105)
    ax2.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    print(f"✅ Distribution plot saved as '{output_file}'")
    plt.close()
    
    # Save individual plots
    output_path = Path(output_file)
    base_name = output_path.stem
    output_dir = output_path.parent
    
    # Steering distribution plot
    fig_steering, ax = plt.subplots(figsize=(7, 5))
    ax.hist(steering, bins=50, range=(-1, 1), weights=weights_steering, 
            color=STEERING_COLOR, alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(steering), color=MEAN_COLOR, linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(steering):.3f}')
    ax.axvline(np.median(steering), color=MEDIAN_COLOR, linestyle='--', linewidth=2, 
               label=f'Median: {np.median(steering):.3f}')
    ax.set_xlabel('Steering')
    ax.set_ylabel('Percentage')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-5, 105)
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    steering_file = output_dir / f"{base_name}_steering.png"
    plt.savefig(steering_file, bbox_inches='tight')
    print(f"✅ Steering distribution saved as '{steering_file}'")
    plt.close()
    
    # Throttle distribution plot
    fig_throttle, ax = plt.subplots(figsize=(7, 5))
    ax.hist(throttle, bins=50, range=(0, 1), weights=weights_throttle, 
            color=THROTTLE_COLOR, alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(throttle), color=MEAN_COLOR, linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(throttle):.3f}')
    ax.axvline(np.median(throttle), color=MEDIAN_COLOR, linestyle='--', linewidth=2, 
               label=f'Median: {np.median(throttle):.3f}')
    ax.set_xlabel('Throttle')
    ax.set_ylabel('Percentage')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-5, 105)
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    throttle_file = output_dir / f"{base_name}_throttle.png"
    plt.savefig(throttle_file, bbox_inches='tight')
    print(f"✅ Throttle distribution saved as '{throttle_file}'")
    plt.close()


# ============================================================================
# Statistics Summary Functions
# ============================================================================

def print_summary_statistics(steering, throttle, inference_times):
    """
    Print comprehensive summary statistics to console.
    
    Args:
        steering: Array of steering values
        throttle: Array of throttle values
        inference_times: Array of inference times (ms)
    """
    print("\n" + "="*70)
    print("📊 SUMMARY STATISTICS")
    print("="*70)
    
    # Action statistics
    print(f"\n🎮 Action Statistics:")
    print(f"   Total actions extracted: {len(steering)}")
    print(f"\n   Steering:")
    print(f"      Range: [{np.min(steering):.4f}, {np.max(steering):.4f}]")
    print(f"      Mean: {np.mean(steering):.4f}")
    print(f"      Median: {np.median(steering):.4f}")
    print(f"\n   Throttle:")
    print(f"      Range: [{np.min(throttle):.4f}, {np.max(throttle):.4f}]")
    print(f"      Mean: {np.mean(throttle):.4f}")
    print(f"      Median: {np.median(throttle):.4f}")
    
    # Inference time statistics
    if len(inference_times) > 0:
        print(f"\n⚡ Inference Performance:")
        print(f"   Total inference samples: {len(inference_times)}")
        
        if len(inference_times) > 1:
            # Exclude first frame (often outlier due to initialization)
            avg_inference = np.mean(inference_times[1:])
            min_inference = np.min(inference_times[1:])
            max_inference = np.max(inference_times[1:])
            
            print(f"   First frame: {inference_times[0]:.2f} ms (initialization, excluded from stats)")
            print(f"   Average: {avg_inference:.2f} ms")
            print(f"   Min: {min_inference:.2f} ms")
            print(f"   Max: {max_inference:.2f} ms")
        else:
            print(f"   Single inference time: {inference_times[0]:.2f} ms")
    else:
        print(f"\n⚡ Inference Performance:")
        print(f"   No inference time data found in log file")
    
    print("\n" + "="*70 + "\n")


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main execution function for command-line usage"""
    
    # Check if filename argument provided
    if len(sys.argv) < 2:
        print("Usage: python jetson_inference_visualization.py <log_file>")
        sys.exit(1)
    
    log_file = sys.argv[1]
    
    # Check if file exists
    log_path = Path(log_file)
    if not log_path.exists():
        print(f"❌ Error: File '{log_file}' not found!")
        sys.exit(1)
    
    # Extract data
    print(f"📖 Reading file: {log_file}")
    steering, throttle, inference_times = data_extraction(log_file)
    
    # Validate data
    if len(steering) == 0 or len(throttle) == 0:
        print("❌ Error: No valid action data found in log file!")
        sys.exit(1)
    
    # Print summary statistics
    print_summary_statistics(steering, throttle, inference_times)
    
    # Generate both visualizations
    output_prefix = log_path.stem
    
    timeseries_file = f"{output_prefix}_timeseries.png"
    plot_timeseries(steering, throttle, timeseries_file)
    
    distribution_file = f"{output_prefix}_distributions.png"
    plot_distributions(steering, throttle, distribution_file)


if __name__ == "__main__":
    main()
