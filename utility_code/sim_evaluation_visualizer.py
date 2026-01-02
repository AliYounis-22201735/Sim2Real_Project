#!/usr/bin/env python3
"""
Evaluation metric visualizer - Extract and plot key metrics for performance analysis
"""
# ====================================================================================
# Import Required Libraries and Modules
# ====================================================================================

import re
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import PercentFormatter
from pathlib import Path

# ====================================================================================
# Plot Styling Configuration
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

# Define colors
STEERING_COLOR = "#DE8F05" # orange
THROTTLE_COLOR = "#0173B2" # blue
CTE_COLOR = "#CC78BC" # purple
VELOCITY_COLOR = "#029E73" # green
REWARD_COLOR = "#CA9161" # brown
DISTRIBUTION_COLOR = "#56B4E9" # light blue
MEAN_COLOR = "#D55E00" # reddish orange
MEDIAN_COLOR = "#0072B2" # blue

# ====================================================================================
# Data Extraction and Plotting Functions
# ====================================================================================

def extract_data(file_path):
    """
    Extract steering, throttle, CTE, forward velocity, reward, and inference time values from the DonkeyCar simulator evaluation log files.
    
    Args:
        file_path (str): Path to the evaluation log file

    Returns:
        tuple: A tuple containing lists of extracted values (steering, throttle, CTE, forward velocity, reward, and inference times)
    """
    steering_values = []
    throttle_values = []
    cte_values = []
    forward_vel_values = []
    reward_values = []
    inference_times = []
    
    with open(file_path, 'r') as f:
        file_content = f.read()
    
    # Extract steering and throttle actions:
    action_pattern = r'Action:\s*\[\[([^\s]+)\s+([^\]]+)\]\]'
    actions = re.findall(action_pattern, file_content)

    for steering_str, throttle_str in actions:
        steering_values.append(float(steering_str.strip()))
        throttle_values.append(float(throttle_str.strip()))
    
    # Extract CTE:
    cte_pattern = r'cte:\s*([-\d.]+)'
    ctes = re.findall(cte_pattern, file_content)
    
    for cte in ctes:
        cte_values.append(float(cte))
    
    # Extract forward velocity:
    forward_vel_pattern = r'forward_vel:\s*([-\d.]+)'
    forward_vels = re.findall(forward_vel_pattern, file_content)
    
    for fv in forward_vels:
        forward_vel_values.append(float(fv))
    
    # Extract reward
    reward_pattern = r'Reward:\s*\[?([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\]?'
    rewards = re.findall(reward_pattern, file_content)
    
    for reward in rewards:
        reward_values.append(float(reward))
    
    # Extract inference times
    inference_time_pattern = r'Inference Time:\s*([-+]?\d*\.?\d+)ms'
    inference_time_matches = re.findall(inference_time_pattern, file_content)
    
    for inference_time in inference_time_matches:
        inference_times.append(float(inference_time))

    return steering_values, throttle_values, cte_values, forward_vel_values, reward_values, inference_times

def plot_action_distributions(steering, throttle, output_file):
    """Plot distribution histograms for steering and throttle actions."""
    # Create a combined figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Steering distribution
    weights_steering = np.ones_like(steering) / len(steering) * 100
    ax1.hist(steering, bins=50, weights=weights_steering, color=STEERING_COLOR, alpha=0.7, edgecolor='black')
    ax1.axvline(np.mean(steering), color=MEAN_COLOR, linestyle='--', linewidth=2, label=f'Mean: {np.mean(steering):.3f}')
    ax1.axvline(np.median(steering), color=MEDIAN_COLOR, linestyle='--', linewidth=2, label=f'Median: {np.median(steering):.3f}')
    ax1.set_xlabel('Steering Value')
    ax1.set_ylabel('Percentage')
    ax1.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Throttle distribution
    weights_throttle = np.ones_like(throttle) / len(throttle) * 100
    ax2.hist(throttle, bins=50, weights=weights_throttle, color=THROTTLE_COLOR, alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(throttle), color=MEAN_COLOR, linestyle='--', linewidth=2, label=f'Mean: {np.mean(throttle):.3f}')
    ax2.axvline(np.median(throttle), color=MEDIAN_COLOR, linestyle='--', linewidth=2, label=f'Median: {np.median(throttle):.3f}')
    ax2.set_xlabel('Throttle Value')
    ax2.set_ylabel('Percentage')
    ax2.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    print(f"✅ Combined histogram distribution plot saved as '{output_file}'")
    plt.close(fig)
    
    # Save individual plots
    output_path = Path(output_file)
    base_name = output_path.stem
    
    # Save steering distribution
    fig_steering, ax_s = plt.subplots(figsize=(10, 5))
    ax_s.hist(steering, bins=50, weights=weights_steering, color=STEERING_COLOR, alpha=0.7, edgecolor='black')
    ax_s.axvline(np.mean(steering), color=MEAN_COLOR, linestyle='--', linewidth=2, label=f'Mean: {np.mean(steering):.3f}')
    ax_s.axvline(np.median(steering), color=MEDIAN_COLOR, linestyle='--', linewidth=2, label=f'Median: {np.median(steering):.3f}')
    ax_s.set_xlabel('Steering Value')
    ax_s.set_ylabel('Percentage')
    ax_s.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax_s.legend()
    ax_s.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    steering_file = output_path.parent / f"{base_name}_steering.png"
    plt.savefig(steering_file, bbox_inches='tight')
    print(f"✅ Steering histogram distribution saved as '{steering_file}'")
    plt.close(fig_steering)
    
    # Save throttle distribution
    fig_throttle, ax_t = plt.subplots(figsize=(10, 5))
    ax_t.hist(throttle, bins=50, weights=weights_throttle, color=THROTTLE_COLOR, alpha=0.7, edgecolor='black')
    ax_t.axvline(np.mean(throttle), color=MEAN_COLOR, linestyle='--', linewidth=2, label=f'Mean: {np.mean(throttle):.3f}')
    ax_t.axvline(np.median(throttle), color=MEDIAN_COLOR, linestyle='--', linewidth=2, label=f'Median: {np.median(throttle):.3f}')
    ax_t.set_xlabel('Throttle Value')
    ax_t.set_ylabel('Percentage')
    ax_t.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax_t.legend()
    ax_t.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    throttle_file = output_path.parent / f"{base_name}_throttle.png"
    plt.savefig(throttle_file, bbox_inches='tight')
    print(f"✅ Throttle histogram distribution saved as '{throttle_file}'")
    plt.close(fig_throttle)

def plot_data(steering, throttle, cte, forward_vel, reward, output_file):
    """Plot steering, throttle, CTE, forward velocity, and reward in separate subplots."""
    # Create combined figure
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(20, 15))
    
    # Plot steering
    ax1.plot(steering, linewidth=1, color=STEERING_COLOR, alpha=0.7)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Steering')
    ax1.set_title('Steering Actions', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    # Plot throttle
    ax2.plot(throttle, linewidth=1, color=THROTTLE_COLOR, alpha=0.7)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Throttle')
    ax2.set_title('Throttle Actions', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    # Plot CTE
    ax3.plot(cte, linewidth=1, color=CTE_COLOR, alpha=0.7)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('CTE (Cross Track Error)')
    ax3.set_title('Cross Track Error (CTE)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    # Plot forward velocity
    ax4.plot(forward_vel, linewidth=1, color=VELOCITY_COLOR, alpha=0.7)
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Forward Velocity')
    ax4.set_title('Forward Velocity', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    # Plot reward
    ax5.plot(reward, linewidth=1, color=REWARD_COLOR, alpha=0.7)
    ax5.set_xlabel('Step')
    ax5.set_ylabel('Reward')
    ax5.set_title('Reward per Step', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    print(f"✅ Combined timeseries plot saved as '{output_file}'")
    plt.close(fig)
    
    # Save individual plots
    output_path = Path(output_file)
    base_name = output_path.stem
    
    # Save steering plot
    fig_s, ax_s = plt.subplots(figsize=(20, 3))
    ax_s.plot(steering, linewidth=1, color=STEERING_COLOR, alpha=0.7)
    ax_s.set_xlabel('Step')
    ax_s.set_ylabel('Steering')
    ax_s.grid(True, alpha=0.3)
    ax_s.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    plt.tight_layout()
    steering_file = output_path.parent / f"{base_name}_steering.png"
    plt.savefig(steering_file, bbox_inches='tight')
    print(f"✅ Steering timeseries plot saved as '{steering_file}'")
    plt.close(fig_s)
    
    # Save throttle plot
    fig_t, ax_t = plt.subplots(figsize=(20, 3))
    ax_t.plot(throttle, linewidth=1, color=THROTTLE_COLOR, alpha=0.7)
    ax_t.set_xlabel('Step')
    ax_t.set_ylabel('Throttle')
    ax_t.grid(True, alpha=0.3)
    ax_t.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    plt.tight_layout()
    throttle_file = output_path.parent / f"{base_name}_throttle.png"
    plt.savefig(throttle_file, bbox_inches='tight')
    print(f"✅ Throttle timeseries plot saved as '{throttle_file}'")
    plt.close(fig_t)
    
    # Save CTE plot
    fig_c, ax_c = plt.subplots(figsize=(20, 3))
    ax_c.plot(cte, linewidth=1, color=CTE_COLOR, alpha=0.7)
    ax_c.set_xlabel('Step')
    ax_c.set_ylabel('CTE (Cross Track Error)')
    ax_c.grid(True, alpha=0.3)
    ax_c.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    plt.tight_layout()
    cte_file = output_path.parent / f"{base_name}_cte.png"
    plt.savefig(cte_file, bbox_inches='tight')
    print(f"✅ CTE timeseries plot saved as '{cte_file}'")
    plt.close(fig_c)
    
    # Save forward velocity plot
    fig_v, ax_v = plt.subplots(figsize=(20, 3))
    ax_v.plot(forward_vel, linewidth=1, color=VELOCITY_COLOR, alpha=0.7)
    ax_v.set_xlabel('Step')
    ax_v.set_ylabel('Forward Velocity')
    ax_v.grid(True, alpha=0.3)
    ax_v.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    plt.tight_layout()
    velocity_file = output_path.parent / f"{base_name}_forward_velocity.png"
    plt.savefig(velocity_file, bbox_inches='tight')
    print(f"✅ Forward velocity timeseries plot saved as '{velocity_file}'")
    plt.close(fig_v)
    
    # Save reward plot
    fig_r, ax_r = plt.subplots(figsize=(20, 3))
    ax_r.plot(reward, linewidth=1, color=REWARD_COLOR, alpha=0.7)
    ax_r.set_xlabel('Step')
    ax_r.set_ylabel('Reward')
    ax_r.grid(True, alpha=0.3)
    ax_r.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    plt.tight_layout()
    reward_file = output_path.parent / f"{base_name}_reward.png"
    plt.savefig(reward_file, bbox_inches='tight')
    print(f"✅ Reward timeseries plot saved as '{reward_file}'")
    plt.close(fig_r)

def main():
    """Main function."""

    input_file = sys.argv[1] # Evaluation file
    
    # Check if file exists
    if not Path(input_file).exists():
        print(f"❌ Error: File '{input_file}' not found!")
        sys.exit(1)
    
    # Extract data
    print(f"📖 Reading file: {input_file}")
    steering, throttle, cte, forward_vel, reward, inference_times = extract_data(input_file)
    
    if not steering or not throttle or not cte or not forward_vel or not reward or not inference_times:
        print("❌ Error: No valid data found in file!")
        sys.exit(1)
    
    print(f"✅ Extracted data:")
    print(f"   Actions: {len(steering)} steering/throttle pairs")
    print(f"   CTE entries: {len(cte)}")
    print(f"   Forward velocity entries: {len(forward_vel)}")
    print(f"   Reward entries: {len(reward)}")
    print(f"   Inference times: {len(inference_times)}")
    print(f"")
    print(f"   Steering range: [{min(steering):.3f}, {max(steering):.3f}]")
    print(f"   Throttle range: [{min(throttle):.3f}, {max(throttle):.3f}]")
    print(f"   CTE range: [{min(cte):.3f}, {max(cte):.3f}]")
    print(f"   Forward velocity range: [{min(forward_vel):.3f}, {max(forward_vel):.3f}]")
    print(f"   Reward range: [{min(reward):.3f}, {max(reward):.3f}]")
    
    # Print inference time statistics
    if len(inference_times) > 0:
        print(f"\n⚡ Inference Performance Statistics:")
        inference_array = np.array(inference_times)
        print(f"   Average: {np.mean(inference_array):.2f} ms")
        print(f"   Min: {np.min(inference_array):.2f} ms")
        print(f"   Max: {np.max(inference_array):.2f} ms")
    
    # Create output filenames
    action_dist_file = Path(input_file).stem + '_action_distributions.png'
    metric_timeseries_file = Path(input_file).stem + '_metric_timeseries.png'
    
    # Plot and save time series data
    plot_data(steering, throttle, cte, forward_vel, reward, metric_timeseries_file)

    # Plot action distributions
    plot_action_distributions(steering, throttle, action_dist_file)


if __name__ == "__main__":
    main()
