#!/usr/bin/env python3
"""
CTE Distribution Comparison Script
Plots overlapping histograms of CTE distributions for image-based and vector-based models.

Usage:
    python cte_distribution_comparator.py <image_log_file> <vector_log_file>
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

# ====================================================================================
# LLNCS / LaTeX-Compatible Font Configuration
# ====================================================================================

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.titlesize": 10,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

# ====================================================================================
# Color Configuration
# ====================================================================================

IMAGE_MODEL_COLOR = '#0173B2'    #  blue
VECTOR_MODEL_COLOR = '#DE8F05'   #  orange
CTE_THRESHOLD = 2.0

# ====================================================================================
# CTE Distribution Comparison Class
# ====================================================================================

class CTEDistributionComparison:
    """
    Extracts CTE values from evaluation logs and plots comparative histograms.
    """
    
    def __init__(self, image_log: str, vector_log: str):
        self.image_log = image_log
        self.vector_log = vector_log
        self.image_cte = []
        self.vector_cte = []
        
    def extract_cte_values(self, log_file: str) -> list:
        """
        Extract all CTE values from a log file.
        
        Args:
            log_file: Path to simulator evaluation log file
            
        Returns:
            List of CTE values (as floats)
        """
        cte_values = []
        cte_pattern = re.compile(r'cte:\s*([-\d.]+)')
        
        try:
            with open(log_file, 'r') as f:
                content = f.read()
                matches = cte_pattern.findall(content)
                cte_values = [float(cte) for cte in matches]
                
            print(f"✓ Extracted {len(cte_values)} CTE values from {os.path.basename(log_file)}")
            return cte_values
            
        except IOError as e:
            raise Exception(f"Could not read log file: {e}")
        except Exception as e:
            raise Exception(f"Error parsing CTE values: {e}")
    
    def load_data(self):
        """Load CTE values extracted from both log files."""
        print("📖 Loading data...")
        self.image_cte = self.extract_cte_values(self.image_log)
        self.vector_cte = self.extract_cte_values(self.vector_log)
        print(f"✓ Data loaded successfully\n")
    
    def plot_histograms(self, output_file: str = "cte_distribution_comparison.png"):
        """
        Plot overlapping histograms of CTE distributions.
        
        Args:
            output_file: Output filename for the plot
        """
        if not self.image_cte or not self.vector_cte:
            print("❌ No data available for plotting!")
            return
        
        # Use normal CTE values (positive and negative)
        image_cte_np = np.array(self.image_cte)
        vector_cte_np = np.array(self.vector_cte)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Determine common bin range for fair comparison
        min_cte = min(np.min(image_cte_np), np.min(vector_cte_np))
        max_cte = max(np.max(image_cte_np), np.max(vector_cte_np))
        bins = np.linspace(min_cte, max_cte, 50)  # Full range, no clipping
        
        # Calculate weights for percentage conversion
        image_weights = np.ones_like(image_cte_np) / len(image_cte_np) * 100
        vector_weights = np.ones_like(vector_cte_np) / len(vector_cte_np) * 100
        
        # Plot overlapping histograms
        ax.hist(image_cte_np, bins=bins, weights=image_weights, alpha=0.6, 
                color=IMAGE_MODEL_COLOR, label='Image-Based Model', 
                edgecolor='black', linewidth=0.5)
        ax.hist(vector_cte_np, bins=bins, weights=vector_weights, alpha=0.6, 
                color=VECTOR_MODEL_COLOR, label='Vector-Based Model', 
                edgecolor='black', linewidth=0.5)
        
        # Add vertical lines at thresholds (positive and negative)
        ax.axvline(CTE_THRESHOLD, color='red', linestyle='--', linewidth=2, 
                   label=f'Threshold (±{CTE_THRESHOLD})', alpha=0.8)
        ax.axvline(-CTE_THRESHOLD, color='red', linestyle='--', linewidth=2, alpha=0.8)
        ax.axvline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
        
        # Labels and title
        ax.set_xlabel('CTE')
        ax.set_ylabel('Percentage (%)')
        # ax.set_title('CTE Distribution Comparison: Image-Based vs Vector-Based Models', pad=10)
        
        # Grid for readability
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        
        # Legend
        ax.legend(loc='upper right', framealpha=0.9)
        
        # Statistics with detailed CTE range breakdown
        image_cte_abs = np.abs(image_cte_np)
        vector_cte_abs = np.abs(vector_cte_np)
        
        # Calculate percentages for different CTE ranges
        image_pct_0 = 100 * np.mean(image_cte_abs <= 0.0)
        image_pct_below_25 = 100 * np.mean(image_cte_abs < 0.25)
        image_pct_below_50 = 100 * np.mean(image_cte_abs < 0.5)
        image_pct_below_100 = 100 * np.mean(image_cte_abs < 1.0)
        image_pct_below_200 = 100 * np.mean(image_cte_abs < 2.0)
        image_pct_below_thresh = 100 * np.mean(image_cte_abs < CTE_THRESHOLD)
        image_pct_above_thresh = 100 * np.mean(image_cte_abs > CTE_THRESHOLD)
        
        vector_pct_0 = 100 * np.mean(vector_cte_abs <= 0.0)
        vector_pct_below_25 = 100 * np.mean(vector_cte_abs < 0.25)
        vector_pct_below_50 = 100 * np.mean(vector_cte_abs < 0.5)
        vector_pct_below_100 = 100 * np.mean(vector_cte_abs < 1.0)
        vector_pct_below_200 = 100 * np.mean(vector_cte_abs < 2.0)
        vector_pct_below_thresh = 100 * np.mean(vector_cte_abs < CTE_THRESHOLD)
        vector_pct_above_thresh = 100 * np.mean(vector_cte_abs > CTE_THRESHOLD)
        
        # Print statistics to console
        print("\n📊 CTE Statistics:")
        print("="*60)
        print("Image Model:")
        print(f"  Mean CTE: {np.mean(image_cte_np):.3f}")
        print(f"  Std Dev: {np.std(image_cte_np):.3f}")
        print(f"  |CTE| = 0: {image_pct_0:.1f}%")
        print(f"  |CTE| < 0.25: {image_pct_below_25:.1f}%")
        print(f"  |CTE| < 0.5: {image_pct_below_50:.1f}%")
        print(f"  |CTE| < 1.0: {image_pct_below_100:.1f}%")
        print(f"  |CTE| < 2.0: {image_pct_below_200:.1f}%")
        print(f"  |CTE| < CTE_THRESHOLD: {image_pct_below_thresh:.1f}%")
        print(f"  |CTE| > CTE_THRESHOLD: {image_pct_above_thresh:.1f}%")
        print()
        print("Vector Model:")
        print(f"  Mean CTE: {np.mean(vector_cte_np):.3f}")
        print(f"  Std Dev: {np.std(vector_cte_np):.3f}")
        print(f"  |CTE| = 0: {vector_pct_0:.1f}%")
        print(f"  |CTE| < 0.25: {vector_pct_below_25:.1f}%")
        print(f"  |CTE| < 0.5: {vector_pct_below_50:.1f}%")
        print(f"  |CTE| < 1.0: {vector_pct_below_100:.1f}%")
        print(f"  |CTE| < 2.0: {vector_pct_below_200:.1f}%")
        print(f"  |CTE| < CTE_THRESHOLD: {vector_pct_below_thresh:.1f}%")
        print(f"  |CTE| > CTE_THRESHOLD: {vector_pct_above_thresh:.1f}%")
        print("="*60)
        
        # Tight layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved histogram to: {output_file}")
        
        plt.close()

# ====================================================================================
# Main Function
# ====================================================================================

def main():
    """Main function with command line argument support."""
    
    parser = argparse.ArgumentParser(
        description="Compare CTE distributions between image-based and vector-based models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "image_log",
        type=str,
        help="Path to image-based model evaluation log file"
    )
    
    parser.add_argument(
        "vector_log",
        type=str,
        help="Path to vector-based model evaluation log file"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="cte_distribution_comparison.png",
        help="Output filename for the histogram plot"
    )
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.image_log):
        print(f"❌ Error: Image log file '{args.image_log}' does not exist!")
        sys.exit(1)
    
    if not os.path.exists(args.vector_log):
        print(f"❌ Error: Vector log file '{args.vector_log}' does not exist!")
        sys.exit(1)
    
    print(f"📊 CTE Distribution Comparison")
    print(f"{'='*60}")
    print(f"Image Model Log:  {args.image_log}")
    print(f"Vector Model Log: {args.vector_log}")
    print(f"Output File:      {args.output}")
    print(f"{'='*60}\n")
    
    try:
        # Create comparator and process
        cte_comparison = CTEDistributionComparison(args.image_log, args.vector_log)
        cte_comparison.load_data()
        cte_comparison.plot_histograms(args.output)
        
        print(f"\n✓ Comparison complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
