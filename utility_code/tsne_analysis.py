#!/usr/bin/env python3
"""
t-SNE Comparison Analysis - compare model input data observations from evaluations in the DonkeyCar simulator and the real world.
The script performs t-SNE dimensionality reduction and creates visualizations to compare the distribution of input observations between the two environments.

Usage Examples:
    # Basic usage with default settings
    python tsne_analysis.py <sim_data.npz> <real_data.npz>
    
    # Custom output file
    python tsne_analysis.py <sim_data.npz> <real_data.npz> -o <custom_plot.png>
    
    # Customized t-SNE parameters
    python tsne_analysis.py <sim_data.npz> <real_data.npz> --perplexity <Perplexity>  --n-iter <N_Iterations>
        
Command-line Arguments:
    simulator_file        Path to simulator t-SNE data file (.npz)
    realworld_file        Path to real-world t-SNE data file (.npz)
    -o, --output          Output plot filename (default: tsne_analysis.png)
    --perplexity          t-SNE perplexity parameter (default: 50)
    --n-iter              Number of t-SNE iterations (default: 5000)
    --max-samples         Maximum samples to use per source (default: 2000)
    --random-state        Random seed for reproducibility (default: 42)
    --pca-components      Number of PCA components before t-SNE (default: 50)
    --learning-rate       Learning rate (default: auto)
    --early-exaggeration  Early exaggeration factor (default: 12)
    --n-jobs              Parallel jobs (default: -1)

Output:
    - 2D visualization showing t-SNE projection of simulator and real-world observations
"""

# ============================================================================
# Import Required Libraries and Modules
# ============================================================================

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# ============================================================================
# LLNCS / LaTeX-Compatible Font Configuration
# ============================================================================

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.titlesize": 10,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

# Set style for better-looking plots
sns.set_style("whitegrid")

# ============================================================================
# Constants
# ============================================================================

DEFAULT_OUTPUT_FILE = 'tsne_analysis.png'
DEFAULT_PERPLEXITY = 50
DEFAULT_N_ITER = 5000
DEFAULT_MAX_SAMPLES = 2000
DEFAULT_RANDOM_STATE = 42
DEFAULT_USE_PCA = True
DEFAULT_PCA_COMPONENTS = 50
DEFAULT_LEARNING_RATE = 'auto'
DEFAULT_EARLY_EXAGGERATION = 12
DEFAULT_N_JOBS = -1
TSNE_COMPONENTS = 2

# ============================================================================
# Helper Functions
# ============================================================================

def load_tsne_data(filepath):
    """
    Load t-SNE data from .npz file.
    
    Args:
        filepath: Path to the .npz file
        
    Returns:
        dict: Dictionary containing model's input observations
    """
    # Verify if the file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ File not found: {filepath}")
    
    file_data = np.load(filepath)
    
    observations = file_data['samples']

    tsne_data = {
        'observations': observations,
    }
    
    print(f"📂 Loaded {filepath}")
    print(f"   • Observations shape: {tsne_data['observations'].shape}")
    
    return tsne_data

# ============================================================================
# t-SNE Analysis and Visualization
# ============================================================================

def run_tsne_analysis(simulator_file, realworld_file, output_file=DEFAULT_OUTPUT_FILE,
                       perplexity=DEFAULT_PERPLEXITY, n_iter=DEFAULT_N_ITER, max_samples=DEFAULT_MAX_SAMPLES, 
                       random_state=DEFAULT_RANDOM_STATE, use_pca=DEFAULT_USE_PCA, pca_components=DEFAULT_PCA_COMPONENTS,
                       learning_rate=DEFAULT_LEARNING_RATE, early_exaggeration=DEFAULT_EARLY_EXAGGERATION, n_jobs=DEFAULT_N_JOBS):
    """
    Load data, perform t-SNE analysis, and plot comparison visualization.
        
    Args:
        simulator_file: Path to simulator data file
        realworld_file: Path to real-world data file
        output_file: Output plot filename
        perplexity: t-SNE perplexity parameter
        n_iter: Number of t-SNE iterations
        max_samples: Maximum samples to use per source
        random_state: Random seed for reproducibility
        use_pca: Whether to apply PCA preprocessing
        pca_components: Number of PCA components to keep
        learning_rate: Learning rate
        early_exaggeration: Early exaggeration factor
        n_jobs: Number of parallel jobs

    Returns:
        dict: Results including embeddings, statistics, and output file path
    """
    print("🔬 Starting t-SNE analysis...")
    print("=" * 60)
    
    # Load data
    sim_data = load_tsne_data(simulator_file)
    real_data = load_tsne_data(realworld_file)

    # Debugging
    print(f"🔍 Simulator data shape: {sim_data['observations'].shape[1]}")
    print(f"🔍 Real-world data shape: {real_data['observations'].shape[1]}")

    # Verify compatibility
    if sim_data['observations'].shape[1] != real_data['observations'].shape[1]:
        raise ValueError(
            f"Observation dimensions don't match! "
            f"Simulator: {sim_data['observations'].shape[1]}, "
            f"Real-world: {real_data['observations'].shape[1]}"
        )
    
    # Extract observations
    sim_obs = sim_data['observations']
    real_obs = real_data['observations']

    # Determine the target sample size (minimum of max_samples and smallest dataset)
    min_dataset_size = min(len(sim_obs), len(real_obs))
    target_samples = min(max_samples, min_dataset_size)
    
    print(f"\n📊 Balancing datasets:")
    print(f"   • Simulator: {len(sim_obs)} observations")
    print(f"   • Real-world: {len(real_obs)} observations")
    print(f"   • Target samples per dataset: {target_samples}")

    # Subsample both datasets to the same size for fair comparison
    if len(sim_obs) > target_samples:
        indices = np.random.choice(len(sim_obs), target_samples, replace=False)
        sim_obs = sim_obs[indices]
        print(f"   ✂️  Subsampled simulator data: {len(sim_data['observations'])} → {target_samples}")
    
    if len(real_obs) > target_samples:
        indices = np.random.choice(len(real_obs), target_samples, replace=False)
        real_obs = real_obs[indices]
        print(f"   ✂️  Subsampled real-world data: {len(real_data['observations'])} → {target_samples}")
    
    # Combine data for joint t-SNE
    combined_obs = np.vstack([sim_obs, real_obs])
    labels = np.array(['Simulator'] * len(sim_obs) + ['Real-world'] * len(real_obs))
    
    print(f"\n Data Preprocessing...")
    print(f"   Combined data shape: {combined_obs.shape}")
    # Debugging
    print(f"   Original dimensions: {combined_obs.shape[1]}")
    
    # Standardize features
    print(f"Standardizing features...")
    scaler = StandardScaler()
    combined_obs_scaled = scaler.fit_transform(combined_obs)
    
    # Apply PCA preprocessing
    data_for_tsne = combined_obs_scaled
    if use_pca and combined_obs.shape[1] > pca_components:
        print(f"Applying PCA to reduce {combined_obs.shape[1]} dims → {pca_components} dims...")
        pca = PCA(n_components=pca_components, random_state=random_state)
        data_for_tsne = pca.fit_transform(combined_obs_scaled)
        explained_var = pca.explained_variance_ratio_.sum()
        print(f"✅ PCA complete - Explained variance: {explained_var:.2%}")
    elif use_pca:
        print(f"Skipping PCA (data already has {combined_obs.shape[1]} dims ≤ {pca_components} PCA components)")
    else:
        print(f"PCA disabled by user")
    
    # Apply t-SNE with optimal parameters
    print(f"\n🔄 Running t-SNE Analysis...")
    print(f"   Parameters:")
    print(f"   • Perplexity: {perplexity}")
    print(f"   • Iterations: {n_iter}")
    print(f"   • Learning rate: {learning_rate}")
    print(f"   • Early exaggeration: {early_exaggeration}")
    print(f"   • Initialization: pca")
    print(f"   • Metric: euclidean")
    print(f"   • Parallel jobs: {n_jobs}")
    
    tsne = TSNE(
        n_components=TSNE_COMPONENTS,
        perplexity=perplexity,
        max_iter=n_iter,
        learning_rate=learning_rate,
        early_exaggeration=early_exaggeration,
        init='pca',
        metric='euclidean',
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=1
    )
    embeddings = tsne.fit_transform(data_for_tsne)
    print(f"✅ t-SNE analysis complete!")
    
    # Split embeddings back
    sim_embeddings = embeddings[:len(sim_obs)]
    real_embeddings = embeddings[len(sim_obs):]
        
    # Create visualization
    print("🎨 Creating visualization...")
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Colour Settings
    SIM_COLOR = '#0173B2'      # Blue - for simulator
    REAL_COLOR = '#DE8F05'     # Orange - for real-world
    
    # Plot: Scatter plot with both distributions
    scatter1 = ax.scatter(
        sim_embeddings[:, 0], sim_embeddings[:, 1],
        c=SIM_COLOR, alpha=0.5, s=20, label='Simulator Observations', edgecolors='none'
    )
    scatter2 = ax.scatter(
        real_embeddings[:, 0], real_embeddings[:, 1],
        c=REAL_COLOR, alpha=0.5, s=20, label='Real-world Observations', edgecolors='none'
    )
        
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    # ax.set_title('t-SNE: Simulator vs Real-world Model Inputs', fontsize=14, fontweight='bold')
    
    # Remove tick labels - t-SNE coordinates are arbitrary, only relative positions matter
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"💾 Saved visualization to: {output_file}")
    
    # Show plot
    plt.show()
    
    print("\n✅ t-SNE comparison analysis complete!")
    
    return {
        'sim_embeddings': sim_embeddings,
        'real_embeddings': real_embeddings,
        'output_file': output_file
    }


# ============================================================================
# Command Line Interface
# ============================================================================

def main():
    """
    Main function to handle command-line execution of t-SNE comparison.
    """
    parser = argparse.ArgumentParser(
        description="Compare model data inputs between simulated and real-world domains using t-SNE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "simulator_file",
        type=str,
        help="Path to simulator t-SNE data file (.npz)"
    )
    
    parser.add_argument(
        "realworld_file",
        type=str,
        help="Path to real-world t-SNE data file (.npz)"
    )
    
    # Optional arguments
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="tsne_analysis.png",
        help="Output plot filename"
    )
    
    parser.add_argument(
        "--perplexity",
        type=int,
        default=DEFAULT_PERPLEXITY,
        help="t-SNE perplexity parameter"
    )
    
    parser.add_argument(
        "--n-iter",
        type=int,
        default=DEFAULT_N_ITER,
        help="Number of t-SNE iterations"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help="Maximum samples to use per domain"
    )
    
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--no-pca",
        action="store_true",
        help="Disable PCA preprocessing (not recommended for high-dimensional data)"
    )
    
    parser.add_argument(
        "--pca-components",
        type=int,
        default=DEFAULT_PCA_COMPONENTS,
        help="Number of PCA components to keep before t-SNE (default: 50)"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=str,
        default=DEFAULT_LEARNING_RATE,
        help="t-SNE learning rate"
    )
    
    parser.add_argument(
        "--early-exaggeration",
        type=float,
        default=DEFAULT_EARLY_EXAGGERATION,
        help="t-SNE early exaggeration factor (default: 12)"
    )
    
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=DEFAULT_N_JOBS,
        help="Number of parallel jobs for t-SNE (-1 for all CPU cores)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.simulator_file):
        print(f"❌ Error: Simulator file '{args.simulator_file}' does not exist!")
        sys.exit(1)
    
    if not os.path.exists(args.realworld_file):
        print(f"❌ Error: Real-world file '{args.realworld_file}' does not exist!")
        sys.exit(1)
    
    # Parse learning rate (can be 'auto' or numeric)
    learning_rate = args.learning_rate
    if learning_rate != 'auto':
        try:
            learning_rate = float(learning_rate)
        except ValueError:
            print(f"⚠️  Invalid learning rate '{args.learning_rate}', using 'auto'")
            learning_rate = 'auto'
    
    # Print configuration
    print("🔬 t-SNE Tool Configuration")
    print("=" * 60)
    print(f"📁 Simulator data: {args.simulator_file}")
    print(f"📁 Real-world data: {args.realworld_file}")
    print(f"📊 Output file: {args.output}")
    print(f"\n⚙️  t-SNE Parameters:")
    print(f"   • Perplexity: {args.perplexity}")
    print(f"   • Iterations: {args.n_iter}")
    print(f"   • Learning rate: {learning_rate}")
    print(f"   • Early exaggeration: {args.early_exaggeration}")
    print(f"   • Parallel jobs: {args.n_jobs}")
    print(f"\n📊 Preprocessing:")
    print(f"   • PCA enabled: {not args.no_pca}")
    if not args.no_pca:
        print(f"   • PCA components: {args.pca_components}")
    print(f"   • Max samples per source: {args.max_samples}")
    print(f"   • Random state: {args.random_state}")
    print("=" * 60 + "\n")
    
    try:
        # Run t-SNE comparison
        run_tsne_analysis(
            simulator_file=args.simulator_file,
            realworld_file=args.realworld_file,
            output_file=args.output,
            perplexity=args.perplexity,
            n_iter=args.n_iter,
            max_samples=args.max_samples,
            random_state=args.random_state,
            use_pca=not args.no_pca,
            pca_components=args.pca_components,
            learning_rate=learning_rate,
            early_exaggeration=args.early_exaggeration,
            n_jobs=args.n_jobs
        )
        
        print("\n🎯 t-SNE Analysis Completed!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n⚠️  t-sne Analysis interrupted by user!")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during T-sne analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()