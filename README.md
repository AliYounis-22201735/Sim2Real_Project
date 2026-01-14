# Sim2Real_Project

A public repository for the code used in the thesis: The Effect of Model Input Representation on Sim2Real Transferability: A Comparative Analysis

---

## CLI Reference for Training, Evaluation, and Visualization Scripts

---

### Table of Contents

- [Training Scripts](#training-scripts)
  - [Image-Based Model Training](#image-based-model-training)
  - [Vector-Based Model Training](#vector-based-model-training)
- [Evaluation Scripts](#evaluation-scripts)
  - [Simulator Evaluation](#simulator-evaluation)
  - [Jetson Nano Evaluation](#jetson-nano-evaluation)
- [Visualization Scripts](#visualization-scripts)
  - [Simulator Evaluation Visualizer](#simulator-evaluation-visualizer)
  - [Jetson Evaluation Visualizer](#jetson-evaluation-visualizer)
  - [Car Path Visualizer](#car-path-visualizer)
  - [CTE Distribution Comparison](#cte-distribution-comparison)
  - [t-SNE Analysis](#t-sne-analysis)

---

### Training Scripts

#### Image-Based Model Training

**Script:** `core_code/image_train.py`

**Description:** Train PPO models using image observations from the DonkeyCar simulator.

**Usage Examples:**

```bash
# Train a new model from scratch
python core_code/image_train.py

# Continue training from a saved model
python core_code/image_train.py --continue-training --model-path <model_path>

# Continue training with specific timesteps
python core_code/image_train.py --continue-training --model-path <model_path> --timesteps <timesteps>
```

**Command-line Arguments:**
- `--continue-training`: Resume training from an existing model
- `--model-path`: Path to the saved model file
- `--timesteps`: Number of training timesteps (optional)

---

#### Vector-Based Model Training

**Script:** `core_code/vec_train.py`

**Description:** Train PPO models using vectorized observations (edge detection features) from the DonkeyCar simulator.

**Usage Examples:**

```bash
# Train a new model from scratch
python core_code/vec_train.py

# Continue training from a saved model
python core_code/vec_train.py --continue-training --model-path <model_path>

# Continue training with specific timesteps
python core_code/vec_train.py --continue-training --model-path <model_path> --timesteps <timesteps>
```

**Command-line Arguments:**
- `--continue-training`: Resume training from an existing model
- `--model-path`: Path to the saved model file
- `--timesteps`: Number of training timesteps (optional)

---

### Evaluation Scripts

#### Simulator Evaluation

**Script:** `core_code/model_evaluator_sim.py`

**Description:** Evaluate trained PPO models in the DonkeyCar simulation environment. Supports both image-based and vector-based models.

**Usage Examples:**

```bash
# Standard evaluation with default settings (10 episodes, 1000 max steps per episode)
python core_code/model_evaluator_sim.py <model_path>

# Customized evaluation with specific episodes and step limits
python core_code/model_evaluator_sim.py <model_path> -e <num_episodes> -s <max_steps>
```

**Command-line Arguments:**
- `model_path` (required): Path to the PPO model file (.zip)
- `-e, --episodes`: Number of evaluation episodes (default: 10)
- `-s, --max-steps`: Maximum steps per episode (default: 1000)

---

#### Jetson Nano Evaluation

**Script:** `core_code/model_evaluator_jetson.py`

**Description:** Evaluate trained PPO models on the Jetson Nano robot car. Integrates with DonkeyCar's `manage.py` infrastructure to control the car driving loop.

**Usage Examples:**

```bash
# Standard evaluation with default settings (5 episodes, 300 max steps)
python core_code/model_evaluator_jetson.py --model <model_path>

# Customized evaluation with specific episodes and step limits
python core_code/model_evaluator_jetson.py --model <model_path> --episodes <num_episodes> --max-steps <max_steps>

# Use custom DonkeyCar configuration file (In case need to chnage any of the included parameters)
python core_code/model_evaluator_jetson.py --model <model_path> --myconfig <custom_config.py>
```

**Command-line Arguments:**
- `--model` (required): Path to the PPO model file (.zip)
- `--episodes`: Number of evaluation episodes (default: 5)
- `--max-steps`: Maximum steps per episode (default: 300)
- `--myconfig`: Custom DonkeyCar configuration file (default: myconfig.py)

---

### Visualization Scripts

#### Simulator Evaluation Visualizer

**Script:** `utility_code/sim_evaluation_visualizer.py`

**Description:** Extract and visualize key metrics from simulator evaluation log files.

**Usage:**

```bash
python utility_code/sim_evaluation_visualizer.py <simulator_evaluation_log_file>
```

---

#### Jetson Evaluation Visualizer

**Script:** `utility_code/jetson_evaluation_visualizer.py`

**Description:** Visualize real-world evaluation data from Jetson Nano runs.

**Usage:**

```bash
python utility_code/jetson_evaluation_visualizer.py <jetson_log_file>
```

---

#### Car Path Visualizer

**Script:** `utility_code/car_path_visualizer.py`

**Description:** Visualize the car's path trajectories during simulator evaluation with CTE-based color gradients.

**Usage:**

```bash
python utility_code/car_path_visualizer.py <simulator_evaluation_log_file>
```

---

#### CTE Distribution Comparison

**Script:** `utility_code/cte_distribution_comparison.py`

**Description:** Compare CTE distributions between image-based and vector-based models. Plots overlapping histograms for comparative analysis.

**Usage:**

```bash
python utility_code/cte_distribution_comparison.py <image_model_log_file> <vector_model_log_file>
```

---

#### t-SNE Analysis

**Script:** `utility_code/tsne_analysis.py`

**Description:** Compare model input data observations from evaluations in the DonkeyCar simulator and the real world using t-SNE dimensionality reduction.

**Usage Examples:**

```bash
# Basic usage with default settings
python utility_code/tsne_analysis.py <sim_data.npz> <real_data.npz>

# Custom output file
python utility_code/tsne_analysis.py <sim_data.npz> <real_data.npz> -o <custom_plot.png>

# Customized t-SNE parameters
python utility_code/tsne_analysis.py <sim_data.npz> <real_data.npz> --perplexity 50 --n-iter 5000
```

**Command-line Arguments:**
- `simulator_file` (required): Path to simulator t-SNE data file (.npz)
- `realworld_file` (required): Path to real-world t-SNE data file (.npz)
- `-o, --output`: Output plot filename (default: tsne_analysis.png)
- `--perplexity`: t-SNE perplexity parameter (default: 50)
- `--n-iter`: Number of t-SNE iterations (default: 5000)
- `--max-samples`: Maximum samples to use per domain (default: 2000)
- `--random-state`: Random seed for reproducibility (default: 42)
- `--no-pca`: Disable PCA preprocessing (not recommended for high-dimensional data)
- `--pca-components`: Number of PCA components before t-SNE (default: 50)
- `--learning-rate`: Learning rate (default: auto)
- `--early-exaggeration`: Early exaggeration factor (default: 12)
- `--n-jobs`: Parallel jobs, -1 for all CPU cores (default: -1)

---

### Notes

- All model paths should point to `.zip` files saved by Stable-Baselines3.
- Log files are automatically generated during evaluation runs.
- Visualization scripts require corresponding evaluation log files as input.