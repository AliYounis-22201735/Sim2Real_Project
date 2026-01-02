#!/usr/bin/env python3
"""
PPO training script for image-based models

Usage Examples:
    # Train new model from scratch
    python image_train.py
    
    # Continue training from saved model
    python image_train.py --continue-training --model-path <model_file>
    
    # Continue training from saved model with specific timesteps
    python image_train.py --continue-training --model-path <model_file> --timesteps <timesteps>
"""
# ============================================================================
# Import Required Libraries and Modules
# ============================================================================

import sys
import os
import time
import datetime
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
import stable_baselines3 as sb3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
import gym
import gym_donkeycar
from gym.wrappers import TimeLimit
import multiprocessing as mp

# Add current directory and parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)

from image_processing_wrapper import ImageProcessingWrapper
from env_flipper import HorizontalFlippingWrapper
from image_processing_config import get_image_processor_training_config, set_training_mode

# ============================================================================
# Add Parent Directories to Path
# ============================================================================

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)

# ============================================================================
# Random Seed Setting Function
# ============================================================================

def set_seed(seed):
    """
    Set random seed for all libraries to ensure reproducibility.

    Args:
        seed (int): Seed number.
    """
    np.random.seed(seed)
    random.seed(seed) 
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"\n✅ Random Seeding Done!")

# ============================================================================
# Device Configuration for PyTorch
# ============================================================================

if torch.cuda.is_available():
    # Use CUDA if available
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    # Use Metal Performance Shaders (MPS) on macOS
    device = torch.device("mps")
else:
    # Fallback to CPU
    device = torch.device("cpu")
print(f"📣  Using {device} device\n")

# ============================================================================
# GPU Optimization Settings
# ============================================================================

# Performance Optimization
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Optimize CPU usage
optimal_threads = min(10, mp.cpu_count() // 2)
torch.set_num_threads(optimal_threads)

# Memory management optimizations
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async CUDA operations
os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'  # Use faster cuDNN v8 API

# ============================================================================
# Multiprocessing Settings
# ============================================================================

try:
    mp.set_start_method('spawn', force=True)
    print("✅  Multiprocessing start method set to 'spawn'\n")
except RuntimeError as e:
    print(f"⚠️  Error in setting multiprocessing start method: {e}")
    pass

# ============================================================================
# Command Line Argument Parser
# ============================================================================

def parse_arguments():
    """Parse command line arguments for training run configuration"""
    parser = argparse.ArgumentParser(
        description='Train PPO model for image-based DonkeyCar control',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train new model from scratch
  python image_train.py
  
  # Continue training from saved model
  python image_train.py --continue-training --model-path <model_file>
  
  # Continue training with additional timesteps
  python image_train.py --continue-training --model-path <model_file> --timesteps <timesteps>
        """
    )
    
    # Continue training arguments
    parser.add_argument(
        '--continue-training', '--continue', '-c',
        action='store_true',
        help='Continue training from a saved model'
    )
    
    parser.add_argument(
        '--model-path', '--model', '-m',
        type=str,
        default='',
        help='Path to the saved model .zip file (required if --continue-training is set)'
    )
    
    parser.add_argument(
        '--timesteps', '-t',
        type=int,
        default=None,
        help='Number of additional timesteps to train (overrides config setting)'
    )
    
    return parser.parse_args()

# ============================================================================
# Training Cycle Configuration Class
# ============================================================================

class TrainingCycleConfig:
    """Training cycle configuration class"""
    
    def __init__(self):
        """Define all training configuration parameters"""

        # Environment Settings
        self.ENV_ID = 'donkey-waveshare-v0'  # Environment ID
        self.NUMBER_OF_ENVS = 1  # Number of parallel environments - set to 1 due to the connection issues with Donkeycar simulator when establishing multiple sessions
        self.HORIZONTAL_FLIPPING_PROBABILITY = 0.0  # Probability of flipping the environment (observation + steering action)

        # DonkeyCar Simulator Settings
        self.MAX_CTE = 2.0  # Maximum allowed cross track error

        # Training Settings
        self.TOTAL_TIMESTEPS = 1000000  # Total timesteps
        self.EVAL_FREQ = 5000  # Model evaluation frequency
        self.NUMBER_OF_EVALS = 5  # Number of evaluation episodes
        self.CHECKPOINT_FREQ = 50000  # Model checkpoint frequency
        self.MAX_EPISODE_STEPS = 1500  # Episode length limit
        
        # Continue Training Settings
        self.CONTINUE_TRAINING = False #Set to True to continue training from a saved model
        self.MODEL_FILE = ""  # Path to the saved model file for continuing training - mandatory if CONTINUE_TRAINING is True

        # Image Processing Settings
        self.STACK_SIZE = 5  # Frame stacking for temporal context

        # PPO Hyperparameters
        self.LEARNING_RATE_INITIAL = 3e-4  # Initial learning rate
        self.LEARNING_RATE_FINAL = 5e-6  # Final learning rate after decay
        self.N_STEPS = 2048  # Number of steps to collect before each policy update
        self.BATCH_SIZE = 128  # Mini-batch size for training
        self.N_EPOCHS = 5  # Number of epochs when optimizing the surrogate loss
        self.GAMMA = 0.99  # Discount factor for future rewards
        self.GAE_LAMBDA = 0.95  # Factor for trade-off of bias vs variance
        self.CLIP_RANGE_INITIAL = 0.3  # Initial clipping range
        self.CLIP_RANGE_FINAL = 0.05  # Final clipping range
        self.ENT_COEF = 0.01  # Entropy coefficient for exploration
        self.VF_COEF = 0.5  # Value function coefficient
        self.MAX_GRAD_NORM = 0.5  # Gradient clipping for stability
        self.USE_SDE = True  # Use generalized State Dependent Exploration (gSDE)
        self.SDE_SAMPLE_FREQ = 5  # Frequency of SDE sampling
        self.TARGET_KL = 0.015  # Target KL divergence for early stopping

        # Model Architecture
        self.PI_NET_ARCH = [256, 64]  # Policy network architecture
        self.VF_NET_ARCH = [256, 64]  # Value network architecture
        self.ORTHO_INIT = True  # Use orthogonal initialization
        self.LOG_STD_INIT = -2.0  # Initial log standard deviation

        # Directory Paths
        self.MODEL_PATH = "./models/ppo_image"  # Directory to save image-based trained models
        self.LOG_PATH = "./logs/ppo_image"  # Directory for saving training logs
        self.CHECKPOINT_PATH = "./checkpoints/ppo_image"  # Directory for saving model checkpoints
        self.TENSORBOARD_PATH = "./tensorboard"  # Directory for saving TensorBoard logs

        # Random Seed
        self.SEED = 42  # Default random seed for reproducibility

    def save_config(self, run_name, description, device, run_path, timesteps, training_duration):
        """Save training cycle configuration to a text file"""
        
        # Ensure the existence of run_path directory
        os.makedirs(run_path, exist_ok=True)

        # Training Parameter Groups
        param_groups = {
            "Environment Settings": [
                "ENV_ID", "NUMBER_OF_ENVS", "HORIZONTAL_FLIPPING_PROBABILITY"
            ],
            "DonkeyCar Simulator Settings": [
                "MAX_CTE"
            ],
            "Training Settings": [
                "TOTAL_TIMESTEPS", "EVAL_FREQ", "NUMBER_OF_EVALS", "CHECKPOINT_FREQ", "MAX_EPISODE_STEPS"
            ],
            "Continue Training Settings": [
                "CONTINUE_TRAINING", "MODEL_FILE"
            ],
            "Image Processing Settings": [
                "STACK_SIZE"
            ],
            "PPO Hyperparameters": [
                "LEARNING_RATE_INITIAL", "LEARNING_RATE_FINAL", "N_STEPS", 
                "BATCH_SIZE", "N_EPOCHS", "GAMMA", "GAE_LAMBDA", 
                "CLIP_RANGE_INITIAL", "CLIP_RANGE_FINAL", "ENT_COEF", 
                "VF_COEF", "MAX_GRAD_NORM", "USE_SDE", "SDE_SAMPLE_FREQ", "TARGET_KL"
            ],
            "Model Architecture": [
                "PI_NET_ARCH", "VF_NET_ARCH", "ORTHO_INIT", "LOG_STD_INIT"
            ],
            "Paths": [
                "MODEL_PATH", "LOG_PATH", "CHECKPOINT_PATH", "TENSORBOARD_PATH"
            ],
            "Random Seed": [
                "SEED"
            ]
        }

        # Save training run configuration to a text file
        config_file = os.path.join(run_path, "training_config.txt")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        # Format training duration as hours, minutes, seconds
        hours = int(training_duration // 3600)
        minutes = int((training_duration % 3600) // 60)
        seconds = int(training_duration % 60)

        with open(config_file, "w") as f:
            f.write(f"Training Configuration - {run_name}\n")
            f.write("=" * 50 + "\n")

            # Write metadata section
            f.write("Training Cycle Metadata:\n")
            f.write("-" * 30 + "\n")
            f.write(f"timestamp: {timestamp}\n")
            f.write(f"run_name: {run_name}\n")
            f.write(f"description: {description}\n")
            f.write(f"device_used: {device}\n")
            f.write(f"trained_timesteps: {timesteps}\n")
            f.write(f"training_duration: {hours}h {minutes}m {seconds}s ({training_duration:.2f} seconds)\n")
            f.write(f"python_version: {sys.version}\n")
            f.write(f"torch_version: {torch.__version__}\n")
            f.write(f"stable_baselines3_version: {sb3.__version__}\n")

            # Write training parameters
            for group_name, params in param_groups.items():
                f.write(f"\n{group_name}:\n")
                f.write("-" * 30 + "\n")
                for param in params:
                    if hasattr(self, param):
                        value = getattr(self, param)
                        # Format large integers with commas, everything else as-is
                        formatted_value = f"{value:,}" if isinstance(value, int) and value >= 1000 else value
                        f.write(f"{param}: {formatted_value}\n")

# Initialize global config instance
cfg = TrainingCycleConfig()

# ============================================================================
# DonkeyCar Simulator Settings
# ============================================================================

sim_conf = {
    "max_cte": cfg.MAX_CTE,
}

# ============================================================================
# Utility Functions
# ============================================================================

def lr_linear_schedule(initial_value=cfg.LEARNING_RATE_INITIAL, final_value=cfg.LEARNING_RATE_FINAL):
    """Learning rate linear schedule"""
    def schedule(progress_remaining):
        return progress_remaining * (initial_value - final_value) + final_value
    return schedule

def clip_range_linear_schedule(initial_value=cfg.CLIP_RANGE_INITIAL, final_value=cfg.CLIP_RANGE_FINAL):
    """Clip range linear schedule"""
    def schedule(progress_remaining):
        return progress_remaining * (initial_value - final_value) + final_value
    return schedule

# ============================================================================
# Environment Creation Function
# ============================================================================

def create_env(env_id=cfg.ENV_ID, n_envs=cfg.NUMBER_OF_ENVS, conf=sim_conf):

    """
    Create and configure vectorized DonkeyCar environment(s)
    
    Args:
        env_id: Environment identifier
        n_envs: Number of parallel environments
        conf: Configuration dictionary for the DonkeyCar simulator environment

    Returns:
        Vectorized environment(s)
    """
    
    def _make_env():
        """Create a single environment with appropriate gym wrappers applied"""
        try:

            # 1. Create base DonkeyCar environment
            env = gym.make(env_id, conf=conf)

            # 2. Apply time limit wrapper
            env = TimeLimit(env, max_episode_steps=cfg.MAX_EPISODE_STEPS)

            # 3. Apply image processing pipeline
            image_config = get_image_processor_training_config()
            env = ImageProcessingWrapper(env,
                                         roi_crop_top=image_config['roi_crop_top'],
                                         target_shape=image_config['target_shape'],
                                         augmentation_probability=image_config['augmentation_probability'],
                                         )

            # 4. Apply horizontal flipping, in case configured
            if cfg.HORIZONTAL_FLIPPING_PROBABILITY > 0.0:
                env = HorizontalFlippingWrapper(env, hflip_prob=cfg.HORIZONTAL_FLIPPING_PROBABILITY)

            # Return the single environment
            return env
            
        except Exception as e:
            print(f"❌ Error creating environment: {e}")
            raise
    
    # 5. Create vectorized environment(s)
    env_manager = [_make_env for _ in range(n_envs)]
    env = DummyVecEnv(env_manager)

    # 6. Apply frame stacking
    env = VecFrameStack(env, n_stack=cfg.STACK_SIZE)

    # 7. Apply image transpose for pytorch (H, W, C) -> (C, H, W)
    env = VecTransposeImage(env)

    # 8. Apply monitoring wrapper
    env = VecMonitor(env)

    # 9. Return the final vectorized environment(s)
    return env

# ============================================================================
# Main Training Function
# ============================================================================

def main():
    """Main training function for image-based models"""
    print("🔄🔄🔄🔄🔄 CALLFLOW: Entering image_train.py - PPO Image Training Pipeline 🔄🔄🔄🔄🔄")

    # Parse command line arguments
    args = parse_arguments()
    
    # Override config with command line arguments if provided
    if args.continue_training:
        cfg.CONTINUE_TRAINING = True
        if args.model_path:
            cfg.MODEL_FILE = args.model_path
        elif not cfg.MODEL_FILE:
            print("\n❌ ERROR: --continue-training requires --model-path argument")
            print("Usage: python image_train.py --continue-training --model-path path/to/model.zip")
            sys.exit(1)
    
    if args.timesteps is not None:
        cfg.TOTAL_TIMESTEPS = args.timesteps
    
    # Display CLI override information if any
    if args.continue_training or args.timesteps is not None:
        print("\n" + "=" * 50)
        if args.continue_training:
            print(f"   🔄 Continue Training: Enabled")
            print(f"   📂 Model Path: {cfg.MODEL_FILE}")
        if args.timesteps is not None:
            print(f"   🐾 Timesteps: {cfg.TOTAL_TIMESTEPS:,}")
        print("=" * 50 + "\n")

    # Set training mode
    set_training_mode(training=True)

    # Get image processing configuration
    image_config = get_image_processor_training_config()

    # Set random seed for reproducibility
    set_seed(cfg.SEED)
    
    # Custom policy architecture
    policy_kwargs = dict(
        net_arch=[
            dict(pi=cfg.PI_NET_ARCH, vf=cfg.VF_NET_ARCH)
        ],
        activation_fn=nn.ReLU,
        ortho_init=cfg.ORTHO_INIT,
        log_std_init=cfg.LOG_STD_INIT,
    )

    # Training setup
    training_start_time = time.time()
    run_time = datetime.datetime.fromtimestamp(training_start_time).strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"ppo_image_{run_time}"
    run_path = f"{cfg.MODEL_PATH}/{run_name}"
    
    # Ensure the existence of necessary directories
    os.makedirs(cfg.LOG_PATH, exist_ok=True)
    os.makedirs(cfg.MODEL_PATH, exist_ok=True)
    os.makedirs(cfg.CHECKPOINT_PATH, exist_ok=True)
    os.makedirs(cfg.TENSORBOARD_PATH, exist_ok=True)

    # Print training configuration
    print(f"\n🚀  Starting DonkeyCar Image Training...")
    print(f"📋  Run: {run_name}")
    print(f"🐾  Total Timesteps: {cfg.TOTAL_TIMESTEPS:,}")
    print(f"🛠️  Number of Environments: {cfg.NUMBER_OF_ENVS}")
    if cfg.STACK_SIZE > 1:
        print(f"📚  Frame Stacking Enabled: {cfg.STACK_SIZE} frames")
    else:
        print(f"📚  Frame Stacking Disabled")
    if image_config['roi_crop_top'] > 0:
        print(f"✂️  Vertical Cropping Enabled: {image_config['roi_crop_top']:.1%} from top")
    else:
        print("✂️  Vertical Cropping Disabled")
    if image_config['augmentation_probability'] > 0:
        print(f"🎨  Augmentation Enabled: {image_config['augmentation_probability']:.1%} probability")
    else:
        print("🎨  Augmentation Disabled")
    if cfg.HORIZONTAL_FLIPPING_PROBABILITY > 0:
        print(f"↔️  Horizontal Flipping Enabled: {cfg.HORIZONTAL_FLIPPING_PROBABILITY:.1%} probability")
    else:
        print(f"↔️  Horizontal Flipping Disabled")
    print(f"🎯  Simulator Max CTE: {cfg.MAX_CTE}")
    print("=" * 50)

    try:
        # Create training environment
        train_env = create_env()
        print(f"✅ {cfg.NUMBER_OF_ENVS} Training environment(s) created successfully")
        print(f"   📊 Observation space: {train_env.observation_space}")
        print(f"   🎮 Action space: {train_env.action_space}")
    except Exception as e:
        print(f"❌ Failed to create training environment(s): {e}")
        return

    # Create evaluation environment - reuse training environment to avoid simulator connection issues
    eval_env = train_env
    print("ℹ️ Using training environment for evaluation (to avoid simulator connection issues when creating multiple environments)")

    # Setup learning rate schedule
    lr_schedule = lr_linear_schedule(cfg.LEARNING_RATE_INITIAL, cfg.LEARNING_RATE_FINAL)

    # Setup clip range schedule
    clip_schedule = clip_range_linear_schedule(cfg.CLIP_RANGE_INITIAL, cfg.CLIP_RANGE_FINAL)

    # Setup callbacks
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=run_path,
        log_path=f"{cfg.LOG_PATH}/{run_name}/",
        eval_freq=cfg.EVAL_FREQ // cfg.NUMBER_OF_ENVS,
        n_eval_episodes=cfg.NUMBER_OF_EVALS,
        deterministic=True,
        render=False,
        verbose=1
    )

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=cfg.CHECKPOINT_FREQ // cfg.NUMBER_OF_ENVS,
        save_path=f"{cfg.CHECKPOINT_PATH}/{run_name}/",
        name_prefix="checkpoint",
        verbose=1
    )

    callbacks = CallbackList([eval_callback, checkpoint_callback])
    
    # Create (fresh training) or load (training continuation) the PPO model
    if cfg.CONTINUE_TRAINING and cfg.MODEL_FILE:
        # Load existing model for continued training
        print("\n" + "=" * 50)
        print("🔄 Continuing Training From Saved Model")
        print("=" * 50)
        print(f"📂 Loading model from: {cfg.MODEL_FILE}")
        
        try:
            ppo_image_model = PPO.load(
                cfg.MODEL_FILE,
                env=train_env,
                device=device,
                # You can optionally override hyperparameters here
                custom_objects={
                    'learning_rate': lr_schedule,
                    'clip_range': clip_schedule,
                }
            )
            print(f"✅ Model loaded successfully!")
            print(f"📊 Current timesteps in loaded model: {ppo_image_model.num_timesteps:,}")
            print(f"🎯 Will train for additional {cfg.TOTAL_TIMESTEPS:,} timesteps")
            print(f"📈 Total timesteps after training: {ppo_image_model.num_timesteps + cfg.TOTAL_TIMESTEPS:,}")
            print("=" * 50 + "\n")
        except Exception as e:
            print(f"❌ Failed to load model from {cfg.MODEL_FILE}")
            print(f"Error: {e}")
            print("⚠️  Creating new model instead...")
            ppo_image_model = PPO(
                policy="CnnPolicy",
                env=train_env,
                verbose=1,
                tensorboard_log=cfg.TENSORBOARD_PATH,
                device=device,
                policy_kwargs=policy_kwargs,
                learning_rate=lr_schedule,
                n_steps=cfg.N_STEPS,
                batch_size=cfg.BATCH_SIZE,
                n_epochs=cfg.N_EPOCHS,
                gamma=cfg.GAMMA,
                gae_lambda=cfg.GAE_LAMBDA,
                clip_range=clip_schedule,
                ent_coef=cfg.ENT_COEF,
                vf_coef=cfg.VF_COEF,
                max_grad_norm=cfg.MAX_GRAD_NORM,
                use_sde=cfg.USE_SDE,
                sde_sample_freq=cfg.SDE_SAMPLE_FREQ,
                target_kl=cfg.TARGET_KL,
            )
    else:
        # Train a new PPO model from scratch
        print("\n" + "=" * 50)
        print("=" * 50 + "\n")
        ppo_image_model = PPO(
            policy="CnnPolicy", # Image-based observations
            env=train_env,
            verbose=1,
            tensorboard_log=cfg.TENSORBOARD_PATH,
            device=device,
            policy_kwargs=policy_kwargs,
            learning_rate=lr_schedule,
            n_steps=cfg.N_STEPS,
            batch_size=cfg.BATCH_SIZE,
            n_epochs=cfg.N_EPOCHS,
            gamma=cfg.GAMMA,
            gae_lambda=cfg.GAE_LAMBDA,
            clip_range=clip_schedule,
            ent_coef=cfg.ENT_COEF,
            vf_coef=cfg.VF_COEF,
            max_grad_norm=cfg.MAX_GRAD_NORM,
            use_sde=cfg.USE_SDE,
            sde_sample_freq=cfg.SDE_SAMPLE_FREQ,
            target_kl=cfg.TARGET_KL,
        )
    
    # Train/ Continue training the PPO model
    try:
        # Start training
        ppo_image_model.learn(
            total_timesteps=cfg.TOTAL_TIMESTEPS,
            callback=callbacks,
            tb_log_name=run_name
        )

        # Save final model
        ppo_image_model.save(f"{run_path}/final_model")

        # Save training cycle summary
        training_duration = time.time() - training_start_time
        cfg.save_config(run_name, "Training completed successfully!", device, run_path, ppo_image_model.num_timesteps, training_duration)

        print("\n" + "=" * 50)
        print("🎉 PPO Image Training Completed Successfully!")
        print(f"⏰  Total Training Time: {training_duration // 3600:.0f}h {(training_duration % 3600) // 60:.0f}m {training_duration % 60:.0f}s")
        print(f"🐾  Timesteps: {ppo_image_model.num_timesteps:,}")
        print(f"💾 Final Model Saved at: {run_path}/final_model.zip")
        print(f"📄 Training Cycle Summary Saved at: {run_path}/training_config.txt")
        print("=" * 50)
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user!")
        try:
            ppo_image_model.save(f"{run_path}/interrupted_model")
            training_duration = time.time() - training_start_time
            cfg.save_config(run_name, "Training interrupted by user!", device, run_path, ppo_image_model.num_timesteps, training_duration)
            print(f"💾 Interrupted model saved at: {run_path}/interrupted_model.zip")
            print(f"📄 Training Cycle Summary Saved at: {run_path}/training_config.txt")
        except Exception as save_error:
            print(f"❌ Could not save interrupted model: {save_error}")

    except Exception as e:
        print(f"\n🚨 Training failed: {e}")
        try:
            ppo_image_model.save(f"{run_path}/failed_model")
            training_duration = time.time() - training_start_time
            cfg.save_config(run_name, f"Training failed due to: {str(e)}", device, run_path, ppo_image_model.num_timesteps, training_duration)
            print(f"💾 Failed model saved at: {run_path}/failed_model.zip")
            print(f"📄 Training Cycle Summary Saved at: {run_path}/training_config.txt")
        except Exception as save_error:
            print(f"❌ Could not save failed model: {save_error}")
        raise
        
    finally:
        # Cleanup
        print("\n🧹  Cleaning up...")
        try:
            train_env.close()
            if eval_env != train_env:
                eval_env.close()
            del train_env
            del eval_env
            torch.cuda.empty_cache()
            print("✅ Cleanup complete")
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")

# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    main()