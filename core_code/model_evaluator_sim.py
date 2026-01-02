#!/usr/bin/env python3
"""
Model evaluation script - Evaluate PPO trained models in the DonkeyCar simulation environment
Evaluates both image-based and vector-based models, logging detailed step information and episode statistics.
Uses lap completion and lap times as the primary performance metrics.

Usage Examples:
    # Standard evaluation with default settings
    python model_evaluator_sim.py <model_path>
    
    # Customized evaluation with configurable number of episodes and max steps per episode
    python model_evaluator_sim.py <model_path> -e <num_episodes> -s <max_steps>
    
Command-line Arguments:
    model_path             Path to the PPO model file
    -e, --episodes         Number of episodes to run the evaluation for (defaults to 10)
    -s, --max-steps        Maximum steps per episode (defaults to 1000)
"""
# ============================================================================
# Import Required Libraries and Modules
# ============================================================================

import os
import sys  
import time
import datetime
import argparse
import numpy as np
from stable_baselines3 import PPO

# Add parent directory to sys.path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from image_processing_config import set_training_mode
from image_train import create_env as create_image_env
from vec_train import create_env as create_vector_env
from model_detector import detect_model

# ============================================================================
# Constants and Parameters
# ============================================================================

# Evaluation Settings
N_EPISODES = 10
EPISODE_MAX_STEPS = 1000
LAP_MIN_STEPS = 20
POSITION_THRESHOLD = 0.5
LAP_DISTANCE_MULTIPLIER = 2  # A multiplier for minimum distance traveled before counting a lap

# t-SNE Data Collection
TSNE_SAMPLE_RATE = 10  # Collect every 10th observation

# Environment Settings
ENV_ID = "donkey-waveshare-v0"
NUMBER_OF_ENVS = 1  # Number of parallel environments

# DonkeyCar Simulator Settings
MAX_CTE = 2.0  # Maximum allowed cross track error

# Simulator configuration dictionary
SIM_CONF = {
    "max_cte": MAX_CTE,
}

# ============================================================================
# Observation Logging Function
# ============================================================================

def log_observation(f, obs, episode_num, step_num):
    """Log raw observation with full statistics"""        
    f.write(f"\n{'='*80}\n")
    f.write(f"Observation - Episode {episode_num} - Step {step_num}\n")
    f.write(f"{'='*80}\n")
    f.write(f"Type:  {type(obs).__name__}\n")
    f.write(f"Shape: {obs.shape}\n")
    f.write(f"Dtype: {obs.dtype}\n")
    f.write(f"\nStatistics:\n")
    f.write(f"  Mean:     {np.mean(obs):.3f}\n")
    f.write(f"  Std Dev:  {np.std(obs):.3f}\n")
    f.write(f"  Min:      {np.min(obs):.3f}\n")
    f.write(f"  Max:      {np.max(obs):.3f}\n")
    f.write(f"  Median:   {np.median(obs):.3f}\n")
    f.write(f"\nObservation Array:\n")
    f.write(f"{repr(obs)}\n\n")

# ============================================================================
# Model Evaluation Function
# ============================================================================

def evaluate_model(model_path, n_episodes=N_EPISODES, max_steps_per_episode=EPISODE_MAX_STEPS, 
                   position_threshold=POSITION_THRESHOLD, min_steps_per_lap=LAP_MIN_STEPS):
    """
    Model evaluation: Run episodes and count successful laps and lap times per episode.
    Logs complete step information to a text file.
    
    Args:
        model_path: Path to the PPO model to evaluate
        n_episodes: Number of episodes to evaluate the model for
        max_steps_per_episode: Maximum steps allowed per episode
        position_threshold: Distance threshold (from starting position) for detecting lap completion
        min_steps_per_lap: Minimum steps from the start position before a lap can be counted
        
    Returns:
        dict: Episode-by-episode evaluation statistics
    """

    # Set evaluation mode (turn off training)
    set_training_mode(False)

    # Initialize episode-level statistics
    episode_stats = []
    eval_start_time = time.time()
    total_laps = 0
    naturally_completed_count = 0  # Track episodes completed naturally

    # Initialize t-SNE data collection parameters
    tsne_collected_samples = []
    tsne_counter = 0

    # Load the PPO model and detect its type and configuration
    try:
        model = PPO.load(model_path)
        model_type, _, _ = detect_model(model_path)
        print(f"🚗 Loaded {model_type} model from {model_path}.")
    except Exception as e:
        print(f"❌ Model loading and detection failed - evaluation aborted: {e}")
        return

    if model_type == 'image':
        env = create_image_env(env_id=ENV_ID, n_envs=NUMBER_OF_ENVS, conf=SIM_CONF)
    elif model_type == 'vector':
        env = create_vector_env(env_id=ENV_ID, n_envs=NUMBER_OF_ENVS, conf=SIM_CONF)
    else:
        raise ValueError(f"❌ Unsupported model type: {model_type}. Must be 'image' or 'vector'.")
    
    # Create an evaluation log file with its name extracted from the training cycle folder name
    model_filename = os.path.basename(os.path.dirname(model_path))
    log_file = f"simulator_evaluation_log_{model_filename}.txt"
    
    print(f"🏁 Starting model evaluation in simulator for {n_episodes} episodes...")
    print(f"⏱️ Max steps per episode: {max_steps_per_episode}")
    print(f"📍 Position threshold: {position_threshold:.1f}")
    print(f"📝 Model evaluation logs saved to: {log_file}")
    print("=" * 60)
    
    # Open log file for writing
    with open(log_file, 'w') as f:
        f.write(f"Episode Logging - {datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}\n")
        f.write(f"Evaluation Episodes: {n_episodes}, Max Steps per Episode: {max_steps_per_episode}, Position Threshold: {position_threshold}, Min Steps per Lap: {min_steps_per_lap}\n")
        f.write("=" * 60 + "\n\n")

        for episode_num in range(1, n_episodes + 1):
            print(f"\n🏎️  Starting Episode {episode_num}/{n_episodes}...")
            f.write(f"\n{'=' * 30} Episode {episode_num}/{n_episodes} {'=' * 30}\n")
            f.write(f"Episode Start Time: {datetime.datetime.now().strftime('%H:%M:%S')}\n")

            # Reset t-SNE counter
            tsne_counter = 0

            # record episode start time
            episode_start_time = time.time()

            # Reset environment for each new episode
            obs = env.reset()
            
            # Log reset observation (step 0)
            log_observation(f, obs, episode_num=episode_num, step_num=0)

            # Initialize episode tracking
            episode_steps = 0
            done = False
            last_lap_time = episode_start_time
            episode_reward = 0.0
            lap_count = 0
            lap_times = []
            max_distance_reached = 0.0
            positions_history = []
            start_x, start_y, start_z = None, None, None  # Starting position - will be set on the first step
            inference_times = []  # Track inference time for each step
            
            # Run the current episode
            while episode_steps < max_steps_per_episode and not done:
                # Measure step inference time
                inference_start = time.time()
                action, _ = model.predict(obs, deterministic=True)
                inference_time = time.time() - inference_start
                inference_times.append(inference_time)
                
                # Collect observations for t-SNE analysis
                if tsne_counter % TSNE_SAMPLE_RATE == 0:
                    # Store a flattened copy of the observation
                    tsne_collected_samples.append(obs.flatten().copy())
                tsne_counter += 1 # Will continue to increment each step, till we reach max steps per episode
                
                # Step through the environment
                obs, reward, done, info = env.step(action)
                episode_steps += 1
                episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward
                
                # Log the observation for this step
                log_observation(f, obs, episode_num=episode_num, step_num=episode_steps)
                
                # Get current position and log complete info dictionary
                if not info or len(info) == 0:
                    raise ValueError(f"❌ No position information at episode {episode_num}, step {episode_steps}!")
                step_info = info[0] if isinstance(info, list) else info
                current_pos = step_info['pos']
                current_x, current_y, current_z = current_pos[0], current_pos[1], current_pos[2]

                # Capture episode starting position on the first step
                if episode_steps == 1:
                    start_x, start_y, start_z = current_x, current_y, current_z
                    positions_history.append((start_x, start_y, start_z))

                # Keep position history (up to last 10 positions)
                if episode_steps > 1:  # Don't add twice on step 1
                    positions_history.append((current_x, current_y, current_z))
                    if len(positions_history) > 10:
                        positions_history.pop(0)
                
                # Calculate the distance from starting position
                distance_from_start = np.sqrt((current_x - start_x)**2 + (current_y - start_y)**2 + (current_z - start_z)**2)
                max_distance_reached = max(max_distance_reached, distance_from_start)
                
                # Add the step computed values to info dictionary
                step_info['distance_from_start'] = distance_from_start
                step_info['max_distance_reached'] = max_distance_reached
                step_info['positions_history'] = positions_history.copy()
                
                # Log complete step information
                step_reward = reward[0] if isinstance(reward, np.ndarray) else reward
                f.write(f"Episode {episode_num} - Step {episode_steps} Log:\n")
                f.write(f"Action: {action}\n")
                f.write(f"Reward: {step_reward:.3f}\n")
                f.write(f"Inference Time: {inference_time*1000:.2f}ms\n")
                f.write(f"Done: {done}\n")
                f.write(f"Distance from start: {distance_from_start:.3f}\n")
                f.write(f"Max distance reached: {max_distance_reached:.3f}\n")
                f.write(f"Step Info Dictionary:\n")
                
                # Log all keys and values in the info dictionary
                for key, value in step_info.items():
                    if key == 'positions_history':
                        f.write(f"  {key}: [\n")
                        for pos in value:
                            f.write(f"    ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}),\n")
                        f.write(f"  ]\n")
                    elif isinstance(value, (list, tuple)) and len(value) == 3 and all(isinstance(v, (int, float)) for v in value):
                        # Format position-like tuples nicely (3-element numeric tuples)
                        f.write(f"  {key}: ({value[0]:.3f}, {value[1]:.3f}, {value[2]:.3f})\n")
                    elif isinstance(value, float):
                        f.write(f"  {key}: {value:.3f}\n")
                    else:
                        f.write(f"  {key}: {value}\n")
                
                # Check for lap completion
                if episode_steps > min_steps_per_lap and distance_from_start < position_threshold:
                    if max_distance_reached > position_threshold * LAP_DISTANCE_MULTIPLIER:  # Must have moved away from the start position
                        lap_count += 1
                        current_time = time.time()
                        lap_time = current_time - last_lap_time  # Time delta from last lap completion
                        lap_times.append(lap_time)
                        
                        lap_message = f"🏁 Lap {lap_count} completed successfully at step {episode_steps} within {lap_time:.2f}s)"
                        print(f"    {lap_message}")
                        f.write("🎉"*30 + "\n")
                        f.write(f"*** {lap_message} ***\n")
                        f.write(f"The information of the step at which the lap was completed: {step_info}\n")
                        f.write("🎉"*30 + "\n")
                        
                        # Update the last lap time for the next lap calculation
                        last_lap_time = current_time
                        # Reset for the next lap completion in the same episode
                        positions_history = [(current_x, current_y, current_z)]
                        max_distance_reached = 0.0  # Reset for next lap
            
            episode_duration = time.time() - episode_start_time
            
            # Calculate inference statistics for this episode
            avg_inference_time = np.mean(inference_times) if inference_times else 0.0
            min_inference_time = np.min(inference_times) if inference_times else 0.0
            max_inference_time = np.max(inference_times) if inference_times else 0.0
            
            # Record current episode stats
            episode_stats.append({
                'episode_number': episode_num,
                'laps_completed': lap_count,
                'steps_taken': episode_steps,
                'episode_duration': episode_duration,
                'episode_reward': episode_reward,
                'lap_times': lap_times,
                'completed_naturally': done,
                'budget_exceeded': episode_steps > max_steps_per_episode,
                'avg_inference_time': avg_inference_time,
                'min_inference_time': min_inference_time,
                'max_inference_time': max_inference_time
            })
            total_laps += lap_count
            
            # Track naturally completed episodes
            if done:
                naturally_completed_count += 1
            
            # Print episode summary
            print(f"  ✅ Episode {episode_num} completed:")
            print(f"     • Laps: {lap_count}")
            print(f"     • Steps: {episode_steps}/{max_steps_per_episode}")
            print(f"     • Duration: {episode_duration:.2f}s")
            print(f"     • Total reward: {episode_reward:.2f}")
            print(f"     • Avg inference time: {avg_inference_time*1000:.2f}ms")
            if lap_times:
                print(f"     • Best lap completed in: {min(lap_times):.2f}s")
            
            # Write episode summary to log
            f.write("=" * 80)
            f.write(f"\nEpisode {episode_num} Summary:\n")
            f.write(f"  Laps completed: {lap_count}\n")
            f.write(f"  Steps taken: {episode_steps}/{max_steps_per_episode}\n")
            f.write(f"  Episode duration: {episode_duration:.2f}s\n")
            f.write(f"  Total reward: {episode_reward:.2f}\n")
            f.write(f"  Inference Time Statistics:\n")
            f.write(f"    • Average: {avg_inference_time*1000:.2f}ms\n")
            f.write(f"    • Min: {min_inference_time*1000:.2f}ms\n")
            f.write(f"    • Max: {max_inference_time*1000:.2f}ms\n")
            f.write(f"  Completed naturally: {done}\n")
            f.write(f"  Budget exceeded: {episode_steps > max_steps_per_episode}\n")
            if lap_times:
                f.write(f"  Lap times: {[f'{t:.2f}s' for t in lap_times]}\n")
            f.write("=" * 80 + "\n\n")
    
    # Calculate evaluation global statistics
    successful_episodes = [ep for ep in episode_stats if ep['laps_completed'] > 0]
    all_lap_times = []
    all_inference_times = []
    all_episode_times = []
    for ep in episode_stats:
        all_lap_times.extend(ep['lap_times'])
        if ep['avg_inference_time'] > 0:
            all_inference_times.append(ep['avg_inference_time'])
        all_episode_times.append(ep['episode_duration'])
    
    evaluation_stats = {
        'n_episodes': n_episodes,
        'successful_episodes': len(successful_episodes),
        'success_rate': len(successful_episodes) / n_episodes,
        'total_laps': total_laps,
        'average_laps_per_episode': total_laps / n_episodes,
        'average_episode_time': np.mean(all_episode_times) if all_episode_times else 0.0,
        'episode_stats': episode_stats,
        'evaluation_session_duration': time.time() - eval_start_time,
        'all_lap_times': all_lap_times,
        'log_file': log_file,
        'naturally_completed_episodes': naturally_completed_count,
        'natural_completion_rate': naturally_completed_count / n_episodes,
    }
    
    # Add inference time statistics
    if all_inference_times:
        evaluation_stats.update({
            'avg_inference_time': np.mean(all_inference_times),
            'min_inference_time': min([ep['min_inference_time'] for ep in episode_stats]),
            'max_inference_time': max([ep['max_inference_time'] for ep in episode_stats]),
        })
    
    if all_lap_times:
        evaluation_stats.update({
            'fastest_lap': min(all_lap_times),
            'slowest_lap': max(all_lap_times),
            'average_lap_time': np.mean(all_lap_times)
        })
    
    # Save t-SNE data
    if tsne_collected_samples:
        tsne_data_file = f"{model_filename}_simulator_tsne_data.npz"
        np.savez(
            tsne_data_file,
            samples=np.array(tsne_collected_samples)
        )
        print(f"💾 Saved {len(tsne_collected_samples)} sample observations for t-SNE analysis to: {tsne_data_file}")
        evaluation_stats['tsne_data_file'] = tsne_data_file
    
    # Final summary
    print("\n" + "=" * 60)
    print("🏆 Model Evaluation Completed!")
    print(f"📊 Evaluation_Stats:")
    print(f"🏎️ {len(successful_episodes)}/{n_episodes} episodes with successful laps")
    print(f"🏎️💥 {naturally_completed_count}/{n_episodes} episodes ended with crashes")
    print(f"🏁 Total laps: {total_laps} (avg: {evaluation_stats['average_laps_per_episode']:.2f} per episode)")
    print(f"⏰ Average episode time: {evaluation_stats['average_episode_time']:.2f}s")
    if all_lap_times:
        print(f"⏱️  Average lap time: {evaluation_stats['average_lap_time']:.2f}s")
        print(f"🏃 Fastest lap: {evaluation_stats['fastest_lap']:.2f}s")
    if all_inference_times:
        print(f"🧠 Inference Performance:")
        print(f"   • Average inference time: {evaluation_stats['avg_inference_time']*1000:.2f}ms")
        print(f"   • Min/Max: {evaluation_stats['min_inference_time']*1000:.2f}ms / {evaluation_stats['max_inference_time']*1000:.2f}ms")
    print(f"📝 Complete info dictionary log saved to: {log_file}")
    
    return evaluation_stats

# ============================================================================
# Command Line Interface Main Function
# ============================================================================

def main():
    """
    Main function to handle command-line interface of the model evaluation script.
    
    Accepts command-line arguments for model path, number of episodes, and max steps per episode.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate PPO trained models in the DonkeyCar simulation environment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required argument
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the PPO model file"
    )
    
    # Optional arguments with defaults from configuration
    parser.add_argument(
        "-e", "--episodes",
        type=int,
        default=N_EPISODES,
        help=f"Number of episodes to evaluate the model for, defaults to {N_EPISODES}"
    )
    
    parser.add_argument(
        "-s", "--max-steps",
        type=int,
        default=EPISODE_MAX_STEPS,
        help=f"Maximum steps per episode, defaults to {EPISODE_MAX_STEPS}"
    )
    
    # Parse arguments
    args = parser.parse_args()
        
    # Print evaluation configuration
    print("🤖 DonkeyCar Model Evaluator")
    print("=" * 60)
    print(f"📁 Model path: {args.model_path}")
    print(f"🔢 Episodes: {args.episodes}")
    print(f"🐾 Max steps per episode: {args.max_steps}")
    print(f"📍 Position threshold: {POSITION_THRESHOLD}")
    print(f"🏁 Min lap steps: {LAP_MIN_STEPS}")
    print(f"🌍 Environment: {ENV_ID}")
    print(f"🎯 Max CTE: {MAX_CTE}")
    print("=" * 60)
    
    try:
        # Run evaluation
        evaluation_stats = evaluate_model(
            model_path=args.model_path,
            n_episodes=args.episodes,
            max_steps_per_episode=args.max_steps,
            position_threshold=POSITION_THRESHOLD,
            min_steps_per_lap=LAP_MIN_STEPS
        )
        
    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user!")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during model evaluation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()