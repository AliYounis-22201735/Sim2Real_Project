#!/usr/bin/env python3
"""
Model evaluation script - Evaluates PPO trained models on Jetson Nano robot car
Runs the model, tracking episode-level metrics including steps taken, steering/throttle statistics, inference time, and episode duration
Integrates with DonkeyCar's 'manage.py' infrastructure framework for Jetson Nano control

Usage Examples:
    # Standard evaluation with default settings (10 episodes, 1000 steps each)
    python model_evaluator_jetson.py --model <model_path>
    
    # Customized evaluation with specific episodes and step limits
    python model_evaluator_jetson.py --model <model_path> --episodes <num_episodes> --max-steps <max_steps>
    
    # Use custom DonkeyCar configuration file (instead of the default myconfig.py)
    python model_evaluator_jetson.py --model <model_path> --myconfig <custom_config.py>

Command-line Arguments:
    --model              Path to the PPO model file (.zip)
    --episodes           Number of episodes to run (default: 10)
    --max-steps          Maximum steps per episode (default: 1000)
    --myconfig           DonkeyCar config file (default: myconfig.py)
"""
# ====================================================================================
# Import Required Libraries and Modules
# ====================================================================================

import os
import sys
import time
import json
import argparse
from datetime import datetime
import donkeycar as dk

# Add current directory and parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)

from rl_pilot import RLPilot # RLPilot class for model inference

# ====================================================================================
# Settings
# ====================================================================================

NUM_EPISODES = 10
MAX_STEPS_PER_EPISODE = 1000
START_DELAY_SECONDS = 5  # Startup delay before evaluation begins
BREAK_BETWEEN_EPISODES_SECONDS = 10  # Rest period between episodes

# ====================================================================================
# Episode Tracker Class
# ====================================================================================
class EpisodeTracker:
    """Tracks episode progress and metrics"""
    
    def __init__(self, num_episodes, max_steps_per_episode):
        print("🔄🔄🔄🔄🔄 CALLFLOW: Entering model_evaluator_jetson.py - Model Evaluation in Jetson Nano 🔄🔄🔄🔄🔄")
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode
        
        # Current episode state
        self.current_episode = 0
        self.current_step = 0
        self.episode_start_time = None
        
        # Episode results
        self.episode_results = []
        
        # Step-level data
        self.step_data = []
        
        # Episode control flags
        self.episode_active = False
        self.episode_stopped = False
        
    def start(self):
        """Kick off model-evaluation episodes"""
        self.episode_active = True
        self.current_episode = 1
        self.current_step = 0
        self.episode_start_time = time.time()
        self.step_data = []
        print(f"\n🚗 Starting Episode {self.current_episode}/{self.num_episodes}...")
    
    def record_step_data(self, steering, throttle, inference_ms):
        """Record metrics for the current step"""
        self.step_data.append({
            'step': self.current_step,
            'steering': steering,
            'throttle': throttle,
            'inference_ms': inference_ms
        })
    
    def step(self, steering=None, throttle=None, inference_ms=None):
        """Record one step"""
        if not self.episode_active:
            return False
            
        self.current_step += 1 # Increment step count
        
        # Record step data if provided
        if steering is not None and throttle is not None and inference_ms is not None:
            self.record_step_data(steering, throttle, inference_ms)
        
        # Print episode progress every 100 steps
        if self.current_step % 100 == 0:
            time_elapsed = time.time() - self.episode_start_time
            print(f" Step {self.current_step}/{self.max_steps_per_episode} | "
                      f"steering={steering:.3f}, throttle={throttle:.3f}, "
                      f"inference={inference_ms:.1f}ms (time elapsed since episode start: {time_elapsed:.1f}s)")

        # Check if the current episode should end
        if self.current_step >= self.max_steps_per_episode or self.episode_stopped:
            self._finish_episode() # Finalize the current episode and prepare for the next, if any
            return False # End episode
        # Otherwise continue
        return True 
    
    def _finish_episode(self):
        """Finish current episode and prepare for the next"""
        episode_duration = time.time() - self.episode_start_time
        
        # Calculate averages from step data
        episode_avg_steering = 0.0
        episode_avg_throttle = 0.0
        episode_avg_inference_ms = 0.0
        
        if self.step_data:
            episode_avg_steering = sum(d['steering'] for d in self.step_data) / len(self.step_data)
            episode_avg_throttle = sum(d['throttle'] for d in self.step_data) / len(self.step_data)
            episode_avg_inference_ms = sum(d['inference_ms'] for d in self.step_data) / len(self.step_data)
        
        # Save result
        result = {
            'episode': self.current_episode,
            'steps': self.current_step,
            'duration': episode_duration,
            'avg_steering': episode_avg_steering,
            'avg_throttle': episode_avg_throttle,
            'avg_inference_ms': episode_avg_inference_ms,
            'step_data': self.step_data  # Include all step data
        }
        self.episode_results.append(result)
        
        print(f"✅ Episode {self.current_episode} completed: {self.current_step} steps in {episode_duration:.1f}s")
        if self.step_data:
            print(f"   Averages: steering={episode_avg_steering:.3f}, throttle={episode_avg_throttle:.3f}, inference={episode_avg_inference_ms:.1f}ms")
        
        # Check if we should continue
        if self.current_episode >= self.num_episodes or self.episode_stopped:
            self.episode_active = False
            print("\n🏁 All episodes completed!")
            return
        
        # Prepare for the next episode
        print(f"⏳ Let's have a break for {BREAK_BETWEEN_EPISODES_SECONDS} seconds before continuing with the next episode...")
        time.sleep(BREAK_BETWEEN_EPISODES_SECONDS)
        
        self.current_episode += 1
        self.current_step = 0
        self.episode_start_time = time.time()
        self.step_data = []  # Reset step data for new episode
        print(f"\n🚗 Starting Episode {self.current_episode}/{self.num_episodes}...")
    
    def save_report(self, model_path):
        """Save model evaluation report to JSON"""
        if not self.episode_results:
            print("⚠️  No evaluation results to save")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = os.path.basename(os.path.dirname(model_path))
        eval_filename = f"jetson_eval_{model_name}_{timestamp}.json"
        
        # Calculate averages
        avg_episode_duration = sum(r['duration'] for r in self.episode_results) / len(self.episode_results)
        avg_episode_steps = sum(r['steps'] for r in self.episode_results) / len(self.episode_results)
        
        report = {
            'model': model_path,
            'timestamp': datetime.now().isoformat(),
            'episodes_completed': len(self.episode_results),
            'episodes_planned': self.num_episodes,
            'avg_duration': avg_episode_duration,
            'avg_steps': avg_episode_steps,
            'episodes': self.episode_results
        }
        
        with open(eval_filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 Evaluation report saved to: {eval_filename}")
        print(f"   Episodes completed: {len(self.episode_results)}/{self.num_episodes} episodes")
        print(f"   Averages: {avg_episode_steps:.0f} steps, {avg_episode_duration:.1f}s per episode")

# ====================================================================================
# Evaluation Pilot - Wraps RLPilot with episode tracking
# ====================================================================================
class RLEvalPilot:
    """Simple wrapper that calls RLPilot and tracks steps"""
    
    def __init__(self, model_path, tracker):
        self.rl_pilot = RLPilot(model_path)
        self.tracker = tracker
        print(f"✅ {self.rl_pilot.model_type} model loaded for evaluation")
    
    def run(self, img_arr):
        """Run inference and track episode progress"""
        # Check if evaluation has started (handles the 5-second startup delay)
        if not self.tracker.episode_active:
            # Return zero commands during startup wait period
            return 0.0, 0.0
        
        # Retrieve action predictions from the model
        steering, throttle, inference_ms = self.rl_pilot.run(img_arr)
        
        # Track step (returns False if episode should end)
        if not self.tracker.step(steering, throttle, inference_ms):
            # Episode ended - stop the car for one frame during transition
            return 0.0, 0.0
        
        return steering, throttle
    
    def shutdown(self):
        """Cleanup"""
        self.rl_pilot.shutdown()

# ====================================================================================
# Evaluation Drive Mode Class
# ====================================================================================
class EvaluationDriveMode:
    """Replaces the standard manage.py DriveMode during model evaluation"""
    
    def __init__(self, tracker, ai_throttle_mult=1.0):
        self.tracker = tracker
        self.ai_throttle_mult = ai_throttle_mult
    
    def run(self, mode, user_steering, user_throttle, pilot_steering, pilot_throttle):
        """Return steering and throttle based on the active mode"""
        
        # Ignore the mode and always use pilot controls during model evaluation
        if self.tracker.episode_active:
            steering = pilot_steering if pilot_steering is not None else 0.0
            throttle = pilot_throttle * self.ai_throttle_mult if pilot_throttle is not None else 0.0
            return steering, throttle
        
        # Otherwise rollback to the standard DriveMode logic - normal mode switching
        if mode == 'user':
            return user_steering, user_throttle
        elif mode == 'local_angle':
            return pilot_steering if pilot_steering else 0.0, user_throttle
        else:
            return (pilot_steering if pilot_steering else 0.0,
                   pilot_throttle * self.ai_throttle_mult if pilot_throttle else 0.0)

# ====================================================================================
# Main Evaluation Function
# ====================================================================================
def run_evaluation(cfg, model_path, episodes, max_steps):
    """
    Run model evaluation by:
    1. Creating a tracker to monitor episodes
    2. Replacing manage.py classes temporarily with evaluation versions
    3. Starting evaluation after a startup delay
    4. Running the normal drive loop with episode tracking
    5. Restoring everything when done
    
    Note: There is a START_DELAY_SECONDS (default: 5s) startup period before
    evaluation begins. During this time, the car returns (0.0, 0.0) commands.
    """
    
    # Create episode tracker
    tracker = EpisodeTracker(episodes, max_steps)
    
    # Import manage.py
    import manage
    from manage import drive
    
    # Save original classes
    original_DriveMode = manage.DriveMode
    original_RLPilot = getattr(manage, 'RLPilot', None)
    
    # Create evaluation pilot
    eval_pilot = RLEvalPilot(model_path, tracker)
    
    # Replace classes with evaluation versions
    manage.DriveMode = lambda ai_mult=1.0: EvaluationDriveMode(tracker, ai_mult)
    manage.RLPilot = lambda path: eval_pilot  # Return same pilot instance
    
    # Start evaluation after delay
    def delayed_start():
        """
        Wait for START_DELAY_SECONDS, then activate episode tracking
        During the delay, RLEvalPilot returns (0.0, 0.0) to keep the car stationary.
        """
        time.sleep(START_DELAY_SECONDS)
        tracker.start()  # Sets episode_active=True to begin evaluation
    
    import threading
    starter = threading.Thread(target=delayed_start, daemon=True)
    starter.start()
    
    print(f"\n🚀 Evaluation starting in {START_DELAY_SECONDS} seconds...")
    print(f"📊 Will run {episodes} episodes, {max_steps} steps each")
    
    try:
        # Run the normal drive loop (managed by tracker)
        drive(cfg, model_path=model_path, use_joystick=False, model_type=None)
        
    except KeyboardInterrupt:
        print("\n⚠️  Model evaluation interrupted by user, exiting...")
        tracker.episode_stopped = True
        
    finally:
        # Restore original classes
        manage.DriveMode = original_DriveMode
        manage.RLPilot = original_RLPilot
        
        # Save results and cleanup
        tracker.save_report(model_path)
        eval_pilot.shutdown()
        
        print("✅ Model evaluation complete on Jetson Nano...")

# ====================================================================================
# Command Line Interface
# ====================================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Evaluate PPO model on the Jetson Nano robot car',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--model', required=True, help='Path to model file (.zip)')
    parser.add_argument('--episodes', type=int, default=NUM_EPISODES, 
                       help=f'Number of episodes to evaluate the model for, default is {NUM_EPISODES}')
    parser.add_argument('--max-steps', type=int, default=MAX_STEPS_PER_EPISODE,
                       help=f'Maximum steps per episode, default is {MAX_STEPS_PER_EPISODE}')
    parser.add_argument('--myconfig', default='myconfig.py',
                       help='DonkeyCar config file')
    
    args = parser.parse_args()
    
    # Validate model exists
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        return
    
    # Load DonkeyCar config
    cfg = dk.load_config(myconfig=args.myconfig)
    
    # Print header
    print("\n" + "="*60)
    print("🤖 PPO Model Evaluation on Jetson Nano")
    print("="*60)
    print(f"Model:  {os.path.basename(args.model)}")
    print(f"Evaluation episodes:   {args.episodes}")
    print(f"Maximum steps per episode: {args.max_steps}")
    print("="*60)
    
    # Run evaluation
    run_evaluation(cfg, args.model, args.episodes, args.max_steps)


if __name__ == '__main__':
    main()
