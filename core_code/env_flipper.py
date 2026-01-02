#!/usr/bin/env python3
"""
Gym Horizontal Flipping Wrapper - flips environment observations and steering actions
"""
# ============================================================================
# Import Required Libraries
# ============================================================================

import numpy as np
import cv2
import gym

# ============================================================================
# Horizontal Flipping Wrapper Class
# ============================================================================

class HorizontalFlippingWrapper(gym.Wrapper):
    """
    A Gym wrapper for horizontal-flipping of observations and steering actions.
    Flipping is managed at episode level, determined at episode start and remains consistent throughout the episode.
    """
    
    def __init__(self, env, hflip_prob):
        """
        Initialize the wrapper.

        Args:
            env: Gym environment to wrap.
            hflip_prob: Probability of flipping episodes.

        Returns:
            None
        """
        super().__init__(env)
        self.hflip_prob = hflip_prob
        self._obs_type = None  # Will be auto-detected
        self._episode_flipped = False  # Episode flipping state

        print("🔄🔄🔄🔄🔄 CALLFLOW: Entering env_flipper.py - Horizontal Flipping Gym Wrapper 🔄🔄🔄🔄🔄")

    def reset(self):
        """Reset the environment and determine episode flipping state."""
        self._episode_flipped = np.random.random() < self.hflip_prob

        obs = self.env.reset()
        
        if self._episode_flipped:
            obs = self._flip_observation(obs)

        return obs

    def step(self, action):
        """Step through the environment."""
        # Use episode's predetermined flipping state to manipulate the steering action
        if self._episode_flipped:
            flipped_action = self._flip_action(action)
        else:
            flipped_action = action

        # Step with the "possibly flipped" action
        obs, reward, done, info = self.env.step(flipped_action)

        # Apply observation flipping based on episode flipping state
        if self._episode_flipped:
            obs = self._flip_observation(obs)
            
        # Add episode flip state to the info dictionary
        info['episode_flipped'] = self._episode_flipped
        
        return obs, reward, done, info

    def _detect_observation_type(self, obs):
        """Auto-detect observation type, whether an image or a vector representation"""
        if self._obs_type is None:
            if len(obs.shape) == 3 and obs.shape[2] == 3:  # Image observation: 3D array - HWC format
                self._obs_type = 'image'
            elif len(obs.shape) == 1:  # Vector observation: 1D array
                self._obs_type = 'vector'
            else:
                self._obs_type = 'unknown'
        return self._obs_type

    def _flip_observation(self, obs):
        """Apply appropriate flipping action based on observation type"""
        obs_type = self._detect_observation_type(obs)
        
        if obs_type == 'image':
            return cv2.flip(obs, 1)  # Horizontal flip
         
        elif obs_type == 'vector':
            if not isinstance(obs, np.ndarray):  # Ensure obs is a numpy array
                return obs 
            flipped_obs = obs.copy()
            for i in range(0, len(flipped_obs) - 2, 3):
                x = flipped_obs[i]
                if not np.isnan(x):
                    # Flip normalized x-coordinate:
                    flipped_obs[i] = 1.0 - x
            return flipped_obs
        else:
            return obs

    def _flip_action(self, action):
        """Flip steering action"""
        action = action.copy()
        action[0] = -action[0]  # Flip steering
        return action