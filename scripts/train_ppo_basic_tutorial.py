#!/usr/bin/env python3
"""
PPO Training for VizDoom Basic - Following Tutorial Exactly

Based on VizDoom-Basic-Tutorial.ipynb
Uses custom VizDoomGym environment with grayscale observations.

Scenario: Shoot the monster in front of you
Actions: 3 (MOVE_LEFT, MOVE_RIGHT, ATTACK)
"""

import os
import cv2
import numpy as np
from vizdoom import DoomGame
import vizdoom as vzd

# Use gymnasium (modern) instead of gym (legacy)
from gymnasium import Env
from gymnasium.spaces import Discrete, Box

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


# ==================== CONFIGURATION ====================
# Training parameters - FROM TUTORIAL
TRAINING_TIMESTEPS = int(1e5)  # 100k steps (same as tutorial)
LEARNING_RATE = 0.0001         # From tutorial
N_STEPS = 2048                 # From tutorial
FRAME_SKIP = 4                 # From tutorial
IMAGE_SHAPE = (100, 160)       # From tutorial (height, width)
CHECKPOINT_FREQ = 10000        # From tutorial

# Paths (PPO_1, PPO_2, etc. folders are created automatically by stable-baselines3)
CHECKPOINT_DIR = './models/VizdoomBasic-v0_basic'
LOG_DIR = './logs/VizdoomBasic-v0_basic'
# =======================================================


class VizDoomGym(Env):
    """
    Custom VizDoom Gym Environment - FROM TUTORIAL

    Uses the raw VizDoom DoomGame() API with grayscale observations.
    """

    def __init__(self, render=False):
        super().__init__()

        # Setup the game
        self.game = DoomGame()
        self.game.load_config(vzd.scenarios_path + "/basic.cfg")

        # Render frame logic
        if not render:
            self.game.set_window_visible(False)
        else:
            self.game.set_window_visible(True)

        # Start the game
        self.game.init()

        # Create the action space and observation space
        # Grayscale image (100, 160, 1) - FROM TUTORIAL
        self.observation_space = Box(
            low=0, high=255,
            shape=(IMAGE_SHAPE[0], IMAGE_SHAPE[1], 1),
            dtype=np.uint8
        )
        self.action_space = Discrete(3)  # 3 actions in basic scenario

    def step(self, action):
        # Specify action and take step
        actions = np.identity(3, dtype=np.uint8)
        reward = self.game.make_action(actions[action], FRAME_SKIP)

        # Get all the other stuff we need to return
        if self.game.get_state():
            state = self.game.get_state().screen_buffer
            state = self._grayscale(state)
            ammo = self.game.get_state().game_variables[0]
            info = {"ammo": ammo}
        else:
            state = np.zeros(self.observation_space.shape, dtype=np.uint8)
            info = {"ammo": 0}

        terminated = self.game.is_episode_finished()
        truncated = False

        return state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        return self._grayscale(state), {}

    def _grayscale(self, observation):
        """Grayscale the game frame and resize it - FROM TUTORIAL"""
        # Convert from CHW to HWC format
        gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (IMAGE_SHAPE[1], IMAGE_SHAPE[0]), interpolation=cv2.INTER_CUBIC)
        state = np.reshape(resize, (IMAGE_SHAPE[0], IMAGE_SHAPE[1], 1))
        return state

    def render(self):
        pass

    def close(self):
        self.game.close()


class TrainAndLoggingCallback(BaseCallback):
    """
    Custom callback for saving model checkpoints - FROM TUTORIAL
    """

    def __init__(self, check_freq, save_path, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, f'best_model_{self.n_calls}')
            self.model.save(model_path)
            if self.verbose:
                print(f"Saved checkpoint: {model_path}")
        return True


def main():
    print("\n" + "="*60)
    print("PPO TRAINING - VizDoom Basic (Following Tutorial)")
    print("="*60)
    print(f"Total timesteps: {TRAINING_TIMESTEPS:,}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"N steps: {N_STEPS}")
    print(f"Frame skip: {FRAME_SKIP}")
    print(f"Image shape: {IMAGE_SHAPE} (grayscale)")
    print(f"Checkpoint frequency: every {CHECKPOINT_FREQ:,} steps")
    print("="*60 + "\n")

    # Create directories
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Create non-rendered environment (like tutorial)
    env = VizDoomGym(render=False)
    env = Monitor(env)  # Add monitoring for rewards
    env = DummyVecEnv([lambda: env])

    # Create callback
    callback = TrainAndLoggingCallback(
        check_freq=CHECKPOINT_FREQ,
        save_path=CHECKPOINT_DIR
    )

    # Create PPO model - FROM TUTORIAL
    model = PPO(
        'CnnPolicy',
        env,
        tensorboard_log=LOG_DIR,
        verbose=1,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS
    )

    print("Starting training...")
    print("Monitor with: tensorboard --logdir=./logs\n")

    try:
        model.learn(
            total_timesteps=TRAINING_TIMESTEPS,
            callback=callback,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted!")

    # Save final model
    final_path = os.path.join(CHECKPOINT_DIR, 'final_model')
    model.save(final_path)

    print("\n" + "="*60)
    print("Training completed!")
    print(f"Final model saved to: {final_path}")
    print("="*60 + "\n")

    env.close()


if __name__ == "__main__":
    main()
