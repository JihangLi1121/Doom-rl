#!/usr/bin/env python3
"""
Optimized PPO Training for VizDoom Deadly Corridor
Based on VizDoom-DeadlyCorridor-Tutorial.ipynb

Key insight from tutorial: Use raw VizDoom DoomGame() API directly
to have full access to game variables for reward shaping.

The gymnasium wrapper doesn't expose HITCOUNT, DAMAGE_TAKEN, etc.
So we create a custom Gym environment like the tutorial does.

Scenario: Navigate corridor, kill 6 monsters, reach the vest
Game variables: HEALTH, DAMAGE_TAKEN, HITCOUNT, SELECTED_WEAPON_AMMO

CURRICULUM LEARNING: Trains on skill 1→2→3→4→5 progressively
"""

import cv2
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import numpy as np
from vizdoom import DoomGame, Mode, ScreenFormat, ScreenResolution
import vizdoom as vzd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor


# ==================== CONFIGURATION ====================
# Training parameters - OPTIMIZED FOR MEMORY
N_STEPS = 2048                 # Reduced for memory (tutorial used 8192 with 1 env)
N_ENVS = 4                     # 4 parallel environments (total batch = 2048*4 = 8192)
FRAME_SKIP = 4                 # Standard frame skip
IMAGE_SHAPE = (100, 160)       # Image shape (height, width) - matches tutorial

# CURRICULUM LEARNING SCHEDULE (like tutorial)
# Train on each difficulty level for specified timesteps
# Tutorial: 400k on skill 1, then 40k each for skills 2-5
CURRICULUM = [
    (1, int(4e5)),   # Skill 1: 400k steps (learn basics)
    (2, int(4e4)),   # Skill 2: 40k steps
    (3, int(4e4)),   # Skill 3: 40k steps
    (4, int(4e4)),   # Skill 4: 40k steps
    (5, int(4e4)),   # Skill 5: 40k steps (nightmare!)
]

# Reward shaping weights (from tutorial)
HITCOUNT_BONUS = 200           # +300 per hit (CRUCIAL! Increased to encourage shooting)
DAMAGE_PENALTY = 10            # -10 per damage taken
AMMO_BONUS = 5                 # +5 for ammo efficiency

# Reward scaling for training stability
REWARD_SCALE = 0.01            # Scale rewards to prevent large gradients

# Debug logging
DEBUG_HITCOUNT = False          # Print when agent hits an enemy
# =======================================================


class VizDoomGym(Env):
    """
    Custom VizDoom Gym Environment (FROM TUTORIAL)

    This directly wraps the VizDoom DoomGame() API to get full access
    to game variables like HITCOUNT, DAMAGE_TAKEN, etc.

    The gymnasium wrapper doesn't expose these, which is why reward
    shaping wasn't working before!
    """

    def __init__(self, render=False, config_path=None, doom_skill=1):
        super().__init__()

        # Setup the game
        self.game = DoomGame()

        # Load the deadly corridor scenario
        if config_path:
            self.game.load_config(config_path)
        else:
            # Use the default deadly corridor config
            self.game.load_config(vzd.scenarios_path + "/deadly_corridor.cfg")

        # Set difficulty BEFORE game.init() - this is critical!
        # doom_skill: 1=easiest, 5=nightmare
        self.game.set_doom_skill(doom_skill)
        self.doom_skill = doom_skill

        # Clear existing game variables and set our own (to ensure correct order)
        self.game.clear_available_game_variables()
        self.game.add_available_game_variable(vzd.GameVariable.HEALTH)
        self.game.add_available_game_variable(vzd.GameVariable.DAMAGE_TAKEN)
        self.game.add_available_game_variable(vzd.GameVariable.HITCOUNT)
        self.game.add_available_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO)

        # Render settings
        if not render:
            self.game.set_window_visible(False)
        else:
            self.game.set_window_visible(True)

        # Set screen format for grayscale processing
        self.game.set_screen_resolution(ScreenResolution.RES_320X240)
        self.game.set_screen_format(ScreenFormat.RGB24)

        # Initialize the game
        self.game.init()

        # Print confirmation that game variables are set up
        print(f"✅ VizDoomGym initialized: doom_skill={doom_skill}, game variables: HEALTH, DAMAGE_TAKEN, HITCOUNT, AMMO")

        # Create observation and action spaces
        # Grayscale image matching tutorial
        self.observation_space = Box(
            low=0, high=255,
            shape=(IMAGE_SHAPE[0], IMAGE_SHAPE[1], 1),
            dtype=np.uint8
        )
        self.action_space = Discrete(7)  # 7 actions in deadly corridor

        # Track game variables for reward shaping
        self.damage_taken = 0
        self.hitcount = 0
        self.ammo = 52  # Starting ammo

        # Track total hits for logging
        self.episode_hits = 0

    def step(self, action):
        # Create one-hot action vector
        actions = np.identity(7, dtype=np.uint8)

        # Take action and get movement reward
        movement_reward = self.game.make_action(actions[action], FRAME_SKIP)

        reward = 0
        terminated = self.game.is_episode_finished()
        truncated = False
        info = {}

        if self.game.get_state():
            # Get screen and convert to grayscale
            state = self.game.get_state().screen_buffer
            state = self._grayscale(state)

            # Get game variables for reward shaping
            game_variables = self.game.get_state().game_variables

            if game_variables is not None and len(game_variables) >= 4:
                health, damage_taken, hitcount, ammo = game_variables[:4]

                # Calculate deltas
                damage_taken_delta = self.damage_taken - damage_taken
                hitcount_delta = hitcount - self.hitcount
                ammo_delta = ammo - self.ammo

                # Update tracked values
                self.damage_taken = damage_taken
                self.hitcount = hitcount
                self.ammo = ammo

                # Reward shaping (FROM TUTORIAL)
                # 1. Movement reward from game
                reward = movement_reward

                # 2. Damage penalty
                reward += damage_taken_delta * DAMAGE_PENALTY

                # 3. Hitcount bonus
                reward += hitcount_delta * HITCOUNT_BONUS

                reward += ammo_delta * AMMO_BONUS

                info = {"hitcount": hitcount, "health": health, "ammo": ammo}
        else:
            # Episode finished - return zero state
            state = np.zeros(self.observation_space.shape, dtype=np.uint8)

        # so tensorboard logs NATURAL rewards
        return state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Log episode hits before reset
        if self.episode_hits > 0:
            pass  # Could add logging here

        # Reset tracking variables
        self.damage_taken = 0
        self.hitcount = 0
        self.ammo = 52
        self.episode_hits = 0

        # Start new episode
        self.game.new_episode()

        # Get initial state
        state = self.game.get_state().screen_buffer
        state = self._grayscale(state)

        return state, {}

    def _grayscale(self, observation):
        """Convert RGB to grayscale and resize (from tutorial)."""
        # observation is in (C, H, W) format from VizDoom
        if len(observation.shape) == 3 and observation.shape[0] == 3:
            # Convert from CHW to HWC
            observation = np.moveaxis(observation, 0, -1)

        # Convert to grayscale
        gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)

        # Resize to target shape
        resized = cv2.resize(gray, (IMAGE_SHAPE[1], IMAGE_SHAPE[0]),
                            interpolation=cv2.INTER_CUBIC)

        # Add channel dimension
        state = np.reshape(resized, (IMAGE_SHAPE[0], IMAGE_SHAPE[1], 1))

        return state

    def render(self):
        pass

    def close(self):
        self.game.close()


class RewardScaleWrapper(gym.Wrapper):
    """Scale rewards for training stability (applied AFTER Monitor)."""
    def __init__(self, env, scale=REWARD_SCALE):
        super().__init__(env)
        self.scale = scale

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward * self.scale, terminated, truncated, info


def make_env(rank, doom_skill=1):
    """
    Create environment with wrappers.
    Wrapper order: VizDoomGym -> Monitor (logs NATURAL rewards) -> RewardScaleWrapper
    """
    def _init():
        env = VizDoomGym(render=False, doom_skill=doom_skill)
        env = Monitor(env)  # Log NATURAL rewards to tensorboard
        env = RewardScaleWrapper(env)  # Scale rewards for training
        return env
    return _init


def create_vec_env(doom_skill, n_envs=N_ENVS):
    """Create vectorized environment for a given skill level."""
    env = DummyVecEnv([make_env(i, doom_skill=doom_skill) for i in range(n_envs)])
    # VecTransposeImage: converts from HWC to CHW format for CNN policies
    env = VecTransposeImage(env)
    return env


def main():
    """
    Main training function with CURRICULUM LEARNING.

    Like the tutorial, we train on progressively harder difficulty levels:
    - Skill 1: Learn the basics (400k steps)
    - Skill 2-5: Fine-tune on harder difficulties (40k each)
    """
    # Calculate total timesteps
    total_timesteps = sum(steps for _, steps in CURRICULUM)

    print("\n" + "="*70)
    print("PPO TRAINING - Deadly Corridor with CURRICULUM LEARNING")
    print("="*70)
    print(f"Parallel environments: {N_ENVS}")
    print(f"Frame skip: {FRAME_SKIP}")
    print(f"Image shape: {IMAGE_SHAPE} (grayscale)")
    print(f"Effective batch: {N_STEPS * N_ENVS} steps per update")
    print("\n📚 CURRICULUM SCHEDULE:")
    for skill, steps in CURRICULUM:
        print(f"   Skill {skill}: {steps:,} steps")
    print(f"   TOTAL: {total_timesteps:,} steps")
    print("\nReward shaping:")
    print(f"  - Hitcount bonus: +{HITCOUNT_BONUS} per hit (CRUCIAL!)")
    print(f"  - Damage penalty: {DAMAGE_PENALTY} per damage")
    print(f"  - Reward scale: {REWARD_SCALE} (for training stability)")
    print("\nUsing: VecTransposeImage + CUSTOM VizDoomGym class")
    print("="*70 + "\n")

    # Start with first skill level
    first_skill, first_steps = CURRICULUM[0]
    print(f"🎮 Creating environments with doom_skill={first_skill}...")
    env = create_vec_env(doom_skill=first_skill)

    # PPO hyperparameters - tuned for better exploration and learning
    agent = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,         # Higher LR for faster learning
        n_steps=N_STEPS,            # 2048 steps per update
        batch_size=256,             # Mini-batch size
        n_epochs=10,                # Number of epochs per update
        gamma=0.99,                 # Standard discount factor
        gae_lambda=0.95,            # Standard GAE lambda
        clip_range=0.2,             # Standard clip range
        ent_coef=0.02,              # Higher entropy for more exploration (shooting!)
        vf_coef=0.5,                # Value function coefficient
        max_grad_norm=0.5,          # Gradient clipping
        tensorboard_log="./logs/VizdoomCorridor-v0_curriculum",
        policy_kwargs=dict(
            net_arch=[dict(pi=[256, 256], vf=[256, 256])],
            normalize_images=True
        )
    )

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=50000 // N_ENVS,
        save_path="./models/VizdoomCorridor-v0_curriculum",
        name_prefix="ppo_corridor"
    )

    print("Starting curriculum training...")
    print("Monitor with: tensorboard --logdir=./logs\n")

    # Entropy schedule: higher for early learning, lower for exploitation
    ENTROPY_SCHEDULE = {
        1: 0.02,    # High exploration on skill 1
        2: 0.015,   # Slightly lower
        3: 0.01,    # Medium
        4: 0.005,   # Low - exploit learned behavior
        5: 0.005,   # Low - exploit learned behavior
    }

    try:
        # Train through each skill level
        for i, (skill, timesteps) in enumerate(CURRICULUM):
            print("\n" + "="*70)
            print(f"⚔️  CURRICULUM STAGE {i+1}/{len(CURRICULUM)}: Skill {skill} for {timesteps:,} steps")
            print("="*70 + "\n")

            # Adjust entropy coefficient based on skill level
            new_ent_coef = ENTROPY_SCHEDULE.get(skill, 0.01)
            agent.ent_coef = new_ent_coef
            print(f"📊 Entropy coefficient set to {new_ent_coef} for skill {skill}")

            if i > 0:
                # Close previous environment
                env.close()

                # Create new environment with current skill level
                print(f"🎮 Creating new environments with doom_skill={skill}...")
                env = create_vec_env(doom_skill=skill)

                # Set new environment on existing model
                agent.set_env(env)

            # Train on this skill level
            # First iteration: reset_num_timesteps=True to create new PPO_X log folder
            # Subsequent iterations: False to continue the same run
            agent.learn(
                total_timesteps=timesteps,
                callback=checkpoint_callback,
                progress_bar=True,
                reset_num_timesteps=(i == 0)  # True for first stage, False for rest
            )

            # Save checkpoint after each skill level
            checkpoint_path = f"./models/VizdoomCorridor-v0_curriculum/skill_{skill}_model"
            agent.save(checkpoint_path)
            print(f"✅ Saved checkpoint: {checkpoint_path}")

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted!")

    # Save final model
    final_path = "./models/VizdoomCorridor-v0_curriculum/final_model"
    agent.save(final_path)

    print("\n" + "="*70)
    print("🎉 CURRICULUM TRAINING COMPLETED!")
    print(f"Final model saved to: {final_path}")
    print("="*70 + "\n")

    env.close()


if __name__ == "__main__":
    main()
