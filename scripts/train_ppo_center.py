#!/usr/bin/env python3
"""
Optimized PPO Training for VizDoom Defend the Center
Specifically tuned for ammo efficiency and kill maximization

Scenario: Defend center, kill respawning monsters with limited ammo
Key challenges:
- Limited ammo (must be efficient)
- Monsters respawn (maximize kills before death)
- 360-degree combat (turn to face enemies)
- Episode ends when you die (inevitable)
- Simple action space (3 actions: turn left, turn right, attack)

Strategy: Encourage kills, penalize ammo waste, reward accuracy
"""

import cv2
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import vizdoom.gymnasium_wrapper  # noqa


# Training parameters - OPTIMIZED FOR DEFEND THE CENTER
TRAINING_TIMESTEPS = int(5e5)  # 500k steps (needs good ammo efficiency)
N_STEPS = 512                  # Moderate rollout (faster updates than 4096, more stable than 128)
N_ENVS = 8                     # 8 parallel environments (match baseline)
FRAME_SKIP = 4                 # Standard frame skip (3 actions don't need precision)
IMAGE_SHAPE = (60, 80)         # Resized image shape

ENV_ID = "VizdoomDefendCenter-v0"


class ScreenOnlyWrapper(gym.ObservationWrapper):
    """
    Extracts only the screen from VizDoom's dict observation space.
    Resizes to smaller size for faster training.
    """
    def __init__(self, env, image_shape=IMAGE_SHAPE):
        super().__init__(env)
        self.image_shape = image_shape
        self.image_shape_reverse = image_shape[::-1]

        # Get original screen shape
        if isinstance(env.observation_space, gym.spaces.Dict):
            original_screen = env.observation_space['screen']
            num_channels = original_screen.shape[-1]
        else:
            num_channels = env.observation_space.shape[-1]

        # Create new observation space with resized shape
        new_shape = (image_shape[0], image_shape[1], num_channels)
        self.observation_space = gym.spaces.Box(0, 255, shape=new_shape, dtype=np.uint8)

    def observation(self, obs):
        # Extract screen from dict if needed
        if isinstance(obs, dict):
            screen = obs['screen']
        else:
            screen = obs

        # Resize for faster processing
        resized = cv2.resize(screen, self.image_shape_reverse)
        return resized


class DefendCenterRewardWrapper(gym.Wrapper):
    """
    Reward shaping for Defend the Center:
    - Moderate kill amplification (encourage kills without instability)
    - Small survival bonus (staying alive = more kill opportunities)
    - NO ammo penalty (let agent learn to shoot freely)
    Applied BEFORE Monitor so natural rewards are logged.
    """
    def __init__(self, env, kill_multiplier=1.5, survival_bonus=0.2):
        super().__init__(env)
        self.kill_multiplier = kill_multiplier
        self.survival_bonus = survival_bonus
        self.episode_kills = 0

    def reset(self, **kwargs):
        self.episode_kills = 0
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        shaped_reward = reward

        # 1. Moderate kill amplification (reward > 0 means kill in this scenario)
        if reward > 0:
            shaped_reward = reward * self.kill_multiplier
            self.episode_kills += 1

        # 2. Small survival bonus (encourage staying alive to get more kills)
        if not terminated:
            shaped_reward += self.survival_bonus

        return obs, shaped_reward, terminated, truncated, info


class RewardScaleWrapper(gym.Wrapper):
    """
    Scale rewards for training stability.
    Applied AFTER Monitor so it doesn't affect logged rewards.
    """
    def __init__(self, env, scale=0.01):
        super().__init__(env)
        self.scale = scale

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        scaled_reward = reward * self.scale
        return obs, scaled_reward, terminated, truncated, info


def make_env(env_id, rank, frame_skip=FRAME_SKIP):
    """
    Create a single environment with all wrappers applied.
    Wrapper order matters for logging natural rewards:
    1. ScreenOnlyWrapper - resize observation
    2. Monitor - logs NATURAL rewards (before shaping) for TensorBoard
    3. DefendCenterRewardWrapper - kill amplification + ammo efficiency (for training only)
    4. RewardScaleWrapper - scale for training stability (for training only)
    """
    def _init():
        env = gym.make(env_id, frame_skip=frame_skip, render_mode='rgb_array')
        env = ScreenOnlyWrapper(env, image_shape=IMAGE_SHAPE)
        env = Monitor(env)  # Logs NATURAL rewards (matches testing)
        env = DefendCenterRewardWrapper(env, kill_multiplier=1.5, survival_bonus=0.2)
        env = RewardScaleWrapper(env, scale=0.01)  # Scale for training
        return env
    return _init


def main():
    """
    Main training function optimized for Defend the Center.
    """

    print("\n" + "="*70)
    print("OPTIMIZED PPO TRAINING - Defend the Center Scenario (v2)")
    print("="*70)
    print(f"Environment: {ENV_ID}")
    print(f"Total timesteps: {TRAINING_TIMESTEPS:,}")
    print(f"Parallel environments: {N_ENVS}")
    print(f"Frame skip: {FRAME_SKIP}")
    print(f"Image shape: {IMAGE_SHAPE}")
    print("\nScenario-specific optimizations:")
    print("  ✅ Moderate kill amplification (1.5x) - stable learning")
    print("  ✅ Survival bonus (+0.2) - stay alive for more kills")
    print("  ✅ NO ammo penalty - let agent explore shooting freely")
    print("  ✅ Moderate rollout (512 steps) - faster updates than baseline")
    print("  ✅ 8 parallel envs - match baseline's parallelization")
    print("  ✅ Frame stacking (4 frames) for monster tracking")
    print("  ✅ Moderate network (192x192) for simple 3-action task")
    print("  ✅ High entropy (0.02) for exploration")
    print("="*70 + "\n")

    # Create parallel environments
    env = SubprocVecEnv([make_env(ENV_ID, i, FRAME_SKIP) for i in range(N_ENVS)])

    # Add frame stacking for temporal information (track monster movement)
    env = VecFrameStack(env, n_stack=4)

    # Create PPO agent with DEFEND-CENTER-OPTIMIZED hyperparameters (v2)
    agent = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=lambda f: 3e-4 * f,  # Linear decay
        n_steps=N_STEPS,                    # Moderate rollout (512)
        batch_size=256,                     # Moderate batch (4096 = 512*8 total, 256 batch)
        n_epochs=10,                        # Standard
        gamma=0.99,                         # Standard discount
        gae_lambda=0.95,                    # Standard GAE
        clip_range=0.2,                     # Standard PPO clip
        clip_range_vf=None,
        ent_coef=0.02,                      # Moderate entropy for 3-action exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.015,                    # Slightly relaxed
        tensorboard_log=f"./logs/{ENV_ID}_center_optimized",
        policy_kwargs=dict(
            net_arch=[dict(pi=[192, 192], vf=[192, 192])],  # Moderate network
            normalize_images=True
        )
    )

    # Setup checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=50000 // N_ENVS,  # Every 50k steps
        save_path=f"./models/{ENV_ID}_center_optimized",
        name_prefix="ppo_center"
    )

    # Train the agent
    print("Starting training...")
    print("Monitor with: tensorboard --logdir=./logs\n")

    try:
        agent.learn(
            total_timesteps=TRAINING_TIMESTEPS,
            callback=checkpoint_callback,
            progress_bar=False  # Disable to avoid tqdm shutdown warnings
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user!")

    # Save final model
    final_path = f"./models/{ENV_ID}_center_optimized/final_model"
    agent.save(final_path)

    print("\n" + "="*70)
    print("Training completed!")
    print(f"Final model saved to: {final_path}")
    print("="*70 + "\n")

    # Close environments
    env.close()


if __name__ == "__main__":
    main()
