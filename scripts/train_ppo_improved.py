"""
PPO Training for VizDoom
1. Optimized hyperparameters
2. Reward shaping
"""

from argparse import ArgumentParser
import cv2
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import vizdoom.gymnasium_wrapper


# Training parameters
TRAINING_TIMESTEPS = int(3e5)
N_STEPS = 2048                
N_ENVS = 4                     
FRAME_SKIP = 4                
IMAGE_SHAPE = (60, 80)         

# Available environments
DEFAULT_ENV = "VizdoomBasic-v0"


class ScreenOnlyWrapper(gym.ObservationWrapper):
    """
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


class KillRewardWrapper(gym.Wrapper):
    """
    Amplify kill rewards without scaling.
    """
    def __init__(self, env, kill_multiplier=2.0):
        super().__init__(env)
        self.kill_multiplier = kill_multiplier

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Amplify kill rewards (reward > 50 indicates kill)
        if reward > 50:
            shaped_reward = reward * self.kill_multiplier
        else:
            shaped_reward = reward

        return obs, shaped_reward, terminated, truncated, info


class RewardScaleWrapper(gym.Wrapper):
    """
    Scale rewards for training stability.
    grabbed from example
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
    Wrapper order matters for logging:
    1. ScreenOnlyWrapper - resize observation
    2. Monitor - logs NATURAL rewards (before shaping) for TensorBoard
    3. KillRewardWrapper - amplify kills (for training only)
    4. RewardScaleWrapper - scale for training stability (for training only)
    """
    def _init():
        env = gym.make(env_id, frame_skip=frame_skip, render_mode='rgb_array')
        env = ScreenOnlyWrapper(env, image_shape=IMAGE_SHAPE) # resize (optained from example)
        env = Monitor(env)  # Logs NATURAL rewards (matches testing)
        env = KillRewardWrapper(env, kill_multiplier=2.0)  # Amplify kills for training
        env = RewardScaleWrapper(env, scale=0.01)  # Scale for training (obtained from example)
        return env
    return _init


def main(args):
    print("PPO training")
    print(f"Environment: {args.env}")
    print(f"Total timesteps: {TRAINING_TIMESTEPS:,}")
    print(f"Parallel environments: {N_ENVS}")
    print(f"Frame skip: {FRAME_SKIP}")
    print(f"Image shape: {IMAGE_SHAPE}")

    env = SubprocVecEnv([make_env(args.env, i, FRAME_SKIP) for i in range(N_ENVS)])

    env = VecFrameStack(env, n_stack=4)

    # Create PPO agent with hyperparameters 
    agent = PPO(
        "CnnPolicy", # CNN because we dont have sound
        env,
        verbose=1, 
        learning_rate=2.5e-4, # add linear decay         
        n_steps=N_STEPS, # larger buffer size collects more data before update
        batch_size=256, # larger batch size, more stable training
        n_epochs=10,                    
        gamma=0.99,                    
        gae_lambda=0.95,                
        clip_range=0.2,                 
        ent_coef=0.015, # exploartion bonus for more exploration               
        vf_coef=0.5,                    
        max_grad_norm=0.5,              
        tensorboard_log=f"./logs/{args.env}_baseline_improved",
        policy_kwargs=dict(
            net_arch=[dict(pi=[128, 128], vf=[128, 128])], # bigger network
            normalize_images=True
        )
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=25000 // N_ENVS,
        save_path=f"./models/{args.env}_baseline_improved",
        name_prefix="ppo_baseline_improved"
    )

    # Train the agent
    print("Starting training...")
    print("Monitor with: tensorboard --logdir=./logs\n")

    try:
        agent.learn(
            total_timesteps=TRAINING_TIMESTEPS,
            callback=checkpoint_callback,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user!")

    # Save final model
    final_path = f"./models/{args.env}_baseline_improved/final_model"
    agent.save(final_path)

    print("Training completed!")
    print(f"Final model saved to: {final_path}")

    env.close()


if __name__ == "__main__":
    parser = ArgumentParser("Basic PPO training for VizDoom")
    parser.add_argument(
        "--env",
        default=DEFAULT_ENV,
        help="VizDoom environment to train on"
    )
    args = parser.parse_args()
    main(args)
