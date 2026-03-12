#!/usr/bin/env python3
"""
DQN Training for VizDoom
Optimized setup matching PPO training structure
"""

import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.monitor import Monitor
from collections import deque
import random
import os
import vizdoom.gymnasium_wrapper  # noqa


# ==================== CONFIGURATION ====================
# Edit these variables to configure training

# Environment to train on
ENV_ID = "VizdoomBasic-v0"  # Options: "VizdoomBasic-v0", "VizdoomCorridor-v0", etc.

# Training parameters (optimized for lower resource usage)
TRAINING_EPISODES = 2000          # Increase to 10000+ for comparable training to PPO
FRAME_SKIP = 4
IMAGE_SHAPE = (60, 80)
BATCH_SIZE = 32                   # Reduced from 64 to save memory
LEARNING_RATE = 1e-4
GAMMA = 0.99
BUFFER_SIZE = 10000               # Reduced from 50k to save RAM (still effective)
TARGET_UPDATE_FREQ = 1000
TRAIN_FREQ = 4                    # Train every 4 steps instead of every step (saves CPU)
WARMUP_STEPS = 1000               # Reduced from 10k for faster start
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

# =======================================================


class ScreenOnlyWrapper(gym.ObservationWrapper):
    """
    Extracts and resizes screen from VizDoom's dict observation space.
    Matches PPO's ScreenOnlyWrapper.
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


class RewardScaleWrapper(gym.Wrapper):
    """
    Scale rewards for training stability.
    Matches PPO's RewardScaleWrapper.
    """
    def __init__(self, env, scale=0.01):
        super().__init__(env)
        self.scale = scale

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        scaled_reward = reward * self.scale
        return obs, scaled_reward, terminated, truncated, info


class CNNFeatureExtractor(nn.Module):
    """CNN backbone for extracting visual features from game frames"""
    def __init__(self, input_channels=3, features_dim=512):
        super(CNNFeatureExtractor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

    def forward(self, x):
        x = x.float() / 255.0  # Normalize to [0, 1]
        return self.conv(x)


class DQNNetwork(nn.Module):
    """Deep Q-Network with CNN backbone"""
    def __init__(self, input_shape, n_actions, features_dim=512):
        super(DQNNetwork, self).__init__()

        self.feature_extractor = CNNFeatureExtractor(
            input_channels=input_shape[0],
            features_dim=features_dim
        )

        # Calculate the size after conv layers
        with torch.no_grad():
            sample_input = torch.zeros(1, *input_shape)
            sample_output = self.feature_extractor(sample_input)
            conv_output_size = sample_output.shape[1]

        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, n_actions)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.fc(features)


class ReplayBuffer:
    """Experience replay buffer for Q-learning (memory optimized)"""
    def __init__(self, capacity=BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Store as uint8 to save memory (will convert to float32 during sampling)
        state = np.array(state, dtype=np.uint8)
        next_state = np.array(next_state, dtype=np.uint8)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN Agent with CNN backbone"""
    def __init__(self, input_shape, n_actions, device='cuda', lr=LEARNING_RATE,
                 gamma=GAMMA, batch_size=BATCH_SIZE):
        self.device = device
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size

        self.policy_net = DQNNetwork(input_shape, n_actions).to(device)
        self.target_net = DQNNetwork(input_shape, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer()

        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_END
        self.epsilon_decay = EPSILON_DECAY

    def select_action(self, state, deterministic=False):
        """Select action using epsilon-greedy policy"""
        if not deterministic and random.random() < self.epsilon:
            return random.randrange(self.n_actions)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(1).item()

    def train_step(self):
        """Perform one step of training"""
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute Q(s,a)
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Compute target Q values (Double DQN)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Compute loss
        loss = F.mse_loss(current_q.squeeze(), target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)  # Gradient clipping
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Update target network with policy network weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        """Save model checkpoint"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)

    def load(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']


def make_env(env_id, frame_skip=FRAME_SKIP):
    """
    Create environment with wrappers (matching PPO structure).
    Wrapper order matters for logging:
    1. Monitor - logs NATURAL rewards (completely unmodified from game)
    2. ScreenOnlyWrapper - resize observation (doesn't affect rewards)
    3. RewardScaleWrapper - scale for training stability (for training only)
    """
    env = gym.make(env_id, frame_skip=frame_skip, render_mode='rgb_array')
    env = Monitor(env)  # First: logs completely natural rewards from game
    env = ScreenOnlyWrapper(env, image_shape=IMAGE_SHAPE)
    env = RewardScaleWrapper(env, scale=0.01)  # Scale for training
    return env


def train_dqn():
    """Train DQN agent with optimized setup"""

    print("\n" + "="*70)
    print("DQN TRAINING FOR VIZDOOM (Optimized for Low Resource Usage)")
    print("="*70)
    print(f"Environment: {ENV_ID}")
    print(f"Episodes: {TRAINING_EPISODES}")
    print(f"Frame skip: {FRAME_SKIP}")
    print(f"Image shape: {IMAGE_SHAPE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Gamma: {GAMMA}")
    print(f"Buffer size: {BUFFER_SIZE} (reduced to save RAM)")
    print(f"Train frequency: Every {TRAIN_FREQ} steps (reduced to save CPU)")
    print(f"Target update frequency: {TARGET_UPDATE_FREQ}")
    print(f"Warmup steps: {WARMUP_STEPS}")
    print("="*70 + "\n")

    # Create save directories
    save_path = f"./models/{ENV_ID}_dqn"
    os.makedirs(save_path, exist_ok=True)

    # Create environment with wrappers
    env = make_env(ENV_ID, frame_skip=FRAME_SKIP)

    # Get observation shape and action space
    # DQN expects (C, H, W) format for PyTorch CNN
    obs_shape = env.observation_space.shape
    obs_shape_chw = (obs_shape[2], obs_shape[0], obs_shape[1])  # Convert HWC to CHW
    n_actions = env.action_space.n

    print(f"Observation shape (HWC): {obs_shape}")
    print(f"Observation shape (CHW for CNN): {obs_shape_chw}")
    print(f"Number of actions: {n_actions}\n")

    # Initialize agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cpu":
        print("WARNING: Training on CPU will be slow. Consider using GPU if available.")
    print()

    agent = DQNAgent(obs_shape_chw, n_actions, device=device)

    # TensorBoard logging
    writer = SummaryWriter(f"./logs/{ENV_ID}_dqn")
    print(f"Monitor with: tensorboard --logdir=./logs\n")

    # Training loop
    print("Starting training...")
    if WARMUP_STEPS > 0:
        print(f"Phase 1: Warmup - collecting {WARMUP_STEPS} steps before training")
    print("="*70 + "\n")

    total_steps = 0
    training_started = False
    episode_count = 0

    for episode in range(TRAINING_EPISODES):
        state, _ = env.reset()
        # Convert HWC to CHW for PyTorch
        state = np.transpose(state, (2, 0, 1))

        episode_reward_scaled = 0  # For training (scaled)
        episode_reward_natural = None  # From Monitor (natural)
        done = False
        step = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Convert next_state to CHW
            next_state = np.transpose(next_state, (2, 0, 1))

            # Store in replay buffer (with scaled reward for training)
            agent.memory.push(state, action, reward, next_state, done)

            # Start training after warmup (train every TRAIN_FREQ steps to save CPU)
            loss = None
            if total_steps >= WARMUP_STEPS and total_steps % TRAIN_FREQ == 0:
                if not training_started:
                    print(f"\nPhase 2: Training started at {total_steps} steps!\n")
                    training_started = True
                loss = agent.train_step()

                # Update target network periodically
                if (total_steps - WARMUP_STEPS) % TARGET_UPDATE_FREQ == 0:
                    agent.update_target_network()

            state = next_state
            episode_reward_scaled += reward
            step += 1
            total_steps += 1

            # Get natural reward from Monitor when episode ends
            if done and 'episode' in info:
                episode_reward_natural = info['episode']['r']
                episode_count += 1

        # Decay epsilon after each episode
        agent.decay_epsilon()

        # Log metrics to TensorBoard (use NATURAL reward from Monitor)
        if episode_reward_natural is not None:
            writer.add_scalar("train/episode_reward", episode_reward_natural, episode_count)
            writer.add_scalar("train/episode_length", step, episode_count)
        else:
            # Fallback if Monitor info not available
            writer.add_scalar("train/episode_reward", episode_reward_scaled, episode)
            writer.add_scalar("train/episode_length", step, episode)

        writer.add_scalar("train/epsilon", agent.epsilon, episode)
        writer.add_scalar("train/total_steps", total_steps, episode)

        if loss is not None:
            writer.add_scalar("train/loss", loss, episode)

        # Print progress every 10 episodes (reduce console overhead)
        if episode % 10 == 0:
            loss_str = f"Loss: {loss:.4f}" if loss is not None else "Warmup"
            buffer_usage = f"{len(agent.memory)}/{BUFFER_SIZE}"
            reward_display = episode_reward_natural if episode_reward_natural is not None else episode_reward_scaled
            print(f"Ep {episode}/{TRAINING_EPISODES} | "
                  f"Reward: {reward_display:.1f} | "
                  f"Len: {step} | "
                  f"Eps: {agent.epsilon:.3f} | "
                  f"{loss_str} | "
                  f"Buf: {buffer_usage} | "
                  f"Total: {total_steps}")

        # Save checkpoint every 100 episodes
        if episode > 0 and episode % 100 == 0:
            checkpoint_path = f"{save_path}/dqn_checkpoint_{episode}.pth"
            agent.save(checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")
            # Clear GPU cache to prevent memory buildup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Save final model
    final_path = f"{save_path}/final_model.pth"
    agent.save(final_path)

    print("\n" + "="*70)
    print("Training completed!")
    print(f"Final model saved to: {final_path}")
    print("="*70 + "\n")

    writer.close()
    env.close()

    return agent


def main():
    """Main training function"""
    train_dqn()


if __name__ == "__main__":
    main()
