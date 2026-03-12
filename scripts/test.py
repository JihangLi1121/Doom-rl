#!/usr/bin/env python3
"""
Test trained PPO models on VizDoom environments
Simply edit the configuration variables below and run the script.
"""

import cv2
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from time import sleep
import vizdoom.gymnasium_wrapper  # noqa
from vizdoom import DoomGame, ScreenFormat, ScreenResolution
import vizdoom as vzd


# ==================== CONFIGURATION ====================
# Edit these variables to test different models/environments

# Environment to test on
ENV_ID = "VizdoomCorridor-v0"  # Options: "VizdoomBasic-v0", "VizdoomCorridor-v0", etc.

# Path to trained model
# MODEL_PATH = "./models/VizdoomBasic-v0_baseline_improved/final_model.zip"
# MODEL_PATH = "./models/VizdoomBasic-v0_baseline/final_model.zip"
# MODEL_PATH = "./models/VizdoomCorridor-v0_baseline/final_model.zip"
# MODEL_PATH = "./models/VizdoomCorridor-v0_corridor_optimized/final_model.zip"
# MODEL_PATH = "./models/VizdoomCorridor-v0_original/final_model.zip"
MODEL_PATH = "./models/VizdoomCorridor-v0_curriculum/final_model.zip"
# MODEL_PATH = "./models/VizdoomBasic-v0_basic/final_model.zip"

# Model type - IMPORTANT: Set this correctly!
# "baseline"  = MultiInputPolicy, no frame stacking, Dict observations
# "improved"  = CnnPolicy, frame stacking, RGB observations
# "corridor"  = CnnPolicy, GRAYSCALE observations, 7 actions (for corridor tutorial model)
# "basic"     = CnnPolicy, GRAYSCALE observations, 3 actions (for basic tutorial model)
MODEL_TYPE = "corridor"  # Options: "baseline", "improved", "corridor", "basic"

# Test parameters
N_EPISODES = 20     # Number of episodes to test
RENDER = False            # Show the game window (False for faster testing)
FRAME_SKIP = 4         # Frame skip for testing (should match training)
IMAGE_SHAPE = (100, 160) # Resized image shape (should match training - corridor uses 100x160)
DETERMINISTIC = True     # Use deterministic actions (True for best performance)
SLEEP_TIME = 0.02        # Sleep between frames for visualization (when RENDER=True)

# Difficulty setting - MUST MATCH TRAINING for corridor models!
DOOM_SKILL = 5          # 1=easiest, 5=nightmare (match your training setting)

# =======================================================


class VizDoomGymCorridorTest(Env):
    """
    Custom VizDoom Gym Environment for TESTING corridor models.
    Matches the training environment exactly - same doom_skill, game variables, etc.
    """
    def __init__(self, render=True, doom_skill=DOOM_SKILL):
        super().__init__()

        self.game = DoomGame()
        self.game.load_config(vzd.scenarios_path + "/deadly_corridor.cfg")

        # Set difficulty BEFORE init - this is critical!
        self.game.set_doom_skill(doom_skill)

        # Render settings
        self.game.set_window_visible(render)
        self.game.set_screen_resolution(ScreenResolution.RES_320X240)
        self.game.set_screen_format(ScreenFormat.RGB24)

        # Initialize
        self.game.init()
        print(f"✅ VizDoomGymCorridorTest initialized with doom_skill={doom_skill}")

        # Spaces - must match training
        self.observation_space = Box(low=0, high=255, shape=(IMAGE_SHAPE[0], IMAGE_SHAPE[1], 1), dtype=np.uint8)
        self.action_space = Discrete(7)

    def step(self, action):
        actions = np.identity(7, dtype=np.uint8)
        reward = self.game.make_action(actions[action], FRAME_SKIP)

        terminated = self.game.is_episode_finished()
        truncated = False
        info = {}

        if self.game.get_state():
            state = self.game.get_state().screen_buffer
            state = self._grayscale(state)
        else:
            state = np.zeros(self.observation_space.shape, dtype=np.uint8)

        return state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        return self._grayscale(state), {}

    def _grayscale(self, observation):
        if len(observation.shape) == 3 and observation.shape[0] == 3:
            observation = np.moveaxis(observation, 0, -1)
        gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (IMAGE_SHAPE[1], IMAGE_SHAPE[0]), interpolation=cv2.INTER_CUBIC)
        return np.reshape(resized, (IMAGE_SHAPE[0], IMAGE_SHAPE[1], 1))

    def render(self):
        pass

    def close(self):
        self.game.close()


class VizDoomGymBasicTest(Env):
    """
    Custom VizDoom Gym Environment for TESTING basic models.
    Matches train_ppo_basic_tutorial.py exactly - grayscale, 3 actions.
    """
    def __init__(self, render=True):
        super().__init__()

        self.game = DoomGame()
        self.game.load_config(vzd.scenarios_path + "/basic.cfg")

        # Render settings
        self.game.set_window_visible(render)

        # Initialize
        self.game.init()
        print(f"✅ VizDoomGymBasicTest initialized")

        # Spaces - must match training (3 actions: MOVE_LEFT, MOVE_RIGHT, ATTACK)
        self.observation_space = Box(low=0, high=255, shape=(IMAGE_SHAPE[0], IMAGE_SHAPE[1], 1), dtype=np.uint8)
        self.action_space = Discrete(3)

    def step(self, action):
        actions = np.identity(3, dtype=np.uint8)
        reward = self.game.make_action(actions[action], FRAME_SKIP)

        terminated = self.game.is_episode_finished()
        truncated = False
        info = {}

        if self.game.get_state():
            state = self.game.get_state().screen_buffer
            state = self._grayscale(state)
            ammo = self.game.get_state().game_variables[0]
            info = {"ammo": ammo}
        else:
            state = np.zeros(self.observation_space.shape, dtype=np.uint8)

        return state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        return self._grayscale(state), {}

    def _grayscale(self, observation):
        if len(observation.shape) == 3 and observation.shape[0] == 3:
            observation = np.moveaxis(observation, 0, -1)
        gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (IMAGE_SHAPE[1], IMAGE_SHAPE[0]), interpolation=cv2.INTER_CUBIC)
        return np.reshape(resized, (IMAGE_SHAPE[0], IMAGE_SHAPE[1], 1))

    def render(self):
        pass

    def close(self):
        self.game.close()


class ScreenOnlyWrapper(gym.ObservationWrapper):
    """
    Extracts only the screen from VizDoom's dict observation space.
    Resizes to smaller size to match training configuration.
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

        # Resize for consistency with training
        resized = cv2.resize(screen, self.image_shape_reverse)
        return resized


class GrayscaleWrapper(gym.ObservationWrapper):
    """
    Grayscale wrapper for corridor models trained with grayscale observations.
    Converts RGB to grayscale and resizes.
    """
    def __init__(self, env, image_shape=IMAGE_SHAPE):
        super().__init__(env)
        self.image_shape = image_shape
        self.image_shape_reverse = image_shape[::-1]

        # Grayscale output (1 channel)
        new_shape = (image_shape[0], image_shape[1], 1)
        self.observation_space = gym.spaces.Box(0, 255, shape=new_shape, dtype=np.uint8)

    def observation(self, obs):
        # Extract screen from dict if needed
        if isinstance(obs, dict):
            screen = obs['screen']
        else:
            screen = obs

        # Convert to grayscale
        if len(screen.shape) == 3 and screen.shape[2] == 3:
            gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
        else:
            gray = screen

        # Resize
        resized = cv2.resize(gray, self.image_shape_reverse, interpolation=cv2.INTER_CUBIC)

        # Add channel dimension
        return np.expand_dims(resized, axis=-1)


class ActionSpaceWrapper(gym.ActionWrapper):
    """
    Limits action space to match the trained model.
    The corridor model was trained with 7 actions but gymnasium wrapper has 8.
    """
    def __init__(self, env, n_actions=7):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(n_actions)

    def action(self, action):
        return action


class DoomSkillWrapper(gym.Wrapper):
    """Sets the doom skill level (difficulty) for the environment."""
    def __init__(self, env, doom_skill=1):
        super().__init__(env)
        self.doom_skill = doom_skill
        self._set_skill()

    def _set_skill(self):
        """Set the doom skill level on the underlying VizDoom game."""
        try:
            game = self.env.unwrapped.game
            game.set_doom_skill(self.doom_skill)
            print(f"✅ Set doom_skill to {self.doom_skill} (1=easiest, 5=nightmare)")
        except Exception as e:
            print(f"⚠️ Could not set doom_skill: {e}")


class BaselineObservationWrapper(gym.ObservationWrapper):
    """
    Wrapper for baseline models - keeps Dict observations but resizes screen.
    Used for MultiInputPolicy models.
    """
    def __init__(self, env, image_shape=IMAGE_SHAPE):
        super().__init__(env)
        self.image_shape = image_shape
        self.image_shape_reverse = image_shape[::-1]

        # Keep Dict observation space but update screen size
        if isinstance(env.observation_space, gym.spaces.Dict):
            original_screen = env.observation_space['screen']
            num_channels = original_screen.shape[-1]
            new_shape = (image_shape[0], image_shape[1], num_channels)

            # Create new Dict observation space with resized screen
            if "audio" in env.observation_space.spaces:
                self.observation_space = gym.spaces.Dict({
                    "screen": gym.spaces.Box(0, 255, shape=new_shape, dtype=np.uint8),
                    "audio": env.observation_space["audio"]
                })
            else:
                self.observation_space = gym.spaces.Dict({
                    "screen": gym.spaces.Box(0, 255, shape=new_shape, dtype=np.uint8)
                })
        else:
            raise ValueError("BaselineObservationWrapper requires Dict observation space")

    def observation(self, obs):
        if isinstance(obs, dict):
            # Resize screen but keep Dict structure
            resized_screen = cv2.resize(obs['screen'], self.image_shape_reverse)
            if "audio" in obs:
                return {"screen": resized_screen, "audio": obs["audio"]}
            else:
                return {"screen": resized_screen}
        return obs


def make_test_env(env_id, frame_skip=FRAME_SKIP, render=RENDER, model_type=MODEL_TYPE):
    """
    Create test environment with appropriate wrappers based on model type.
    """
    if model_type == "corridor":
        # Corridor: Use CUSTOM environment that matches training exactly!
        # This ensures doom_skill is set BEFORE game.init()
        env = VizDoomGymCorridorTest(render=render, doom_skill=DOOM_SKILL)
        return env

    if model_type == "basic":
        # Basic: Use CUSTOM environment that matches train_ppo_basic_tutorial.py
        env = VizDoomGymBasicTest(render=render)
        return env

    # For other model types, use gymnasium wrapper
    render_mode = 'human' if render else 'rgb_array'
    env = gym.make(env_id, frame_skip=frame_skip, render_mode=render_mode)

    if model_type == "baseline":
        # Baseline: Keep Dict observations (for MultiInputPolicy)
        env = BaselineObservationWrapper(env, image_shape=IMAGE_SHAPE)
    else:
        # Improved: Convert to Box observations (for CnnPolicy)
        env = ScreenOnlyWrapper(env, image_shape=IMAGE_SHAPE)

    return env


def test_model():
    """
    Test a trained PPO model on the specified environment.
    """
    # Determine settings based on model type
    # NOTE: "corridor" models trained with train_ppo_corridor.py do NOT use frame stacking (matches tutorial)
    use_frame_stack = (MODEL_TYPE == "improved")  # Only improved uses frame stacking

    print("\n" + "="*70)
    print("TESTING TRAINED PPO MODEL")
    print("="*70)
    print(f"Environment: {ENV_ID}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Model type: {MODEL_TYPE}")
    print(f"Number of test episodes: {N_EPISODES}")
    print(f"Frame skip: {FRAME_SKIP}")
    print(f"Render: {RENDER}")
    print(f"Frame stacking: {use_frame_stack}")
    print(f"Deterministic: {DETERMINISTIC}")
    print("="*70 + "\n")

    # Create test environment
    env = make_test_env(ENV_ID, frame_skip=FRAME_SKIP, render=RENDER, model_type=MODEL_TYPE)

    # Wrap in DummyVecEnv for compatibility
    env = DummyVecEnv([lambda: env])

    # Add frame stacking ONLY for improved models
    if use_frame_stack:
        env = VecFrameStack(env, n_stack=4)
        print("Using frame stacking (4 frames) for improved model")

    # Load trained model
    print(f"\nLoading model from: {MODEL_PATH}")
    try:
        model = PPO.load(MODEL_PATH, env=env)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nMake sure:")
        print("  1. The MODEL_PATH is correct")
        print("  2. The model was trained with stable-baselines3 PPO")
        print("  3. MODEL_TYPE matches your training config:")
        print("     - 'baseline' for MultiInputPolicy (original baseline)")
        print("     - 'improved' for CnnPolicy (improved/optimized versions)")
        return

    # Test the model
    print(f"\n{'='*70}")
    print("Starting test episodes...")
    print(f"{'='*70}\n")

    episode_rewards = []
    episode_lengths = []

    for episode in range(N_EPISODES):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0

        print(f"Episode {episode + 1}/{N_EPISODES}")

        while not done:
            # Get action from model
            action, _states = model.predict(obs, deterministic=DETERMINISTIC)

            # Take action
            obs, reward, done, info = env.step(action)

            episode_reward += reward[0]
            episode_length += 1

            # Slow down for visualization if rendering
            if RENDER:
                sleep(SLEEP_TIME)

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        print(f"  Reward: {episode_reward:.2f}, Length: {episode_length} steps")

    # Calculate statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)

    # Print summary
    print(f"\n{'='*70}")
    print("TEST RESULTS")
    print(f"{'='*70}")
    print(f"Episodes: {N_EPISODES}")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Min reward: {np.min(episode_rewards):.2f}")
    print(f"Max reward: {np.max(episode_rewards):.2f}")
    print(f"Mean episode length: {mean_length:.2f} ± {std_length:.2f}")
    print(f"{'='*70}\n")

    env.close()


if __name__ == "__main__":
    test_model()
