#!/usr/bin/env python3
"""
Video Recording Script for VizDoom RL Models
Records agent gameplay with slow motion and action overlay
"""

import os
import cv2
import numpy as np
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import vizdoom.gymnasium_wrapper  # noqa
from vizdoom import DoomGame, ScreenFormat, ScreenResolution
import vizdoom as vzd


# ==================== CONFIGURATION ====================
# Environment settings
# ENV_ID = "VizdoomBasic-v0"  # Options: "VizdoomBasic-v0", "VizdoomCorridor-v0"
ENV_ID = "VizdoomCorridor-v0"
FRAME_SKIP = 4
IMAGE_SHAPE = (60, 80)

# Difficulty setting - MUST MATCH TRAINING!
# If you trained with DOOM_SKILL=1, test with DOOM_SKILL=1
DOOM_SKILL = 5  # 1=easiest, 5=nightmare

# Model to record (set model_type to match training)
MODEL_PATH = "./models/VizdoomBasic-v0_baseline/final_model.zip"
# MODEL_PATH = "./models/VizdoomBasic-v0_baseline_improved/final_model.zip"
# MODEL_PATH = "./models/VizdoomCorridor-v0_curriculum/final_model.zip"
MODEL_TYPE = "baseline"  # "baseline", "improved", or "corridor" (grayscale)

# Recording settings
NUM_EPISODES = 20          # Number of episodes to record
SLOW_MOTION_FACTOR = 2    # Duplicate each frame N times (higher = slower)
OUTPUT_FPS = 30           # Output video FPS (effective FPS = OUTPUT_FPS / SLOW_MOTION_FACTOR)
VIDEO_SCALE = 4           # Scale up video for better visibility (4x = 320x240 -> 1280x960)
SHOW_ACTION_OVERLAY = True  # Show action text on video
OUTPUT_DIR = "./videos"
# =======================================================


# Action names for different environments
ACTION_NAMES = {
    "VizdoomBasic-v0": ["MOVE_LEFT", "MOVE_RIGHT", "ATTACK"],
    "VizdoomCorridor-v0": ["ATTACK", "MOVE_RIGHT", "MOVE_LEFT", "MOVE_FORWARD", "MOVE_BACKWARD", "TURN_RIGHT", "TURN_LEFT"],
    "VizdoomDefendCenter-v0": ["TURN_LEFT", "TURN_RIGHT", "ATTACK"],
}


class VizDoomGymBasicRecord(Env):
    """
    Custom VizDoom Gym Environment for RECORDING basic models with Dict observations.
    For baseline models trained with MultiInputPolicy.
    """
    def __init__(self):
        super().__init__()

        self.game = DoomGame()
        self.game.load_config(vzd.scenarios_path + "/basic.cfg")

        # Render settings
        self.game.set_window_visible(False)
        self.game.set_screen_resolution(ScreenResolution.RES_320X240)
        self.game.set_screen_format(ScreenFormat.RGB24)

        # Initialize
        self.game.init()
        print(f"✅ VizDoomGymBasicRecord initialized")

        # Spaces - Dict observation with RGB screen (channels-first for SB3)
        self.observation_space = gym.spaces.Dict({
            'screen': Box(low=0, high=255, shape=(3, IMAGE_SHAPE[0], IMAGE_SHAPE[1]), dtype=np.uint8)
        })
        self.action_space = Discrete(4)  # Gymnasium wrapper default: 4 actions

        # Frame capture for video recording
        self.last_frame = None

    def step(self, action):
        actions = np.identity(4, dtype=np.uint8)
        reward = self.game.make_action(actions[action], FRAME_SKIP)

        terminated = self.game.is_episode_finished()
        truncated = False
        info = {}

        if self.game.get_state():
            state = self.game.get_state().screen_buffer
            self._capture_frame(state)
            obs = self._process_observation(state)
        else:
            obs = {'screen': np.zeros((3, IMAGE_SHAPE[0], IMAGE_SHAPE[1]), dtype=np.uint8)}

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        self._capture_frame(state)
        return self._process_observation(state), {}

    def _process_observation(self, observation):
        """Process observation to Dict format with resized RGB screen."""
        # observation is in CHW format from VizDoom
        if len(observation.shape) == 3 and observation.shape[0] == 3:
            observation = np.moveaxis(observation, 0, -1)  # CHW -> HWC
        # Resize
        resized = cv2.resize(observation, (IMAGE_SHAPE[1], IMAGE_SHAPE[0]), interpolation=cv2.INTER_CUBIC)
        # Back to CHW for SB3
        transposed = np.transpose(resized, (2, 0, 1))
        return {'screen': transposed}

    def _capture_frame(self, frame):
        """Capture raw frame for video recording."""
        if frame is not None:
            self.last_frame = frame
            if len(self.last_frame.shape) == 3 and self.last_frame.shape[0] == 3:
                self.last_frame = np.transpose(self.last_frame, (1, 2, 0))

    def get_frame(self):
        return self.last_frame

    def render(self):
        pass

    def close(self):
        self.game.close()


class VizDoomGymCorridorRecord(Env):
    """
    Custom VizDoom Gym Environment for RECORDING corridor models.
    Sets doom_skill BEFORE game.init() and includes frame capture.
    """
    def __init__(self, doom_skill=DOOM_SKILL):
        super().__init__()

        self.game = DoomGame()
        self.game.load_config(vzd.scenarios_path + "/deadly_corridor.cfg")

        # Set difficulty BEFORE init - this is critical!
        self.game.set_doom_skill(doom_skill)

        # Render settings
        self.game.set_window_visible(False)
        self.game.set_screen_resolution(ScreenResolution.RES_320X240)
        self.game.set_screen_format(ScreenFormat.RGB24)

        # Initialize
        self.game.init()
        print(f"✅ VizDoomGymCorridorRecord initialized with doom_skill={doom_skill}")

        # Spaces - must match training
        self.observation_space = Box(low=0, high=255, shape=(IMAGE_SHAPE[0], IMAGE_SHAPE[1], 1), dtype=np.uint8)
        self.action_space = Discrete(7)

        # Frame capture for video recording
        self.last_frame = None

    def step(self, action):
        actions = np.identity(7, dtype=np.uint8)
        reward = self.game.make_action(actions[action], FRAME_SKIP)

        terminated = self.game.is_episode_finished()
        truncated = False
        info = {}

        if self.game.get_state():
            state = self.game.get_state().screen_buffer
            self._capture_frame(state)
            state = self._grayscale(state)
        else:
            state = np.zeros(self.observation_space.shape, dtype=np.uint8)

        return state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        self._capture_frame(state)
        return self._grayscale(state), {}

    def _grayscale(self, observation):
        if len(observation.shape) == 3 and observation.shape[0] == 3:
            observation = np.moveaxis(observation, 0, -1)
        gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (IMAGE_SHAPE[1], IMAGE_SHAPE[0]), interpolation=cv2.INTER_CUBIC)
        return np.reshape(resized, (IMAGE_SHAPE[0], IMAGE_SHAPE[1], 1))

    def _capture_frame(self, frame):
        """Capture raw frame for video recording."""
        if frame is not None:
            self.last_frame = frame
            if len(self.last_frame.shape) == 3 and self.last_frame.shape[0] == 3:
                self.last_frame = np.transpose(self.last_frame, (1, 2, 0))

    def get_frame(self):
        return self.last_frame

    def render(self):
        pass

    def close(self):
        self.game.close()


class ScreenOnlyWrapper(gym.ObservationWrapper):
    """Extracts and resizes screen from VizDoom's dict observation space."""
    def __init__(self, env, image_shape=IMAGE_SHAPE):
        super().__init__(env)
        self.image_shape = image_shape
        self.image_shape_reverse = image_shape[::-1]

        if isinstance(env.observation_space, gym.spaces.Dict):
            original_screen = env.observation_space['screen']
            num_channels = original_screen.shape[-1]
        else:
            num_channels = env.observation_space.shape[-1]

        new_shape = (image_shape[0], image_shape[1], num_channels)
        self.observation_space = gym.spaces.Box(0, 255, shape=new_shape, dtype=np.uint8)

    def observation(self, obs):
        if isinstance(obs, dict):
            screen = obs['screen']
        else:
            screen = obs
        resized = cv2.resize(screen, self.image_shape_reverse)
        return resized


class GrayscaleWrapper(gym.ObservationWrapper):
    """Converts RGB to grayscale for corridor models trained with grayscale."""
    def __init__(self, env, image_shape=IMAGE_SHAPE):
        super().__init__(env)
        self.image_shape = image_shape
        self.image_shape_reverse = image_shape[::-1]

        # Grayscale = 1 channel
        new_shape = (image_shape[0], image_shape[1], 1)
        self.observation_space = gym.spaces.Box(0, 255, shape=new_shape, dtype=np.uint8)

    def observation(self, obs):
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
    """Limits action space to match the trained model (7 actions for corridor)."""
    def __init__(self, env, n_actions=7):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(n_actions)

    def action(self, action):
        # Pass action through directly (model outputs 0-6, env expects 0-6)
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
    """Wrapper for baseline models that use Dict observations."""
    def __init__(self, env, image_shape=IMAGE_SHAPE):
        super().__init__(env)
        self.image_shape = image_shape
        self.image_shape_reverse = image_shape[::-1]

        if isinstance(env.observation_space, gym.spaces.Dict):
            original_screen = env.observation_space['screen']
            num_channels = original_screen.shape[-1]
        else:
            num_channels = env.observation_space.shape[-1]

        new_shape = (num_channels, image_shape[0], image_shape[1])
        self.observation_space = gym.spaces.Dict({
            'screen': gym.spaces.Box(0, 255, shape=new_shape, dtype=np.uint8)
        })

    def observation(self, obs):
        if isinstance(obs, dict):
            screen = obs['screen']
        else:
            screen = obs
        resized = cv2.resize(screen, self.image_shape_reverse)
        transposed = np.transpose(resized, (2, 0, 1))
        return {'screen': transposed}


class FrameCaptureWrapper(gym.Wrapper):
    """Wrapper to capture raw frames for video recording."""
    def __init__(self, env):
        super().__init__(env)
        self.last_frame = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._capture_frame()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._capture_frame()
        return obs, reward, terminated, truncated, info

    def _capture_frame(self):
        """Capture raw frame from VizDoom game."""
        game = self.env.unwrapped.game
        if game.is_episode_finished():
            return
        state = game.get_state()
        if state is not None:
            self.last_frame = state.screen_buffer
            if self.last_frame is not None:
                # Convert from CHW to HWC if needed
                if len(self.last_frame.shape) == 3 and self.last_frame.shape[0] == 3:
                    self.last_frame = np.transpose(self.last_frame, (1, 2, 0))

    def get_frame(self):
        return self.last_frame


def add_text_overlay(frame, text, position="bottom"):
    """Add text overlay to frame."""
    frame = frame.copy()
    h, w = frame.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2

    # Get text size
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Position
    if position == "bottom":
        x = 10
        y = h - 20
    elif position == "top":
        x = 10
        y = 30
    else:
        x, y = position

    # Draw background rectangle
    cv2.rectangle(frame, (x - 5, y - text_h - 5), (x + text_w + 5, y + 5), (0, 0, 0), -1)

    # Draw text
    cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness)

    return frame


def record_episode(env, model, action_names, episode_num, slow_factor=SLOW_MOTION_FACTOR):
    """Record a single episode and return frames."""
    frames = []
    actions_taken = []

    obs = env.reset()
    done = False
    total_reward = 0
    step = 0

    # Get the environment that has get_frame() method
    # For custom corridor env, it's directly on base_env
    # For wrapper-based envs, we need to find the FrameCaptureWrapper
    capture_env = env.envs[0]
    while hasattr(capture_env, 'env') and not hasattr(capture_env, 'get_frame'):
        capture_env = capture_env.env

    while not done:
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)
        action_idx = action[0] if isinstance(action, np.ndarray) else action

        # Get current frame before step
        raw_frame = capture_env.get_frame()

        if raw_frame is not None:
            # Scale up frame
            scaled_frame = cv2.resize(raw_frame, None, fx=VIDEO_SCALE, fy=VIDEO_SCALE,
                                     interpolation=cv2.INTER_NEAREST)

            # Add action overlay if enabled
            if SHOW_ACTION_OVERLAY:
                action_name = action_names[action_idx] if action_idx < len(action_names) else f"ACTION_{action_idx}"
                overlay_text = f"Step: {step} | Action: {action_name} | Reward: {total_reward:.0f}"
                scaled_frame = add_text_overlay(scaled_frame, overlay_text, "top")

                # Add episode info at bottom
                ep_text = f"Episode {episode_num}"
                scaled_frame = add_text_overlay(scaled_frame, ep_text, "bottom")

            # Duplicate frame for slow motion
            for _ in range(slow_factor):
                frames.append(scaled_frame)

        actions_taken.append(action_idx)

        # Step environment
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]
        step += 1

        if done[0]:
            # Get final reward from info if available
            if 'episode' in info[0]:
                total_reward = info[0]['episode']['r']
            break

    return frames, total_reward, step


def main():
    """Main function to record agent gameplay videos."""
    print("\n" + "="*70)
    print("VIZDOOM VIDEO RECORDER")
    print("="*70)
    print(f"Environment: {ENV_ID}")
    print(f"Model: {MODEL_PATH}")
    print(f"Model type: {MODEL_TYPE}")
    print(f"Episodes to record: {NUM_EPISODES}")
    print(f"Slow motion factor: {SLOW_MOTION_FACTOR}x")
    print(f"Output FPS: {OUTPUT_FPS}")
    print(f"Video scale: {VIDEO_SCALE}x")
    print("="*70 + "\n")

    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(OUTPUT_DIR, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Get action names for this environment
    action_names = ACTION_NAMES.get(ENV_ID, [f"ACTION_{i}" for i in range(10)])

    # Create environment with frame capture
    print("Creating environment...")

    if MODEL_TYPE == "corridor":
        # Corridor model: Use CUSTOM environment that sets doom_skill BEFORE init!
        base_env = VizDoomGymCorridorRecord(doom_skill=DOOM_SKILL)
        env = DummyVecEnv([lambda: base_env])
    elif MODEL_TYPE == "baseline":
        # Baseline model: Use CUSTOM environment with Dict observations (MultiInputPolicy)
        base_env = VizDoomGymBasicRecord()
        env = DummyVecEnv([lambda: base_env])
    else:  # improved (RGB)
        base_env = gym.make(ENV_ID, frame_skip=FRAME_SKIP, render_mode='rgb_array')
        base_env = FrameCaptureWrapper(base_env)
        base_env = ScreenOnlyWrapper(base_env, image_shape=IMAGE_SHAPE)
        env = DummyVecEnv([lambda: base_env])
        env = VecFrameStack(env, n_stack=4)

    # Load model
    print(f"Loading model from {MODEL_PATH}...")
    model = PPO.load(MODEL_PATH, env=env)

    # Record episodes and track the best one
    all_episode_frames = []  # Store frames for each episode separately
    episode_rewards = []
    best_reward = float('-inf')
    best_episode_idx = 0

    for ep in range(NUM_EPISODES):
        print(f"\nRecording episode {ep + 1}/{NUM_EPISODES}...")
        frames, reward, steps = record_episode(env, model, action_names, ep + 1)
        all_episode_frames.append(frames)
        episode_rewards.append(reward)
        print(f"  Episode {ep + 1}: Reward = {reward:.0f}, Steps = {steps}, Frames = {len(frames)}")

        # Track best episode
        if reward > best_reward:
            best_reward = reward
            best_episode_idx = ep

    env.close()

    # Save video of BEST episode only
    best_frames = all_episode_frames[best_episode_idx]
    env_name = ENV_ID.replace("-v0", "").replace("Vizdoom", "")
    if best_frames:
        video_path = os.path.join(output_dir, f"{env_name}_{MODEL_TYPE}_best_run.mp4")

        print(f"\n🏆 Best episode: {best_episode_idx + 1} with reward {best_reward:.0f}")
        print(f"Saving best run video to {video_path}...")

        # Get frame dimensions
        h, w = best_frames[0].shape[:2]

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, OUTPUT_FPS, (w, h))

        for frame in best_frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        out.release()

        print(f"\nVideo saved successfully!")
        print(f"  Path: {video_path}")
        print(f"  Total frames: {len(best_frames)}")
        print(f"  Duration: {len(best_frames) / OUTPUT_FPS:.1f} seconds")
        print(f"  Resolution: {w}x{h}")

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Episodes tested: {NUM_EPISODES}")
    print(f"Average reward: {np.mean(episode_rewards):.1f} ± {np.std(episode_rewards):.1f}")
    print(f"Best episode: {best_episode_idx + 1} with reward {best_reward:.0f}")
    print(f"Video saved: {output_dir}/{env_name}_{MODEL_TYPE}_best_run.mp4")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
