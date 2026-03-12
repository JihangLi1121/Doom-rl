#!/usr/bin/env python3
"""
Visualization Script for VizDoom RL Models
Compares baseline vs improved models for Basic and Corridor scenarios
Outputs 4 graphs:
1. Training comparison - Basic (baseline vs improved)
2. Training comparison - Corridor (baseline vs improved)
3. Evaluation comparison - Basic (baseline vs improved)
4. Evaluation comparison - Corridor (baseline vs improved)
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import gymnasium as gym
import cv2
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import vizdoom.gymnasium_wrapper  # noqa


# ==================== CONFIGURATION ====================
# Paths to TensorBoard log directories
LOGS_DIR = "./logs"

# Model paths for evaluation
MODELS = {
    "basic": {
        "baseline": "./models/VizdoomBasic-v0_baseline/final_model.zip",
        "improved": "./models/VizdoomBasic-v0_baseline_improved/final_model.zip"
    },
    "corridor": {
        "baseline": "./models/VizdoomCorridor-v0_baseline/final_model.zip",
        "improved": "./models/VizdoomCorridor-v0_corridor_optimized/final_model.zip"
    }
}

# TensorBoard log directories
LOG_DIRS = {
    "basic": {
        "baseline": "./logs/VizdoomBasic-v0_baseline",
        "improved": "./logs/VizdoomBasic-v0_baseline_improved"
    },
    "corridor": {
        "baseline": "./logs/VizdoomCorridor-v0_baseline",
        "improved": "./logs/VizdoomCorridor-v0_corridor_optimized"
    }
}

# Reward normalization factors (to undo reward amplification in old logs)
REWARD_NORMALIZATION = {
    "basic": {
        "baseline": 1.0,
        "improved": 2.0
    },
    "corridor": {
        "baseline": 1.0,
        "improved": 1.0
    }
}

# Evaluation settings
EVAL_EPISODES = 20  # Number of episodes to evaluate each model
FRAME_SKIP = 4
IMAGE_SHAPE = (60, 80)

# Output settings
OUTPUT_DIR = "./visualizations"
# =======================================================


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


def read_tensorboard_logs(log_dir, tag="rollout/ep_rew_mean"):
    """Read training metrics from TensorBoard logs."""
    if not os.path.exists(log_dir):
        print(f"Warning: Log directory not found: {log_dir}")
        return None, None

    all_steps = []
    all_values = []

    # Find all subdirectories AND the main directory
    dirs_to_read = [log_dir]  # Always include main dir
    for item in os.listdir(log_dir):
        item_path = os.path.join(log_dir, item)
        if os.path.isdir(item_path):
            dirs_to_read.append(item_path)

    print(f"  Scanning {len(dirs_to_read)} directories in {os.path.basename(log_dir)}")

    for dir_path in dirs_to_read:
        try:
            ea = event_accumulator.EventAccumulator(dir_path)
            ea.Reload()

            available_tags = ea.Tags().get('scalars', [])
            if not available_tags:
                continue

            # Find the right tag
            current_tag = None
            for try_tag in [tag, 'rollout/ep_rew_mean', 'train/episode_reward', 'ep_rew_mean', 'episode_reward']:
                if try_tag in available_tags:
                    current_tag = try_tag
                    break

            if current_tag is None:
                continue

            events = ea.Scalars(current_tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]

            if steps:
                all_steps.extend(steps)
                all_values.extend(values)
                print(f"    Read {len(steps)} points from {os.path.basename(dir_path)} (tag: {current_tag})")

        except Exception:
            continue

    if not all_steps:
        print(f"Warning: No data found in {log_dir}")
        return None, None

    # Sort by steps and remove duplicates (keep latest value for each step)
    sorted_indices = np.argsort(all_steps)
    steps_sorted = np.array(all_steps)[sorted_indices]
    values_sorted = np.array(all_values)[sorted_indices]

    # Remove duplicate steps (keep last occurrence)
    unique_steps, unique_indices = np.unique(steps_sorted[::-1], return_index=True)
    unique_indices = len(steps_sorted) - 1 - unique_indices

    print(f"  Total: {len(unique_steps)} unique data points (max step: {unique_steps.max() if len(unique_steps) > 0 else 0})")

    return steps_sorted[np.sort(unique_indices)], values_sorted[np.sort(unique_indices)]


def smooth_data(data, window=10):
    """Apply moving average smoothing to data."""
    if len(data) < window:
        return data
    # Use 'same' mode to keep the same length as input
    return np.convolve(data, np.ones(window)/window, mode='same')


def evaluate_model(model_path, env_id, model_type="improved", num_episodes=EVAL_EPISODES):
    """Evaluate a trained model and return rewards."""
    if not os.path.exists(model_path):
        print(f"Warning: Model not found: {model_path}")
        return None

    print(f"Evaluating: {model_path}")

    try:
        # Create environment based on model type
        env = gym.make(env_id, frame_skip=FRAME_SKIP, render_mode='rgb_array')

        if model_type == "baseline":
            env = BaselineObservationWrapper(env, image_shape=IMAGE_SHAPE)
            env = DummyVecEnv([lambda: env])
            # No frame stacking for baseline
        else:  # improved
            env = ScreenOnlyWrapper(env, image_shape=IMAGE_SHAPE)
            env = DummyVecEnv([lambda: env])
            env = VecFrameStack(env, n_stack=4)

        # Load model
        model = PPO.load(model_path, env=env)

        # Evaluate
        rewards = []
        for ep in range(num_episodes):
            obs = env.reset()
            episode_reward = 0
            done = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward[0]

                if done[0]:
                    # Get natural reward from info if available
                    if 'episode' in info[0]:
                        episode_reward = info[0]['episode']['r']
                    rewards.append(episode_reward)
                    break

        env.close()
        return np.array(rewards)

    except Exception as e:
        print(f"Error evaluating model {model_path}: {e}")
        return None


def create_training_comparison_plot(scenario, baseline_data, improved_data, output_path):
    """Create training comparison plot for a scenario."""
    fig, ax = plt.subplots(figsize=(12, 6))

    baseline_norm = REWARD_NORMALIZATION.get(scenario, {}).get("baseline", 1.0)
    improved_norm = REWARD_NORMALIZATION.get(scenario, {}).get("improved", 1.0)

    final_baseline = None
    final_improved = None

    if baseline_data[0] is not None:
        steps_b, values_b = baseline_data
        values_b_normalized = values_b / baseline_norm
        smoothed_b = smooth_data(values_b_normalized, window=20)
        steps_smooth_b = steps_b[:len(smoothed_b)]
        ax.plot(steps_smooth_b, smoothed_b, 'c-', label='Baseline', linewidth=2, alpha=0.9)
        ax.fill_between(steps_smooth_b, smoothed_b * 0.95, smoothed_b * 1.05, color='cyan', alpha=0.2)
        final_baseline = np.mean(values_b_normalized[-50:]) if len(values_b_normalized) >= 50 else np.mean(values_b_normalized)

    if improved_data[0] is not None:
        steps_i, values_i = improved_data
        values_i_normalized = values_i / improved_norm
        smoothed_i = smooth_data(values_i_normalized, window=20)
        steps_smooth_i = steps_i[:len(smoothed_i)]
        ax.plot(steps_smooth_i, smoothed_i, color='orange', linestyle='-', label='Improved', linewidth=2, alpha=0.9)
        ax.fill_between(steps_smooth_i, smoothed_i * 0.95, smoothed_i * 1.05, color='orange', alpha=0.2)
        final_improved = np.mean(values_i_normalized[-50:]) if len(values_i_normalized) >= 50 else np.mean(values_i_normalized)

    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Episode Reward', fontsize=12)
    ax.set_title(f'Training Comparison - {scenario.capitalize()}', fontsize=14)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 300000)  # Limit x-axis to 300k steps

    stats_text = ""
    if final_baseline is not None:
        stats_text += f"Baseline Final: {final_baseline:.1f}\n"
    if final_improved is not None:
        stats_text += f"Improved Final: {final_improved:.1f}"
        if final_baseline is not None and final_baseline != 0:
            improvement = ((final_improved - final_baseline) / abs(final_baseline)) * 100
            stats_text += f"\nImprovement: {improvement:+.1f}%"

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_evaluation_comparison_plot(scenario, baseline_rewards, improved_rewards, output_path):
    """Create evaluation comparison plot for a scenario."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    labels = ['Baseline', 'Improved']
    means = []
    stds = []
    colors = ['cyan', 'orange']

    if baseline_rewards is not None:
        means.append(np.mean(baseline_rewards))
        stds.append(np.std(baseline_rewards))
    else:
        means.append(0)
        stds.append(0)

    if improved_rewards is not None:
        means.append(np.mean(improved_rewards))
        stds.append(np.std(improved_rewards))
    else:
        means.append(0)
        stds.append(0)

    x = np.arange(len(labels))
    bars = ax1.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Average Episode Reward', fontsize=12)
    ax1.set_title(f'Evaluation Comparison - {scenario.capitalize()}\n({EVAL_EPISODES} Episodes)', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax1.annotate(f'{mean:.1f}±{std:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    data_to_plot = []
    box_labels = []
    box_colors = []

    if baseline_rewards is not None:
        data_to_plot.append(baseline_rewards)
        box_labels.append('Baseline')
        box_colors.append('cyan')

    if improved_rewards is not None:
        data_to_plot.append(improved_rewards)
        box_labels.append('Improved')
        box_colors.append('orange')

    if data_to_plot:
        bp = ax2.boxplot(data_to_plot, labels=box_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

    ax2.set_ylabel('Episode Reward', fontsize=12)
    ax2.set_title(f'Reward Distribution - {scenario.capitalize()}', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add improvement percentage
    if baseline_rewards is not None and improved_rewards is not None:
        improvement = ((means[1] - means[0]) / abs(means[0])) * 100 if means[0] != 0 else 0
        fig.suptitle(f'Improvement: {improvement:+.1f}%', fontsize=12, y=1.02, fontweight='bold',
                    color='green' if improvement > 0 else 'red')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Main function to generate all visualization graphs."""
    print("\n" + "="*70)
    print("VIZDOOM MODEL VISUALIZATION")
    print("="*70)
    print("Generating comparison graphs for Basic and Corridor scenarios")
    print("="*70 + "\n")

    # Create timestamped output directory to avoid overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(OUTPUT_DIR, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    # Store results for summary
    results = {}

    # ========================
    # TRAINING COMPARISONS
    # ========================
    print("\n--- TRAINING COMPARISONS (from TensorBoard logs) ---\n")

    for scenario in ["basic", "corridor"]:
        print(f"\nProcessing {scenario.upper()} training logs...")

        baseline_log = LOG_DIRS[scenario]["baseline"]
        improved_log = LOG_DIRS[scenario]["improved"]

        baseline_data = read_tensorboard_logs(baseline_log)
        improved_data = read_tensorboard_logs(improved_log)

        output_path = os.path.join(output_dir, f"training_{scenario}.png")
        create_training_comparison_plot(scenario, baseline_data, improved_data, output_path)

        results[f"{scenario}_training"] = {
            "baseline": baseline_data,
            "improved": improved_data
        }

    # ========================
    # EVALUATION COMPARISONS
    # ========================
    print("\n--- EVALUATION COMPARISONS (running models) ---\n")

    for scenario, env_id in [("basic", "VizdoomBasic-v0"), ("corridor", "VizdoomCorridor-v0")]:
        print(f"\nEvaluating {scenario.upper()} models...")

        baseline_path = MODELS[scenario]["baseline"]
        improved_path = MODELS[scenario]["improved"]

        # Evaluate baseline
        print(f"  Evaluating baseline model...")
        baseline_rewards = evaluate_model(baseline_path, env_id, model_type="baseline")

        # Evaluate improved
        print(f"  Evaluating improved model...")
        improved_rewards = evaluate_model(improved_path, env_id, model_type="improved")

        output_path = os.path.join(output_dir, f"evaluation_{scenario}.png")
        create_evaluation_comparison_plot(scenario, baseline_rewards, improved_rewards, output_path)

        results[f"{scenario}_eval"] = {
            "baseline": baseline_rewards,
            "improved": improved_rewards
        }

    # ========================
    # PRINT SUMMARY
    # ========================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    for scenario in ["basic", "corridor"]:
        print(f"\n{scenario.upper()}:")

        # Training summary
        train_data = results.get(f"{scenario}_training", {})
        if train_data.get("baseline") and train_data["baseline"][0] is not None:
            final_baseline = np.mean(train_data["baseline"][1][-20:])
            print(f"  Training - Baseline Final Reward: {final_baseline:.1f}")
        if train_data.get("improved") and train_data["improved"][0] is not None:
            final_improved = np.mean(train_data["improved"][1][-20:])
            print(f"  Training - Improved Final Reward: {final_improved:.1f}")

        # Evaluation summary
        eval_data = results.get(f"{scenario}_eval", {})
        if eval_data.get("baseline") is not None:
            print(f"  Evaluation - Baseline: {np.mean(eval_data['baseline']):.1f} ± {np.std(eval_data['baseline']):.1f}")
        if eval_data.get("improved") is not None:
            print(f"  Evaluation - Improved: {np.mean(eval_data['improved']):.1f} ± {np.std(eval_data['improved']):.1f}")

        # Improvement
        if eval_data.get("baseline") is not None and eval_data.get("improved") is not None:
            baseline_mean = np.mean(eval_data['baseline'])
            improved_mean = np.mean(eval_data['improved'])
            if baseline_mean != 0:
                improvement = ((improved_mean - baseline_mean) / abs(baseline_mean)) * 100
                print(f"  Improvement: {improvement:+.1f}%")

    print("\n" + "="*70)
    print(f"All graphs saved to: {output_dir}/")
    print("  - training_basic.png")
    print("  - training_corridor.png")
    print("  - evaluation_basic.png")
    print("  - evaluation_corridor.png")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
