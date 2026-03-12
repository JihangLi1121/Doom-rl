#!/usr/bin/env python3
"""
Test script to verify VizDoom RL environment setup
"""

import sys

def test_imports():
    """Test if all required packages are installed"""
    print("Testing package imports...")
    
    tests = {
        "PyTorch": lambda: __import__('torch'),
        "TensorFlow": lambda: __import__('tensorflow'),
        "VizDoom": lambda: __import__('vizdoom'),
        "Gymnasium": lambda: __import__('gymnasium'),
        "Stable-Baselines3": lambda: __import__('stable_baselines3'),
        "PyTorch Geometric": lambda: __import__('torch_geometric'),
        "OpenCV": lambda: __import__('cv2'),
        "NumPy": lambda: __import__('numpy'),
        "Pandas": lambda: __import__('pandas'),
        "Matplotlib": lambda: __import__('matplotlib'),
        "TensorBoard": lambda: __import__('tensorboard'),
    }
    
    failed = []
    for name, test_func in tests.items():
        try:
            test_func()
            print(f"✓ {name}")
        except ImportError as e:
            print(f"✗ {name}: {e}")
            failed.append(name)
    
    return len(failed) == 0


def test_cuda():
    """Test CUDA availability"""
    print("\nTesting CUDA...")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    except Exception as e:
        print(f"Error testing PyTorch CUDA: {e}")
    
    try:
        import tensorflow as tf
        print(f"\nTensorFlow version: {tf.__version__}")
        gpus = tf.config.list_physical_devices('GPU')
        print(f"TensorFlow GPU devices: {len(gpus)}")
        for gpu in gpus:
            print(f"  {gpu}")
    except Exception as e:
        print(f"Error testing TensorFlow GPU: {e}")


def test_vizdoom():
    """Test VizDoom environment"""
    print("\nTesting VizDoom...")
    
    try:
        import gymnasium as gym
        from vizdoom import gymnasium_wrapper
        
        print("Creating VizdoomBasic-v0 environment...")
        env = gym.make("VizdoomBasic-v0")
        
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        
        print("Running random episode...")
        obs, info = env.reset()
        
        # Handle dict or array observations
        if isinstance(obs, dict):
            print(f"Observation type: Dict")
            print(f"  Screen shape: {obs['screen'].shape}")
            if 'gamevariables' in obs:
                print(f"  Game variables shape: {obs['gamevariables'].shape}")
        else:
            print(f"Observation shape: {obs.shape}")
        
        total_reward = 0
        for step in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        print(f"Episode finished in {step + 1} steps")
        print(f"Total reward: {total_reward}")
        
        env.close()
        print("✓ VizDoom test passed")
        return True
        
    except Exception as e:
        print(f"✗ VizDoom test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_stable_baselines3():
    """Test Stable-Baselines3"""
    print("\nTesting Stable-Baselines3...")
    
    try:
        import gymnasium as gym
        from vizdoom import gymnasium_wrapper
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from gymnasium.wrappers import FlattenObservation
        
        print("Creating environment and PPO model...")
        
        # Create environment and flatten dict observations
        def make_env():
            env = gym.make("VizdoomBasic-v0", frame_skip=4)
            # Flatten dict observations to work with CnnPolicy
            env = FlattenObservation(env)
            return env
        
        env = DummyVecEnv([make_env])
        
        model = PPO("MlpPolicy", env, verbose=0)
        print(f"Model created: {type(model).__name__}")
        print(f"Policy: {type(model.policy).__name__}")
        
        print("Testing short training run (100 steps)...")
        model.learn(total_timesteps=100)
        
        print("✓ Stable-Baselines3 test passed")
        return True
        
    except Exception as e:
        print(f"✗ Stable-Baselines3 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_custom_networks():
    """Test custom network architectures"""
    print("\nTesting custom networks...")
    
    try:
        import torch
        import torch.nn as nn
        
        # Define test networks inline
        class TestCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=8, stride=4),
                    nn.ReLU(),
                    nn.Flatten()
                )
            
            def forward(self, x):
                return self.conv(x.float() / 255.0)
        
        print("Testing CNN...")
        cnn = TestCNN()
        dummy_input = torch.randn(1, 3, 120, 160)
        output = cnn(dummy_input)
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        
        print("✓ Custom networks test passed")
        return True
        
    except Exception as e:
        print(f"✗ Custom networks test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 80)
    print("VizDoom RL Environment Test Suite")
    print("=" * 80)
    
    results = []
    
    # Test imports
    results.append(("Package Imports", test_imports()))
    
    # Test CUDA
    test_cuda()
    
    # Test VizDoom
    results.append(("VizDoom", test_vizdoom()))
    
    # Test Stable-Baselines3
    results.append(("Stable-Baselines3", test_stable_baselines3()))
    
    # Test custom networks
    results.append(("Custom Networks", test_custom_networks()))
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name:30s} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Environment is ready for training.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
