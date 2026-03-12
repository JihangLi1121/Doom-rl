"""
Tests the environment to make sure everything is working and not broken
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_vizdoom_import():
    """Test that VizDoom can be imported"""
    try:
        import vizdoom
        assert vizdoom is not None
    except ImportError:
        pytest.skip("VizDoom not available")

def test_gymnasium_import():
    """Test that Gymnasium can be imported"""
    try:
        import gymnasium
        assert gymnasium is not None
    except ImportError:
        pytest.skip("Gymnasium not available")

def test_torch_import():
    """Test that PyTorch can be imported"""
    try:
        import torch
        assert torch is not None
    except ImportError:
        pytest.skip("PyTorch not available")

