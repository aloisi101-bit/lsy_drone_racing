"""Simple test script for AttitudeMPCAdaptive controller.

Tests:
1. Controller instantiation
2. Gate change detection logic
3. Adaptive trajectory generation
"""

import os
import sys
from pathlib import Path

# Set scipy environment variable before importing anything
os.environ["SCIPY_ARRAY_API"] = "1"

import numpy as np

# Add workspace to path
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

from lsy_drone_racing.control.attitude_mpc_adaptive import AttitudeMPCAdaptive


class MockConfig:
    """Mock config object for testing."""
    class Env:
        freq = 50
        control_mode = "attitude"
    
    class Sim:
        drone_model = "cf21B_500"  # Valid drone model from config
    
    env = Env()
    sim = Sim()


def test_gate_change_detection():
    """Test _check_replanning_needed() logic."""
    print("\n=== Test: Gate Change Detection ===")
    
    # Create mock observation
    n_gates = 4
    obs = {
        "pos": np.array([0.0, 0.0, 0.5]),
        "quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "vel": np.array([0.0, 0.0, 0.0]),
        "ang_vel": np.array([0.0, 0.0, 0.0]),
        "target_gate": 0,
        "gates_pos": np.random.randn(n_gates, 3),
        "gates_quat": np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n_gates, 1)),
        "gates_visited": np.array([False, False, False, False]),
        "obstacles_pos": np.random.randn(2, 3),
        "obstacles_visited": np.array([False, False]),
    }
    
    config = MockConfig()
    
    try:
        controller = AttitudeMPCAdaptive(obs, {}, config)
        print("✓ Controller instantiated successfully")
    except Exception as e:
        print(f"✗ Controller instantiation failed: {e}")
        return False
    
    # Test 1: No gates visited initially
    gates_visited = np.array([False, False, False, False])
    replanning_needed, discovered = controller._check_replanning_needed(gates_visited)
    assert not replanning_needed, "Should not replan when no gates discovered"
    assert len(discovered) == 0, "Should have no discovered gates"
    print("✓ Test 1: No replanning when gates unvisited")
    
    # Test 2: First gate discovered
    gates_visited = np.array([True, False, False, False])
    replanning_needed, discovered = controller._check_replanning_needed(gates_visited)
    assert replanning_needed, "Should replan when gate is discovered"
    assert len(discovered) == 1, "Should detect 1 newly discovered gate"
    assert discovered[0] == 0, "Should be gate 0"
    controller._prev_gates_visited = gates_visited.copy()
    print("✓ Test 2: Replanning triggered on gate discovery")
    
    # Test 3: Second gate discovered
    gates_visited = np.array([True, True, False, False])
    replanning_needed, discovered = controller._check_replanning_needed(gates_visited)
    assert replanning_needed, "Should replan when new gate discovered"
    assert len(discovered) == 1, "Should detect 1 newly discovered gate"
    assert discovered[0] == 1, "Should be gate 1"
    controller._prev_gates_visited = gates_visited.copy()
    print("✓ Test 3: Replanning triggered on second gate discovery")
    
    # Test 4: No new gates
    gates_visited = np.array([True, True, False, False])
    replanning_needed, discovered = controller._check_replanning_needed(gates_visited)
    assert not replanning_needed, "Should not replan when no new gates"
    assert len(discovered) == 0, "Should have no newly discovered gates"
    print("✓ Test 4: No replanning when no new gates")
    
    return True


def test_trajectory_generation():
    """Test _generate_adaptive_trajectory() logic."""
    print("\n=== Test: Adaptive Trajectory Generation ===")
    
    n_gates = 4
    obs = {
        "pos": np.array([0.0, 0.0, 0.5]),
        "quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "vel": np.array([0.0, 0.0, 0.0]),
        "ang_vel": np.array([0.0, 0.0, 0.0]),
        "target_gate": 0,
        "gates_pos": np.array([
            [0.5, 0.5, 0.7],
            [1.0, 0.5, 0.9],
            [1.0, -0.5, 1.1],
            [0.5, -0.5, 0.8],
        ]),
        "gates_quat": np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n_gates, 1)),
        "gates_visited": np.array([True, False, False, False]),
        "obstacles_pos": np.random.randn(2, 3),
        "obstacles_visited": np.array([False, False]),
    }
    
    config = MockConfig()
    controller = AttitudeMPCAdaptive(obs, {}, config)
    
    # Test 1: Trajectory with single discovered gate
    drone_pos = obs["pos"].copy()
    gates_pos = obs["gates_pos"]
    discovered_indices = [0]  # Only gate 0 discovered
    obstacles = obs["obstacles_pos"]
    
    try:
        pos, vel, yaw = controller._generate_adaptive_trajectory(
            drone_pos, gates_pos, discovered_indices, obstacles
        )
        assert pos.shape[1] == 3, "Position trajectory should have 3 dimensions"
        assert vel.shape[1] == 3, "Velocity trajectory should have 3 dimensions"
        assert len(yaw) == len(pos), "Yaw should match trajectory length"
        print("✓ Test 1: Trajectory generated for single gate")
    except Exception as e:
        print(f"✗ Trajectory generation failed: {e}")
        return False
    
    # Test 2: Trajectory with multiple discovered gates
    discovered_indices = [0, 1, 2]
    try:
        pos, vel, yaw = controller._generate_adaptive_trajectory(
            drone_pos, gates_pos, discovered_indices, obstacles
        )
        # Verify trajectory passes near first gate
        first_gate = gates_pos[0]
        distances = np.linalg.norm(pos - first_gate, axis=1)
        min_distance = np.min(distances)
        print(f"  - Min distance to gate 0: {min_distance:.4f} m")
        assert min_distance < 1.0, "Trajectory should pass near discovered gates"
        print("✓ Test 2: Trajectory generated for multiple gates")
    except Exception as e:
        print(f"✗ Multi-gate trajectory generation failed: {e}")
        return False
    
    return True


def test_observation_structure():
    """Verify expected observation structure."""
    print("\n=== Test: Observation Structure ===")
    
    # Create realistic observation
    n_gates = 4
    n_obstacles = 2
    obs = {
        "pos": np.array([0.0, 0.0, 0.5], dtype=np.float32),
        "quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "vel": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "ang_vel": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "target_gate": 0,
        "gates_pos": np.random.randn(n_gates, 3).astype(np.float32),
        "gates_quat": np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n_gates, 1)).astype(np.float32),
        "gates_visited": np.array([False, False, False, False]),
        "obstacles_pos": np.random.randn(n_obstacles, 3).astype(np.float32),
        "obstacles_visited": np.array([False, False]),
    }
    
    config = MockConfig()
    
    try:
        controller = AttitudeMPCAdaptive(obs, {}, config)
        
        # Verify internal state initialized correctly
        assert controller._prev_gates_visited.shape == obs["gates_visited"].shape
        assert len(controller._discovered_gates_indices) == 0
        assert controller._current_trajectory_index == 0
        
        print("✓ Observation structure validated")
        print("✓ Controller state initialized correctly")
        return True
    except Exception as e:
        print(f"✗ Observation validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("AttitudeMPCAdaptive Validation Tests")
    print("=" * 60)
    
    results = []
    
    results.append(("Observation Structure", test_observation_structure()))
    results.append(("Gate Change Detection", test_gate_change_detection()))
    results.append(("Trajectory Generation", test_trajectory_generation()))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:.<40} {status}")
    
    all_passed = all(result for _, result in results)
    if all_passed:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
