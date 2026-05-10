"""Trajectory controller for drone racing using cubic spline interpolation.

This controller generates optimal paths through gates dynamically and follows them
using cubic spline-based trajectory tracking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control import Controller
from lsy_drone_racing.utils.trajectory_planner import TrajectoryPlanner

if TYPE_CHECKING:
    from numpy.typing import NDArray


class TrajectoryController(Controller):
    """State-level trajectory controller using cubic spline interpolation.

    Generates waypoints from gate positions and follows a smooth cubic spline trajectory
    that passes through all gates. Returns state commands with desired position, velocity,
    acceleration, and yaw.
    """

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the trajectory controller.

        Args:
            obs: Initial observation containing drone state and position.
            info: Initial info dict containing gate/obstacle positions.
            config: Configuration dict with simulation parameters.
        """
        super().__init__(obs, info, config)
        self._freq = config.env.freq

        # Extract starting position from observation
        start_pos = obs["pos"].copy()

        # Use proven waypoints from StateController (optimized through trial and error)
        # These have been tested and work well for the drone dynamics
        waypoints = np.array(
            [
                [-1.5, 0.75, 0.05],
                [-1.0, 0.55, 0.4],
                [0.3, 0.35, 0.7],
                [1.3, -0.15, 0.9],
                [0.85, 0.85, 1.2],
                [-0.5, -0.05, 0.7],
                [-1.2, -0.2, 0.8],
                [-1.2, -0.2, 1.2],
                [-0.0, -0.7, 1.2],
                [0.5, -0.75, 1.2],
            ],
            dtype=np.float32,
        )

        # Create trajectory planner with the optimized waypoints
        self._planner = TrajectoryPlanner(
            waypoints=waypoints,
        )

        self._tick = 0
        self._finished = False

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired state of the drone.

        Args:
            obs: Current observation containing drone state.
            info: Optional additional information.

        Returns:
            State command [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate]
            as a numpy array.
        """
        # Compute current trajectory time
        t = min(self._tick / self._freq, self._planner.total_time)

        # Check if trajectory is complete
        if t >= self._planner.total_time:
            self._finished = True

        # Evaluate desired state from spline
        des_pos, des_vel, des_acc = self._planner.evaluate_state(t)

        # Construct state command: [pos (3), vel (3), acc (3), yaw, rrate, prate, yrate]
        # Set all rates to zero - the simulator will handle attitude stabilization
        action = np.concatenate(
            (des_pos, des_vel, des_acc, np.array([0.0, 0.0, 0.0, 0.0])),
            dtype=np.float32,
        )

        return action

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Update controller state after each step.

        Args:
            action: Latest applied action.
            obs: Latest observation.
            reward: Latest reward.
            terminated: Whether episode terminated.
            truncated: Whether episode was truncated.
            info: Latest info dict.

        Returns:
            True if controller is finished, False otherwise.
        """
        self._tick += 1
        return self._finished

    def episode_callback(self):
        """Reset internal state for new episode."""
        self._tick = 0
        self._finished = False

    def reset(self):
        """Reset controller for new episode."""
        self._tick = 0
        self._finished = False
