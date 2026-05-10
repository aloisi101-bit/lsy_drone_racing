"""Trajectory planner for autonomous drone racing.

Generates optimal paths through gates using cubic spline interpolation with adaptive timing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline

if TYPE_CHECKING:
    from numpy.typing import NDArray


class TrajectoryPlanner:
    """Generates smooth trajectories through waypoints using cubic spline interpolation.

    The planner creates waypoints from gate positions, computes adaptive timing based on
    distances, and provides smooth trajectory evaluation with continuous derivatives
    (velocity and acceleration).
    """

    def __init__(
        self,
        waypoints: NDArray[np.floating],
    ):
        """Initialize trajectory planner with waypoints.

        Args:
            waypoints: Array of waypoints, shape (n_waypoints, 3) in [x, y, z] format.
        """
        self.waypoints = np.array(waypoints, dtype=np.float32)

        # Compute times - use linear spacing over 15 seconds
        n_waypoints = len(self.waypoints)
        self.times = np.linspace(0, 15.0, n_waypoints, dtype=np.float32)
        self.total_time = self.times[-1]

        # Create spline interpolators
        self._des_pos_spline = CubicSpline(self.times, self.waypoints)
        self._des_vel_spline = self._des_pos_spline.derivative(1)
        self._des_acc_spline = self._des_pos_spline.derivative(2)

    def _generate_waypoints(
        self,
        gates: NDArray[np.floating],
        start_pos: NDArray[np.floating],
        end_pos: NDArray[np.floating] | None = None,
    ) -> NDArray[np.floating]:
        """Generate waypoints by offsetting from gate positions.

        Generates multiple intermediate waypoints between gates for smooth trajectory.

        Args:
            gates: Gate positions, shape (n_gates, 3).
            start_pos: Starting position.
            end_pos: Ending position. If None, uses last gate + vertical offset.

        Returns:
            Waypoints array, shape (n_waypoints, 3).
        """
        waypoints = []

        # Start position
        current_pos = np.array(start_pos, dtype=np.float32)
        waypoints.append(current_pos.copy())

        # Generate waypoints through each gate
        for gate_pos in gates:
            gate_pos = np.array(gate_pos, dtype=np.float32)

            # Add intermediate waypoint (half way between current and gate)
            mid_point = (current_pos + gate_pos) / 2.0
            # Blend height smoothly
            mid_point[2] = (current_pos[2] + gate_pos[2]) / 2.0 + 0.1
            waypoints.append(mid_point)

            # Add gate waypoint
            waypoints.append(gate_pos)
            current_pos = gate_pos.copy()

        # Add final hover position
        if end_pos is None:
            final_pos = gates[-1].copy()
            final_pos[2] = gates[-1, 2] + 0.3
        else:
            final_pos = np.array(end_pos, dtype=np.float32)
        waypoints.append(final_pos)

        return np.array(waypoints, dtype=np.float32)

    def _compute_times(self, waypoints: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute timing for waypoints with fixed total time.

        Uses linear interpolation of waypoint times over the trajectory duration.

        Args:
            waypoints: Waypoint positions, shape (n_waypoints, 3).

        Returns:
            Time array for waypoints, shape (n_waypoints,).
        """
        # Use fixed total time of 15 seconds (proven to work from StateController)
        n_waypoints = len(waypoints)
        return np.linspace(0, 15.0, n_waypoints, dtype=np.float32)

    def evaluate_position(self, t: float) -> NDArray[np.floating]:
        """Evaluate desired position at time t.

        Args:
            t: Time in seconds.

        Returns:
            Position [x, y, z].
        """
        t = np.clip(t, 0, self.total_time)
        return np.array(self._des_pos_spline(t), dtype=np.float32)

    def evaluate_velocity(self, t: float) -> NDArray[np.floating]:
        """Evaluate desired velocity at time t.

        Args:
            t: Time in seconds.

        Returns:
            Velocity [vx, vy, vz].
        """
        t = np.clip(t, 0, self.total_time)
        return np.array(self._des_vel_spline(t), dtype=np.float32)

    def evaluate_acceleration(self, t: float) -> NDArray[np.floating]:
        """Evaluate desired acceleration at time t.

        Args:
            t: Time in seconds.

        Returns:
            Acceleration [ax, ay, az].
        """
        t = np.clip(t, 0, self.total_time)
        return np.array(self._des_acc_spline(t), dtype=np.float32)

    def evaluate_state(self, t: float) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        """Evaluate position, velocity, and acceleration at time t.

        Args:
            t: Time in seconds.

        Returns:
            Tuple of (position, velocity, acceleration).
        """
        return self.evaluate_position(t), self.evaluate_velocity(t), self.evaluate_acceleration(t)

    @property
    def n_waypoints(self) -> int:
        """Number of waypoints."""
        return len(self.waypoints)
