"""Enhanced attitude controller with dynamic gate-based waypoint generation.

This controller improves upon the basic AttitudeController by dynamically generating
waypoints based on actual gate positions, enabling it to handle randomized gate
positions (Level 2) while maintaining performance on Levels 0-1.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from drone_models.core import load_params
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


class AttitudeControllerUpgrade(Controller):
    """Attitude controller with dynamic gate-based waypoint generation.

    Adapts the trajectory in real-time based on actual gate positions that become
    visible during flight. This enables successful completion of Level 2 tracks with
    randomized gate positions.
    """

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the attitude controller.

        Args:
            obs: The initial observation of the environment's state.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
        """
        super().__init__(obs, info, config)
        self._freq = config.env.freq

        # Load drone parameters for gravity compensation
        drone_params = load_params(config.sim.physics, config.sim.drone_model)
        self.drone_mass = drone_params["mass"]

        # PID gains (same as original AttitudeController)
        self.kp = np.array([0.4, 0.4, 1.25])
        self.ki = np.array([0.05, 0.05, 0.05])
        self.kd = np.array([0.2, 0.2, 0.4])
        self.ki_range = np.array([2.0, 2.0, 0.4])
        self.i_error = np.zeros(3)
        self.g = 9.81

        # Gate tracking
        self.gates_detected = False
        
        # self.target_gate_detected = np.zeros(10, dtype=bool)  # Track which gates we've seen
        num_gates = len(obs["gates_pos"])
        self.target_gate_detected = np.zeros(num_gates, dtype=bool)
        
        self._replan_tick = 0  # Track when trajectory was last replanned
        self._t_total = 20.0  # Trajectory duration in seconds

        # Initialize with nominal waypoints (from original AttitudeController)
        self._init_waypoints_from_gates(obs)

        self._tick = 0
        self._finished = False

    def _init_waypoints_from_gates(self, obs: dict[str, NDArray[np.floating]]) -> None:
        """Initialize trajectory from gate positions in observation.

        Args:
            obs: The observation dictionary containing gate positions.
        """
        # Extract gate positions from observation
        gates_pos = obs["gates_pos"]  # Shape: (n_gates, 3)
        start_pos = obs["pos"]

        # Generate waypoints through gates
        waypoints = self._generate_waypoints_from_gates(gates_pos, start_pos)

        # Create cubic spline trajectory with longer duration for smoother execution
        n_waypoints = len(waypoints)
        times = np.linspace(0, 20.0, n_waypoints)  # Extended to 20 seconds for smoother trajectory
        self._des_pos_spline = CubicSpline(times, waypoints)
        self._des_vel_spline = self._des_pos_spline.derivative()
        self._t_total = 20.0  # Track the total trajectory time

    def _generate_waypoints_from_gates(
        self, gates_pos: NDArray[np.floating], start_pos: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Generate waypoints by placing points through each gate.

        Creates intermediate waypoints (approach, center, exit) for each gate to ensure
        smooth passage through the racing track.

        Args:
            gates_pos: Gate positions, shape (n_gates, 3) in [x, y, z] format.
            start_pos: Starting position [x, y, z].

        Returns:
            Waypoints array, shape (n_waypoints, 3).
        """
        waypoints = [start_pos.copy()]

        for gate_pos in gates_pos:
            gate_pos = np.array(gate_pos, dtype=np.float32)

            # Approach waypoint: slightly above and before gate
            approach = gate_pos.copy()
            approach[2] += 0.05  # Smaller vertical offset for smoother trajectory
            waypoints.append(approach)

            # Gate center waypoint: pass through gate
            waypoints.append(gate_pos.copy())

            # Exit waypoint: below gate for smooth continuation
            exit_pt = gate_pos.copy()
            exit_pt[2] -= 0.05  # Smaller vertical offset for smoother trajectory
            waypoints.append(exit_pt)

        # Final hover position: above last gate
        final_pos = gates_pos[-1].copy()
        final_pos[2] += 0.5  # Hover 50cm above last gate
        waypoints.append(final_pos)

        return np.array(waypoints, dtype=np.float32)

    def _should_replan(self, obs: dict[str, NDArray[np.floating]]) -> bool:
        """Check if trajectory should be replanned based on new gate information.

        Args:
            obs: Current observation containing gate positions and visited flags.

        Returns:
            True if replanning is needed, False otherwise.
        """
        gates_visited = obs["gates_visited"]

        # Replan if any gates just became visible
        if np.any(gates_visited) and not self.gates_detected:
            return True

        # Replan if next target gate just became visible
        target_gate = obs["target_gate"]
        if target_gate >= 0 and target_gate < len(gates_visited):
            if gates_visited[target_gate] and not self.target_gate_detected[target_gate]:
                return True

        return False

    def _update_trajectory(self, obs: dict[str, NDArray[np.floating]]) -> None:
        """Update trajectory with new gate positions from observation.

        Args:
            obs: Current observation containing actual gate positions.
        """
        gates_pos = obs["gates_pos"]
        gates_visited = obs["gates_visited"]
        current_pos = obs["pos"]

        # Generate new waypoints from current gate positions
        waypoints = self._generate_waypoints_from_gates(gates_pos, current_pos)

        # Recompute spline with new waypoints
        n_waypoints = len(waypoints)
        times = np.linspace(0, self._t_total, n_waypoints)
        self._des_pos_spline = CubicSpline(times, waypoints)
        self._des_vel_spline = self._des_pos_spline.derivative()

        # Update gate detection tracking
        self.gates_detected = True
        self.target_gate_detected[gates_visited] = True
        self._replan_tick = self._tick  # Reset trajectory timer after replan

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: The current observation of the environment.
            info: Optional additional information as a dictionary.

        Returns:
            The attitude command [roll, pitch, yaw, thrust] as a numpy array.
        """
        # Check if we should replan trajectory based on newly detected gates
        if self._should_replan(obs):
            self._update_trajectory(obs)

        # Compute time along the trajectory
        t = min((self._tick - self._replan_tick) / self._freq, self._t_total)
        if t >= self._t_total:
            self._finished = True

        # Evaluate desired state from spline
        des_pos = self._des_pos_spline(t)
        des_vel = self._des_vel_spline(t)
        des_yaw = 0.0

        # Calculate position and velocity errors
        pos_error = des_pos - obs["pos"]
        vel_error = des_vel - obs["vel"]

        # Update integral error with anti-windup
        self.i_error += pos_error * (1 / self._freq)
        self.i_error = np.clip(self.i_error, -self.ki_range, self.ki_range)

        # Compute desired thrust using PID control
        target_thrust = np.zeros(3)
        target_thrust += self.kp * pos_error
        target_thrust += self.ki * self.i_error
        target_thrust += self.kd * vel_error
        target_thrust[2] += self.drone_mass * self.g

        # Get current drone orientation
        z_axis = R.from_quat(obs["quat"]).as_matrix()[:, 2]

        # Project desired thrust onto current z-axis to get thrust magnitude
        thrust_desired = target_thrust.dot(z_axis)

        # Compute desired orientation using geometric control
        z_axis_desired = target_thrust / np.linalg.norm(target_thrust)
        x_c_des = np.array([math.cos(des_yaw), math.sin(des_yaw), 0.0])
        y_axis_desired = np.cross(z_axis_desired, x_c_des)
        y_axis_desired /= np.linalg.norm(y_axis_desired)
        x_axis_desired = np.cross(y_axis_desired, z_axis_desired)

        # Convert desired orientation to Euler angles
        R_desired = np.vstack([x_axis_desired, y_axis_desired, z_axis_desired]).T
        euler_desired = R.from_matrix(R_desired).as_euler("xyz", degrees=False)

        action = np.concatenate([euler_desired, [thrust_desired]], dtype=np.float32)

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
        """Increment the tick counter.

        Returns:
            True if the controller is finished, False otherwise.
        """
        self._tick += 1
        return self._finished

    def episode_callback(self):
        """Reset the internal state for a new episode."""
        self.i_error[:] = 0
        self._tick = 0
        self._finished = False
        self.gates_detected = False

        #self.target_gate_detected[:] = False
        self.target_gate_detected.fill(False)

        self._replan_tick = 0

    def reset(self):
        """Reset controller state."""
        self.i_error[:] = 0
        self._tick = 0
        self._finished = False
