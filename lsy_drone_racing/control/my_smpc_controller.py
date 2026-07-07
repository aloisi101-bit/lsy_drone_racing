"""Sampling-based MPC (MPPI) controller for drone racing.

The controller builds three waypoints per gate (before, center, after) along the gate's
forward normal and drives the drone through them with a Model Predictive Path Integral
solver. The through-axis always points along the gate's local +x, because the environment
only credits a pass when the drone crosses the gate from local -x to +x
(see ``lsy_drone_racing.envs.utils.gate_passed``).
"""

from __future__ import annotations

import logging
import os
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from crazyflow.sim.visualize import draw_line, draw_points
from drone_models.core import load_params
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.mppi_engine import build_mppi_solver

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray


class MySMPCController(Controller):
    """MPPI controller that flies through the gates via per-gate waypoints."""

    WAYPOINT_OFFSET = 0.30  # pre/post waypoint distance from the gate center [m]
    WAYPOINT_ADVANCE_RADIUS = 0.20  # advance to the next waypoint within this distance [m]
    OBSTACLE_SAFETY_RADIUS = 0.50  # waypoints are pushed out of this bubble around obstacles [m]

    def __init__(self, obs: dict, info: dict, config: dict):
        """Initialize the controller.

        Args:
            obs: Initial environment observation.
            info: Additional reset information from the environment.
            config: Environment and simulation configuration.
        """
        super().__init__(obs, info, config)

        # Timing and physical parameters.
        self.dt_pred = 0.04  # MPPI prediction step [s]
        self._tick = 0
        self._finished = False

        drone_params = load_params(config.sim.physics, config.sim.drone_model)
        self.drone_mass = float(drone_params["mass"])
        self.g = 9.81
        self.hover_thrust = self.drone_mass * self.g

        # Track geometry: nominal gate poses and obstacle positions from the config.
        self.obstacle_positions = np.array([o["pos"] for o in config.env.track.obstacles])
        gates_config = config.env.track.gates
        self.num_gates = len(gates_config)
        self.init_gates_pos = [g["pos"] for g in gates_config]
        self.init_gates_rpy = [g["rpy"] for g in gates_config]
        self._update_track_geometry(
            self.init_gates_pos, self.init_gates_rpy, self.obstacle_positions
        )
        self.waypoint_idx = 0

        # MPPI configuration.
        self._horizon = 20  # prediction steps (~0.4 s at the prediction step)
        self.temperature = 0.50  # softmax temperature; lower = sharper trajectory selection
        # Per-action exploration std. Yaw is a small angle offset (it was a rate before, so
        # the old 0.30 now perturbs the predicted heading ~25x more and must be scaled down).
        self.noise_std = jnp.array([0.15, 0.15, 0.02, 0.1])
        self.engine = build_mppi_solver(self._cost_fn, self._dynamics_fn)
        self.nominal_actions = self._initial_nominal_actions()
        self.rng_key = jax.random.PRNGKey(42)

        # Final action clip on the MPPI output. Yaw is bounded to the sim's [-pi/2, pi/2]
        # attitude range so the command we send is interpreted exactly as the model predicts.
        self._min_action = jnp.array([-0.5, -0.5, -np.pi / 2, 0.0])
        self._max_action = jnp.array([0.5, 0.5, np.pi / 2, self.drone_mass * self.g * 2.5])

        self.planned_trajectory = None
        self.debug = getattr(config, "debug", False)
        self._logger = self._setup_logger()

    def _initial_nominal_actions(self) -> jax.Array:
        """Return the warm-start action sequence: slight forward tilt and a lift bias."""
        nominal = jnp.zeros((self._horizon, 4))
        nominal = nominal.at[:, 0].set(0.1)  # gentle forward pitch
        nominal = nominal.at[:, 3].set(self.hover_thrust * 1.5)  # extra lift on cold start
        return nominal

    def _setup_logger(self) -> logging.Logger:
        """Create a file logger writing debug output next to this module."""
        logger = logging.getLogger("lsy_smpc_debug")
        if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
            log_path = os.path.join(os.path.dirname(__file__), "smpc_debug.log")
            handler = logging.FileHandler(log_path, mode="a")
            handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _shift_waypoint_away_from_obstacles(
        self, waypoint: NDArray, approach_dir: NDArray
    ) -> NDArray:
        """Push a waypoint radially out of any obstacle safety bubble it lands inside.

        Repeats a few times so a waypoint squeezed between several obstacles settles
        outside all of their bubbles instead of only the last one checked.
        """
        shifted = waypoint.copy()
        for _ in range(5):
            conflict = False
            for obstacle in self.obstacle_positions:
                obstacle_xy = np.asarray(obstacle[:2])
                offset = shifted[:2] - obstacle_xy
                if np.linalg.norm(offset) >= self.OBSTACLE_SAFETY_RADIUS:
                    continue
                conflict = True
                if np.linalg.norm(offset) < 1e-6:
                    # Waypoint sits on the obstacle: push sideways relative to the approach.
                    lateral = np.cross(approach_dir, [0.0, 0.0, 1.0])[:2]
                    offset = lateral if np.linalg.norm(lateral) > 1e-6 else np.array([1.0, 0.0])
                push_dir = offset / np.linalg.norm(offset)
                shifted[:2] = obstacle_xy + push_dir * 0.20
            if not conflict:
                break
        return shifted

    def _update_track_geometry(
        self, gates_pos: NDArray, gates_rpy: NDArray, obstacles: NDArray
    ) -> None:
        """Rebuild waypoints and gate collision frames from the latest poses.

        Places three waypoints per gate along its forward normal (pre, center, post) and
        stores each gate's pose and forward direction for the collision cost.
        """
        self.obstacle_positions = np.asarray(obstacles)

        waypoints, waypoint_dirs, gate_poses, gate_forwards = [], [], [], []
        for gate_idx in range(self.num_gates):
            gate_pos = np.asarray(gates_pos[gate_idx], dtype=float)
            gate_rpy = np.asarray(gates_rpy[gate_idx], dtype=float)

            # Gate forward normal (local +x), flattened into the XY plane for stability.
            forward = R.from_euler("xyz", gate_rpy).apply([1.0, 0.0, 0.0])
            forward[2] = 0.0
            norm = np.linalg.norm(forward)
            forward = forward / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])

            gate_poses.append(gate_pos)
            gate_forwards.append(forward)

            pre_wp = self._shift_waypoint_away_from_obstacles(
                gate_pos - forward * self.WAYPOINT_OFFSET, forward
            )
            post_wp = self._shift_waypoint_away_from_obstacles(
                gate_pos + forward * self.WAYPOINT_OFFSET, forward
            )
            waypoints.extend([pre_wp, gate_pos.copy(), post_wp])
            waypoint_dirs.extend([forward, forward, forward])

        self.waypoints = jnp.array(waypoints)
        self.waypoint_dirs = jnp.array(waypoint_dirs)
        self.gate_poses = jnp.array(gate_poses)
        self.gate_forwards = jnp.array(gate_forwards)

    def _get_live_gates_rpy(self, obs: dict, info: dict, num_gates: int) -> NDArray:
        """Extract gate orientations (roll/pitch/yaw) from whatever the env provides."""
        # Preferred: quaternions.
        gates_quat = obs.get("gates_quat", info.get("gates_quat"))
        if gates_quat is not None and len(gates_quat) > 0:
            try:
                return R.from_quat(gates_quat).as_euler("xyz", degrees=False)
            except ValueError:
                pass

        # Pose arrays [x, y, z, qx, qy, qz, qw].
        gates_pose = obs.get("gates_pose", info.get("gates_pose"))
        if gates_pose is not None and len(gates_pose) > 0 and len(gates_pose[0]) == 7:
            try:
                return R.from_quat([p[3:] for p in gates_pose]).as_euler("xyz", degrees=False)
            except ValueError:
                pass

        # Explicit RPY arrays.
        for key in ("gates_rpy", "gate_rpy"):
            rpy = obs.get(key, info.get(key))
            if rpy is not None and len(rpy) > 0:
                return np.array(rpy)

        # Yaw-only overrides.
        gates_yaw = obs.get("gates_yaw", info.get("gates_yaw"))
        if gates_yaw is not None and len(gates_yaw) > 0:
            rpys = np.zeros((num_gates, 3))
            rpys[:, 2] = gates_yaw
            return rpys

        return np.array(self.init_gates_rpy)

    @staticmethod
    def _dynamics_fn(state: jax.Array, action: jax.Array, params: dict) -> jax.Array:
        """Integrate one prediction step of the attitude-controlled quadrotor.

        Roll, pitch, and yaw are absolute attitude setpoints: roll/pitch are tracked by a
        first-order loop, while yaw is applied directly. This matches the simulator, whose
        attitude command is ``[roll, pitch, yaw, collective_thrust]`` with yaw an angle in
        [-pi/2, pi/2] (see crazyflow.control.control.Control.attitude). Thrust acts along
        the resulting body-z axis. Returns the next 9D state ``[pos(3), vel(3), rpy(3)]``.
        """
        dt = params["dt_pred"]
        mass = params["mass"]
        g = params["g"]

        pos = state[0:3]
        vel = state[3:6]
        roll, pitch = state[6], state[7]  # current yaw is unused: yaw is an absolute setpoint

        cmd_roll = jnp.clip(action[0], -0.45, 0.45)
        cmd_pitch = jnp.clip(action[1], -0.45, 0.45)
        cmd_yaw = jnp.clip(action[2], -jnp.pi / 2, jnp.pi / 2)
        cmd_thrust = jnp.clip(action[3], 0.0, mass * g * 2.5)

        # First-order attitude tracking of roll/pitch; yaw is an absolute angle setpoint.
        attitude_gain = 5.0
        next_roll = roll + (cmd_roll - roll) * attitude_gain * dt
        next_pitch = pitch + (cmd_pitch - pitch) * attitude_gain * dt
        next_yaw = cmd_yaw

        # Thrust direction: the body-z axis expressed in the world frame.
        cy, sy = jnp.cos(next_yaw), jnp.sin(next_yaw)
        cr, sr = jnp.cos(next_roll), jnp.sin(next_roll)
        cp, sp = jnp.cos(next_pitch), jnp.sin(next_pitch)
        body_z = jnp.array([cy * sp * cr + sy * sr, sy * sp * cr - cy * sr, cp * cr])

        acc = body_z * cmd_thrust / mass + jnp.array([0.0, 0.0, -g])
        next_vel = vel + acc * dt
        next_pos = pos + vel * dt + 0.5 * acc * dt**2
        next_rpy = jnp.array([next_roll, next_pitch, next_yaw])

        return jnp.concatenate([next_pos, next_vel, next_rpy])

    @staticmethod
    def _cost_fn(
        state: jax.Array, action: jax.Array, next_state: jax.Array, params: dict
    ) -> jax.Array:
        """Compute the stage cost for one predicted step."""
        pos = next_state[0:3]
        vel = next_state[3:6]
        target = params["target"]
        gate_dir = params["gate_dir"]
        obstacles = params["obstacles"]

        diff = pos - target
        dist_to_target = jnp.linalg.norm(diff)
        xy_dist = jnp.linalg.norm(diff[:2])
        z_dist = jnp.abs(diff[2])

        # 1. Attraction to the target waypoint (Z weighted harder to hold altitude).
        cost = xy_dist**2 * 125.0 + z_dist**2 * 150.0

        # 2. Cross-track "tube": penalize lateral deviation while approaching the gate.
        longitudinal_dist = jnp.dot(diff, gate_dir)
        cross_track_error = jnp.linalg.norm(diff - longitudinal_dist * gate_dir)
        funnel_radius = 0.10 + jnp.maximum(0.0, -longitudinal_dist)
        cost += jnp.where(
            (cross_track_error > funnel_radius) & (longitudinal_dist < 0.0),
            (cross_track_error - funnel_radius) * 200.0,
            0.0,
        )
        cost += jnp.where(longitudinal_dist < 0.0, cross_track_error * 50.0, 0.0)

        # 3. Gate frame and gate stand collision penalties (signed-distance based).
        gate_poses = params["gate_poses"]
        gate_forwards = params["gate_forwards"]
        if gate_poses.shape[0] > 0:
            rel_poses = pos[None, :] - gate_poses

            # Orthonormal gate frame; fall back to Y-up for near-vertical forwards.
            up = jnp.array([0.0, 0.0, 1.0])
            near_vertical = jnp.abs(jnp.sum(gate_forwards * up, axis=1))[:, None] > 0.99
            up_vecs = jnp.where(
                near_vertical, jnp.array([0.0, 1.0, 0.0]), jnp.tile(up, (gate_forwards.shape[0], 1))
            )
            gate_y = jnp.cross(gate_forwards, up_vecs)
            gate_y = gate_y / (jnp.linalg.norm(gate_y, axis=1, keepdims=True) + 1e-6)
            gate_z = jnp.cross(gate_y, gate_forwards)

            dx = jnp.sum(rel_poses * gate_forwards, axis=1)
            dy = jnp.sum(rel_poses * gate_y, axis=1)
            dz = jnp.sum(rel_poses * gate_z, axis=1)
            max_yz = jnp.maximum(jnp.abs(dy), jnp.abs(dz))

            # Signed distance to the square frame (opening half-width 0.28, wall 0.08).
            depth_yz = 0.08 - jnp.abs(max_yz - 0.28)
            depth_x = 0.05 - jnp.abs(dx)
            exterior_dist = jnp.sqrt(
                jnp.maximum(0.0, -depth_yz) ** 2 + jnp.maximum(0.0, -depth_x) ** 2 + 1e-8
            )
            interior_depth = jnp.minimum(depth_yz, depth_x)
            sdf = jnp.where((depth_yz > 0) & (depth_x > 0), -interior_depth, exterior_dist)
            cost += jnp.sum(jnp.where(sdf < 0.10, (0.10 - sdf) * 100000.0, 0.0))

            # Gate stand (post below the opening) with a downward escape gradient.
            dist_to_stand = jnp.linalg.norm(pos[:2][None, :] - gate_poses[:, :2], axis=1)
            depth_below_gate = jnp.maximum(0.0, (gate_poses[:, 2] - 0.15) - pos[2])
            cost += jnp.sum(
                jnp.where(
                    (depth_below_gate > 0.0) & (dist_to_stand < 0.20),
                    (0.20 - dist_to_stand) * 1000.0 + depth_below_gate * 500.0,
                    0.0,
                )
            )

        # 4. Velocity shaping: align velocity with the target and gate direction, cap speed.
        speed = jnp.linalg.norm(vel) + 1e-5
        vel_dir = vel / speed
        target_dir = -diff / (dist_to_target + 1e-5)
        cost += (1.0 - jnp.dot(vel_dir, target_dir)) * 20.0 * speed
        cost += (1.0 - jnp.dot(vel_dir, gate_dir)) * 10.0 * speed
        cost += speed**2 * 0.05

        # 5. Obstacle repulsion (cylindrical, XY only).
        if obstacles.shape[0] > 0:
            dist_to_obs = jnp.linalg.norm(obstacles[:, :2] - pos[:2][None, :], axis=1)
            cost += jnp.sum(jnp.where(dist_to_obs < 0.20, (0.20 - dist_to_obs) * 10000.0, 0.0))

        # 6. Keep the drone inside the arena bounds.
        out_of_bounds = (
            (pos[0] < -2.4)
            | (pos[0] > 2.4)
            | (pos[1] < -1.4)
            | (pos[1] > 1.4)
            | (pos[2] < 0.05)
            | (pos[2] > 1.95)
        )
        cost += jnp.where(out_of_bounds, 100000.0, 0.0)

        # 7. Actuator effort.
        cost += (action[0] ** 2 + action[1] ** 2) * 0.5
        cost += action[2] ** 2 * 2.0
        cost += action[3] ** 2 * 0.1

        return cost

    @partial(jax.jit, static_argnums=(0,))
    def _rollout_traj(self, state: jax.Array, action_seq: jax.Array, params: dict) -> jax.Array:
        """Roll out the nominal action sequence into a predicted position trajectory."""

        def step(carry_state: jax.Array, action: jax.Array) -> tuple:
            next_state = self._dynamics_fn(carry_state, action, params)
            return next_state, next_state[:3]

        _, positions = jax.lax.scan(step, state, action_seq)
        return positions

    def compute_control(self, obs: dict, info: dict | None = None) -> NDArray:
        """Compute the attitude command [roll, pitch, yaw, thrust] for this step."""
        if info is None:
            info = {}

        # Pull the latest (possibly sensor-revealed) gate and obstacle poses.
        gates_pos = obs.get("gates_pos", info.get("gates_pos"))
        if gates_pos is None:
            gates_pos = getattr(self, "_last_gates_pos", self.init_gates_pos)
        else:
            self._last_gates_pos = [np.asarray(g).copy() for g in gates_pos]

        gates_rpy = self._get_live_gates_rpy(obs, info, self.num_gates)
        obstacles = obs.get(
            "obstacles_pos",
            info.get("obstacles_pos", obs.get("obstacle_pos", self.obstacle_positions)),
        )
        self._update_track_geometry(gates_pos, gates_rpy, obstacles)

        # Advance to the next waypoint once close enough to the current one.
        pos = np.asarray(obs["pos"])
        self.waypoint_idx = min(self.waypoint_idx, len(self.waypoints) - 1)
        target = np.asarray(self.waypoints[self.waypoint_idx])
        close_enough = np.linalg.norm(target - pos) < self.WAYPOINT_ADVANCE_RADIUS
        if close_enough and self.waypoint_idx < len(self.waypoints) - 1:
            self.waypoint_idx += 1
            target = np.asarray(self.waypoints[self.waypoint_idx])

        rpy = R.from_quat(obs["quat"]).as_euler("xyz", degrees=False)
        current_state = jnp.concatenate(
            [jnp.asarray(obs["pos"]), jnp.asarray(obs["vel"]), jnp.asarray(rpy)]
        )

        if len(self.obstacle_positions) > 0:
            obstacles = jnp.asarray(self.obstacle_positions)
        else:
            obstacles = jnp.zeros((0, 3))
        params = {
            "dt_pred": self.dt_pred,
            "mass": self.drone_mass,
            "g": self.g,
            "target": jnp.asarray(target),
            "gate_dir": jnp.asarray(self.waypoint_dirs[self.waypoint_idx]),
            "obstacles": obstacles,
            "gate_poses": self.gate_poses,
            "gate_forwards": self.gate_forwards,
        }

        self.rng_key, subkey = jax.random.split(self.rng_key)
        optimal_action, self.nominal_actions, min_cost, mean_cost = self.engine(
            subkey, current_state, self.nominal_actions, self.temperature, self.noise_std, params
        )

        self.planned_trajectory = np.asarray(
            jax.device_get(self._rollout_traj(current_state, self.nominal_actions, params))
        )

        action = jnp.clip(optimal_action, self._min_action, self._max_action)
        if self.debug:
            self._print_debug(obs, target, [min_cost, mean_cost], action)
        return np.asarray(jax.device_get(action), dtype=np.float32)

    def render_callback(self, sim: Sim) -> None:
        """Draw the waypoints, current target, approach funnel, and planned trajectory."""
        if len(self.waypoints) > 0:
            draw_points(sim, np.asarray(self.waypoints), rgba=(0.0, 1.0, 0.0, 0.3), size=0.03)

        if self.waypoint_idx < len(self.waypoints):
            target = np.asarray(self.waypoints[self.waypoint_idx])
            draw_points(sim, target.reshape(1, -1), rgba=(1.0, 0.0, 0.0, 1.0), size=0.05)
            self._draw_funnel(sim, target, np.asarray(self.waypoint_dirs[self.waypoint_idx]))

        if self.planned_trajectory is not None:
            draw_line(sim, self.planned_trajectory, rgba=(0.0, 0.0, 1.0, 1.0))

    def _draw_funnel(self, sim: Sim, target: NDArray, gate_dir: NDArray) -> None:
        """Draw the conical approach funnel of rings leading into the gate."""
        base_radius = 0.15
        slope = 1.0
        offsets = [0.0, -0.5, -1.0, -1.5]

        up = np.array([0.0, 0.0, 1.0])
        if np.abs(np.dot(gate_dir, up)) > 0.99:
            up = np.array([0.0, 1.0, 0.0])
        u = np.cross(gate_dir, up)
        u = u / (np.linalg.norm(u) + 1e-6)
        v = np.cross(gate_dir, u)

        angles = np.linspace(0, 2 * np.pi, 20)
        rings = []
        for offset in offsets:
            radius = base_radius + abs(offset) * slope
            center = target + gate_dir * offset
            ring = center + np.array([np.cos(a) * u + np.sin(a) * v for a in angles]) * radius
            draw_line(sim, ring, rgba=(1.0, 1.0, 0.0, 0.5))
            rings.append(ring)

        # Connect the outermost and innermost rings to suggest the funnel walls.
        for i in [0, 5, 10, 15]:
            draw_line(sim, np.vstack([rings[-1][i], rings[0][i]]), rgba=(1.0, 1.0, 0.0, 0.3))

    def step_callback(
        self,
        action: NDArray,
        obs: dict,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Advance the tick counter and report whether the episode has ended."""
        self._tick += 1
        if terminated or truncated:
            self._finished = True
        return self._finished

    def episode_callback(self) -> None:
        """Reset per-episode state at the start of a new run."""
        self._tick = 0
        self._finished = False
        self.waypoint_idx = 0
        self.nominal_actions = self._initial_nominal_actions()
        self.rng_key = jax.random.PRNGKey(42)

    def _print_debug(self, obs: dict, target: NDArray, costs: list, action: jax.Array) -> None:
        """Print and log the current tracking state for debugging."""
        pos = np.asarray(obs["pos"])
        costs_np = np.asarray(jax.device_get(costs))
        action_np = np.asarray(jax.device_get(action))
        dist_to_target = np.linalg.norm(pos - target)
        if len(self.obstacle_positions) > 0:
            nearest_obs = np.linalg.norm(self.obstacle_positions[:, :2] - pos[:2], axis=1).min()
        else:
            nearest_obs = float("inf")

        message = (
            f"[SMPC] pos={pos.round(3)} tgt={target.round(3)} "
            f"dist={dist_to_target:.3f} obs={nearest_obs:.3f} "
            f"cost_min/mean={costs_np.min():.1f}/{costs_np.mean():.1f} "
            f"act={action_np.round(3)} wp={self.waypoint_idx} gate={self.waypoint_idx // 3}"
        )
        print(message)
        self._logger.info(message)
