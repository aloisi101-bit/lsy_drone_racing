"""SMPC Controller for Drone Racing - Fixed Level 0 Implementation with proper MPPI."""
from __future__ import annotations

from typing import TYPE_CHECKING
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import os
import logging
from scipy.spatial.transform import Rotation as R
from drone_models.core import load_params

from crazyflow.sim.visualize import draw_line, draw_points
from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.mppi_engine import build_mppi_solver

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray


class MySMPCController(Controller):
    """Sampling-based MPC controller with corrected dynamics and cost function."""

    def __init__(self, obs: dict, info: dict, config: dict):
        super().__init__(obs, info, config)

        # --- Environment & Physical Parameters ---
        self._freq = config.env.freq
        self.dt = 1.0 / self._freq
        self.dt_pred = 0.04
        self._tick = 0
        self._finished = False

        drone_params = load_params(config.sim.physics, config.sim.drone_model)
        self.drone_mass = float(drone_params["mass"])
        self.g = 9.81
        self.hover_thrust = self.drone_mass * self.g

        # --- Track Setup ---
        self.obstacles_np = np.array([o["pos"] for o in config.env.track.obstacles])
        self.spawn_pos = np.array(obs["pos"], dtype=float)
        self.current_drone_pos = self.spawn_pos.copy()  # Initialize for gate ordering
        self.last_gate_order = None  # Track gate order to avoid recalc every frame
        self.last_gate_positions = None  # Track gate positions to detect changes
        self.last_gate_rpys = None  # Track gate orientations to detect changes
        
        gates_config = config.env.track.gates
        self.num_gates = len(gates_config)

        # Store nominal positions and orientations safely
        self.init_gates_pos = [g["pos"] for g in gates_config]
        self.init_gates_rpy = [g["rpy"] for g in gates_config]

        self._update_track_geometry(self.init_gates_pos, self.init_gates_rpy, self.obstacles_np)

        self.active_wp_idx = 0

        # --- Logging ---
        log_dir = os.path.dirname(__file__)
        log_path = os.path.join(log_dir, "smpc_debug.log")
        self._logger = logging.getLogger("lsy_smpc_debug")
        if not any(isinstance(h, logging.FileHandler) for h in self._logger.handlers):
            fh = logging.FileHandler(log_path, mode="a")
            fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
            self._logger.addHandler(fh)
            self._logger.setLevel(logging.INFO)

        self.debug = getattr(config, "debug", True)

        # --- MPPI Configuration ---
        self.H = 10 # Reasonable horizon (0.5s at 50Hz env)
        self.temperature = 0.08 # Lower temperature for sharper selection of low-cost trajectories
        
        # Noise exploration: [roll, pitch, yaw_rate, thrust]
        self.noise_std = jnp.array([0.15, 0.15, 0.30, 0.1])

        self.engine = build_mppi_solver(self._cost_fn, self._dynamics_fn)

        # Initialize nominal with gentle forward bias (helps cold start)
        self.nominal_seq = jnp.zeros((self.H, 4))
        self.nominal_seq = self.nominal_seq.at[:, 0].set(0.01)  # Gentle forward tilt
        self.nominal_seq = self.nominal_seq.at[:, 3].set(self.hover_thrust * 1.5)  # Slight lift
        self.rng_key = jax.random.PRNGKey(42)

        self.planned_trajectory = None

    def _compute_optimal_gate_order(self, gates_pos, current_pos):
        """
        Gates MUST be flown in their fixed, predefined index order (0, 1, 2, ...).
        Returns: strict sequential list of gate indices [0, 1, ..., num_gates - 1]
        """
        return list(range(len(gates_pos)))

    def _shift_waypoint_away_from_obstacles(self, waypoint, approach_dir, is_post=False):
        """
        If a waypoint is too close to an obstacle, push it radially outward away from
        the obstacle's core instead of just shifting it purely longitudinally.
        This prevents squeezing the drone and clipping the turning radius.
        """
        shifted = waypoint.copy()
        
        # Iteratively push away from multiple close obstacles if necessary
        for _ in range(5):
            conflict = False
            point_xy = shifted[:2]
            for obs_pos in self.obstacles_np:
                obs_xy = np.array(obs_pos[:2])
                dist = np.linalg.norm(point_xy - obs_xy)
                
                # 0.50m is a safe bubble radius (obstacle=0.015m, drone=~0.1m + buffer)
                if dist < 0.50:
                    conflict = True
                    push_dir = point_xy - obs_xy
                    norm = np.linalg.norm(push_dir)
                    
                    if norm < 1e-6:
                        # Dead center of obstacle: push laterally relative to approach dir
                        up = np.array([0.0, 0.0, 1.0])
                        lat = np.cross(approach_dir, up)[:2]
                        push_dir = lat if np.linalg.norm(lat) > 1e-6 else np.array([1.0, 0.0])
                        norm = np.linalg.norm(push_dir)
                        
                    push_dir = push_dir / norm
                    # Push waypoint outwards to be exactly 0.23m away from obstacle center
                    shifted[:2] = obs_xy + push_dir * 0.20
            
            if not conflict:
                break
                
        return shifted

    def _update_track_geometry(self, live_gates_pos, live_gates_rpy, live_obstacles):
        """
        Rebuilds the MPC track geometry. Now strictly monitors BOTH positions and 
        orientations (RPY) to ensure waypoints immediately rotate when the gate's true
        yaw is revealed by the sensor suite.
        """
        self.obstacles_np = np.array(live_obstacles)
        
        current_gate_positions = np.array([g[:2] for g in live_gates_pos])  # 2D positions only
        current_gate_rpys = np.array(live_gates_rpy)
        gates_moved = False
        
        if getattr(self, 'last_gate_positions', None) is None or getattr(self, 'last_gate_rpys', None) is None:
            gates_moved = True
        else:
            # Rebuild track if any gate moved > 1mm OR rotated > 0.001 rad
            pos_diff = np.linalg.norm(current_gate_positions - self.last_gate_positions, axis=1)
            rpy_diff = np.linalg.norm(current_gate_rpys - self.last_gate_rpys, axis=1)
            if np.any(pos_diff > 0.001) or np.any(rpy_diff > 0.001):
                gates_moved = True
        
        if gates_moved or getattr(self, 'last_gate_order', None) is None:
            gate_order = self._compute_optimal_gate_order(live_gates_pos, self.current_drone_pos)
            self.last_gate_order = gate_order
            self.last_gate_positions = current_gate_positions.copy()
            self.last_gate_rpys = current_gate_rpys.copy()
        else:
            gate_order = self.last_gate_order
        
        self.gate_order = gate_order
        
        waypoints = []
        gate_dirs = []
        all_gate_poses = []
        all_gate_dirs = []
        
        for sequence_idx, gate_idx in enumerate(gate_order):
            gate_pos = np.array(live_gates_pos[gate_idx], dtype=float)
            gate_rpy = np.array(live_gates_rpy[gate_idx], dtype=float)

            # Extract gate forward direction from its orientation
            rot = R.from_euler("xyz", gate_rpy)
            gate_forward = rot.apply(np.array([1.0, 0.0, 0.0]))
            gate_forward[2] = 0.0  # Keep in XY plane for waypoint stability
            norm = np.linalg.norm(gate_forward)
            gate_forward = gate_forward / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])

            all_gate_poses.append(gate_pos.copy())
            all_gate_dirs.append(gate_forward.copy())

            # --- ROBUST MACRO-FLOW SIGN CALCULATION ---
            # Use normalized approach vector + heavily down-weighted exit vector.
            # This ensures the funnel ALWAYS faces the incoming drone (even on 180-deg U-turns)
            # while still breaking dot-product instability on exact 90-degree turns.
            if sequence_idx == 0:
                prev_pos = self.spawn_pos[:2]
            else:
                prev_gate_idx = gate_order[sequence_idx - 1]
                prev_pos = np.array(live_gates_pos[prev_gate_idx][:2], dtype=float)

            if sequence_idx < len(gate_order) - 1:
                next_gate_idx = gate_order[sequence_idx + 1]
                next_pos = np.array(live_gates_pos[next_gate_idx][:2], dtype=float)
            else:
                curr_pos = gate_pos[:2]
                next_pos = curr_pos + (curr_pos - prev_pos)

            # 1. Normalized Approach Vector
            approach_vec = gate_pos[:2] - prev_pos
            app_norm = np.linalg.norm(approach_vec)
            if app_norm > 1e-6:
                approach_vec = approach_vec / app_norm
            else:
                approach_vec = gate_forward[:2]

            # 2. Normalized Exit Vector
            exit_vec = next_pos - gate_pos[:2]
            ext_norm = np.linalg.norm(exit_vec)
            if ext_norm > 1e-6:
                exit_vec = exit_vec / ext_norm
            else:
                exit_vec = gate_forward[:2]
            
            # Combine: Approach dominates entirely, Exit vector only tips the scale on a 90-deg tie
            macro_flow_dir = approach_vec + 0.1 * exit_vec
            macro_flow_norm = np.linalg.norm(macro_flow_dir)
            
            if macro_flow_norm > 1e-6:
                macro_flow_dir = macro_flow_dir / macro_flow_norm
            else:
                macro_flow_dir = gate_forward[:2]

            # Compare macro flow to the gate's local X-axis
            sign = 1.0 if np.dot(gate_forward[:2], macro_flow_dir) >= 0.0 else -1.0
            wp_axis = gate_forward * sign  # True centerline vector

            pre_offset = 0.40
            post_offset = 0.40
            
            pre_wp = gate_pos - wp_axis * pre_offset
            center_wp = gate_pos.copy()
            post_wp = gate_pos + wp_axis * post_offset
            
            pre_wp = self._shift_waypoint_away_from_obstacles(pre_wp, wp_axis, is_post=False)
            post_wp = self._shift_waypoint_away_from_obstacles(post_wp, wp_axis, is_post=True)
            
            waypoints.extend([pre_wp, center_wp, post_wp])
            gate_dirs.extend([wp_axis, wp_axis, wp_axis])

        self.waypoints = jnp.array(waypoints)
        self.gate_dirs = jnp.array(gate_dirs)
        self.all_gate_poses_jnp = jnp.array(all_gate_poses)
        self.all_gate_dirs_jnp = jnp.array(all_gate_dirs)

    def _get_live_gates_rpy(self, obs, info, num_gates):
        """Safely extracts gate orientations from various possible environment configurations."""
        # 1. Try Quaternions (often most stable formatting)
        gates_quat = obs.get("gates_quat", info.get("gates_quat", None))
        if gates_quat is not None and len(gates_quat) > 0:
            try:
                return R.from_quat(gates_quat).as_euler("xyz", degrees=False)
            except Exception:
                pass
                
        # 2. Try Pose array [x,y,z, qx,qy,qz,qw]
        gates_pose = obs.get("gates_pose", info.get("gates_pose", None))
        if gates_pose is not None and len(gates_pose) > 0 and len(gates_pose[0]) == 7:
            try:
                quats = [p[3:] for p in gates_pose]
                return R.from_quat(quats).as_euler("xyz", degrees=False)
            except Exception:
                pass
                
        # 3. Try explicitly named RPY arrays
        gates_rpy = obs.get("gates_rpy", info.get("gates_rpy", None))
        if gates_rpy is not None and len(gates_rpy) > 0:
            return np.array(gates_rpy)
            
        gate_rpy = obs.get("gate_rpy", info.get("gate_rpy", None))
        if gate_rpy is not None and len(gate_rpy) > 0:
            return np.array(gate_rpy)
            
        # 4. Try just Yaw overrides
        gates_yaw = obs.get("gates_yaw", info.get("gates_yaw", None))
        if gates_yaw is not None and len(gates_yaw) > 0:
            rpys = np.zeros((num_gates, 3))
            rpys[:, 2] = gates_yaw
            return rpys
            
        return np.array(self.init_gates_rpy)

    @staticmethod
    def _dynamics_fn(state: jnp.ndarray, action: jnp.ndarray, params: dict) -> jnp.ndarray:
        """Proper kinematic rigid body dynamics with correct physics."""
        dt = params.get("dt_pred", params["dt"])
        mass = params["mass"]
        g = params["g"]

        pos = state[0:3]
        vel = state[3:6]
        rpy = state[6:9]

        roll, pitch, yaw = rpy[0], rpy[1], rpy[2]

        cmd_roll = jnp.clip(action[0], -0.45, 0.45)
        cmd_pitch = jnp.clip(action[1], -0.45, 0.45)
        cmd_yaw_rate = jnp.clip(action[2], -2.0, 2.0)
        cmd_thrust = jnp.clip(action[3], 0.0, mass * g * 2.5)

        roll_rate = (cmd_roll - roll) * 5.0
        pitch_rate = (cmd_pitch - pitch) * 5.0
        yaw_rate = cmd_yaw_rate

        next_roll = roll + roll_rate * dt
        next_pitch = pitch + pitch_rate * dt
        next_yaw = yaw + yaw_rate * dt

        cy, sy = jnp.cos(next_yaw), jnp.sin(next_yaw)
        cr, sr = jnp.cos(next_roll), jnp.sin(next_roll)
        cp, sp = jnp.cos(next_pitch), jnp.sin(next_pitch)

        bz_x = cy * sp * cr + sy * sr
        bz_y = sy * sp * cr - cy * sr
        bz_z = cp * cr

        thrust_acc = jnp.array([bz_x, bz_y, bz_z]) * cmd_thrust / mass
        gravity_acc = jnp.array([0.0, 0.0, -g])
        drag_acc = 0.0 * vel

        total_acc = thrust_acc + gravity_acc + drag_acc

        next_vel = vel + total_acc * dt
        next_pos = pos + vel * dt + 0.5 * total_acc * dt * dt

        return jnp.concatenate([next_pos, next_vel, jnp.array([next_roll, next_pitch, next_yaw])])

    @staticmethod
    def _cost_fn(state: jnp.ndarray, action: jnp.ndarray, next_state: jnp.ndarray, params: dict) -> jnp.ndarray:
        """Stage cost metric mapping optimized trajectories."""
        pos = next_state[0:3]
        vel = next_state[3:6]
        target = params["target"]
        g_dir = params["g_dir"]
        obstacles = params["obstacles"]

        # 1. Positional Attraction Cost
        # Strong pull to target, heavily weighted in Z to fight Level 3 gravity/downdrafts
        diff = pos - target
        dist_to_target = jnp.linalg.norm(diff)
        xy_dist = jnp.linalg.norm(diff[:2])
        z_dist = jnp.abs(diff[2])

        cost = (xy_dist**2) * 60.0 + (z_dist**2) * 100.0
        
        # 2. Cross-Track Error (Virtual Tube)
        rel_pos = pos - target
        longitudinal_dist = jnp.dot(rel_pos, g_dir)
        cross_track_vec = rel_pos - (longitudinal_dist * g_dir)
        cross_track_error = jnp.linalg.norm(cross_track_vec)

        funnel_radius = 0.10 + jnp.maximum(0.0, -longitudinal_dist) * 1.0
        
        raw_tube_penalty = jnp.where(
            (cross_track_error > funnel_radius) & (longitudinal_dist < 0.0), 
            (cross_track_error - funnel_radius) * 50.0, 
            0.0
        )
        cost += raw_tube_penalty
        
        cost += jnp.where(longitudinal_dist < 0.0, cross_track_error  * 100.0, 0.0)

        # 3. GLOBAL STRICT GATE FRAME & STAND PENALTY (True SDF Formulation)
        all_gate_poses = params.get("all_gate_poses")
        all_gate_dirs = params.get("all_gate_dirs")

        if all_gate_poses is not None and all_gate_poses.shape[0] > 0:
            rel_poses = pos[None, :] - all_gate_poses 

            up = jnp.array([0.0, 0.0, 1.0])
            dots = jnp.sum(all_gate_dirs * up, axis=1)
            
            up_vecs = jnp.where(
                jnp.abs(dots)[:, None] > 0.99, 
                jnp.array([0.0, 1.0, 0.0]), 
                jnp.tile(up, (all_gate_dirs.shape[0], 1))
            )
            
            g_y = jnp.cross(all_gate_dirs, up_vecs)
            g_y = g_y / (jnp.linalg.norm(g_y, axis=1, keepdims=True) + 1e-6)
            g_z = jnp.cross(g_y, all_gate_dirs)

            dx = jnp.sum(rel_poses * all_gate_dirs, axis=1)
            dy = jnp.sum(rel_poses * g_y, axis=1)
            dz = jnp.sum(rel_poses * g_z, axis=1)
            
            max_yz = jnp.maximum(jnp.abs(dy), jnp.abs(dz))
            
            # Distance from the exact center of the PVC material
            d_yz = jnp.maximum(0.0, jnp.abs(max_yz - 0.28) - 0.08)
            d_x = jnp.maximum(0.0, jnp.abs(dx) - 0.05)
            
            ext_dist = jnp.sqrt(d_yz**2 + d_x**2 + 1e-8)
            
            depth_yz = 0.08 - jnp.abs(max_yz - 0.28)
            depth_x = 0.05 - jnp.abs(dx)
            interior_depth = jnp.minimum(depth_yz, depth_x)
            
            sdf = jnp.where(
                (depth_yz > 0) & (depth_x > 0),
                -interior_depth,
                ext_dist
            )
            
            frame_collision = jnp.where(
                sdf < 0.10,
                (0.10 - sdf) * 100000.0,
                0.0
            )
            cost += jnp.sum(frame_collision)

            # GATE STAND PENALTY (With Vertical Escape Gradient)
            dist_to_stand_xy = jnp.linalg.norm(pos[:2][None, :] - all_gate_poses[:, :2], axis=1)
            depth_below_gate = jnp.maximum(0.0, (all_gate_poses[:, 2] - 0.15) - pos[2])
            
            stand_collision = jnp.where(
                (depth_below_gate > 0.0) & (dist_to_stand_xy < 0.20),
                (0.20 - dist_to_stand_xy) * 1000.0 + (depth_below_gate * 500.0),
                0.0
            )
            cost += jnp.sum(stand_collision)

        # 4. Velocity Alignment & Speed Limit
        speed = jnp.linalg.norm(vel) + 1e-5
        vel_dir = vel / speed
        target_dir = -diff / (dist_to_target + 1e-5) 
               
        # Multiply the target_dir penalty by the fade multiplier
        cost += (1.0 - jnp.dot(vel_dir, target_dir)) * 12.0 * speed 
        
        # The g_dir penalty remains fully active to keep the drone pointing forward 
        cost += (1.0 - jnp.dot(vel_dir, g_dir)) * 8.0 * speed 
        
        # Allow the drone to fly fast without huge penalties
        cost += (speed ** 2) * 0.15

        # 5. Obstacle Avoidance
        if obstacles.shape[0] > 0:
            obs_xy = obstacles[:, :2]
            pos_xy = pos[:2]
            dist_to_obs = jnp.linalg.norm(obs_xy - pos_xy[None, :], axis=1)
            
          
            repulsion_field = jnp.sum(
                jnp.where(
                    dist_to_obs < 0.15, 
                    (0.15 - dist_to_obs) * 500.0, 
                    0.0
                )
            )
            cost += repulsion_field
        
        # 6. Out-of-Bounds
        oob = (pos[0] < -2.4) | (pos[0] > 2.4) | (pos[1] < -1.4) | (pos[1] > 1.4) | (pos[2] < 0.05) | (pos[2] > 1.95)
        cost += jnp.where(oob, 100000.0, 0.0)

        # 7. Actuator Smoothing
        cost += (action[0]**2 + action[1]**2) * 0.5    
        cost += (action[2]**2) * 2.0                  
        cost += (action[3]** 2) * 0.1

        return cost
    
    @partial(jax.jit, static_argnums=(0,))
    def _rollout_traj(self, state: jnp.ndarray, action_seq: jnp.ndarray, params: dict) -> jnp.ndarray:
        """Forward simulates action sequence for visualization."""
        def scan_fn(s, a):
            next_s = self._dynamics_fn(s, a, params)
            return next_s, next_s[:3]
        _, pos_seq = jax.lax.scan(scan_fn, state, action_seq)
        return pos_seq

    def compute_control(self, obs: dict, info: dict | None = None) -> np.ndarray:
        """Main control loop."""
        if info is None:
            info = {}
            
        # 1. --- LIVE SENSOR UPDATE ---
        live_gates_pos = obs.get("gates_pos", info.get("gates_pos", None))
        if live_gates_pos is None:
            # Prevent re-ordering desync if env fails to pass positional list once initialized
            if hasattr(self, 'last_live_gates_pos_3d'):
                live_gates_pos = self.last_live_gates_pos_3d
            else:
                live_gates_pos = self.init_gates_pos
        else:
            self.last_live_gates_pos_3d = [np.array(g).copy() for g in live_gates_pos]

        live_gates_rpy = self._get_live_gates_rpy(obs, info, self.num_gates)
        
        live_obstacles = obs.get("obstacles_pos", info.get("obstacles_pos", 
                            obs.get("obstacle_pos", info.get("obstacle_pos", self.obstacles_np))))
        
        # Instantly shift targets and mathematical penalty volumes precisely onto the true rotated frame
        self._update_track_geometry(live_gates_pos, live_gates_rpy, live_obstacles)
        
        # 2. --- TARGET ACQUISITION ---
        pos = np.array(obs["pos"])
        self.current_drone_pos = pos.copy()

        if self.active_wp_idx >= len(self.waypoints):
            self.active_wp_idx = len(self.waypoints) - 1

        current_target = np.array(self.waypoints[self.active_wp_idx])
        vec_to_target = current_target - pos
        dist_to_wp = np.linalg.norm(vec_to_target)

        if dist_to_wp < 0.20 and self.active_wp_idx < len(self.waypoints) - 1:
            self.active_wp_idx += 1
            current_target = np.array(self.waypoints[self.active_wp_idx])

        rpy = R.from_quat(obs["quat"]).as_euler("xyz", degrees=False)
        current_state = jnp.array([
            obs["pos"][0], obs["pos"][1], obs["pos"][2],
            obs["vel"][0], obs["vel"][1], obs["vel"][2],
            rpy[0], rpy[1], rpy[2]
        ])

        
        params = {
            "dt": self.dt,
            "dt_pred": self.dt_pred,
            "mass": self.drone_mass,
            "g": self.g,
            "hover_thrust": self.hover_thrust,
            "target": jnp.array(current_target),
            "g_dir": jnp.array(self.gate_dirs[self.active_wp_idx]),
            "obstacles": jnp.array(self.obstacles_np) if len(self.obstacles_np) > 0 else jnp.array([[]]),
            "all_gate_poses": self.all_gate_poses_jnp,
            "all_gate_dirs": self.all_gate_dirs_jnp
        }

        # MPPI optimization
        self.rng_key, subkey = jax.random.split(self.rng_key)

        optimal_action, self.nominal_seq, min_cost, mean_cost = self.engine(
            subkey,
            current_state,
            self.nominal_seq,
            self.dt,
            self.drone_mass,
            self.g,
            self.temperature,
            self.noise_std,
            params,
            dt_pred=self.dt_pred
        )

        action = optimal_action

        pos_seq = self._rollout_traj(current_state, self.nominal_seq, params)
        self.planned_trajectory = np.array(jax.device_get(pos_seq))

        min_action = jnp.array([-0.5, -0.5, -2.0, 0.0])
        max_action = jnp.array([0.5, 0.5, 2.0, self.drone_mass * self.g * 2.5])
        action = jnp.clip(action, min_action, max_action)

        if self.debug:
            self._print_debug(obs, current_target, [min_cost, mean_cost], action)

        return np.array(jax.device_get(action), dtype=np.float32)

    def render_callback(self, sim: Sim):
        """Visualize trajectory, target, waypoints, and the 3D Geometric Funnel."""
        if hasattr(self, 'waypoints') and len(self.waypoints) > 0:
            draw_points(sim, np.array(self.waypoints), rgba=(0.0, 1.0, 0.0, 0.3), size=0.03)

        if hasattr(self, 'active_wp_idx') and self.active_wp_idx < len(self.waypoints):
            target_pos = np.array(self.waypoints[self.active_wp_idx])
            g_dir = np.array(self.gate_dirs[self.active_wp_idx])

            draw_points(sim, target_pos.reshape(1, -1), rgba=(1.0, 0.0, 0.0, 1.0), size=0.05)

            base_gate_radius = 0.15
            funnel_slope = 1.0
            offsets = [0.0, -0.5, -1.0, -1.5]

            up = np.array([0.0, 0.0, 1.0])
            if np.abs(np.dot(g_dir, up)) > 0.99:
                up = np.array([0.0, 1.0, 0.0])
            
            u = np.cross(g_dir, up)
            u = u / (np.linalg.norm(u) + 1e-6)
            v = np.cross(g_dir, u)

            angles = np.linspace(0, 2 * np.pi, 20)
            ring_points_list = []

            for offset_dist in offsets:
                current_radius = base_gate_radius + (abs(offset_dist) * funnel_slope)
                ring_center = target_pos + (g_dir * offset_dist)
                circle_offsets = np.array([np.cos(a) * u + np.sin(a) * v for a in angles]) * current_radius
                ring_points = ring_center + circle_offsets
                
                draw_line(sim, ring_points, rgba=(1.0, 1.0, 0.0, 0.5))
                ring_points_list.append(ring_points)

            if len(ring_points_list) > 1:
                smallest_ring = ring_points_list[0]
                largest_ring = ring_points_list[-1]
                
                for i in [0, 5, 10, 15]: 
                    start_pt = largest_ring[i]
                    end_pt = smallest_ring[i]
                    draw_line(sim, np.vstack([start_pt, end_pt]), rgba=(1.0, 1.0, 0.0, 0.3))

        if hasattr(self, 'planned_trajectory') and self.planned_trajectory is not None:
            draw_line(sim, self.planned_trajectory, rgba=(0.0, 0.0, 1.0, 1.0))

    def step_callback(self, action, obs, reward, terminated, truncated, info) -> bool:
        self._tick += 1
        if terminated or truncated:
            self._finished = True
        return self._finished

    def episode_callback(self):
        self._tick = 0
        self._finished = False
        self.active_wp_idx = 0
        self.thrust_integrator = 0.0
        self.nominal_seq = jnp.zeros((self.H, 4))
        self.nominal_seq = self.nominal_seq.at[:, 3].set(self.hover_thrust)
        self.rng_key = jax.random.PRNGKey(42)

    def _print_debug(self, obs, target_np, traj_costs, action):
        """Log debug information."""
        pos = obs["pos"]
        try:
            costs_np = np.array(jax.device_get(traj_costs))
            act_np = np.array(jax.device_get(action))
            dist_to_target = np.linalg.norm(pos - target_np)

            if len(self.obstacles_np) > 0:
                dists_obs = np.linalg.norm(self.obstacles_np[:, :2] - pos[:2], axis=1)
                nearest_obs = dists_obs.min()
            else:
                nearest_obs = 999.0

            gate_idx = self.active_wp_idx // 3
            msg = (
                f"[SMPC] pos={pos.round(3)} tgt={target_np.round(3)} "
                f"dist={dist_to_target:.3f} obs={nearest_obs:.3f} "
                f"cost_min/mean={costs_np.min():.1f}/{costs_np.mean():.1f} "
                f"act={act_np.round(3)} wp={self.active_wp_idx} gate={gate_idx}"
            )
            print(msg)
            try:
                self._logger.info(msg)
            except Exception:
                pass
        except Exception:
            pass