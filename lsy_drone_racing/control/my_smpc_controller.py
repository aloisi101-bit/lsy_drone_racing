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
#from lsy_drone_racing.control.mppi_engine_backup_marek import MPPIEngine
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
        # Gates only reveal their true (randomized) pose once the drone is within this
        # range; before that, obs reports the nominal pose. See compute_control's
        # caution-zone speed cap, which slows the approach so there's more real time to
        # react once the true pose snaps in.
        self.sensor_range = float(getattr(config.env, "sensor_range", 0.7))
        
        gates_config = config.env.track.gates
        self.num_gates = len(gates_config)

       # Store nominal positions and orientations safely
        self.init_gates_pos = [g["pos"] for g in gates_config]
        self.init_gates_rpy = [g["rpy"] for g in gates_config]

        self._update_track_geometry(self.init_gates_pos, self.init_gates_rpy, self.obstacles_np)
        self._cached_gates_pos = np.array(self.init_gates_pos)
        self._cached_gates_rpy = np.array(self.init_gates_rpy)

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

        self._episode_count = 0

        # --- MPPI Configuration ---
        self.H = 10  # 0.8 s lookahead at dt_pred=0.04
        self.K = 5000  # Number of parallel trajectory samples

        # Noise exploration: [roll, pitch, yaw_rate, thrust]
        self.noise_std = jnp.array([0.15, 0.15, 0.30, 0.15])

        self.engine = build_mppi_solver(self._cost_fn, self._dynamics_fn, K=self.K)

        # Initialize nominal with gentle forward bias (helps cold start)
        self.nominal_seq = jnp.zeros((self.H, 4))
        self.nominal_seq = self.nominal_seq.at[:, 0].set(0.05)  # Gentle forward tilt
        self.nominal_seq = self.nominal_seq.at[:, 3].set(self.hover_thrust * 1.1)  # Slight lift
        self.rng_key = jax.random.PRNGKey(42)
        self.last_action = jnp.array([0.0, 0.0, 0.0, self.hover_thrust])

        self.planned_trajectory = None

    def _update_track_geometry(self, live_gates_pos, live_gates_rpy, live_obstacles):
        """Rebuilds the MPC track geometry based on dynamic sensor observations."""
        self.obstacles_np = np.array(live_obstacles)

        waypoints = []
        gate_dirs = []
        all_gate_poses = []
        all_gate_dirs = []

        for i in range(len(live_gates_pos)):
            gate_pos = np.array(live_gates_pos[i], dtype=float)
            gate_rpy = np.array(live_gates_rpy[i], dtype=float)

            rot = R.from_euler("xyz", gate_rpy)
            forward_dir = rot.apply(np.array([1.0, 0.0, 0.0]))
            forward_dir[2] = 0.0  # Keep trajectory in XY plane
            norm = np.linalg.norm(forward_dir)
            forward_dir = forward_dir / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])

            all_gate_poses.append(gate_pos.copy())
            all_gate_dirs.append(forward_dir.copy())

            pre_wp = gate_pos - forward_dir * 0.2
            center_wp = gate_pos.copy()
            post_wp = gate_pos + forward_dir * 0.8

            waypoints.extend([pre_wp, center_wp, post_wp])
            gate_dirs.extend([forward_dir, forward_dir, forward_dir])

        # Push to class state
        self.waypoints = jnp.array(waypoints)
        self.gate_dirs = jnp.array(gate_dirs)
        self.all_gate_poses_jnp = jnp.array(all_gate_poses)
        self.all_gate_dirs_jnp = jnp.array(all_gate_dirs)


    @staticmethod
    def _dynamics_fn(state: jnp.ndarray, action: jnp.ndarray, params: dict) -> jnp.ndarray:
        """Proper kinematic rigid body dynamics with correct physics.
        
        Args:
            state: 9D state [pos_xyz, vel_xyz, rpy]
            action: 4D action [roll_cmd, pitch_cmd, yaw_rate, thrust]
            params: dict with dt, dt_pred, mass, g
            
        Returns:
            next_state: 9D state after one timestep
        """
        dt = params.get("dt_pred", params["dt"])
        mass = params["mass"]
        g = params["g"]

        pos = state[0:3]
        vel = state[3:6]
        rpy = state[6:9]

        roll, pitch, yaw = rpy[0], rpy[1], rpy[2]

        # Clamp attitude commands to reasonable limits
        cmd_roll = jnp.clip(action[0], -0.45, 0.45)
        cmd_pitch = jnp.clip(action[1], -0.45, 0.45)
        cmd_yaw_rate = jnp.clip(action[2], -2.0, 2.0)  # rad/s limit
        cmd_thrust = jnp.clip(action[3], 0.0, mass * g * 2.5)

        roll_rate = (cmd_roll - roll) * 5.0
        pitch_rate = (cmd_pitch - pitch) * 5.0
        yaw_rate = cmd_yaw_rate

        # Integrate attitudes
        next_roll = roll + roll_rate * dt
        next_pitch = pitch + pitch_rate * dt
        next_yaw = yaw + yaw_rate * dt

        # Compute rotation matrix from integrated RPY
        cy, sy = jnp.cos(next_yaw), jnp.sin(next_yaw)
        cr, sr = jnp.cos(next_roll), jnp.sin(next_roll)
        cp, sp = jnp.cos(next_pitch), jnp.sin(next_pitch)

        # Body Z-axis in world frame (where thrust acts)
        # This is column 3 of the rotation matrix (from body to world)
        bz_x = cy * sp * cr + sy * sr
        bz_y = sy * sp * cr - cy * sr
        bz_z = cp * cr

        # Thrust acceleration in world frame
        thrust_acc = jnp.array([bz_x, bz_y, bz_z]) * cmd_thrust / mass

        # Gravity and drag
        gravity_acc = jnp.array([0.0, 0.0, -g])
        drag_coeff = 0.08  # Light aerodynamic drag — prevents runaway speed in rollouts
        drag_acc = -drag_coeff * vel

        total_acc = thrust_acc + gravity_acc + drag_acc

        # Integrate velocity and position
        next_vel = vel + total_acc * dt
        next_pos = pos + vel * dt + 0.5 * total_acc * dt * dt

        return jnp.concatenate([next_pos, next_vel, jnp.array([next_roll, next_pitch, next_yaw])])

    @staticmethod
    def _cost_fn(state: jnp.ndarray, action: jnp.ndarray, next_state: jnp.ndarray, params: dict) -> jnp.ndarray:
        """Stage cost metric mapping optimized trajectories.
        
        Implements PA-MPPI cost function with:
        - Positional attraction to target
        - Cross-track error (funnel constraints)
        - Gate frame penalties
        - Velocity alignment
        - Obstacle avoidance
        - Arena bounds enforcement
        - Actuator smoothing
        
        Args:
            state: Current 9D state [pos, vel, rpy]
            action: Control input [roll, pitch, yaw_rate, thrust]
            next_state: Next state after dynamics
            params: Cost parameters dict
            
        Returns:
            Scalar cost value
        """
        pos = next_state[0:3]
        vel = next_state[3:6]
        target = params["target"]
        g_dir = params["g_dir"]
        obstacles = params["obstacles"]
       

        

        # 1. Positional Attraction Cost
        diff = pos - target
        dist_to_target = jnp.linalg.norm(diff)
        
        xy_dist = jnp.linalg.norm(diff[:2])
        z_dist = jnp.abs(diff[2])

        raw_pos_cost = (xy_dist**2) * 50.0 + (z_dist**2) * 40.0
        
        # CLIP RESTORED: Capped at 600.0 to prevent waypoint shock.
        #cost = jnp.clip(raw_pos_cost, 0.0, 600.0)
        cost = raw_pos_cost  # No clipping, let the optimizer feel the full gradient to the target.
        
       # 2. Cross-Track Error (Virtual Tube)
        rel_pos = pos - target
        longitudinal_dist = jnp.dot(rel_pos, g_dir)
        cross_track_vec = rel_pos - (longitudinal_dist * g_dir)
        cross_track_error = jnp.linalg.norm(cross_track_vec)

        # A. STRICT GEOMETRIC FUNNEL (The Hard Wall)
        # Prevents the drone from flying too wide
        longitudinal_dist_abs = jnp.abs(longitudinal_dist)
        funnel_radius = 0.10 + (longitudinal_dist_abs * 1.0)

        raw_tube_penalty = jnp.where(cross_track_error > funnel_radius, (cross_track_error - funnel_radius) * 5000.0, 0.0)
        #cost += jnp.clip(raw_tube_penalty, 0.0, 400.0) 
        cost += raw_tube_penalty  # No clipping, let the optimizer feel the full gradient to stay in the funnel.\


        # --- B. CENTERLINE ATTRACTION (The Gravity Well) ---
        # Gently but constantly pulls the drone into the absolute dead center of the track.
        # Because this is not capped, the optimizer will always try to minimize it.
        cost += cross_track_error * 15.0
        # ---------------------------------------------------------
        # 3. GLOBAL STRICT GATE FRAME PENALTY (All Gates at once)
        # ---------------------------------------------------------
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

            # --- GATE FRAME COLLISION BARRIER ---
            # Matches the real MuJoCo collision boxes in envs/assets/gate.xml: a safe
            # square hole of half-width ~0.20m, frame material from ~0.20-0.36m, and a
            # paper-thin depth (~0.01m) along the gate normal. The previous formula
            # multiplied three sub-1 fractional terms together, which structurally
            # capped its own peak penalty around ~150 regardless of the nominal 250,000
            # weight — far too weak to discourage an off-center pass (obstacle
            # hard-crash is 80,000; this is the dominant cause of gate-frame collisions
            # in evaluation). This barrier scales a single quadratic term directly, so
            # the weight controls the realized magnitude.
            safe_half = 0.16  # tighter than the real 0.20m hole: margin for drone body
            outer_half = 0.40  # a bit past the real 0.36m frame edge
            thickness = 0.15  # depth margin around the (paper-thin) real frame

            lat_excess = jnp.maximum(jnp.abs(dy) - safe_half, jnp.abs(dz) - safe_half)
            lat_excess = jnp.maximum(lat_excess, 0.0)
            in_outer = (jnp.abs(dy) < outer_half) & (jnp.abs(dz) < outer_half)
            near_plane = jnp.abs(dx) < thickness

            frame_collision = jnp.where(in_outer & near_plane, (lat_excess ** 2) * 100000.0, 0.0)

            cost += jnp.sum(frame_collision)
        # ---------------------------------------------------------
        # 4. Velocity — fast in open space, controlled on gate approach
        speed = jnp.linalg.norm(vel) + 1e-5
        vel_dir = vel / speed
        target_dir = -diff / (dist_to_target + 1e-5)

        cost += (1.0 - jnp.dot(vel_dir, target_dir)) * 10.0 * speed
        cost += (1.0 - jnp.dot(vel_dir, g_dir)) * 8.0 * speed

        cost += (speed ** 2) * 3.0

        # Yaw alignment: penalize sideways flight with fixed weight (no speed scaling —
        # speed-scaled yaw cost causes MPPI to systematically prefer slow trajectories)
        yaw = next_state[8]
        vel_xy = vel[:2]
        speed_xy = jnp.linalg.norm(vel_xy) + 1e-5
        desired_yaw = jnp.arctan2(vel_xy[1], vel_xy[0])
        yaw_err = jnp.arctan2(jnp.sin(yaw - desired_yaw), jnp.cos(yaw - desired_yaw))
        cost += (yaw_err ** 2) * 0.5

        # Next-waypoint lookahead: gently pull toward the waypoint after current target.
        # This is the key fix for gate transitions (e.g. gate 3 → gate 4 180° turn):
        # while flying toward gate 3 post_wp, the planner already curves toward gate 4 pre_wp.
        next_diff = pos - params["next_target"]
        cost += jnp.linalg.norm(next_diff) * 2.0


        # 5. Obstacle Avoidance
        if obstacles.shape[0] > 0 and obstacles.shape[1] > 0:
            obs_xy = obstacles[:, :2]
            pos_xy = pos[:2]
            dist_to_obs = jnp.linalg.norm(obs_xy - pos_xy[None, :], axis=1)

            repulsion_field = jnp.sum(
                jnp.where(
                    dist_to_obs < 0.25,
                    (0.25 - dist_to_obs) ** 2 * 6000.0,
                    0.0
                )
            )
            hard_crash = jnp.sum(jnp.where(dist_to_obs < 0.10, 80000.0, 0.0))
            cost += repulsion_field + hard_crash
        
        # 6. Out-of-Bounds Arena Guardrail Constraints
        oob = (pos[0] < -2.4) | (pos[0] > 2.4) | (pos[1] < -1.4) | (pos[1] > 1.4) | (pos[2] < 0.05) | (pos[2] > 1.95)
        cost += jnp.where(oob, 500.0, 0.0)

        # 7. Actuator Smoothing
        cost += (action[0]**2 + action[1]**2) * 5.0
        cost += ((action[3] - params["hover_thrust"]) ** 2) * 5.0

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

        def _get(key, default):
            """Safe lookup: obs first, then info, then default."""
            if key in obs:
                return obs[key]
            if info is not None and key in info:
                return info[key]
            return default

        # 1. --- LIVE SENSOR UPDATE ---
        live_gates_pos = _get("gates_pos", self.all_gate_poses_jnp)

        # Safely extract orientations: Check for RPY, then Yaw, then fallback to nominal config
        if "gates_rpy" in obs or (info is not None and "gates_rpy" in info):
            live_gates_rpy = _get("gates_rpy", self.init_gates_rpy)
        elif "gates_yaw" in obs or (info is not None and "gates_yaw" in info):
            yaws = _get("gates_yaw", None)
            live_gates_rpy = np.zeros((len(live_gates_pos), 3))
            live_gates_rpy[:, 2] = yaws
        else:
            live_gates_rpy = self.init_gates_rpy

        live_obstacles = _get("obstacles_pos", self.obstacles_np)

        # Only recompute geometry when gate data actually changes (gates are fixed mid-episode)
        gates_pos_arr = np.array(live_gates_pos)
        gates_rpy_arr = np.array(live_gates_rpy)
        if not (np.array_equal(gates_pos_arr, self._cached_gates_pos) and
                np.array_equal(gates_rpy_arr, self._cached_gates_rpy)):
            self._update_track_geometry(live_gates_pos, live_gates_rpy, live_obstacles)
            self._cached_gates_pos = gates_pos_arr
            self._cached_gates_rpy = gates_rpy_arr
        
        # 2. --- TARGET ACQUISITION ---
        pos = np.array(obs["pos"])

        # Sync waypoint index with env's gate tracker to prevent desync at high speed.
        env_gate = int(obs.get("target_gate", -1))
        if env_gate != -1:
            expected_wp = env_gate * 3
            if self.active_wp_idx < expected_wp:
                self.active_wp_idx = expected_wp

        # If we've gone through all waypoints, loop back to last one
        if self.active_wp_idx >= len(self.waypoints):
            self.active_wp_idx = len(self.waypoints) - 1

        current_target = np.array(self.waypoints[self.active_wp_idx])
        vec_to_target = current_target - pos
        dist_to_wp = np.linalg.norm(vec_to_target)

        # Tight waypoint switching: just distance, no "passed plane" logic
        # This prevents overshooting
        if abs(dist_to_wp) < 0.15 and self.active_wp_idx < len(self.waypoints) - 1:
            self.active_wp_idx += 1
            current_target = np.array(self.waypoints[self.active_wp_idx])

        # Get current state
        rpy = R.from_quat(obs["quat"]).as_euler("xyz", degrees=False)
        current_state = jnp.array([
            obs["pos"][0], obs["pos"][1], obs["pos"][2],
            obs["vel"][0], obs["vel"][1], obs["vel"][2],
            rpy[0], rpy[1], rpy[2]
        ])

        active_gate_idx = self.active_wp_idx // 3

        # Next waypoint for lookahead cost — clamped so last waypoint points to itself
        next_wp_idx = min(self.active_wp_idx + 1, len(self.waypoints) - 1)
        next_target = np.array(self.waypoints[next_wp_idx])

        # Pack simulation parameters
        params = {
            "dt": self.dt,
            "dt_pred": self.dt_pred,
            "mass": self.drone_mass,
            "g": self.g,
            "hover_thrust": self.hover_thrust,
            "target": jnp.array(current_target),
            "next_target": jnp.array(next_target),
            "g_dir": jnp.array(self.gate_dirs[self.active_wp_idx]),
            "obstacles": jnp.array(self.obstacles_np) if len(self.obstacles_np) > 0 else jnp.array([[]]),
            "active_gate_pos": jnp.array(self.all_gate_poses_jnp[active_gate_idx]),
            "active_gate_dir": jnp.array(self.all_gate_dirs_jnp[active_gate_idx]),
            "sensor_range": self.sensor_range,

            # Penalize the active gate's frame plus the one just passed: the env credits a
            # gate pass as soon as the drone's center crosses the gate plane within bounds,
            # but the drone's body can still be clearing the frame edge for a few more cm —
            # dropping the just-passed gate's frame penalty at that exact moment leaves a
            # window where nothing discourages grazing it on the way out.
            "all_gate_poses": self.all_gate_poses_jnp[max(0, active_gate_idx - 1):active_gate_idx + 1],
            "all_gate_dirs": self.all_gate_dirs_jnp[max(0, active_gate_idx - 1):active_gate_idx + 1],
        }

        # MPPI optimization
        self.rng_key, subkey = jax.random.split(self.rng_key)

        # Call the JIT-compiled solver directly.
        # Notice we pass lambda (0.08) and our noise_std here now.
        optimal_action, self.nominal_seq, min_cost, mean_cost = self.engine(
            subkey,
            current_state,
            self.nominal_seq,
            self.dt,
            self.drone_mass,
            self.g,
            0.08,             # lambda
            self.noise_std,
            params,           # <--- SEE CRITICAL WARNING BELOW
            dt_pred=self.dt_pred
        )

        action = optimal_action
        self.last_action = optimal_action

       # Calculate visualization trajectory using the newly updated nominal sequence
        pos_seq = self._rollout_traj(current_state, self.nominal_seq, params)
        self.planned_trajectory = np.array(jax.device_get(pos_seq))

        # Saturate actions — bounds must match engine lower_bound/upper_bound
        min_action = jnp.array([-0.5, -0.5, -2.0, self.drone_mass * self.g * 0.5])
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

            # Draw the active target point (solid red)
            draw_points(sim, target_pos.reshape(1, -1), rgba=(1.0, 0.0, 0.0, 1.0), size=0.05)

            # --- NEW: Draw 3D Wireframe Geometric Funnel ---
            base_gate_radius = 0.15
            funnel_slope = 1.0  # Must match the dist_to_target multiplier in _cost_fn
            
            # Draw rings at these distances extending BACKWARDS from the gate
            # e.g., 0.0m, 0.5m, 1.0m, and 1.5m away
            offsets = [0.0, -0.5, -1.0, -1.5]

            # 1. Calculate two vectors (u, v) perpendicular to the gate direction
            up = np.array([0.0, 0.0, 1.0])
            if np.abs(np.dot(g_dir, up)) > 0.99:
                up = np.array([0.0, 1.0, 0.0]) # Fallback if gate points straight up
            
            u = np.cross(g_dir, up)
            u = u / (np.linalg.norm(u) + 1e-6)
            v = np.cross(g_dir, u)

            angles = np.linspace(0, 2 * np.pi, 20)
            ring_points_list = []

            # 2. Draw rings with expanding radii
            for offset_dist in offsets:
                # Calculate the exact radius the cost function sees at this distance
                current_radius = base_gate_radius + (abs(offset_dist) * funnel_slope)
                
                ring_center = target_pos + (g_dir * offset_dist)
                circle_offsets = np.array([np.cos(a) * u + np.sin(a) * v for a in angles]) * current_radius
                ring_points = ring_center + circle_offsets
                
                draw_line(sim, ring_points, rgba=(1.0, 1.0, 0.0, 0.5))
                ring_points_list.append(ring_points)

            # 3. Draw longitudinal connecting lines (the "walls" of the funnel)
            if len(ring_points_list) > 1:
                smallest_ring = ring_points_list[0]   # At the gate
                largest_ring = ring_points_list[-1]   # Furthest away
                
                for i in [0, 5, 10, 15]: 
                    start_pt = largest_ring[i]
                    end_pt = smallest_ring[i]
                    draw_line(sim, np.vstack([start_pt, end_pt]), rgba=(1.0, 1.0, 0.0, 0.3))
            # -----------------------------------

        

        # Draw the MPPI planned trajectory horizon (blue line)
        if hasattr(self, 'planned_trajectory') and self.planned_trajectory is not None:
            draw_line(sim, self.planned_trajectory, rgba=(0.0, 0.0, 1.0, 1.0))

    def step_callback(self, action, obs, reward, terminated, truncated, info) -> bool:
        self._tick += 1
        if terminated or truncated:
            self._finished = True
        return self._finished

    def episode_callback(self):
        self._episode_count += 1
        self._tick = 0
        self._finished = False
        self.active_wp_idx = 0
        self.nominal_seq = jnp.zeros((self.H, 4))
        self.nominal_seq = self.nominal_seq.at[:, 3].set(self.hover_thrust)
        # Use a different seed each episode so the 20 evaluation runs explore different trajectories
        self.rng_key = jax.random.PRNGKey(self._episode_count)
        self.last_action = jnp.array([0.0, 0.0, 0.0, self.hover_thrust])

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
