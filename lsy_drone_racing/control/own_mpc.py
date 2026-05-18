"""This module implements an example MPC using attitude control for a quadrotor.

It utilizes the collective thrust interface for drone control to compute control commands based on
current state observations and desired waypoints.

The waypoints are generated using cubic spline interpolation from a set of predefined waypoints.
The trajectory adapts when:
- Gate positions detected from sensors differ from hard-coded positions
- Obstacles are detected and need to be avoided

The trajectory is recomputed periodically based on sensor feedback.
"""

from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING

import numpy as np
import scipy
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from drone_models.core import load_params
from drone_models.so_rpy import symbolic_dynamics_euler
from drone_models.utils.rotation import ang_vel2rpy_rates
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


def create_acados_model(parameters: dict) -> AcadosModel:
    """Creates an acados model from a symbolic drone_model."""
    # For more info on the models, check out https://github.com/utiasDSL/drone-models
    X_dot, X, U, _ = symbolic_dynamics_euler(
        mass=parameters["mass"],
        gravity_vec=parameters["gravity_vec"],
        J=parameters["J"],
        J_inv=parameters["J_inv"],
        acc_coef=parameters["acc_coef"],
        cmd_f_coef=parameters["cmd_f_coef"],
        rpy_coef=parameters["rpy_coef"],
        rpy_rates_coef=parameters["rpy_rates_coef"],
        cmd_rpy_coef=parameters["cmd_rpy_coef"],
    )

    # Initialize the nonlinear model for NMPC formulation
    model = AcadosModel()
    model.name = "basic_example_mpc"
    model.f_expl_expr = X_dot
    model.f_impl_expr = None
    model.x = X
    model.u = U

    return model


def create_ocp_solver(
    Tf: float, N: int, parameters: dict, verbose: bool = False
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Creates an acados Optimal Control Problem and Solver."""
    ocp = AcadosOcp()

    # Set model
    ocp.model = create_acados_model(parameters)

    # Get Dimensions
    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()
    ny = nx + nu
    ny_e = nx

    # Set dimensions
    ocp.solver_options.N_horizon = N

    ## Set Cost
    # For more Information regarding Cost Function Definition in Acados:
    # https://github.com/acados/acados/blob/main/docs/problem_formulation/problem_formulation_ocp_mex.pdf
    #

    # Cost Type
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    # Weights
    # State weights
    Q = np.diag(
        [
            50.0,  # pos
            50.0,  # pos
            400.0,  # pos
            1.0,  # rpy
            1.0,  # rpy
            1.0,  # rpy
            10.0,  # vel
            10.0,  # vel
            10.0,  # vel
            5.0,  # drpy
            5.0,  # drpy
            5.0,  # drpy
        ]
    )
    # Input weights (reference is upright orientation and hover thrust)
    R = np.diag(
        [
            1.0,  # rpy
            1.0,  # rpy
            1.0,  # rpy
            50.0,  # thrust
        ]
    )

    Q_e = Q.copy()
    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = Q_e

    Vx = np.zeros((ny, nx))
    Vx[0:nx, 0:nx] = np.eye(nx)  # Select all states
    ocp.cost.Vx = Vx

    Vu = np.zeros((ny, nu))
    Vu[nx : nx + nu, :] = np.eye(nu)  # Select all actions
    ocp.cost.Vu = Vu

    Vx_e = np.zeros((ny_e, nx))
    Vx_e[0:nx, 0:nx] = np.eye(nx)  # Select all states
    ocp.cost.Vx_e = Vx_e

    # Set initial references. We will overwrite these later to track the trajectory
    ocp.cost.yref, ocp.cost.yref_e = np.zeros((ny,)), np.zeros((ny_e,))

    # Set State Constraints (rpy < 30°)
    ocp.constraints.lbx = np.array([-0.5, -0.5, -0.5])
    ocp.constraints.ubx = np.array([0.5, 0.5, 0.5])
    ocp.constraints.idxbx = np.array([3, 4, 5])

    # Set Input Constraints (rpy < 30°)
    ocp.constraints.lbu = np.array([-0.5, -0.5, -0.5, parameters["thrust_min"] * 4])
    ocp.constraints.ubu = np.array([0.5, 0.5, 0.5, parameters["thrust_max"] * 4])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    # We have to set x0 even though we will overwrite it later on.
    ocp.constraints.x0 = np.zeros((nx))

    # Solver Options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"  # FULL_, PARTIAL_ ,_HPIPM, _QPOASES
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"  # SQP, SQP_RTI
    ocp.solver_options.tol = 1e-6

    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_warm_start = 1

    ocp.solver_options.qp_solver_iter_max = 20
    ocp.solver_options.nlp_solver_max_iter = 50

    # set prediction horizon
    ocp.solver_options.tf = Tf

    acados_ocp_solver = AcadosOcpSolver(
        ocp,
        json_file="c_generated_code/lsy_example_mpc.json",
        verbose=verbose,
        build=True,
        generate=True,
    )

    return acados_ocp_solver, ocp


class AttitudeMPC(Controller):
    """Example of a MPC using the collective thrust and attitude interface."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the attitude controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
        """
        super().__init__(obs, info, config)
        self._N = 25
        self._dt = 1 / config.env.freq
        self._T_HORIZON = self._N * self._dt

        # Same waypoints as in the trajectory controller. Determined by trial and error.
        self._hard_coded_waypoints = np.array(
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
            ]
        )
        self._t_total = 15  # s
        # Level 2 Optimization: 5x faster updates (0.1s instead of 0.5s) to react quickly to randomized gates
        self._recompute_interval = 0.1  # recompute trajectory every 0.1 seconds
        self._recompute_steps = int(self._recompute_interval * config.env.freq)

        # Initialize current waypoints and build trajectories
        self._current_waypoints = self._hard_coded_waypoints.copy()
        self._adapted_gate_indices = set()
        self._last_recompute_tick = 0
        self._sensor_range = config.env.sensor_range

        # Track gate detections for predictive adaptation
        self._detected_gates = {}  # gate_idx -> detected position

        self._build_waypoint_trajectories(config.env.freq)

        self.drone_params = load_params("so_rpy", config.sim.drone_model)
        self._acados_ocp_solver, self._ocp = create_ocp_solver(
            self._T_HORIZON, self._N, self.drone_params
        )
        
        self._nx = self._ocp.model.x.rows()
        self._nu = self._ocp.model.u.rows()
        self._ny = self._nx + self._nu
        self._ny_e = self._nx

        self._tick = 0
        self._tick_max = len(self._waypoints_pos) - 1 - self._N
        self._config = config
        self._finished = False

    def _build_waypoint_trajectories(self, freq: int):
        """Build spline-based waypoint trajectories from current waypoints."""
        t = np.linspace(0, self._t_total, len(self._current_waypoints))
        des_pos_spline = CubicSpline(t, self._current_waypoints)
        des_vel_spline = des_pos_spline.derivative()

        # Pre-compute waypoints at control frequency
        n_waypoints = int(freq * self._t_total)
        t_waypoints = np.linspace(0, self._t_total, n_waypoints)
        self._waypoints_pos = des_pos_spline(t_waypoints)
        self._waypoints_vel = des_vel_spline(t_waypoints)
        self._waypoints_yaw = self._waypoints_pos[:, 0] * 0

        # Update tick_max based on new waypoint count
        self._tick_max = len(self._waypoints_pos) - 1 - self._N

    def _distance(self, pos1: NDArray[np.floating], pos2: NDArray[np.floating]) -> float:
        """Compute Euclidean distance between two 3D points."""
        return float(np.linalg.norm(pos1 - pos2))

    def _get_waypoint_index_for_gate(self, gate_idx: int) -> int:
        """Map a gate index to a waypoint index based on spatial proximity.

        Correct mapping based on gate and waypoint positions:
        - Gate 0 [0.5, 0.25, 0.7] → Waypoint 2 [0.3, 0.35, 0.7]
        - Gate 1 [1.05, 0.75, 1.2] → Waypoint 4 [0.85, 0.85, 1.2]
        - Gate 2 [-1.0, -0.25, 0.7] → Waypoint 1 [-1.0, 0.55, 0.4]
        - Gate 3 [0.0, -0.75, 1.2] → Waypoint 8 [-0.0, -0.7, 1.2]
        """
        gate_to_waypoint = {
            0: 2,  # Gate 0 → Waypoint 2
            1: 4,  # Gate 1 → Waypoint 4
            2: 1,  # Gate 2 → Waypoint 1
            3: 8,  # Gate 3 → Waypoint 8
        }
        return gate_to_waypoint.get(gate_idx, gate_idx)

    def _add_safety_margin_to_waypoint(
        self, waypoint: NDArray[np.floating], obstacles_pos: NDArray[np.floating],
        safety_distance: float = 0.2
    ) -> NDArray[np.floating]:
        """Add conservative safety margin to waypoint."""
        if len(obstacles_pos) == 0:
            # Even without obstacles, add small margin for level 2
            return waypoint + np.array([0.08, 0.08, 0.0])

        offset = np.zeros(3)
        has_threat = False

        for obs_pos in obstacles_pos:
            dist = self._distance(waypoint, obs_pos)
            if dist < safety_distance:
                has_threat = True
                if dist > 0.01:
                    direction = (waypoint - obs_pos) / dist
                    # Stronger repulsion for level 2
                    offset += direction * (safety_distance - dist) / (safety_distance * 0.7)

        # Cap the offset to prevent drastic changes
        offset_magnitude = np.linalg.norm(offset)
        if offset_magnitude > 0.01:
            max_offset = 0.25
            offset = offset / offset_magnitude * min(offset_magnitude, max_offset)
        elif not has_threat:
            # Small default margin
            offset = np.array([0.08, 0.08, 0.0])

        return waypoint + offset

    def _is_obstacle_near_waypoint(
        self, waypoint: NDArray[np.floating], obstacles_pos: NDArray[np.floating],
        safety_distance: float = 0.35
    ) -> bool:
        """Check if any obstacle is within safety_distance of a waypoint."""
        for obs_pos in obstacles_pos:
            if self._distance(waypoint, obs_pos) < safety_distance:
                return True
        return False

    def _offset_waypoint_from_obstacles(
        self, waypoint: NDArray[np.floating], obstacles_pos: NDArray[np.floating],
        safety_distance: float = 0.35, offset_magnitude: float = 0.25
    ) -> NDArray[np.floating]:
        """Offset a waypoint away from nearby obstacles with stronger repulsion for Level 2."""
        if len(obstacles_pos) == 0:
            return waypoint.copy()

        offset = np.zeros(3)
        for obs_pos in obstacles_pos:
            dist = self._distance(waypoint, obs_pos)
            if dist < safety_distance and dist > 0.01:
                direction = (waypoint - obs_pos) / dist
                # Stronger offset for level 2
                offset += direction * (safety_distance - dist) / (safety_distance * 0.6)

        offset_magnitude_actual = np.linalg.norm(offset)
        if offset_magnitude_actual > 0.01:
            # Use slightly larger offset for level 2
            offset = offset / offset_magnitude_actual * offset_magnitude

        return waypoint + offset

    def _update_trajectory(self, obs: dict[str, NDArray[np.floating]]) -> bool:
        """Update waypoints based on sensor data with smooth adaptation."""
        waypoints_changed = False
        gates_pos = obs["gates_pos"]
        gates_visited = obs.get("gates_visited", np.zeros(len(gates_pos), dtype=bool))
        obstacles_pos = obs.get("obstacles_pos", [])
        
        # Calculate roughly which waypoint we are currently approaching
        current_progress_ratio = self._tick / self._tick_max
        current_wp_idx = int(current_progress_ratio * len(self._current_waypoints))

        # 1. GATE ADAPTATION
        for gate_idx, (gate_visited, sensed_pos) in enumerate(zip(gates_visited, gates_pos)):
            # BUG FIX: Skip gates we have ALREADY visited. Only adapt to future gates.
            if gate_visited:
                continue

            waypoint_idx = self._get_waypoint_index_for_gate(gate_idx)
            
            # Only adapt waypoints that are strictly IN FRONT of the drone
            if waypoint_idx > current_wp_idx and waypoint_idx < len(self._current_waypoints):
                hard_coded_pos = self._hard_coded_waypoints[waypoint_idx]
                difference_magnitude = self._distance(sensed_pos, hard_coded_pos)

                if difference_magnitude > 0.05: # Minimum threshold to adapt
                    # Blend the adaptation so the MPC doesn't receive a step-input shock
                    target_pos = hard_coded_pos + (sensed_pos - hard_coded_pos) * 0.8
                    current_pos = self._current_waypoints[waypoint_idx]
                    
                    # Exponential Smoothing (Alpha = 0.3): Slowly drag the waypoint to the new target
                    alpha = 0.3 
                    smoothed_pos = current_pos * (1.0 - alpha) + target_pos * alpha

                    if self._distance(current_pos, smoothed_pos) > 0.01:
                        self._current_waypoints[waypoint_idx] = smoothed_pos
                        self._adapted_gate_indices.add(waypoint_idx)
                        waypoints_changed = True

        # 2. OBSTACLE AVOIDANCE
        if len(obstacles_pos) > 0:
            # Only check waypoints in front of the drone
            for i in range(current_wp_idx + 1, len(self._current_waypoints)):
                original_wp = self._current_waypoints[i]

                if self._is_obstacle_near_waypoint(original_wp, obstacles_pos, safety_distance=0.3):
                    offset_target = self._offset_waypoint_from_obstacles(
                        original_wp, obstacles_pos, safety_distance=0.3, offset_magnitude=0.25
                    )
                    
                    # Smooth the obstacle offset as well
                    alpha = 0.4
                    smoothed_offset = original_wp * (1.0 - alpha) + offset_target * alpha
                    
                    if self._distance(original_wp, smoothed_offset) > 0.01:
                        self._current_waypoints[i] = smoothed_offset
                        waypoints_changed = True

        # Rebuild the spline if changes occurred
        if waypoints_changed:
            self._build_waypoint_trajectories(int(1 / self._dt))

        return waypoints_changed

    def _update_trajectory_old(self, obs: dict[str, NDArray[np.floating]]) -> bool:
        """Update waypoints based on sensor data with CONSERVATIVE adaptation for Level 2.

        Key principle: Don't directly use sensed gate positions as waypoints.
        Instead, use them to inform safe adjustments to hard-coded waypoints.

        Returns True if trajectory was recomputed, False otherwise.
        """
        waypoints_changed = False
        gates_pos = obs["gates_pos"]
        gates_visited = obs["gates_visited"]
        obstacles_pos = obs["obstacles_pos"]
        obstacles_visited = obs["obstacles_visited"]

        # LEVEL 2 OPTIMIZATION 1: Conservatively adapt based on detected gate offsets
        # Don't use sensed position directly—use it to adjust hard-coded waypoint
        for gate_idx, (gate_visited, sensed_pos) in enumerate(zip(gates_visited, gates_pos)):
            if not gate_visited:
                continue

            waypoint_idx = self._get_waypoint_index_for_gate(gate_idx)
            if waypoint_idx >= len(self._current_waypoints):
                continue

            hard_coded_pos = self._hard_coded_waypoints[waypoint_idx]
            sensed_difference = sensed_pos - hard_coded_pos
            difference_magnitude = self._distance(sensed_pos, hard_coded_pos)

            # Only apply partial adjustment (30-50% of detected offset) to avoid overshoot
            # This keeps us closer to the safe hard-coded waypoint
            if difference_magnitude > 0.1:
                # Apply 40% of the detected offset instead of 100%
                conservative_adjustment = sensed_difference * 0.4
                adjusted_waypoint = hard_coded_pos + conservative_adjustment

                if waypoint_idx not in self._adapted_gate_indices:
                    self._current_waypoints[waypoint_idx] = adjusted_waypoint
                    self._adapted_gate_indices.add(waypoint_idx)
                    waypoints_changed = True

        # LEVEL 2 OPTIMIZATION 2: Aggressive obstacle avoidance
        detected_obstacles = obstacles_pos[obstacles_visited]
        if len(detected_obstacles) > 0:
            for i in range(len(self._current_waypoints)):
                original_wp = self._current_waypoints[i]

                # Check for obstacles and offset if needed
                if self._is_obstacle_near_waypoint(original_wp, detected_obstacles, safety_distance=0.2):
                    offset_wp = self._offset_waypoint_from_obstacles(
                        original_wp, detected_obstacles, safety_distance=0.2, offset_magnitude=0.25
                    )
                    if self._distance(original_wp, offset_wp) > 0.01:
                        self._current_waypoints[i] = offset_wp
                        waypoints_changed = True

        if waypoints_changed:
            self._build_waypoint_trajectories(int(1 / self._dt))

        return waypoints_changed

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The orientation as roll, pitch, yaw angles, and the collective thrust
            [r_des, p_des, y_des, t_des] as a numpy array.
        """
        # Periodically check and update trajectory based on sensor data
        if self._tick - self._last_recompute_tick >= self._recompute_steps:
            self._update_trajectory(obs)
            self._last_recompute_tick = self._tick

        i = min(self._tick, self._tick_max)
        if self._tick >= self._tick_max:
            self._finished = True

        # Setting initial state
        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        obs["drpy"] = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        x0 = np.concatenate((obs["pos"], obs["rpy"], obs["vel"], obs["drpy"]))
        self._acados_ocp_solver.set(0, "lbx", x0)
        self._acados_ocp_solver.set(0, "ubx", x0)

        # Setting state reference
        yref = np.zeros((self._N, self._ny))
        yref[:, 0:3] = self._waypoints_pos[i : i + self._N]  # position
        # zero roll, pitch
        yref[:, 5] = self._waypoints_yaw[i : i + self._N]  # yaw
        yref[:, 6:9] = self._waypoints_vel[i : i + self._N]  # velocity
        # zero drpy

        # Setting input reference (index > self._nx)
        # zero rpy
        # hover thrust
        yref[:, 15] = self.drone_params["mass"] * -self.drone_params["gravity_vec"][-1]
        for j in range(self._N):
            self._acados_ocp_solver.set(j, "yref", yref[j])

        # Setting final state reference
        yref_e = np.zeros((self._ny_e))
        yref_e[0:3] = self._waypoints_pos[i + self._N]  # position
        # zero roll, pitch
        yref_e[5] = self._waypoints_yaw[i + self._N]  # yaw
        yref_e[6:9] = self._waypoints_vel[i + self._N]  # velocity
        # zero drpy
        self._acados_ocp_solver.set(self._N, "y_ref", yref_e)

        # Solving problem and getting first input
        self._acados_ocp_solver.solve()
        u0 = self._acados_ocp_solver.get(0, "u")

        return u0

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment the tick counter."""
        self._tick += 1

        return self._finished

    def episode_callback(self):
        """Reset the internal state."""
        self._tick = 0
        self._current_waypoints = self._hard_coded_waypoints.copy()
        self._build_waypoint_trajectories(int(1 / self._dt))
        self._adapted_gate_indices = set()
        self._last_recompute_tick = 0
        self._detected_gates = {}  # Reset detected gates
