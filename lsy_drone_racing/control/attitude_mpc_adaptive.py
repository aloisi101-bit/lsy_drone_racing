"""Adaptive MPC controller with reactive trajectory replanning for gate detection.

This module extends the standard attitude MPC to handle level 2 complexity by:
1. Detecting gates/obstacles as they enter sensor range
2. Regenerating the reference trajectory through discovered gates
3. Updating MPC waypoints reactively (only when gates change state)

The controller maintains backward compatibility: if no gates are discovered,
it falls back to nominal waypoints (same as original AttitudeMPC).
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
    ocp.solver_options.nlp_solver_type = "SQP"  # SQP, SQP_RTI
    ocp.solver_options.tol = 1e-6

    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_warm_start = 1

    ocp.solver_options.qp_solver_iter_max = 20
    ocp.solver_options.nlp_solver_max_iter = 50

    # set prediction horizon
    ocp.solver_options.tf = Tf

    acados_ocp_solver = AcadosOcpSolver(
        ocp,
        json_file="c_generated_code/lsy_example_mpc_adaptive.json",
        verbose=verbose,
        build=True,
        generate=True,
    )

    return acados_ocp_solver, ocp


class AttitudeMPCAdaptive(Controller):
    """Adaptive MPC with reactive trajectory replanning for gate-based navigation.
    
    This controller detects when gates/obstacles enter sensor range and regenerates
    the reference trajectory to pass through discovered gates in order. The MPC
    solver continuously tracks this adaptive trajectory.
    """

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the adaptive attitude controller.

        Args:
            obs: The initial observation of the environment's state.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
        """
        super().__init__(obs, info, config)
        self._N = 25
        self._dt = 1 / config.env.freq
        self._T_HORIZON = self._N * self._dt

        # Nominal waypoints (fallback if gates not detected)
        self._nominal_waypoints = np.array(
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
        
        # Initialize with nominal trajectory
        t = np.linspace(0, self._t_total, len(self._nominal_waypoints))
        self._des_pos_spline = CubicSpline(t, self._nominal_waypoints)
        self._des_vel_spline = self._des_pos_spline.derivative()
        self._waypoints_pos = self._des_pos_spline(
            np.linspace(0, self._t_total, int(config.env.freq * self._t_total))
        )
        self._waypoints_vel = self._des_vel_spline(
            np.linspace(0, self._t_total, int(config.env.freq * self._t_total))
        )
        self._waypoints_yaw = self._waypoints_pos[:, 0] * 0

        # State tracking for adaptive replanning
        self._prev_gates_visited = np.zeros(obs["gates_visited"].shape, dtype=bool)
        self._discovered_gates_indices = []  # Ordered list of detected gate indices
        self._replanning_triggered = False
        self._current_trajectory_index = 0  # Track position in current trajectory

        # Obstacle tracking for soft constraint tuning
        self._detected_obstacles = {}  # Map: obstacle_idx -> (visited, position)
        n_obstacles = obs["obstacles_pos"].shape[0]
        for idx in range(n_obstacles):
            self._detected_obstacles[idx] = {
                "visited": False,
                "pos": obs["obstacles_pos"][idx].copy(),
            }

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

    def _check_replanning_needed(
        self, gates_visited: NDArray[np.floating]
    ) -> tuple[bool, list[int]]:
        """Check if any gates transitioned from unvisited to visited.
        
        Args:
            gates_visited: Current gates_visited array from observation.
            
        Returns:
            Tuple of (replanning_needed, newly_discovered_indices)
        """
        gates_changed = gates_visited != self._prev_gates_visited
        newly_discovered = np.where(gates_changed & gates_visited)[0]
        
        replanning_needed = len(newly_discovered) > 0
        newly_discovered_list = newly_discovered.tolist() if replanning_needed else []
        
        return replanning_needed, newly_discovered_list

    def _generate_adaptive_trajectory(
        self,
        drone_pos: NDArray[np.floating],
        gates_pos: NDArray[np.floating],
        discovered_gate_indices: list[int],
        obstacle_positions: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        """Regenerate trajectory through discovered gates.
        
        Args:
            drone_pos: Current drone position [x, y, z].
            gates_pos: All gate positions (n_gates, 3); discovered gates have true positions.
            discovered_gate_indices: Ordered list of gate indices that have been discovered.
            obstacle_positions: All obstacle positions (n_obstacles, 3).
            
        Returns:
            Tuple of (waypoints_pos, waypoints_vel, waypoints_yaw) for new trajectory.
        """
        # Build waypoint list: start -> discovered gates (in order) -> end
        waypoint_list = [drone_pos.copy()]
        
        # Add gate centers in discovered order
        for gate_idx in discovered_gate_indices:
            gate_pos = gates_pos[gate_idx].copy()
            waypoint_list.append(gate_pos)
        
        # Add final nominal waypoint if not already there
        final_nominal = self._nominal_waypoints[-1].copy()
        if len(waypoint_list) < 4:  # Need at least a few waypoints for spline
            waypoint_list.append(final_nominal)
        
        waypoint_array = np.array(waypoint_list)
        
        # Create new spline through discovered gates
        n_waypoints = len(waypoint_array)
        # Distribute time based on distances between waypoints
        distances = np.linalg.norm(np.diff(waypoint_array, axis=0), axis=1)
        cumulative_dist = np.concatenate(([0], np.cumsum(distances)))
        
        # Scale to fit within available time, or use fixed time for now
        if cumulative_dist[-1] > 0:
            t_new = (cumulative_dist / cumulative_dist[-1]) * self._t_total
        else:
            t_new = np.linspace(0, self._t_total, n_waypoints)
        
        # Create cubic spline through new waypoints
        try:
            des_pos_spline = CubicSpline(t_new, waypoint_array)
            des_vel_spline = des_pos_spline.derivative()
            
            # Sample trajectory at control frequency
            t_samples = np.linspace(0, self._t_total, int(self._config.env.freq * self._t_total))
            waypoints_pos = des_pos_spline(t_samples)
            waypoints_vel = des_vel_spline(t_samples)
            
            # Yaw remains zero (pointing forward)
            waypoints_yaw = np.zeros(len(t_samples))
            
            return waypoints_pos, waypoints_vel, waypoints_yaw
            
        except Exception as e:
            # Fallback to nominal if spline fails
            print(f"Warning: Adaptive trajectory generation failed ({e}), using nominal")
            return self._waypoints_pos, self._waypoints_vel, self._waypoints_yaw

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: The current observation of the environment.
            info: Optional additional information as a dictionary.

        Returns:
            The orientation as roll, pitch, yaw angles, and the collective thrust
            [r_des, p_des, y_des, t_des] as a numpy array.
        """
        # ===== PHASE 1: DETECT GATE CHANGES AND TRIGGER REPLANNING =====
        replanning_needed, newly_discovered = self._check_replanning_needed(
            obs["gates_visited"]
        )
        
        if replanning_needed:
            # Add newly discovered gates to ordered list
            for gate_idx in newly_discovered:
                if gate_idx not in self._discovered_gates_indices:
                    self._discovered_gates_indices.append(gate_idx)
            
            # Regenerate trajectory through discovered gates
            new_pos, new_vel, new_yaw = self._generate_adaptive_trajectory(
                obs["pos"],
                obs["gates_pos"],
                self._discovered_gates_indices,
                obs["obstacles_pos"],
            )
            
            # Update internal trajectory
            self._waypoints_pos = new_pos
            self._waypoints_vel = new_vel
            self._waypoints_yaw = new_yaw
            self._tick_max = len(self._waypoints_pos) - 1 - self._N
            self._current_trajectory_index = 0  # Reset to start of new trajectory
            self._replanning_triggered = True
        
        # Update visited gates for next iteration
        self._prev_gates_visited = obs["gates_visited"].copy()
        
        # ===== PHASE 2: STANDARD MPC TRACKING =====
        i = min(self._current_trajectory_index, self._tick_max)
        if self._current_trajectory_index >= self._tick_max:
            self._finished = True

        # Setting initial state
        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        obs["drpy"] = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        x0 = np.concatenate((obs["pos"], obs["rpy"], obs["vel"], obs["drpy"]))
        self._acados_ocp_solver.set(0, "lbx", x0)
        self._acados_ocp_solver.set(0, "ubx", x0)

        # Setting state reference from adaptive trajectory
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
        """Increment the trajectory index."""
        self._current_trajectory_index += 1

        return self._finished

    def episode_callback(self):
        """Reset controller state for new episode."""
        self._tick = 0
        self._current_trajectory_index = 0
        self._discovered_gates_indices = []
        self._prev_gates_visited = np.zeros(self._prev_gates_visited.shape, dtype=bool)
        self._replanning_triggered = False
