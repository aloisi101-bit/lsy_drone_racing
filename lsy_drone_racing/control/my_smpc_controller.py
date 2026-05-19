"""
*** IMPORTANT NOTE:***
Static vs. Dynamic Arrays in JAX: JAX relies heavily on static array sizes to compile efficiently. When you pass obs["obstacles_pos"] to your JIT-compiled functions in Level 2, the number of detected obstacles might change from step to step (e.g., 0 obstacles seen, then 2 seen). JAX will trigger a slow recompilation every time the array size changes. You must pad your obstacle array to a fixed maximum size (e.g., shape (5, 3)) and use a dummy position (like [999.0, 999.0, 999.0]) for empty slots.

Action Space Mapping: You noted # Output exactly what the 'attitude' control mode expects. Double-check lsy_drone_racing/envs/drone_race.py. The attitude control mode in drone models usually maps inputs as [thrust, roll, pitch, yaw] or [roll, pitch, yaw, thrust]. Make sure your jax_dynamics unwraps the array in the exact same order your controller returns it to the environment!
"""


"""
Main Controller Logic: Track parsing, Target logic, and Cost Function definition.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from jax import random
from jax import random
from typing import TYPE_CHECKING

from lsy_drone_racing.control.controller import Controller
from drone_models.core import load_params

# FIXED: Absolute import to bypass importlib dynamic loading issues
from lsy_drone_racing.control.mppi_engine import build_mppi_solver

# FIXED: Absolute import to bypass importlib dynamic loading issues
from lsy_drone_racing.control.mppi_engine import build_mppi_solver

if TYPE_CHECKING:
    from numpy.typing import NDArray

class MPPIControllerJAX(Controller):
    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        
        # 1. Physics Data
        # 1. Physics Data
        self.drone_params = load_params("so_rpy", config.sim.drone_model)
        self.mass = float(self.drone_params["mass"])
        self.g = float(abs(self.drone_params["gravity_vec"][2]))
        self.dt = 1.0 / config.env.freq
        
        # 2. MPPI Config
        self.T = 30  
        # 2. MPPI Config
        self.T = 30  
        self.lam = 0.1
        self.noise_std = jnp.array([0.1, 0.1, 0.1, 0.5])
        
        # 3. Track Data 
        # 3. Track Data 
        self.gates = jnp.array([g["pos"] for g in config.env.track.gates], dtype=jnp.float64)
        self.gate_tolerance = 0.4
        
        # 4. INJECT COST FUNCTION
        self.mppi_step = build_mppi_solver(self.evaluate_trajectory_cost)
        
        # 5. Initialize dynamic states (target_idx, U_nominal, rng_key)
        # 4. INJECT COST FUNCTION
        self.mppi_step = build_mppi_solver(self.evaluate_trajectory_cost)
        
        # 5. Initialize dynamic states (target_idx, U_nominal, rng_key)
        self.episode_reset()

    # ========================================================
    # THE COST FUNCTION (Sandro's Workspace)
    # ========================================================
    @staticmethod
    def evaluate_trajectory_cost(state: jnp.ndarray, action: jnp.ndarray, target_pos: jnp.ndarray) -> float:
        """Evaluates how good or bad a specific predicted state is."""
        pos = state[0:3]
        
        # Reward: Distance to target gate
        dist_to_gate = jnp.linalg.norm(pos - target_pos)
        
        # Penalty: Aggressive attitude flying
        attitude_penalty = jnp.linalg.norm(action[0:3]) 
        
        return (dist_to_gate * 15.0) + (attitude_penalty * 0.05)
    # ========================================================

    # ========================================================
    # THE COST FUNCTION (Sandro's Workspace)
    # ========================================================
    @staticmethod
    def evaluate_trajectory_cost(state: jnp.ndarray, action: jnp.ndarray, target_pos: jnp.ndarray) -> float:
        """Evaluates how good or bad a specific predicted state is."""
        pos = state[0:3]
        
        # Reward: Distance to target gate
        dist_to_gate = jnp.linalg.norm(pos - target_pos)
        
        # Penalty: Aggressive attitude flying
        attitude_penalty = jnp.linalg.norm(action[0:3]) 
        
        return (dist_to_gate * 15.0) + (attitude_penalty * 0.05)
    # ========================================================

    def compute_control(self, obs: dict[str, NDArray[np.floating]], info: dict | None = None) -> NDArray[np.floating]:
        # CRITICAL: Cast to float64 immediately
        # CRITICAL: Cast to float64 immediately
        current_pos = jnp.array(obs["pos"], dtype=jnp.float64)
        current_vel = jnp.array(obs["vel"], dtype=jnp.float64)
        current_state = jnp.concatenate([current_pos, current_vel])
        
        # Update Target Gate Progression
        # Update Target Gate Progression
        if jnp.linalg.norm(current_pos - self.gates[self.target_idx]) < self.gate_tolerance:
            if self.target_idx < len(self.gates) - 1:
                self.target_idx += 1
                
                
        target_pos = self.gates[self.target_idx]
        
        # Advance RNG
        # Advance RNG
        self.rng_key, subkey = random.split(self.rng_key)
        
        # Call the JIT-Compiled Engine
        optimal_action, self.U_nominal = self.mppi_step(
        # Call the JIT-Compiled Engine
        optimal_action, self.U_nominal = self.mppi_step(
            subkey, 
            current_state, 
            self.U_nominal, 
            self.dt, 
            self.mass, 
            self.g, 
            self.lam, 
            self.noise_std,
            target_pos
            self.noise_std,
            target_pos
        )
        
        # Cast back to float32 for Simulator
        action = np.array(optimal_action, dtype=np.float32)
        # Cast back to float32 for Simulator
        action = np.array(optimal_action, dtype=np.float32)
        
        # Failsafe: Add 10% extra thrust to hover so it doesn't instantly hit the floor if NaNs occur
        if np.isnan(action).any():
            action = np.array([0.0, 0.0, 0.0, self.mass * self.g * 1.1], dtype=np.float32)
            
        return action
        # Failsafe: Add 10% extra thrust to hover so it doesn't instantly hit the floor if NaNs occur
        if np.isnan(action).any():
            action = np.array([0.0, 0.0, 0.0, self.mass * self.g * 1.1], dtype=np.float32)
            
        return action

    def episode_reset(self):
        """CRITICAL: Called by __init__ and between runs by the simulator."""
        """CRITICAL: Called by __init__ and between runs by the simulator."""
        self.target_idx = 0
        self.rng_key = random.PRNGKey(42)
        
        hover_seq = jnp.zeros((self.T, 4), dtype=jnp.float64)
        hover_seq = jnp.zeros((self.T, 4), dtype=jnp.float64)
        hover_seq = hover_seq.at[:, 3].set(self.mass * self.g)
        self.U_nominal = hover_seq