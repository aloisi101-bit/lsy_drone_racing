"""
JAX-accelerated MPPI Controller for Level 0
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit, vmap
from jax.lax import scan
from typing import TYPE_CHECKING
from scipy.spatial.transform import Rotation as R
jax.config.update("jax_enable_x64", True) # <--- ADD THIS LINE

from lsy_drone_racing.control.controller import Controller
from drone_models.core import load_params

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray

# ==========================================
# 1. JIT-Compiled Core Math & Physics
# ==========================================

@jax.jit
def jax_dynamics(state: jnp.ndarray, action: jnp.ndarray, dt: float, mass: float, g: float) -> jnp.ndarray:
    """
    Simulates the rigid-body attitude dynamics.
    state: [x, y, z, vx, vy, vz]
    action: [roll, pitch, yaw, thrust]
    """
    p = state[0:3]
    v = state[3:6]
    
    roll, pitch, yaw, thrust = action[0], action[1], action[2], action[3]
    
    # Calculate body Z-axis in world frame (Z-Y-X Euler convention)
    sr, cr = jnp.sin(roll), jnp.cos(roll)
    sp, cp = jnp.sin(pitch), jnp.cos(pitch)
    sy, cy = jnp.sin(yaw), jnp.cos(yaw)
    
    z_b = jnp.array([
        sr * sy + cr * cy * sp,
        cr * sp * sy - cy * sr,
        cr * cp
    ])
    
    # Linear acceleration
    acc = (thrust / mass) * z_b - jnp.array([0.0, 0.0, g])
    
    next_v = v + acc * dt
    next_p = p + v * dt + 0.5 * acc * (dt ** 2)
    
    return jnp.concatenate([next_p, next_v])

@jax.jit
def _mppi_rollout(
    rng_key: jnp.ndarray, 
    current_state: jnp.ndarray, 
    U_nominal: jnp.ndarray, 
    target_pos: jnp.ndarray,
    dt: float, 
    mass: float, 
    g: float, 
    lam: float, 
    noise_std: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    The core MPPI optimization loop.
    """
    K = 2500  # Number of samples
    T = U_nominal.shape[0]
    
    # Generate batched noise: (K, T, 4)
    noise = random.normal(rng_key, (K, T, 4)) * noise_std
    
    def rollout_fn(carry, step_data):
        state = carry
        nominal_action, step_noise = step_data
        
        # Apply noise and clamp limits: [roll, pitch, yaw, thrust]
        action = nominal_action + step_noise
        
        # Optional: You can enforce strict constraints here similar to acados
        # action = jnp.clip(action, jnp.array([-0.5, -0.5, -0.5, 0.0]), jnp.array([0.5, 0.5, 0.5, mass * g * 2.5]))
        
        next_state = jax_dynamics(state, action, dt, mass, g)
        
        # Evaluate Stage Cost
        dist_to_gate = jnp.linalg.norm(next_state[0:3] - target_pos)
        attitude_penalty = jnp.linalg.norm(action[0:3]) * 0.01
        stage_cost = (dist_to_gate * 10.0) + attitude_penalty
        
        return next_state, stage_cost

    def simulate_single_trajectory(single_noise_sequence):
        # scan replaces the Python 'for' loop over the horizon T
        final_state, costs = scan(
            f=rollout_fn, 
            init=current_state, 
            xs=(U_nominal, single_noise_sequence)
        )
        # Add Terminal Cost
        terminal_cost = jnp.linalg.norm(final_state[0:3] - target_pos) * 100.0
        return jnp.sum(costs) + terminal_cost

    # Vectorize the trajectory simulation over the K dimension
    batched_simulate = vmap(simulate_single_trajectory, in_axes=(0,))
    costs = batched_simulate(noise)
    
    # Softmax weighting
    beta = jnp.min(costs)
    weights = jnp.exp((-1.0 / lam) * (costs - beta))
    weights = weights / jnp.sum(weights)
    
    # Compute the weighted average to update the nominal control sequence
    # weights: (K,), noise: (K, T, 4) -> sum over K yields (T, 4)
    weighted_noise = jnp.sum(weights[:, None, None] * noise, axis=0)
    U_updated = U_nominal + weighted_noise
    
    # Extract optimal first action and shift sequence (Warm Start)
    optimal_action = U_updated[0]
    U_next = jnp.roll(U_updated, shift=-1, axis=0)
    
    # Pad the final step with hover thrust [0, 0, 0, hover_thrust]
    hover_action = jnp.array([0.0, 0.0, 0.0, mass * g])
    U_next = U_next.at[-1].set(hover_action)
    
    return optimal_action, U_next


# ==========================================
# 2. Controller Class Integration
# ==========================================

class MPPIControllerJAX(Controller):
    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        
        # Load physical drone parameters
        self.drone_params = load_params("so_rpy", config.sim.drone_model)
        self.mass = float(self.drone_params["mass"])
        self.g = float(abs(self.drone_params["gravity_vec"][2]))
        self.dt = 1.0 / config.env.freq
        
        # MPPI Configuration
        self.T = 30  # Horizon
        self.lam = 0.1
        # Noise profile: [roll, pitch, yaw, thrust]
        self.noise_std = jnp.array([0.1, 0.1, 0.1, 0.5])
        
        # JAX Random Number Generator state
        self.rng_key = random.PRNGKey(42)
        
        # Parse Level 0 track gates
        self.gates = jnp.array([g["pos"] for g in config.env.track.gates], dtype=jnp.float64)
        self.gate_tolerance = 0.4
        
        self.episode_reset()

    def compute_control(self, obs: dict[str, NDArray[np.floating]], info: dict | None = None) -> NDArray[np.floating]:
        # 1. Construct State Array
        current_pos = jnp.array(obs["pos"], dtype=jnp.float64)
        current_vel = jnp.array(obs["vel"], dtype=jnp.float64)
        current_state = jnp.concatenate([current_pos, current_vel])
        
        # 2. Track Progress
        if jnp.linalg.norm(current_pos - self.gates[self.target_idx]) < self.gate_tolerance:
            if self.target_idx < len(self.gates) - 1:
                self.target_idx += 1
        
        target_pos = self.gates[self.target_idx]
        
        # 3. Advance JAX RNG
        self.rng_key, subkey = random.split(self.rng_key)
        
        # 4. Execute JIT-compiled MPPI
        optimal_action, self.U_nominal = _mppi_rollout(
            subkey, 
            current_state, 
            self.U_nominal, 
            target_pos,
            self.dt, 
            self.mass, 
            self.g, 
            self.lam, 
            self.noise_std
        )
        
        # Output exactly what the 'attitude' control mode expects:
        # [thrust, roll, pitch, yaw] -> Note the swapped order required by the environment
        r, p, y, t = np.array(optimal_action)
        
        # Normalize thrust based on environment constraints (mass * gravity approx 1.0 depending on scaling)
        thrust_normalized = np.clip((t / self.mass) / 25.0, 0.0, 1.0) 
        
        return np.array([thrust_normalized, r, p, y])

    def episode_reset(self):
        """Reset planner state between simulation runs."""
        self.target_idx = 0
        self.rng_key = random.PRNGKey(42)
        
        # Initialize the nominal control sequence with hover commands
        hover_seq = jnp.zeros((self.T, 4))
        hover_seq = hover_seq.at[:, 3].set(self.mass * self.g)
        self.U_nominal = hover_seq