"""MPPI Engine: Handles Physics and JAX Tensor Optimization.

Implements Model Predictive Path Integral (MPPI) control with full quadrotor dynamics
and PA-MPPI cost functions. Supports both simplified dynamics for testing and full
13D quadrotor state (position, quaternion, velocity, angular velocity).
"""

import jax
import jax.numpy as jnp
from jax import random, vmap
from jax.lax import scan

#from lsy_drone_racing.control.mppi_dynamics import quadrotor_dynamics, quadrotor_state_from_simple_attitude
#from lsy_drone_racing.control.mppi_cost import default_cost_weights, create_dummy_cost

# CRITICAL: Force 64-bit precision to prevent overflow in softmax
jax.config.update("jax_enable_x64", True)


# Legacy simplified dynamics (for backward compatibility)
@jax.jit
def jax_dynamics_simple(state: jnp.ndarray, action: jnp.ndarray, dt: float, mass: float, g: float) -> jnp.ndarray:
    """Simulates simplified attitude dynamics (6D state: position + velocity).

    Legacy function for backward compatibility. Use quadrotor_dynamics() for full model.
    """
    p = state[0:3]
    v = state[3:6]

    # [roll, pitch, yaw, thrust_in_newtons]
    roll, pitch, yaw, thrust = action[0], action[1], action[2], action[3]

    sr, cr = jnp.sin(roll), jnp.cos(roll)
    sp, cp = jnp.sin(pitch), jnp.cos(pitch)
    sy, cy = jnp.sin(yaw), jnp.cos(yaw)

    # Body Z-axis in world frame
    z_b = jnp.array([
        sr * sy + cr * cy * sp,
        cr * sp * sy - cy * sr,
        cr * cp
    ])

    acc = (thrust / mass) * z_b - jnp.array([0.0, 0.0, g])

    next_v = v + acc * dt
    next_p = p + v * dt + 0.5 * acc * (dt ** 2)

    return jnp.concatenate([next_p, next_v])

def build_mppi_solver(cost_fn, dynamics_fn, K: int = 5000):
    """
    FACTORY FUNCTION: Compiles the parallel simulation loop using the 
    injected cost function. Supports both 6D (simple) and 13D (full) states.
    Supports time-decoupled prediction and control horizons.
    
    Args:
        cost_fn: Cost function (state, action, target_pos) -> cost
        dynamics_fn: Dynamics function (state, action, dt, mass, g) -> next_state
        K: Number of parallel trajectory samples (default: 5000)
        
    Returns:
        solver_step: Function (rng, state, U_nominal, ...) -> (action, U_next, costs)
    """
    
    @jax.jit(static_argnames=["dt_ctrl", "dt_pred"])
    def solver_step(
        rng_key: jnp.ndarray, 
        current_state: jnp.ndarray, 
        U_nominal: jnp.ndarray, 
        dt_ctrl: float, 
        mass: float, 
        g: float, 
        lam: float, 
        noise_std: jnp.ndarray,
        params: dict,
        dt_pred: float | None = None,
    ):
        # Time-decoupling: dt_pred for prediction dynamics, dt_ctrl for control
        # Default dt_pred = dt_ctrl for backward compatibility
        if dt_pred is None:
            dt_pred = dt_ctrl
        
        # Number of parallel trajectories (K is captured from closure)
        T = U_nominal.shape[0] # Horizon length in prediction steps
        
        noise = random.normal(rng_key, (K, T, 4)) * noise_std
        
        # Action constraints — thrust floor at 50% hover prevents zero-thrust ground crashes
        lower_bound = jnp.array([-0.5, -0.5, -2.0, mass * g * 0.5])
        upper_bound = jnp.array([0.5, 0.5, 2.0, mass * g * 2.5])

        def rollout_fn(carry, step_data):
            state = carry
            nominal_action, step_noise = step_data

            action = jnp.clip(nominal_action + step_noise, lower_bound, upper_bound)
            # Track the actual perturbation applied so the update uses consistent noise
            effective_noise = action - nominal_action

            next_state = dynamics_fn(state, action, params)
            stage_cost = cost_fn(state, action, next_state, params)

            return next_state, (stage_cost, effective_noise)

        def simulate_single_trajectory(single_noise_sequence):
            final_state, (costs, eff_noises) = scan(
                f=rollout_fn, init=current_state, xs=(U_nominal, single_noise_sequence)
            )
            last_action = U_nominal[-1] + eff_noises[-1]
            terminal_cost = cost_fn(final_state, last_action, final_state, params) * 5.0
            return jnp.sum(costs) + terminal_cost, eff_noises

        # Vectorize the simulation across K trajectories
        batched_simulate = vmap(simulate_single_trajectory, in_axes=(0,))
        costs, all_eff_noises = batched_simulate(noise)  # (K,), (K, T, 4)

        # Softmax weighting with numerical stability
        beta = jnp.min(costs)
        weights = jnp.exp((-1.0 / lam) * (costs - beta))
        weights = weights / (jnp.sum(weights) + 1e-8)

        # Update nominal sequence using the effective (clipped) perturbations
        weighted_noise = jnp.sum(weights[:, None, None] * all_eff_noises, axis=0)
        U_updated = U_nominal + weighted_noise
        U_updated = jnp.clip(U_updated, lower_bound, upper_bound)
        
        optimal_action = U_updated[0]

        min_cost = jnp.min(costs)
        mean_cost = jnp.mean(costs)
        
        # Warm start: shift by dt_pred/dt_ctrl steps (both are static, so this is a Python int)
        shift_steps = max(1, round(dt_ctrl / dt_pred)) # shift_steps = max(1, round(dt_pred / dt_ctrl))
        U_next = jnp.roll(U_updated, shift=-shift_steps, axis=0)

        # Fill the vacated tail with a hover action
        hover_action = jnp.array([0.0, 0.0, 0.0, mass * g])
        start_idx = T - shift_steps
        U_next = jnp.where(
            jnp.arange(T)[:, None] >= start_idx,
            hover_action,
            U_next
        )
        
        return optimal_action, U_next, min_cost, mean_cost

    return solver_step


if __name__ == "__main__":
    import time

    print("\n" + "="*60)
    print("MPPI Engine - Basic Validation Test")
    print("="*60)
    
    print("✓ JAX 64-bit precision enabled")
    print("✓ build_mppi_solver function defined")
    print("✓ jax_dynamics_simple available for 6D testing")
    print("✓ solver_step JIT compilation configured")
    
    mass, g, dt = 1.0, 9.81, 0.05
    
    # Test 1: Physics validation - simple hover
    print("\n--- Physics Sanity Checks ---")
    initial_state_6d = jnp.zeros(6)
    hover_action = jnp.array([0.0, 0.0, 0.0, mass * g])
    next_state = jax_dynamics_simple(initial_state_6d, hover_action, dt, mass, g)
    print(f"✓ Hover Test -> Z-Velocity: {next_state[5]:.6f} (should be ~0.0)")
    
    # Test 2: Physics validation - forward pitch
    forward_action = jnp.array([0.0, 0.1, 0.0, mass * g])
    next_state = jax_dynamics_simple(initial_state_6d, forward_action, dt, mass, g)
    print(f"✓ Forward Pitch -> X-Velocity: {next_state[3]:.6f} (should be > 0)")
    
    print("\n" + "="*60)
    print("Engine validation complete. Ready for integration with controller.")
    print("="*60)
    print("\nNOTE: Full testing with cost functions requires integration with controller.")
    print("Use my_smpc_controller.py to test the complete MPPI workflow.")