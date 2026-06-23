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

def build_mppi_solver(cost_fn, dynamics_fn):
    """
    FACTORY FUNCTION: Compiles the parallel simulation loop using the 
    injected cost function. Supports both 6D (simple) and 13D (full) states.
    Supports time-decoupled prediction and control horizons.
    
    Args:
        cost_fn: Cost function (state, action, target_pos) -> cost
        dynamics_fn: Dynamics function (state, action, dt, mass, g) -> next_state
        
    Returns:
        solver_step: Function (rng, state, U_nominal, ...) -> (action, U_next)
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
        #target_pos: jnp.ndarray,
        # Optional full dynamics parameters
        inertia_diag: jnp.ndarray | None = None,
        max_thrust_per_motor: float = 0.3,
        dt_pred: float | None = None,
    ):
        # Time-decoupling: dt_pred for prediction dynamics, dt_ctrl for control
        # Default dt_pred = dt_ctrl for backward compatibility
        if dt_pred is None:
            dt_pred = dt_ctrl
        
        K = 10000  # Number of parallel trajectories
        T = U_nominal.shape[0] # Horizon length in prediction steps
        
        noise = random.normal(rng_key, (K, T, 4)) * noise_std
        
        # Action constraints
        lower_bound = jnp.array([-0.5, -0.5, -3.14, 0.0])
        upper_bound = jnp.array([0.5, 0.5, 3.14, mass * g * 2.5])

        # Choose dynamics based on state dimensionality
        state_dim = current_state.shape[0]
        use_full_dynamics = state_dim == 13 and inertia_diag is not None

        def rollout_fn(carry, step_data):
            state = carry
            nominal_action, step_noise = step_data
            
            # Apply noise and strictly clip
            action = nominal_action + step_noise
            action = jnp.clip(action, lower_bound, upper_bound)
            
            # Use appropriate dynamics with dt_pred for prediction
            # if use_full_dynamics:
            #     next_state = quadrotor_dynamics(
            #         state, action, dt_pred, mass, g, inertia_diag, max_thrust_per_motor
            #     )
            # else:
            #     next_state = jax_dynamics_simple(state, action, dt_pred, mass, g)
            next_state = dynamics_fn(state, action, params)
            

            
            # ---> EVALUATE THE INJECTED COST FUNCTION <---
            stage_cost = cost_fn(state, action, next_state,params)
            
            return next_state, stage_cost

        def simulate_single_trajectory(single_noise_sequence):
            final_state, costs = scan(f=rollout_fn, init=current_state, xs=(U_nominal, single_noise_sequence))
            
            # Add terminal cost using the same injected function
            terminal_cost = cost_fn(final_state, U_nominal[-1], final_state,params) * 5.0
            return jnp.sum(costs) + terminal_cost

        # Vectorize the simulation across K trajectories
        batched_simulate = vmap(simulate_single_trajectory, in_axes=(0,))
        costs = batched_simulate(noise)
        
        # Softmax weighting with numerical stability
        beta = jnp.min(costs)
        weights = jnp.exp((-1.0 / lam) * (costs - beta))
        weights = weights / (jnp.sum(weights) + 1e-8)
        
        # Update nominal sequence
        weighted_noise = jnp.sum(weights[:, None, None] * noise, axis=0)
        U_updated = U_nominal + weighted_noise
        U_updated = jnp.clip(U_updated, lower_bound, upper_bound)
        
        optimal_action = U_updated[0]

        min_cost = jnp.min(costs)
        mean_cost = jnp.mean(costs)
        
        # Warm start shift (simplified: always shift by 1 step)
        U_next = jnp.roll(U_updated, shift=-1, axis=0)
        hover_action = jnp.array([0.0, 0.0, 0.0, mass * g])
        U_next = U_next.at[-1].set(hover_action)
        
        return optimal_action, U_next, min_cost, mean_cost

    return solver_step


if __name__ == "__main__":
    import time

    # Print the active hardware backend for debugging
    available_devices = jax.devices()
    backend_type = available_devices[0].platform.upper()
    print(f"Active Backend: {backend_type}")
    print(f"Available Devices: {available_devices}")

    # STEP 1: Test with simplified 6D dynamics
    print("\n" + "="*60)
    print("TEST 1: Simplified 6D Dynamics (Legacy)")
    print("="*60)
    
    mass, g, dt = 1.0, 9.81, 0.05
    initial_state_6d = jnp.zeros(6)
    
    print("--- Physics Sanity Checks ---")
    
    # Test 1A: Perfect Hover
    hover_action = jnp.array([0.0, 0.0, 0.0, mass * g])
    next_state = jax_dynamics_simple(initial_state_6d, hover_action, dt, mass, g)
    print(f"Hover Test -> Z-Velocity should be 0.0: {next_state[5]:.6f}")
    
    # Test 1B: Forward Pitch
    forward_action = jnp.array([0.0, 0.1, 0.0, mass * g])
    next_state = jax_dynamics_simple(initial_state_6d, forward_action, dt, mass, g)
    print(f"Forward Pitch -> X-Velocity should be > 0: {next_state[3]:.6f}")

    # STEP 2: Test with full 13D dynamics
    print("\n" + "="*60)
    print("TEST 2: Full 13D Quadrotor Dynamics")
    print("="*60)
    
    inertia_diag = jnp.array([1.395e-5, 1.436e-5, 2.173e-5])
    max_thrust_per_motor = 0.3
    
    initial_state_13d = quadrotor_state_from_simple_attitude(
        jnp.array([0.0, 0.0, 0.0]),
        jnp.array([0.0, 0.0, 0.0])
    )
    
    print("Full state dimensions:", initial_state_13d.shape)
    print(f"Position: {initial_state_13d[0:3]}")
    print(f"Quaternion: {initial_state_13d[3:7]}")
    print(f"Velocity: {initial_state_13d[7:10]}")
    print(f"Angular velocity: {initial_state_13d[10:13]}")
    
    next_state_13d = quadrotor_dynamics(
        initial_state_13d, hover_action, dt, mass, g, inertia_diag, max_thrust_per_motor
    )
    print(f"\nAfter one step (hover):")
    print(f"Position: {next_state_13d[0:3]}")
    print(f"Velocity: {next_state_13d[7:10]} (should be ~[0, 0, 0])")

    # STEP 3: Test cost functions
    print("\n" + "="*60)
    print("TEST 3: PA-MPPI Cost Functions")
    print("="*60)
    
    target_pos = jnp.array([5.0, 0.0, 0.0])
    dummy_cost = create_dummy_cost(target_pos)
    cost = dummy_cost(initial_state_13d, hover_action, target_pos)
    print(f"Dummy cost at origin: {cost:.4f}")
    
    weights = default_cost_weights("approach")
    print(f"Cost weights (approach mode): {list(weights.keys())}")

    # STEP 4: MPPI Compilation and Execution
    print("\n" + "="*60)
    print("TEST 4: MPPI Solver Compilation")
    print("="*60)
    
    print("Compiling solver with simplified dynamics... (This might take a few seconds)")
    t0 = time.time()
    my_mppi_step = build_mppi_solver(dummy_cost)
    
    # Setup inputs
    rng = random.PRNGKey(42)
    current_state = jnp.zeros(6)
    U_nominal = jnp.zeros((30, 4))
    U_nominal = U_nominal.at[:, 3].set(mass * g)
    target_pos = jnp.array([5.0, 0.0, 0.0])
    noise_std = jnp.array([0.1, 0.1, 0.1, 0.5])
    
    # Trigger JIT compilation with backward-compatible default (dt_pred = dt_ctrl)
    opt_action, next_U = my_mppi_step(
        rng, current_state, U_nominal, dt, mass, g, 0.1, noise_std, target_pos
    )
    
    print(f"Compilation finished in {time.time() - t0:.2f} seconds!")
    print(f"Optimal First Action: Roll={opt_action[0]:.3f}, Pitch={opt_action[1]:.3f}, " +
          f"Yaw={opt_action[2]:.3f}, Thrust={opt_action[3]:.3f}")

    # STEP 5: Closed-loop flight simulation
    print("\n" + "="*60)
    print("TEST 5: Closed-Loop Flight Simulation (15 steps)")
    print("="*60)
    
    sim_state = jnp.zeros(6)
    target = jnp.array([10.0, 5.0, 2.0])
    print(f"Target: {target}\n")
    
    for step in range(15):
        rng, subkey = random.split(rng)
        opt_action, U_nominal = my_mppi_step(
            subkey, sim_state, U_nominal, dt, mass, g, 0.1, noise_std, target
        )
        sim_state = jax_dynamics_simple(sim_state, opt_action, dt, mass, g)
        pos = sim_state[0:3]
        dist = jnp.linalg.norm(pos - target)
        print(f"Step {step:02d} | Pos: [{pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:6.2f}] | Dist: {dist:6.2f}")

    print("\n" + "="*60)
    print("All engine tests completed successfully!")
    print("="*60)

    # STEP 6: Test time-decoupled prediction and control horizons
    print("\n" + "="*60)
    print("TEST 6: Time-Decoupled Prediction/Control Horizons")
    print("="*60)
    
    # Test case: 2x longer horizon with dt_pred = 2*dt_ctrl
    dt_pred = 0.1  # Prediction step = 0.1s
    dt_ctrl = 0.05 # Control step = 0.05s
    horizon_steps = 60  # 60 prediction steps
    
    print(f"dt_pred (prediction) = {dt_pred}s")
    print(f"dt_ctrl (control) = {dt_ctrl}s")
    print(f"Horizon length = {horizon_steps} prediction steps")
    print(f"Prediction horizon = {horizon_steps * dt_pred:.2f}s")
    print(f"Control horizon (effective) = {horizon_steps * dt_pred:.2f}s")
    
    # Create MPPI solver
    my_mppi_decoupled = build_mppi_solver(dummy_cost)
    
    # Setup inputs with longer horizon
    rng = random.PRNGKey(99)
    current_state = jnp.zeros(6)
    U_nominal_long = jnp.zeros((horizon_steps, 4))
    U_nominal_long = U_nominal_long.at[:, 3].set(mass * g)
    target_pos = jnp.array([10.0, 0.0, 0.0])
    
    print("\nRunning MPPI step with time-decoupling...")
    t0 = time.time()
    opt_action, next_U_decoupled = my_mppi_decoupled(
        rng, current_state, U_nominal_long, dt_ctrl, mass, g, 0.1, noise_std, target_pos, 
        dt_pred=dt_pred
    )
    elapsed = time.time() - t0
    
    print(f"Execution time: {elapsed:.2f}s")
    print(f"Next U shape: {next_U_decoupled.shape}")
    print(f"Shift ratio (dt_pred / dt_ctrl) = {dt_pred / dt_ctrl:.1f}x")
    
    # Verify that the decoupling works correctly
    shift_steps = jnp.maximum(1, jnp.round(dt_pred / dt_ctrl).astype(jnp.int32))
    print(f"Expected shift steps: {shift_steps}")
    
    # Test closed-loop with time-decoupling (5 steps)
    print("\nClosed-loop test with time-decoupling (5 steps):")
    sim_state = jnp.zeros(6)
    target = jnp.array([5.0, 0.0, 0.0])
    
    for step in range(5):
        rng, subkey = random.split(rng)
        opt_action, U_nominal_long = my_mppi_decoupled(
            subkey, sim_state, U_nominal_long, dt_ctrl, mass, g, 0.1, noise_std, target,
            dt_pred=dt_pred
        )
        # Simulate actual dynamics with dt_ctrl
        sim_state = jax_dynamics_simple(sim_state, opt_action, dt_ctrl, mass, g)
        pos = sim_state[0:3]
        dist = jnp.linalg.norm(pos - target)
        print(f"Step {step:02d} | Pos: [{pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:6.2f}] | Dist: {dist:6.2f}")
    
    print("\n" + "="*60)
    print("All engine tests completed successfully!")
    print("="*60)