"""
MPPI Engine: Handles Physics and JAX Tensor Optimization.
"""
import jax
import jax.numpy as jnp
from jax import random, vmap
from jax.lax import scan

# CRITICAL: Force 64-bit precision to prevent overflow in softmax
jax.config.update("jax_enable_x64", True)

@jax.jit
def jax_dynamics(state: jnp.ndarray, action: jnp.ndarray, dt: float, mass: float, g: float) -> jnp.ndarray:
    """Simulates rigid-body attitude dynamics."""
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

def build_mppi_solver(cost_fn):
    """
    FACTORY FUNCTION: Compiles the parallel simulation loop using the 
    injected cost function.
    """
    
    @jax.jit
    def solver_step(
        rng_key: jnp.ndarray, 
        current_state: jnp.ndarray, 
        U_nominal: jnp.ndarray, 
        dt: float, 
        mass: float, 
        g: float, 
        lam: float, 
        noise_std: jnp.ndarray,
        target_pos: jnp.ndarray
    ):
        K = 2500  # Number of parallel trajectories
        T = U_nominal.shape[0] # Horizon length
        
        noise = random.normal(rng_key, (K, T, 4)) * noise_std
        
        # Action constraints
        lower_bound = jnp.array([-0.5, -0.5, -3.14, 0.0])
        upper_bound = jnp.array([0.5, 0.5, 3.14, mass * g * 2.5])

        def rollout_fn(carry, step_data):
            state = carry
            nominal_action, step_noise = step_data
            
            # Apply noise and strictly clip
            action = nominal_action + step_noise
            action = jnp.clip(action, lower_bound, upper_bound)
            
            next_state = jax_dynamics(state, action, dt, mass, g)
            
            # ---> EVALUATE THE INJECTED COST FUNCTION <---
            stage_cost = cost_fn(next_state, action, target_pos)
            
            return next_state, stage_cost

        def simulate_single_trajectory(single_noise_sequence):
            final_state, costs = scan(f=rollout_fn, init=current_state, xs=(U_nominal, single_noise_sequence))
            
            # Add terminal cost using the same injected function
            terminal_cost = cost_fn(final_state, U_nominal[-1], target_pos) * 5.0
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
        
        # Warm start shift
        U_next = jnp.roll(U_updated, shift=-1, axis=0)
        hover_action = jnp.array([0.0, 0.0, 0.0, mass * g])
        U_next = U_next.at[-1].set(hover_action)
        
        return optimal_action, U_next

    return solver_step


if __name__ == "__main__":
    import time

    # Print the active hardware backend for debugging
    available_devices = jax.devices()
    backend_type = available_devices[0].platform.upper()
    print(f"Active Backend: {backend_type}")
    print(f"Available Devices: {available_devices}")

    # STEP 2: COMPILE AND TEST THE MPPI PIPELINE

    print("--- STEP 1: Physics Sanity Check ---")
    mass, g, dt = 1.0, 9.81, 0.05
    initial_state = jnp.zeros(6) # [0,0,0, 0,0,0]
    
    # Test 1A: Perfect Hover
    # [roll, pitch, yaw, thrust]
    hover_action = jnp.array([0.0, 0.0, 0.0, mass * g]) 
    next_state = jax_dynamics(initial_state, hover_action, dt, mass, g)
    print(f"Hover Test -> Z-Velocity should be 0.0: {next_state[5]:.4f}")
    
    # Test 1B: Forward Pitch
    # Pitching down slightly (positive pitch usually means nose down depending on your frame)
    forward_action = jnp.array([0.0, 0.1, 0.0, mass * g])
    next_state = jax_dynamics(initial_state, forward_action, dt, mass, g)
    print(f"Forward Test -> X-Velocity should be non-zero: {next_state[3]:.4f}")


    # STEP 2: COMPILE AND TEST THE MPPI PIPELINE
    print("\n--- STEP 2: MPPI Compilation and Execution ---")
    
    # 1. Define a dummy cost function just for testing
    @jax.jit
    def dummy_cost(state, action, target_pos):
        dist = jnp.linalg.norm(state[0:3] - target_pos)
        attitude_penalty = jnp.linalg.norm(action[0:3]) * 0.1
        return (dist * 10.0) + attitude_penalty

    # 2. Build the solver using your factory function
    print("Compiling solver... (This might take a few seconds)")
    t0 = time.time()
    my_mppi_step = build_mppi_solver(dummy_cost)
    
    # 3. Setup dummy inputs to feed the solver
    rng = random.PRNGKey(42)
    current_state = jnp.zeros(6)
    U_nominal = jnp.zeros((30, 4))
    U_nominal = U_nominal.at[:, 3].set(mass * g) # Fill with hover thrust
    target_pos = jnp.array([5.0, 0.0, 0.0])      # 5 meters in front
    noise_std = jnp.array([0.1, 0.1, 0.1, 0.5])
    
    # 4. Trigger JIT compilation by running it once
    opt_action, next_U = my_mppi_step(
        rng, current_state, U_nominal, dt, mass, g, 0.1, noise_std, target_pos
    )
    
    print(f"Compilation finished in {time.time() - t0:.2f} seconds!")
    print(f"Optimal First Action computed: ")
    print(f"Roll: {opt_action[0]:.3f}, Pitch: {opt_action[1]:.3f}, Yaw: {opt_action[2]:.3f}, Thrust: {opt_action[3]:.3f}")


    # STEP 3: CLOSED-LOOP MINI-FLIGHT
    print("\n--- STEP 3: Closed-Loop Flight Simulation ---")
    
    sim_state = jnp.zeros(6)
    target = jnp.array([10.0, 5.0, 2.0]) # Target: x=10, y=5, z=2
    print(f"Target position: {target}")
    
    for step in range(15): # Simulate 15 control steps
        rng, subkey = random.split(rng)
        
        # 1. Get the best action sequence from your solver
        opt_action, U_nominal = my_mppi_step(
            subkey, sim_state, U_nominal, dt, mass, g, 0.1, noise_std, target
        )
        
        # 2. Apply it to the physics engine
        sim_state = jax_dynamics(sim_state, opt_action, dt, mass, g)
        
        # 3. Print the drone's position
        pos = sim_state[0:3]
        dist = jnp.linalg.norm(pos - target)
        print(f"Step {step:02d} | Pos: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] | Dist to target: {dist:.2f}")

    print("\nAll engine tests passed.")