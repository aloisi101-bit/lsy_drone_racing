"""Sandro's highly parallelized JAX MPPI Engine."""
from functools import partial
import jax
import jax.numpy as jnp

class MPPIEngine:
    def __init__(self, num_samples: int, horizon: int, dim_u: int, 
                 lambda_: float, noise_std: jnp.ndarray):
        """Initialize the MPPI engine hyperparameters."""
        self.K = num_samples
        self.H = horizon
        self.dim_u = dim_u
        self.lambda_ = lambda_
        self.noise_std = noise_std
        self.noise_cov = noise_std ** 2

    @partial(jax.jit, static_argnums=(0,), static_argnames=('dynamics_fn', 'cost_fn', 'debug'))
    def optimize(self, state_init, nominal_seq, rng_key, params, dynamics_fn, cost_fn, debug: bool = True):
        """
        Runs one optimization step of the MPPI algorithm.
        JIT-compiled to ensure maximum performance.
        """
        # 1. Generate Gaussian noise for all samples across the horizon
        noise = jax.random.normal(rng_key, shape=(self.K, self.H, self.dim_u)) * self.noise_std
        if debug:
            jax.debug.print("MPPI: K={K}, H={H}, dim_u={dim_u}, lambda={lam}", K=self.K, H=self.H, dim_u=self.dim_u, lam=self.lambda_)
            jax.debug.print("noise stats mean={mean:.6f}, std={std:.6f}, min={min:.6f}, max={max:.6f}",
                            mean=jnp.mean(noise), std=jnp.std(noise), min=jnp.min(noise), max=jnp.max(noise))
        
        # 2. Add noise to the nominal control sequence
        # nominal_seq shape: (H, dim_u) -> perturbed shape: (K, H, dim_u)
        perturbed_action_seqs = nominal_seq + noise

        # Clip sampled actions to safe bounds so MPPI does not explore impossible
        # control values that would produce many out-of-bounds trajectories.
        # We use conservative limits: roll/pitch in [-0.5,0.5], yaw in [-pi,pi],
        # thrust in [0, mass*g*2.5]. The mass/g values are taken from params.
        try:
            max_thrust = params["mass"] * params["g"] * 2.5
        except Exception:
            max_thrust = 10.0
        min_action = jnp.array([-0.5, -0.5, -jnp.pi, 0.0])
        max_action = jnp.array([0.5, 0.5, jnp.pi, max_thrust])
        # Broadcast clipping across K,H dims
        perturbed_action_seqs = jnp.clip(perturbed_action_seqs, min_action, max_action)
        if debug:
            jax.debug.print("perturbed_action_seqs mean={mean:.6f}, std={std:.6f}",
                            mean=jnp.mean(perturbed_action_seqs), std=jnp.std(perturbed_action_seqs))

        # 3. Define the step function for jax.lax.scan
        def scan_fn(current_state, action):
            next_state = dynamics_fn(current_state, action, params)
            step_cost = cost_fn(current_state, action, next_state, params)
            return next_state, step_cost

        # 4. Define the rollout for a SINGLE trajectory
        def single_rollout(action_seq):
            # scan loops over the sequence of actions
            _, step_costs = jax.lax.scan(scan_fn, state_init, action_seq)
            total_cost = jnp.sum(step_costs)
            return total_cost

        # 5. Vectorize the rollout across all K samples
        batch_rollout = jax.vmap(single_rollout)
        trajectory_costs = batch_rollout(perturbed_action_seqs)
        if debug:
            jax.debug.print("trajectory_costs stats min={min:.6f}, max={max:.6f}, mean={mean:.6f}, best_idx={idx}",
                            min=jnp.min(trajectory_costs), max=jnp.max(trajectory_costs),
                            mean=jnp.mean(trajectory_costs), idx=jnp.argmin(trajectory_costs))

        # 6. Calculate MPPI weights
        beta = jnp.min(trajectory_costs)
        # Shift costs for numerical stability before exp
        shifted_costs = trajectory_costs - beta
        
        # Weight formula: exp(-1/lambda * [S(V) + control_penalty])
        # Note: Control penalty is often included directly in the step_cost for simplicity 
        # but standard MPPI adds a noise-dependent term here. We'll keep it simple:
        weights = jnp.exp(-shifted_costs / self.lambda_)
        if debug:
            jax.debug.print("shifted_costs stats min={min:.6f}, max={max:.6f}, mean={mean:.6f}",
                            min=jnp.min(shifted_costs), max=jnp.max(shifted_costs), mean=jnp.mean(shifted_costs))
        weight_sum = jnp.sum(weights)
        weights = weights / (weight_sum + 1e-8)
        if debug:
            jax.debug.print("weights summary sum={sum:.6f}, max={max:.6f}, min={min:.6f}",
                            sum=jnp.sum(weights), max=jnp.max(weights), min=jnp.min(weights))

        # 7. Compute the optimal control sequence via weighted sum
        # weights shape: (K,), noise shape: (K, H, dim_u)
        optimal_noise = jnp.sum(weights[:, None, None] * noise, axis=0)
        optimal_seq = nominal_seq + optimal_noise
        if debug:
            jax.debug.print("optimal_noise mean={mean:.6f}, std={std:.6f}", mean=jnp.mean(optimal_noise), std=jnp.std(optimal_noise))
            # show first action of nominal and optimal sequences for quick comparison
            jax.debug.print("nominal first action {nom}", nom=nominal_seq[0])
            jax.debug.print("optimal first action {opt}", opt=optimal_seq[0])

        return optimal_seq, trajectory_costs