"""MPPI engine: JAX-accelerated Model Predictive Path Integral optimization.

Builds a JIT-compiled solver step that samples many noisy action sequences, rolls each
one through an injected dynamics model, scores them with an injected cost function, and
returns the softmax-weighted optimal action sequence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import random, vmap
from jax.lax import scan

if TYPE_CHECKING:
    from collections.abc import Callable

# Force 64-bit precision so the softmax over trajectory costs does not overflow.
jax.config.update("jax_enable_x64", True)

NUM_SAMPLES = 20000  # parallel noisy action sequences sampled per solver step
TERMINAL_COST_WEIGHT = 5.0  # extra weight on the final-state cost of each rollout


def build_mppi_solver(cost_fn: Callable, dynamics_fn: Callable) -> Callable:
    """Build a JIT-compiled MPPI solver step for the given cost and dynamics.

    Args:
        cost_fn: Stage cost ``(state, action, next_state, params) -> cost``.
        dynamics_fn: Discrete dynamics ``(state, action, params) -> next_state``.

    Returns:
        A function ``(rng_key, state, nominal_actions, temperature, noise_std, params)``
        that returns ``(optimal_action, next_nominal_actions, min_cost, mean_cost)``.
    """

    @jax.jit
    def solver_step(
        rng_key: jax.Array,
        current_state: jax.Array,
        nominal_actions: jax.Array,
        temperature: float,
        noise_std: jax.Array,
        params: dict,
    ) -> tuple:
        horizon = nominal_actions.shape[0]
        mass, g = params["mass"], params["g"]
        # Action bounds [roll, pitch, yaw, thrust]; yaw is an angle in the sim's [-pi/2, pi/2].
        lower_bound = jnp.array([-0.5, -0.5, -jnp.pi / 2, 0.0])
        upper_bound = jnp.array([0.5, 0.5, jnp.pi / 2, mass * g * 2.5])

        noise = random.normal(rng_key, (NUM_SAMPLES, horizon, 4)) * noise_std

        def rollout(state: jax.Array, step: tuple) -> tuple:
            nominal_action, step_noise = step
            action = jnp.clip(nominal_action + step_noise, lower_bound, upper_bound)
            next_state = dynamics_fn(state, action, params)
            return next_state, cost_fn(state, action, next_state, params)

        def trajectory_cost(noise_sequence: jax.Array) -> jax.Array:
            final_state, stage_costs = scan(
                rollout, current_state, (nominal_actions, noise_sequence)
            )
            terminal_cost = cost_fn(final_state, nominal_actions[-1], final_state, params)
            return jnp.sum(stage_costs) + TERMINAL_COST_WEIGHT * terminal_cost

        costs = vmap(trajectory_cost)(noise)

        # Softmax weighting, shifted by the minimum cost for numerical stability.
        min_cost = jnp.min(costs)
        weights = jnp.exp(-(costs - min_cost) / temperature)
        weights = weights / (jnp.sum(weights) + 1e-8)

        updated_actions = nominal_actions + jnp.sum(weights[:, None, None] * noise, axis=0)
        updated_actions = jnp.clip(updated_actions, lower_bound, upper_bound)
        optimal_action = updated_actions[0]

        # Warm-start the next step: shift the sequence forward, append a hover action.
        next_actions = jnp.roll(updated_actions, shift=-1, axis=0)
        next_actions = next_actions.at[-1].set(jnp.array([0.0, 0.0, 0.0, mass * g]))

        return optimal_action, next_actions, min_cost, jnp.mean(costs)

    return solver_step
