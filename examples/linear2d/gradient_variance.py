from functools import partial
import numpy as np
import pandas as pd

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import random
from flax.linen.initializers import constant
import matplotlib.pyplot as plt

from ppomdp.smc import smc, backward_tracing
from ppomdp.policy.arch import LinearGaussDecoder
from ppomdp.policy.linear import create_linear_policy
from ppomdp.smc.utils import systematic_resampling
from experiments.common import get_pomdp


def accumulate_log_prob(params, policy, actions, particles, weights):
    """Compute the negative log-likelihood loss for a single trajectory using scan."""
    # actions: (T, action_dim)
    # particles: (T, M, state_dim)
    # weights: (T, M)

    def acc_grad(carry, inputs):
        _actions, _particles, _weights = inputs
        log_prob = policy.log_prob(
            _actions[None, ...],
            _particles[None, ...],
            _weights[None, ...],
            params
        )
        return carry + log_prob, log_prob

    init_log_prob = jnp.zeros((1,))
    log_prob, _ = jax.lax.scan(
        acc_grad,
        init=init_log_prob,
        xs=(
            actions[1:],
            particles[:-1],
            weights[:-1]
        )
    )
    return log_prob


def compute_policy_gradient(params, policy, actions, particles, weights):
    def loss_fn(_params):
        log_prob = accumulate_log_prob(_params, policy, actions, particles, weights)
        return -1.0 * jnp.squeeze(log_prob)

    _, grads = jax.value_and_grad(loss_fn)(params)
    return grads


def compute_single_trajectory_gradient_and_norm(params, policy, actions, particles, weights):
    """Compute gradient and its norm for a single trajectory."""
    def loss_fn(_params):
        log_prob = accumulate_log_prob(_params, policy, actions, particles, weights)
        return -1.0 * jnp.squeeze(log_prob)

    _, grads = jax.value_and_grad(loss_fn)(params)

    # Flatten gradient and compute norm
    from jax.flatten_util import ravel_pytree
    flat_gradient, _ = ravel_pytree(grads)
    gradient_norm = jnp.linalg.norm(flat_gradient)

    return flat_gradient, gradient_norm


# JIT the entire vectorized computation
@partial(jax.jit, static_argnames=("policy",))
def compute_all_trajectory_gradients(params, policy, all_actions, all_particles, all_weights):
    """JIT-compiled function to compute gradients for all trajectories."""
    vectorized_fn = jax.vmap(
        compute_single_trajectory_gradient_and_norm,
        in_axes=(None, None, 1, 1, 1),
        out_axes=0
    )
    return vectorized_fn(params, policy, all_actions, all_particles, all_weights)


def run_single_experiment(env, policy, params, tempering, seed, num_history_particles, num_belief_particles, slew_rate_penalty):
    """Run a single SMC experiment and return the gradient covariance trace."""

    # Set up random key
    rng_key = random.PRNGKey(seed)
    key, smc_key = random.split(rng_key)

    # Run SMC
    history_states, belief_states, belief_infos, log_marginal = smc(
        rng_key=smc_key,
        num_time_steps=env.num_time_steps,
        num_history_particles=num_history_particles,
        num_belief_particles=num_belief_particles,
        belief_prior=env.belief_prior,
        policy_prior=policy,
        policy_prior_params=params,
        trans_model=env.trans_model,
        obs_model=env.obs_model,
        reward_fn=env.reward_fn,
        slew_rate_penalty=slew_rate_penalty,
        tempering=tempering,
        history_resample_fn=systematic_resampling,
        belief_resample_fn=systematic_resampling,
    )

    # Backward tracing to get training trajectories
    key, trace_key = random.split(key)
    traced_history, traced_belief, _ = backward_tracing(
        trace_key, history_states, belief_states, belief_infos
    )

    # Compute gradients for all trajectories
    gradient_vectors, gradient_norms = compute_all_trajectory_gradients(
        params,
        policy,
        traced_history.actions,
        traced_belief.particles,
        traced_belief.weights
    )

    # Compute trace of covariance matrix (first 3x3 block)
    trace = jnp.trace(jnp.cov(gradient_vectors, rowvar=False)[:3, :3])
    return float(trace), float(log_marginal)


def main():
    print("Evaluating SMC Trajectory Gradient Variance - Multiple Tempering & Seeds")
    print("=" * 70)

    # Get environment
    env = get_pomdp("linear-2d")

    print("Environment Configuration:")
    print(f"State dimension: {env.state_dim}")
    print(f"Action dimension: {env.action_dim}")
    print(f"Observation dimension: {env.obs_dim}")
    print(f"Time horizon: {env.num_time_steps}")
    print()

    # Experiment configuration
    tempering_values = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0, 5.0, 10.0]
    num_seeds = 10
    seeds = list(range(10, 10 + num_seeds))  # Seeds 10-19

    # SMC Configuration
    num_history_particles = 128
    num_belief_particles = 32
    slew_rate_penalty = 0.001

    print("Experiment Configuration:")
    print(f"Tempering values: {tempering_values}")
    print(f"Number of seeds: {num_seeds}")
    print(f"Seeds: {seeds}")
    print(f"History particles: {num_history_particles}")
    print(f"Belief particles: {num_belief_particles}")
    print(f"Slew rate penalty: {slew_rate_penalty}")
    print()

    # Create policy
    decoder = LinearGaussDecoder(
        output_dim=env.action_dim,
        init_log_std=constant(jnp.log(2.3)),
    )
    policy = create_linear_policy(decoder=decoder)

    # Initialize policy parameters
    init_key = random.PRNGKey(27)
    params = policy.init(
        rng_key=init_key,
        particle_dim=env.state_dim,
        action_dim=env.action_dim,
        batch_size=num_history_particles,
        num_particles=num_belief_particles,
    )

    # Storage for results
    results = {}

    print("Running experiments...")
    print("-" * 50)

    # Run experiments for each tempering value
    for temp_idx, tempering in enumerate(tempering_values):
        print(f"Tempering: {tempering} ({temp_idx+1}/{len(tempering_values)})")

        traces = []
        log_marginals = []

        for seed_idx, seed in enumerate(seeds):
            print(f"  Seed {seed} ({seed_idx+1}/{num_seeds})...", end=" ", flush=True)

            trace, log_marginal = run_single_experiment(
                env, policy, params, tempering, seed,
                num_history_particles, num_belief_particles, slew_rate_penalty
            )
            traces.append(trace)
            log_marginals.append(log_marginal)
            print(f"trace={trace:.3f}, log_marg={log_marginal:.3f}")

        traces_array = np.array(traces)
        log_marg_array = np.array(log_marginals)

        trace_mean = np.mean(traces_array)
        trace_sem = np.std(traces_array) / np.sqrt(len(traces_array))
        log_marg_mean = np.mean(log_marg_array)
        log_marg_sem = np.std(log_marg_array) / np.sqrt(len(log_marg_array))

        results[tempering] = {
            'trace_mean': trace_mean,
            'trace_sem': trace_sem,
            'log_marg_mean': log_marg_mean,
            'log_marg_sem': log_marg_sem,
            'n_successful': len(traces),
            'traces': traces_array,
            'log_marginals': log_marg_array
        }

        print(f"  → Trace: {trace_mean:.4f} ± {trace_sem:.4f} (n={len(traces)})")
        print()

    # Print final results table
    print("Final Results:")
    print("=" * 70)
    print(f"{'Tempering':<12} {'Trace Mean':<12} {'Trace SEM':<12} {'LogMarg Mean':<12} {'LogMarg SEM':<12} {'N':<3}")
    print("-" * 70)

    for tempering in tempering_values:
        r = results[tempering]
        print(f"{tempering:<12.1e} {r['trace_mean']:<12.4f} {r['trace_sem']:<12.4f} "
                f"{r['log_marg_mean']:<12.3f} {r['log_marg_sem']:<12.3f} {r['n_successful']:<3}")

    print()
    print("Legend:")
    print("  Trace Mean/SEM: Mean ± SEM of gradient covariance trace across seeds")
    print("  LogMarg Mean/SEM: Mean ± SEM of SMC log marginal likelihood")
    print("  N: Number of successful runs")

    # Save tempering vs covariance data to CSV
    print("\nSaving tempering vs covariance data to CSV...")
    
    # Extract data for CSV
    csv_data = []
    for tempering in tempering_values:
        csv_data.append({
            'Temperature': float(tempering),
            'Mean': float(results[tempering]['trace_mean']),
            'SEM': float(results[tempering]['trace_sem'])
        })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(csv_data)
    csv_filename = 'tempering_covariance_trace.csv'
    df.to_csv(csv_filename, index=False)
    print(f"Data saved to: {csv_filename}")
    print(f"Columns: {list(df.columns)}")
    print(f"Total rows: {len(df)}")

    # Plotting
    print("\nGenerating plots...")

    # Extract data in correct order
    trace_means = [results[temp]['trace_mean'] for temp in tempering_values]
    trace_sems = [results[temp]['trace_sem'] for temp in tempering_values]
    log_marg_means = [results[temp]['log_marg_mean'] for temp in tempering_values]
    log_marg_sems = [results[temp]['log_marg_sem'] for temp in tempering_values]

    # Plot 1: Gradient Covariance Trace
    plt.figure(figsize=(10, 6))
    plt.errorbar(tempering_values, trace_means, yerr=trace_sems,
                 fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=8,
                 color='blue', label='Trace Mean ± SEM')
    plt.xscale('log')
    plt.xlabel('Tempering (log scale)', fontsize=12)
    plt.ylabel('Gradient Covariance Trace', fontsize=12)
    plt.title('Gradient Covariance Trace vs Tempering', fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('gradient_variance_trace.png', dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.errorbar(tempering_values, log_marg_means, yerr=log_marg_sems,
                 fmt='s-', capsize=5, capthick=2, linewidth=2, markersize=8,
                 color='red', label='Log Marginal Mean ± SEM')
    plt.xscale('log')
    plt.xlabel('Tempering (log scale)', fontsize=12)
    plt.ylabel('SMC Log Marginal Likelihood', fontsize=12)
    plt.title('SMC Log Marginal Likelihood vs Tempering', fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('gradient_variance_log_marginal.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Plots saved as 'gradient_variance_trace.png' and 'gradient_variance_log_marginal.png'")


if __name__ == "__main__":
    main()
