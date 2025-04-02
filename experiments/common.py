import argparse
from datetime import datetime

import matplotlib.pyplot as plt
from jax import Array

from ppomdp.envs import pomdps
from ppomdp.envs.core import POMDPEnv


def get_cmd_args():
    """Command line arguments common to all algorithms."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="Seed for PRNGs", default=0)
    parser.add_argument(
        "--env",
        type=str,
        help="Environment name",
        choices=[
            "pendulum",
            "cartpole",
            "target-interception",
            "light-dark-1d",
            "light-dark-2d",
        ],
    )
    parser.add_argument(
        "--log_dir", type=str, help="Logging directory", default=f"logs/{timestamp}"
    )
    return parser.parse_args()


def get_env(env_name: str) -> POMDPEnv:
    if env_name == "pendulum":
        return pomdps.PendulumEnv
    elif env_name == "cartpole":
        return pomdps.CartPoleEnv
    elif env_name == "target-interception":
        return pomdps.TargetEnv
    elif env_name == "light-dark-1d":
        return pomdps.LightDark1DEnv
    else:
        return pomdps.LightDark2DEnv


def plot_trajectory(env_name: str, states: Array, actions: Array):
    if env_name == "pendulum":
        fig, axs = plt.subplots(3, 1, figsize=(10, 10))
        fig.suptitle("Simulated trajectory")

        axs[0].plot(states[:, 0])
        axs[0].set_ylabel("Angle")
        axs[0].grid(True)

        axs[1].plot(states[:, 1])
        axs[1].set_ylabel("Angular velocity")
        axs[1].grid(True)

        axs[2].plot(actions[:, 0])
        axs[2].set_ylabel("Action")
        axs[2].set_xlabel("Time")
        axs[2].grid(True)

        plt.tight_layout()
        plt.show()
    elif env_name == "cartpole":
        fig, axs = plt.subplots(3, 1, figsize=(10, 10))
        fig.suptitle("Simulated trajectory")

        axs[0].plot(states[:, 1])
        axs[0].set_ylabel("Angle")
        axs[0].grid(True)

        axs[1].plot(states[:, 3])
        axs[1].set_ylabel("Angular velocity")
        axs[1].grid(True)

        axs[2].plot(actions[:, 0])
        axs[2].set_ylabel("Action")
        axs[2].set_xlabel("Time")
        axs[2].grid(True)

        plt.tight_layout()
        plt.show()
    elif env_name == "target-interception":
        plt.figure()
        plt.plot(states[:, 0], states[:, 2], label="Trajectory")
        plt.plot(
            [-200], [100], "o", color="black", markersize=10, label="Starting point"
        )
        plt.plot([0], [0], "o", color="orange", markersize=10, label="Target")
        plt.plot([-200, 0], [100, 0], "r--")
        plt.title("Simulated trajectory")
        plt.xlabel("x")
        plt.ylabel("y")
        ax = plt.gca()
        ax.set_aspect("equal", adjustable="box")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    elif env_name == "light-dark-2d":
        plt.figure()
        plt.title("Simulated trajectory")
        plt.plot(states[:, 0], states[:, 1], "g-")
        plt.plot(2, 2, "ro", label="Starting location")
        plt.plot(0, 0, "rx", label="Target location")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.axis("equal")
        plt.show()

        # Plot actions.
        plt.figure()
        plt.plot(actions[:, 0])
        plt.plot(actions[:, 1])
        plt.xlabel("Time")
        plt.ylabel("Action")
        plt.show()
    else:
        raise ValueError("Invalid environment name.")
