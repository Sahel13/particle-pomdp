import time
import uuid

from jax import Array
import matplotlib.pyplot as plt

from ppomdp.envs import pomdps
from ppomdp.envs.core import POMDPEnv


def get_unique_identifier() -> str:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"-{timestamp}-{unique_id}"


def get_pomdp(env_name: str) -> POMDPEnv:
    if env_name == "pendulum":
        return pomdps.PendulumEnv
    elif env_name == "cartpole":
        return pomdps.CartPoleEnv
    elif env_name == "triangulation":
        return pomdps.TriangulationEnv
    elif env_name == "light-dark-2d":
        return pomdps.LightDark2DEnv
    elif env_name == "linear-2d":
        return pomdps.Linear2DEnv
    else:
        raise NotImplementedError


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
    elif env_name == "triangulation":
        plt.figure()
        plt.plot(states[:, 0], states[:, 2], label="Trajectory")
        plt.plot([-200], [100], "o", color="black", markersize=10, label="Starting point")
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
    elif env_name == "linear-2d":
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Linear 2D System Trajectory")

        # Position over time
        axs[0, 0].plot(states[:, 0])
        axs[0, 0].set_ylabel("Position")
        axs[0, 0].set_xlabel("Time")
        axs[0, 0].grid(True)
        axs[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Target')
        axs[0, 0].legend()

        # Velocity over time
        axs[0, 1].plot(states[:, 1])
        axs[0, 1].set_ylabel("Velocity")
        axs[0, 1].set_xlabel("Time")
        axs[0, 1].grid(True)
        axs[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Target')
        axs[0, 1].legend()

        # Control actions over time
        axs[1, 0].plot(actions[:, 0])
        axs[1, 0].set_ylabel("Control Action")
        axs[1, 0].set_xlabel("Time")
        axs[1, 0].grid(True)
        axs[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)

        # Phase space trajectory (position vs velocity)
        axs[1, 1].plot(states[:, 0], states[:, 1], 'b-', linewidth=2, label='Trajectory')
        axs[1, 1].plot(states[0, 0], states[0, 1], 'go', markersize=8, label='Start')
        axs[1, 1].plot(0, 0, 'ro', markersize=8, label='Target')
        axs[1, 1].set_xlabel("Position")
        axs[1, 1].set_ylabel("Velocity")
        axs[1, 1].grid(True)
        axs[1, 1].legend()
        axs[1, 1].set_title("Phase Space")

        plt.tight_layout()
        plt.show()
    else:
        raise ValueError("Invalid environment name.")
