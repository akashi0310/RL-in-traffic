"""
train.py
========
Train a DQN agent to control the traffic light at the intersection.

Usage
-----
  # Default training (Bernoulli traffic, p=0.05):
  python train.py

  # With SUMO-GUI so you can watch the simulation:
  python train.py --gui

  # Resume from a checkpoint:
  python train.py --resume checkpoints/dqn_ep100.pt

  # Custom episode count and Bernoulli probability:
  python train.py --episodes 300 --prob 0.08
"""

import os
import argparse
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend so it works headlessly
import matplotlib.pyplot as plt

from env.traffic_env import TrafficEnv
from agent.dqn_agent import DQNAgent

REWARD_SCALE = 1e-3


# -- Helpers -------------------------------------------------------------------
def moving_average(values, window=10):
    if len(values) < window:
        return np.array(values)
    return np.convolve(values, np.ones(window) / window, mode="valid")


def plot_training(rewards, losses, wait_parts=None, count_parts=None, save_path="training_results.png"):
    """
    Parameters
    ----------
    rewards : list[float]
        Per-episode total reward.
    losses : list[float]
        Per-episode average loss.
    wait_parts : list[float], optional
        Lead-vehicle waiting component of reward.
    count_parts : list[float], optional
        Halting vehicle count component of reward.
    """
    n_rows = 3 if wait_parts is not None else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 4.5 * n_rows), facecolor="#1e1e2e")
    for ax in axes:
        ax.set_facecolor("#2a2a3e")
        ax.tick_params(colors="#cdd6f4")
        for spine in ax.spines.values():
            spine.set_color("#313244")
        ax.xaxis.label.set_color("#cdd6f4")
        ax.yaxis.label.set_color("#cdd6f4")
        ax.title.set_color("#cba6f7")

    # Reward panel
    episodes = np.arange(1, len(rewards) + 1)
    axes[0].scatter(episodes, rewards, color="#94e2d5", alpha=0.35, s=14)
    if len(rewards) >= 5:
        sm = moving_average(rewards, window=5)
        axes[0].plot(episodes[4:], sm, color="#94e2d5", linewidth=2.2,
                     label="Moving avg (5 ep)")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward")
    axes[0].set_title("Training Reward  (higher = less waiting time)")
    axes[0].legend(facecolor="#313244", labelcolor="#cdd6f4",
                   fontsize=9, loc="lower right")
    axes[0].grid(True, color="#313244", alpha=0.6)

    # Loss panel
    axes[1].plot(losses, color="#f38ba8", alpha=0.5, linewidth=1)
    if len(losses) >= 10:
        axes[1].plot(np.arange(10, len(losses) + 1), moving_average(losses),
                     color="#f38ba8", linewidth=2.2, label="Moving avg (10 ep)")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Loss (Huber)")
    axes[1].set_title("Training Loss")
    axes[1].legend(facecolor="#313244", labelcolor="#cdd6f4")
    axes[1].grid(True, color="#313244", alpha=0.6)

    # Component panel (if provided)
    if wait_parts is not None and count_parts is not None:
        episodes = np.arange(1, len(rewards) + 1)
        # Use stacked or side-by-side? Let's use two lines for clarity
        axes[2].plot(episodes, wait_parts, color="#fab387", alpha=0.7, label="Wait Component")
        axes[2].plot(episodes, count_parts, color="#89b4fa", alpha=0.7, label="Count Component")
        if len(wait_parts) >= 5:
            axes[2].plot(episodes[4:], moving_average(wait_parts, 5), color="#fab387", linewidth=2)
            axes[2].plot(episodes[4:], moving_average(count_parts, 5), color="#89b4fa", linewidth=2)
        axes[2].set_xlabel("Episode")
        axes[2].set_ylabel("Component Reward (Unscaled)")
        axes[2].set_title("Reward Breakdown (Harmonic Components)")
        axes[2].legend(facecolor="#313244", labelcolor="#cdd6f4")
        axes[2].grid(True, color="#313244", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  [plot saved] {save_path}")
    plt.close(fig)


# -- Main training loop -------------------------------------------------------
def train(num_episodes: int = 200, use_gui: bool = False, resume: str = None,
          bernoulli_p: float = 0.05, reward_mode: str = "harmonic",
          scenario: str = None):
    sumo_cfg = os.path.join("data", "new_sim.sumocfg")
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    # None -> cycle through all scenarios round-robin; else pin to one
    scenario_cycle = (TrafficEnv.SCENARIOS if scenario is None else (scenario,))
    env = TrafficEnv(sumo_cfg=sumo_cfg, use_gui=use_gui, port=8813,
                     bernoulli_p=bernoulli_p, reward_mode=reward_mode,
                     scenario=scenario_cycle[0])
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size)

    if resume and os.path.isfile(resume):
        agent.load(resume)

    losses_log = []
    rewards_log = []
    wait_log = []
    count_log = []
    best_reward = -float("inf")

    # Unique identifier for this session's plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_plot_path = f"results_{timestamp}.png"

    print(f"\n{'='*60}")
    print(f"  DQN Traffic Signal Control - Training")
    print(f"  Episodes    : {num_episodes}")
    print(f"  Bernoulli p : {bernoulli_p}")
    print(f"  GUI         : {use_gui}")
    print(f"  Device      : {agent.device}")
    print(f"{'='*60}\n")

    for ep in range(1, num_episodes + 1):
        env.set_scenario(scenario_cycle[(ep - 1) % len(scenario_cycle)])
        state = env.reset()
        total_reward = 0.0
        ep_wait = 0.0
        ep_count = 0.0
        ep_losses = []

        while True:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            reward *= REWARD_SCALE
            agent.store(state, action, reward, next_state, done)
            loss = agent.update()
            if loss is not None:
                ep_losses.append(loss)
            total_reward += reward
            ep_wait += info.get("wait_part", 0.0)
            ep_count += info.get("count_part", 0.0)
            state = next_state
            if done:
                break

        avg_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
        rewards_log.append(total_reward)
        losses_log.append(avg_loss)
        wait_log.append(ep_wait)
        count_log.append(ep_count)

        # Console log
        bar = "=" * int(30 * ep / num_episodes) + "-" * (30 - int(30 * ep / num_episodes))
        print(
            f"  Ep {ep:>4}/{num_episodes}  [{bar}]  "
            f"scen={env.scenario:<10}  "
            f"reward={total_reward:>10.1f}  "
            f"eps={agent.epsilon:.3f}  "
            f"loss={avg_loss:.4f}"
        )

        # Checkpoint every 50 episodes
        if ep % 50 == 0:
            agent.save(os.path.join(ckpt_dir, f"dqn_ep{ep}.pt"))

        # Save best model based on the average of the last 4 scenarios (one full cycle)
        if ep >= 4 and ep % len(scenario_cycle) == 0:
            cycle_avg = np.mean(rewards_log[-len(scenario_cycle):])
            if cycle_avg > best_reward:
                best_reward = cycle_avg
                agent.save(os.path.join(ckpt_dir, "dqn_best.pt"))
                print(f"  [new best] cycle avg = {best_reward:.1f}")

    agent.save(os.path.join(ckpt_dir, "dqn_final.pt"))
    plot_training(rewards_log, losses_log, wait_log, count_log, save_path=session_plot_path)

    print(f"\n  Training complete. Best reward: {best_reward:.1f}")
    return agent, rewards_log


# -- Entry point ---------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN traffic signal agent")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of training episodes (default: 100)")
    parser.add_argument("--gui", action="store_true",
                        help="Use SUMO-GUI (slow but visual)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--prob", type=float, default=0.05,
                        help="Bernoulli spawn probability per lane (default: 0.05)")
    parser.add_argument("--reward-mode", type=str, default="harmonic", choices=["wait", "count", "harmonic"],
                        help="Reward metric: 'wait' for waiting time, 'count' for halting vehicles")
    parser.add_argument("--scenario", type=str, default=None,
                        choices=list(TrafficEnv.SCENARIOS),
                        help="Pin training to a single scenario (default: cycle all)")
    args = parser.parse_args()

    train(num_episodes=args.episodes, use_gui=args.gui, resume=args.resume,
          bernoulli_p=args.prob, reward_mode=args.reward_mode,
          scenario=args.scenario)
