"""
train.py
========
Train a DQN agent to control the traffic light at intersection J8.

Usage
-----
  # Fast headless training (recommended):
  python train.py

  # With SUMO-GUI so you can watch the simulation:
  python train.py --gui

  # Resume from a checkpoint:
  python train.py --resume checkpoints/dqn_ep100.pt

  # Custom episode count:
  python train.py --episodes 300
"""

import os
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend so it works headlessly
import matplotlib.pyplot as plt

from env.traffic_env import TrafficEnv
from agent.dqn_agent import DQNAgent


# ── Helpers ───────────────────────────────────────────────────────────────────
def moving_average(values, window=10):
    if len(values) < window:
        return np.array(values)
    return np.convolve(values, np.ones(window) / window, mode="valid")


def plot_training(rewards, losses, save_path="training_results.png"):
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), facecolor="#1e1e2e")
    for ax in axes:
        ax.set_facecolor("#2a2a3e")
        ax.tick_params(colors="#cdd6f4")
        for spine in ax.spines.values():
            spine.set_color("#313244")
        ax.xaxis.label.set_color("#cdd6f4")
        ax.yaxis.label.set_color("#cdd6f4")
        ax.title.set_color("#cba6f7")

    eps = np.arange(1, len(rewards) + 1)

    # Reward panel
    axes[0].plot(eps, rewards, color="#89b4fa", alpha=0.3, linewidth=1)
    if len(rewards) >= 10:
        sm = moving_average(rewards)
        axes[0].plot(np.arange(10, len(rewards) + 1), sm,
                     color="#89b4fa", linewidth=2.0, label="Moving avg (10 ep)")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward")
    axes[0].set_title("Training Reward  (higher = less waiting time)")
    axes[0].legend(facecolor="#313244", labelcolor="#cdd6f4")
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

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  [plot saved] {save_path}")
    plt.close(fig)


# ── Main training loop ────────────────────────────────────────────────────────
def train(num_episodes: int = 200, use_gui: bool = False, resume: str = None):
    sumo_cfg = os.path.join("data", "sim_rl.sumocfg")
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    env   = TrafficEnv(sumo_cfg=sumo_cfg, use_gui=use_gui, port=8813)
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size)

    if resume and os.path.isfile(resume):
        agent.load(resume)

    rewards_log, losses_log = [], []
    best_reward = -float("inf")

    print(f"\n{'='*60}")
    print(f"  DQN Traffic Signal Control — Training")
    print(f"  Episodes : {num_episodes}")
    print(f"  GUI      : {use_gui}")
    print(f"  Device   : {agent.device}")
    print(f"{'='*60}\n")

    for ep in range(1, num_episodes + 1):
        state = env.reset()
        total_reward = 0.0
        ep_losses = []

        while True:
            action              = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store(state, action, reward, next_state, done)
            loss = agent.update()
            if loss is not None:
                ep_losses.append(loss)
            total_reward += reward
            state = next_state
            if done:
                break

        avg_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
        rewards_log.append(total_reward)
        losses_log.append(avg_loss)

        # Console log
        bar = "█" * int(30 * ep / num_episodes) + "░" * (30 - int(30 * ep / num_episodes))
        print(
            f"  Ep {ep:>4}/{num_episodes}  [{bar}]  "
            f"reward={total_reward:>10.1f}  "
            f"ε={agent.epsilon:.3f}  "
            f"loss={avg_loss:.4f}"
        )

        # Checkpoint every 50 episodes
        if ep % 50 == 0:
            agent.save(os.path.join(ckpt_dir, f"dqn_ep{ep}.pt"))

        # Save best model
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save(os.path.join(ckpt_dir, "dqn_best.pt"))

    agent.save(os.path.join(ckpt_dir, "dqn_final.pt"))
    plot_training(rewards_log, losses_log)

    print(f"\n  Training complete. Best reward: {best_reward:.1f}")
    return agent, rewards_log


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN traffic signal agent")
    parser.add_argument("--episodes", type=int, default=200,
                        help="Number of training episodes (default: 200)")
    parser.add_argument("--gui", action="store_true",
                        help="Use SUMO-GUI (slow but visual)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    train(num_episodes=args.episodes, use_gui=args.gui, resume=args.resume)
