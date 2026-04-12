"""
train.py
========
Train a DQN agent to control the traffic light at intersection J8.

Usage
-----
  # Train on all scenarios (default):
  python train.py

  # Train on specific scenarios:
  python train.py --scenarios normal crowded_all

  # With SUMO-GUI so you can watch the simulation:
  python train.py --gui

  # Resume from a checkpoint:
  python train.py --resume checkpoints/dqn_ep100.pt

  # Custom episode count:
  python train.py --episodes 300

Available scenarios: normal, crowded_all, crowded_ns, crowded_ew, fluctuate, all
"""

import os
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend so it works headlessly
import matplotlib.pyplot as plt

from collections import defaultdict

from env.traffic_env import TrafficEnv, SCENARIOS
from agent.dqn_agent import DQNAgent

REWARD_SCALE = 1e-3


# -- Helpers -------------------------------------------------------------------
def moving_average(values, window=10):
    if len(values) < window:
        return np.array(values)
    return np.convolve(values, np.ones(window) / window, mode="valid")


SCENARIO_COLORS = {
    "normal_traffic.rou.xml":        "#89b4fa",   # blue
    "crowded_all.rou.xml":           "#f38ba8",   # red
    "crowded_single.rou.xml":        "#a6e3a1",   # green
    "crowded_single_ew.rou.xml":     "#fab387",   # orange
    "crowded_fluctuate.rou.xml":     "#cba6f7",   # purple
    "bernoulli.rou.xml":             "#94e2d5",   # teal
}

SCENARIO_LABELS = {
    "normal_traffic.rou.xml":        "normal",
    "crowded_all.rou.xml":           "crowded_all",
    "crowded_single.rou.xml":        "crowded_ns",
    "crowded_single_ew.rou.xml":     "crowded_ew",
    "crowded_fluctuate.rou.xml":     "fluctuate",
    "bernoulli.rou.xml":             "bernoulli",
}


def plot_training(scenario_rewards, losses, save_path="training_results.png"):
    """
    Parameters
    ----------
    scenario_rewards : dict[str, list[tuple[int, float]]]
        {route_file: [(episode_num, reward), ...]}
    losses : list[float]
        Per-episode average loss (all scenarios combined).
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), facecolor="#1e1e2e")
    for ax in axes:
        ax.set_facecolor("#2a2a3e")
        ax.tick_params(colors="#cdd6f4")
        for spine in ax.spines.values():
            spine.set_color("#313244")
        ax.xaxis.label.set_color("#cdd6f4")
        ax.yaxis.label.set_color("#cdd6f4")
        ax.title.set_color("#cba6f7")

    # Reward panel — one line per scenario
    for route_file, ep_rewards in scenario_rewards.items():
        if not ep_rewards:
            continue
        episodes = [er[0] for er in ep_rewards]
        rewards  = [er[1] for er in ep_rewards]
        color = SCENARIO_COLORS.get(route_file, "#cdd6f4")
        label = SCENARIO_LABELS.get(route_file, route_file)

        axes[0].scatter(episodes, rewards, color=color, alpha=0.25, s=12)
        if len(rewards) >= 5:
            sm = moving_average(rewards, window=5)
            sm_x = np.array(episodes[4:])  # align with 'valid' mode output
            axes[0].plot(sm_x, sm, color=color, linewidth=2.0,
                         label=f"{label} (avg 5)")

    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward")
    axes[0].set_title("Training Reward per Scenario  (higher = less waiting time)")
    axes[0].legend(facecolor="#313244", labelcolor="#cdd6f4",
                   fontsize=8, loc="lower right")
    axes[0].grid(True, color="#313244", alpha=0.6)

    # Loss panel — combined across all scenarios
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


# -- Main training loop -------------------------------------------------------
def train(num_episodes: int = 200, use_gui: bool = False, resume: str = None,
          scenarios: list = None, bernoulli_p: float = 0.05, 
          reward_mode: str = "wait"):
    sumo_cfg = os.path.join("data", "new_sim.sumocfg")
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    env   = TrafficEnv(sumo_cfg=sumo_cfg, use_gui=use_gui, port=8813,
                       scenarios=scenarios, bernoulli_p=bernoulli_p,
                       reward_mode=reward_mode)
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size)

    if resume and os.path.isfile(resume):
        agent.load(resume)

    losses_log = []
    scenario_rewards = defaultdict(list)   # {route_file: [(ep, reward), ...]}
    best_reward = -float("inf")

    scenario_str = ", ".join(scenarios) if scenarios else "default (from .sumocfg)"
    print(f"\n{'='*60}")
    print(f"  DQN Traffic Signal Control - Training")
    print(f"  Episodes  : {num_episodes}")
    print(f"  Scenarios : {scenario_str}")
    print(f"  GUI       : {use_gui}")
    print(f"  Device    : {agent.device}")
    print(f"{'='*60}\n")

    for ep in range(1, num_episodes + 1):
        state = env.reset()
        total_reward = 0.0
        ep_losses = []

        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            reward *= REWARD_SCALE
            agent.store(state, action, reward, next_state, done)
            loss = agent.update()
            if loss is not None:
                ep_losses.append(loss)
            total_reward += reward
            state = next_state
            if done:
                break

        avg_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
        scenario_key = env._current_scenario or "default"
        scenario_rewards[scenario_key].append((ep, total_reward))
        losses_log.append(avg_loss)

        # Console log
        scenario_tag = f"  [{env._current_scenario}]" if env._current_scenario else ""
        bar = "=" * int(30 * ep / num_episodes) + "-" * (30 - int(30 * ep / num_episodes))
        print(
            f"  Ep {ep:>4}/{num_episodes}  [{bar}]  "
            f"reward={total_reward:>10.1f}  "
            f"eps={agent.epsilon:.3f}  "
            f"loss={avg_loss:.4f}"
            f"{scenario_tag}"
        )

        # Checkpoint every 50 episodes
        if ep % 50 == 0:
            agent.save(os.path.join(ckpt_dir, f"dqn_ep{ep}.pt"))

        # Save best model
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save(os.path.join(ckpt_dir, "dqn_best.pt"))

    agent.save(os.path.join(ckpt_dir, "dqn_final.pt"))
    plot_training(scenario_rewards, losses_log)

    print(f"\n  Training complete. Best reward: {best_reward:.1f}")
    return agent, scenario_rewards


# -- Entry point ---------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN traffic signal agent")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of training episodes (default: 200)")
    parser.add_argument("--gui", action="store_true",
                        help="Use SUMO-GUI (slow but visual)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--scenarios", nargs="+", default=["all"],
                        help="Scenarios to mix during training "
                             "(default: all). Options: normal, crowded_all, "
                             "crowded_ns, crowded_ew, fluctuate, bernoulli, all")
    parser.add_argument("--prob", type=float, default=0.05,
                        help="Bernoulli spawn probability per lane (default: 0.05)")
    parser.add_argument("--reward-mode", type=str, default="wait", choices=["wait", "count"],
                        help="Reward metric: 'wait' for waiting time, 'count' for halting vehicles")
    args = parser.parse_args()

    train(num_episodes=args.episodes, use_gui=args.gui, resume=args.resume,
          scenarios=args.scenarios, bernoulli_p=args.prob,
          reward_mode=args.reward_mode)
