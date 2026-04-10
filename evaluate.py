"""
evaluate.py
===========
Compare trained RL agent vs the original static timed controller.

Usage
-----
  # Evaluate best checkpoint with SUMO-GUI:
  python evaluate.py

  # Specify a different model:
  python evaluate.py --model checkpoints/dqn_ep150.pt

  # Run 5 trials each:
  python evaluate.py --runs 5

  # Headless mode:
  python evaluate.py --no-gui
"""

import os
import argparse
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── TraCI imports (same guard as traffic_env) ─────────────────────────────────
import sys
if "SUMO_HOME" in os.environ:
    sys.path.insert(0, os.path.join(os.environ["SUMO_HOME"], "tools"))
import traci
import subprocess

from env.traffic_env import TrafficEnv
from agent.dqn_agent import DQNAgent


# ── RL evaluation ─────────────────────────────────────────────────────────────
def run_rl(model_path: str, sumo_cfg: str, use_gui: bool, num_runs: int):
    print(f"\n--- RL Agent ({num_runs} run(s)) ---")
    env   = TrafficEnv(sumo_cfg=sumo_cfg, use_gui=use_gui, port=8813)
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size)
    agent.load(model_path)
    agent.epsilon = 0.0      # greedy policy — no exploration

    rewards = []
    for run in range(1, num_runs + 1):
        state = env.reset()
        total_reward = 0.0
        while True:
            action = agent.select_action(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
        rewards.append(total_reward)
        print(f"  Run {run}: reward = {total_reward:.1f}")

    return rewards


# ── Static controller evaluation ──────────────────────────────────────────────
def run_static(sumo_cfg: str, use_gui: bool, num_runs: int):
    """
    Let SUMO run its built-in static 2-phase plan without RL interference.
    We still use TraCI to collect waiting times, but never call setPhase.
    """
    print(f"\n--- Static Controller ({num_runs} run(s)) ---")
    sumo_cfg_abs = os.path.abspath(sumo_cfg)
    sumo_dir     = os.path.dirname(sumo_cfg_abs)
    binary       = "sumo-gui" if use_gui else "sumo"
    port         = 8814      # different port to avoid conflicts
    lanes        = TrafficEnv.LANES
    max_sim_secs = TrafficEnv.MAX_STEPS * TrafficEnv.STEP_SIZE   # 3600 s

    rewards = []
    for run in range(1, num_runs + 1):
        cmd = [
            binary, "-c", sumo_cfg_abs,
            "--remote-port", str(port),
            "--waiting-time-memory", "3600",
            "--no-step-log", "true",
            "--no-warnings", "true",
            "--duration-log.disable", "true",
        ]
        if use_gui:
            cmd += ["--start", "--delay", "50"]

        proc = subprocess.Popen(
            cmd, cwd=sumo_dir,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(1.2)
        traci.init(port)

        total_reward = 0.0
        for _ in range(max_sim_secs):
            traci.simulationStep()
            total_reward -= sum(traci.lane.getWaitingTime(l) for l in lanes)

        traci.close()
        proc.wait(timeout=10)

        rewards.append(total_reward)
        print(f"  Run {run}: reward = {total_reward:.1f}")

    return rewards


# ── Comparison plot ───────────────────────────────────────────────────────────
def plot_comparison(rl_rewards, static_rewards,
                    save_path="evaluation_results.png"):
    labels   = ["RL Agent\n(DQN)", "Static\nController"]
    means    = [np.mean(rl_rewards), np.mean(static_rewards)]
    stds     = [np.std(rl_rewards),  np.std(static_rewards)]
    colors   = ["#89b4fa", "#f38ba8"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="#1e1e2e")

    # ── Bar chart: mean reward ------------------------------------------------
    ax = axes[0]
    ax.set_facecolor("#2a2a3e")
    bars = ax.bar(labels, means, color=colors, width=0.45,
                  yerr=stds, capsize=8, error_kw={"ecolor": "#cdd6f4", "lw": 2})
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(stds) * 0.05,
                f"{m:.0f}", ha="center", va="bottom",
                color="#cdd6f4", fontsize=11, fontweight="bold")
    ax.set_title("Mean Cumulative Reward (higher = better)",
                 color="#cba6f7", fontsize=12)
    ax.set_ylabel("Total Reward (− waiting time)", color="#cdd6f4")
    ax.tick_params(colors="#cdd6f4")
    for sp in ax.spines.values():
        sp.set_color("#313244")
    ax.grid(axis="y", color="#313244", alpha=0.6)

    # ── Box plot: distribution ------------------------------------------------
    ax2 = axes[1]
    ax2.set_facecolor("#2a2a3e")
    bp = ax2.boxplot(
        [rl_rewards, static_rewards],
        labels=labels,
        patch_artist=True,
        medianprops={"color": "#cdd6f4", "linewidth": 2},
        whiskerprops={"color": "#cdd6f4"},
        capprops={"color": "#cdd6f4"},
        flierprops={"marker": "o", "markerfacecolor": "#cdd6f4", "markersize": 5},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_title("Reward Distribution", color="#cba6f7", fontsize=12)
    ax2.set_ylabel("Total Reward", color="#cdd6f4")
    ax2.tick_params(colors="#cdd6f4")
    for sp in ax2.spines.values():
        sp.set_color("#313244")
    ax2.grid(axis="y", color="#313244", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  [plot saved] {save_path}")
    plt.close(fig)

    # Summary table
    improvement = (np.mean(rl_rewards) - np.mean(static_rewards)) / abs(np.mean(static_rewards)) * 100
    print("\n" + "=" * 50)
    print(f"  RL Agent   - mean reward : {np.mean(rl_rewards):.1f} +/- {np.std(rl_rewards):.1f}")
    print(f"  Static     - mean reward : {np.mean(static_rewards):.1f} +/- {np.std(static_rewards):.1f}")
    print(f"  Improvement              : {improvement:+.1f}%")
    print("=" * 50)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RL vs Static controller")
    parser.add_argument("--model",  type=str, default="checkpoints/dqn_best.pt",
                        help="Path to trained model checkpoint")
    parser.add_argument("--runs",   type=int, default=3,
                        help="Number of evaluation runs per controller")
    parser.add_argument("--gui", dest="gui", action="store_true",
                        help="Use SUMO-GUI (default behavior)")
    parser.add_argument("--no-gui", dest="gui", action="store_false",
                        help="Disable SUMO-GUI (headless)")
    parser.set_defaults(gui=True)
    args = parser.parse_args()

    sumo_cfg = os.path.join("data", "new_sim.sumocfg")

    if not os.path.isfile(args.model):
        print(f"[ERROR] Model not found: {args.model}")
        print("  Run  python train.py  first to train a model.")
        raise SystemExit(1)

    rl_rewards     = run_rl(args.model, sumo_cfg, args.gui, args.runs)
    static_rewards = run_static(sumo_cfg, args.gui, args.runs)
    plot_comparison(rl_rewards, static_rewards)
