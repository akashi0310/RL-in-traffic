"""
evaluate.py
===========
Compare the trained RL agent vs the original static timed controller on the
Bernoulli traffic scenario.

Usage
-----
  # Evaluate best checkpoint:
  python evaluate.py

  # Specify a different model:
  python evaluate.py --model checkpoints/dqn_ep150.pt

  # Run 5 trials:
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

import sys
if "SUMO_HOME" in os.environ:
    sys.path.insert(0, os.path.join(os.environ["SUMO_HOME"], "tools"))
import traci
import subprocess

from env.traffic_env import (TrafficEnv, BERNOULLI_ROUTE_FILE,
                             compute_reward)
from agent.dqn_agent import DQNAgent

REWARD_SCALE = 1e-3


# -- RL evaluation -------------------------------------------------------------
def run_rl(model_path: str, sumo_cfg: str, use_gui: bool, num_runs: int,
           bernoulli_p: float = 0.05, reward_mode: str = "count",
           scenario: str = "uniform"):
    print(f"\n--- RL Agent ({num_runs} run(s), scenario={scenario}) ---")
    env = TrafficEnv(sumo_cfg=sumo_cfg, use_gui=use_gui, port=8813,
                     bernoulli_p=bernoulli_p, reward_mode=reward_mode,
                     eval_mode=True, scenario=scenario)
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size)
    agent.load(model_path)
    agent.epsilon = 0.0

    rewards = []
    for run in range(1, num_runs + 1):
        state = env.reset()
        total_reward = 0.0
        while True:
            action = agent.select_action(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward * REWARD_SCALE
            if done:
                break
        rewards.append(total_reward)
        print(f"  Run {run}: reward = {total_reward:.1f}")

    return rewards


# -- Static controller evaluation ----------------------------------------------
def run_static(sumo_cfg: str, use_gui: bool, num_runs: int,
               bernoulli_p: float = 0.05, reward_mode: str = "count",
               scenario: str = "uniform"):
    print(f"\n--- Static Controller ({num_runs} run(s), scenario={scenario}) ---")
    sumo_cfg_abs = os.path.abspath(sumo_cfg)
    sumo_dir     = os.path.dirname(sumo_cfg_abs)
    binary       = "sumo-gui" if use_gui else "sumo"
    port         = 8814
    lanes        = TrafficEnv.ALL_LANES
    max_sim_secs = TrafficEnv.MAX_STEPS * TrafficEnv.STEP_SIZE

    # Reuse the env's Bernoulli route generator so both controllers see the same spec
    gen_env = TrafficEnv(sumo_cfg=sumo_cfg, bernoulli_p=bernoulli_p,
                         reward_mode=reward_mode, scenario=scenario)

    rewards = []
    for run in range(1, num_runs + 1):
        gen_env._generate_bernoulli_routes()

        cmd = [
            binary, "-c", sumo_cfg_abs,
            "--route-files", BERNOULLI_ROUTE_FILE,
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
            step_reward = compute_reward(lanes, TrafficEnv.LEFT_LANES, mode=reward_mode)
            total_reward += step_reward * REWARD_SCALE

        # drain phase: keep stepping (no more spawns) until network is clear
        drain_cap = 200 * TrafficEnv.STEP_SIZE
        drained = 0
        while traci.simulation.getMinExpectedNumber() > 0 and drained < drain_cap:
            traci.simulationStep()
            drained += 1

        traci.close()
        proc.wait(timeout=10)

        rewards.append(total_reward)
        print(f"  Run {run}: reward = {total_reward:.1f}")

    return rewards


# -- Comparison plot -----------------------------------------------------------
def plot_comparison(rl_rewards, static_rewards, save_path="evaluation_results.png"):
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="#1e1e2e")
    ax.set_facecolor("#2a2a3e")

    rl_mean, static_mean = np.mean(rl_rewards), np.mean(static_rewards)
    rl_std, static_std = np.std(rl_rewards), np.std(static_rewards)

    labels = ["RL\nAgent", "Static\nCtrl"]
    means = [rl_mean, static_mean]
    stds = [rl_std, static_std]
    colors = ["#89b4fa", "#f38ba8"]

    bars = ax.bar(labels, means, color=colors, width=0.45,
                  yerr=stds, capsize=8,
                  error_kw={"ecolor": "#cdd6f4", "lw": 2})
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(stds) * 0.05,
                f"{m:.0f}", ha="center", va="bottom",
                color="#cdd6f4", fontsize=10, fontweight="bold")

    improvement = ((rl_mean - static_mean) / abs(static_mean) * 100
                   if static_mean != 0 else 0)
    ax.set_title(f"Bernoulli traffic  ({improvement:+.1f}%)",
                 color="#cba6f7", fontsize=12)
    ax.set_ylabel("Total Reward", color="#cdd6f4")
    ax.tick_params(colors="#cdd6f4")
    for sp in ax.spines.values():
        sp.set_color("#313244")
    ax.grid(axis="y", color="#313244", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  [plot saved] {save_path}")
    plt.close(fig)


# -- Entry point ---------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate RL vs Static controller on Bernoulli traffic")
    parser.add_argument("--model", type=str, default="checkpoints/dqn_best.pt",
                        help="Path to trained model checkpoint")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of evaluation runs")
    parser.add_argument("--gui", dest="gui", action="store_true",
                        help="Use SUMO-GUI")
    parser.add_argument("--no-gui", dest="gui", action="store_false",
                        help="Disable SUMO-GUI (headless)")
    parser.add_argument("--prob", type=float, default=0.04,
                        help="Bernoulli spawn probability per lane (default: 0.04)")
    parser.add_argument("--reward-mode", type=str, default="count", choices=["wait", "count", "harmonic"],
                        help="Reward metric for evaluation")
    parser.add_argument("--scenario", type=str, default="uniform",
                        choices=list(TrafficEnv.SCENARIOS),
                        help="Traffic scenario (default: uniform)")
    parser.set_defaults(gui=True)
    args = parser.parse_args()

    sumo_cfg = os.path.join("data", "new_sim.sumocfg")

    if not os.path.isfile(args.model):
        print(f"[ERROR] Model not found: {args.model}")
        print("  Run  python train.py  first to train a model.")
        raise SystemExit(1)

    all_rl_rewards = []
    all_static_rewards = []

    print(f"\n{'='*60}")
    print(f"  Evaluating performance across all 4 scenarios")
    print(f"  Scenarios : {TrafficEnv.SCENARIOS}")
    print(f"  Runs/scen : {args.runs}")
    print(f"{'='*60}")

    for scen in TrafficEnv.SCENARIOS:
        scen_rl = run_rl(args.model, sumo_cfg, args.gui, args.runs,
                         bernoulli_p=args.prob, reward_mode=args.reward_mode,
                         scenario=scen)
        scen_static = run_static(sumo_cfg, args.gui, args.runs,
                                 bernoulli_p=args.prob, reward_mode=args.reward_mode,
                                 scenario=scen)
        
        all_rl_rewards.extend(scen_rl)
        all_static_rewards.extend(scen_static)

    rl_avg = np.mean(all_rl_rewards)
    static_avg = np.mean(all_static_rewards)
    improvement = ((rl_avg - static_avg) / abs(static_avg) * 100
                   if static_avg != 0 else 0)

    print(f"\n{'='*60}")
    print(f"  GRAND AVERAGE RESULTS")
    print(f"  RL Agent    : {rl_avg:>8.1f} +/- {np.std(all_rl_rewards):.1f}")
    print(f"  Static Ctrl : {static_avg:>8.1f} +/- {np.std(all_static_rewards):.1f}")
    print(f"  Improvement : {improvement:>+8.1f}%")
    print(f"{'='*60}\n")

    plot_comparison(all_rl_rewards, all_static_rewards)
