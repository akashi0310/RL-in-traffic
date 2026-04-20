import os
import argparse
import time
import subprocess
import numpy as np

import config
from env.traffic_env import TrafficEnv, compute_reward
from agent.dqn_agent import DQNAgent
from utils.plots import plot_comparison

import sys
if "SUMO_HOME" in os.environ:
    sys.path.insert(0, os.path.join(os.environ["SUMO_HOME"], "tools"))
import traci


def run_rl(model_path: str, use_gui: bool, num_runs: int,
           bernoulli_p: float = 0.05,
           scenario: str = "uniform"):
    print(f"\n--- RL Agent ({num_runs} run(s), scenario={scenario}) ---")
    env = TrafficEnv(use_gui=use_gui, bernoulli_p=bernoulli_p, 
                     eval_mode=True, scenario=scenario, max_steps=config.EVAL_STEPS)
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size)
    agent.load(model_path)

    rewards = []
    stats = {"departed": [], "arrived": []}
    
    for run in range(1, num_runs + 1):
        state = env.reset()
        agent.reset_history()
        total_reward = 0.0
        while True:
            action = agent.select_action(state, training=False)
            state, reward, done, info = env.step(action)
            total_reward += reward * config.REWARD_SCALE
            if done:
                stats["departed"].append(info.get("cum_departed", 0))
                stats["arrived"].append(info.get("cum_arrived", 0))
                break
        rewards.append(total_reward)
        print(f"  Run {run}: reward = {total_reward:.1f}, passing = {stats['arrived'][-1]}/{stats['departed'][-1]}")

    return rewards, stats


def run_static(use_gui: bool, num_runs: int,
               bernoulli_p: float = 0.05,
               scenario: str = "uniform"):
    """
    Evaluates the default SUMO controller (static) using the same reward logic.
    """
    print(f"\n--- Static Controller ({num_runs} run(s), scenario={scenario}) ---")
    
    gen_env = TrafficEnv(bernoulli_p=bernoulli_p, scenario=scenario, max_steps=config.EVAL_STEPS)
    sumo_cfg_abs = gen_env.sumo_cfg
    sumo_dir = gen_env.sumo_dir
    from env.traffic_env import BERNOULLI_ROUTE_FILE
    
    binary = "sumo-gui" if use_gui else "sumo"
    port = 8814
    lanes = TrafficEnv.ALL_LANES
    left_lanes = TrafficEnv.LEFT_LANES

    rewards = []
    stats = {"departed": [], "arrived": []}
    
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
        cum_dep = 0
        cum_arr = 0
        
        while traci.simulation.getTime() < config.EVAL_DURATION:
            traci.simulationStep()
            cum_dep += traci.simulation.getDepartedNumber()
            cum_arr += traci.simulation.getArrivedNumber()
            
            step_reward = compute_reward(lanes, left_lanes)
            total_reward += step_reward * config.REWARD_SCALE

        traci.close()
        proc.wait(timeout=10)

        rewards.append(total_reward)
        stats["departed"].append(cum_dep)
        stats["arrived"].append(cum_arr)
        print(f"  Run {run}: reward = {total_reward:.1f}, passing = {cum_arr}/{cum_dep}")

    return rewards, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RL vs Static controller")
    parser.add_argument("--model", type=str, default=os.path.join(config.CHECKPOINT_DIR, "dqn_best.pt"),
                        help="Path to trained model checkpoint")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of evaluation runs")
    parser.add_argument("--gui", action="store_true", help="Use SUMO-GUI")
    parser.add_argument("--prob", type=float, default=0.04,
                        help="Bernoulli spawn probability")
    parser.add_argument("--scenario", type=str, default="uniform",
                        choices=list(TrafficEnv.SCENARIOS),
                        help="Traffic scenario")
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print(f"[ERROR] Model not found: {args.model}")
        print("  Run  python train.py  first to train a model.")
        raise SystemExit(1)

    print(f"\n{'='*60}")
    print(f"  Evaluation - Scenario: {args.scenario}")
    print(f"  Runs       : {args.runs}")
    print(f"{'='*60}")

    rl_rewards, rl_stats = run_rl(args.model, args.gui, args.runs,
                                  bernoulli_p=args.prob,
                                  scenario=args.scenario)
    
    static_rewards, static_stats = run_static(args.gui, args.runs,
                                              bernoulli_p=args.prob,
                                              scenario=args.scenario)

    rl_avg = np.mean(rl_rewards)
    static_avg = np.mean(static_rewards)
    improvement = ((rl_avg - static_avg) / abs(static_avg) * 100
                   if static_avg != 0 else 0)

    # Throughput ratios
    rl_ratio = np.mean(rl_stats["departed"]) / np.mean(rl_stats["arrived"]) if np.mean(rl_stats["arrived"]) != 0 else 0
    static_ratio = np.mean(static_stats["departed"]) / np.mean(static_stats["arrived"]) if np.mean(static_stats["arrived"]) != 0 else 0

    print(f"\n{'='*60}")
    print(f"  SUMMARY RESULTS")
    print(f"  RL Agent    : Reward={rl_avg:>8.1f} | Ratio={rl_ratio:.3f} (Injected/Completed)")
    print(f"  Static Ctrl : Reward={static_avg:>8.1f} | Ratio={static_ratio:.3f} (Injected/Completed)")
    print(f"  Improvement : {improvement:>+8.1f}%")
    print(f"{'='*60}\n")

    plot_comparison(rl_rewards, static_rewards)
