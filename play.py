"""
play.py
=======
Visual replay of the trained DQN agent using SUMO-GUI.

Usage
-----
  # Play best checkpoint with default scenario:
  python play.py

  # Play a specific scenario:
  python play.py --scenario crowded_all

  # Play a specific checkpoint:
  python play.py --model checkpoints/dqn_ep200.pt

Available scenarios: normal, crowded_all, crowded_ns, crowded_ew, fluctuate
"""

import os
import argparse

import numpy as np

from env.traffic_env import TrafficEnv, SCENARIOS
from agent.dqn_agent import DQNAgent

ACTION_NAMES = ["STAY", "FORWARD (+1)", "DIAGONAL (+2)", "BACKWARD (+3)"]
PHASE_NAMES = ["N-S straight", "E-W straight", "N-S turn", "E-W turn"]
REWARD_SCALE = 1e-3


def play(model_path: str, scenario: str = None):
    sumo_cfg = os.path.join("data", "new_sim.sumocfg")
    scenarios = [scenario] if scenario else None
    env = TrafficEnv(sumo_cfg=sumo_cfg, use_gui=True, port=8813,
                     scenarios=scenarios)
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size)
    agent.load(model_path)
    agent.epsilon = 0.0  # fully greedy

    scenario_str = scenario or "default (from .sumocfg)"
    print(f"\n{'='*62}")
    print(f"  DQN Traffic Agent - Visual Replay")
    print(f"  Model    : {model_path}")
    print(f"  Scenario : {scenario_str}")
    print(f"  Device   : {agent.device}")
    print(f"{'='*62}\n")

    state = env.reset()
    total_reward = 0.0
    step = 0

    print(f"  {'Step':>4}  {'Action':<18}  {'Phase':<16}  {'Reward':>10}  {'Cumulative':>10}")
    print(f"  {'─'*4}  {'─'*18}  {'─'*16}  {'─'*10}  {'─'*10}")

    while True:
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        reward *= REWARD_SCALE
        total_reward += reward
        step += 1

        phase_idx = int(next_state[8]) if not done else -1
        phase_name = PHASE_NAMES[phase_idx] if phase_idx >= 0 else "—"

        print(
            f"  {step:>4}  {ACTION_NAMES[action]:<18}  {phase_name:<16}  "
            f"{reward:>+10.2f}  {total_reward:>10.2f}"
        )

        state = next_state
        if done:
            break

    print(f"\n  {'='*62}")
    print(f"  Episode finished - Total reward: {total_reward:.2f}  ({step} steps)")
    print(f"  {'='*62}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual replay of trained DQN agent")
    parser.add_argument(
        "--model", type=str, default="checkpoints/dqn_best.pt",
        help="Path to checkpoint (default: checkpoints/dqn_best.pt)",
    )
    parser.add_argument(
        "--scenario", type=str, default=None,
        choices=list(SCENARIOS.keys()),
        help="Traffic scenario to play (default: uses .sumocfg route file)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print(f"[ERROR] Checkpoint not found: {args.model}")
        print("  Run  python train.py  first.")
        raise SystemExit(1)

    play(args.model, args.scenario)
