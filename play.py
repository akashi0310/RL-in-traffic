import os
import argparse

import config
from env.traffic_env import TrafficEnv
from agent.dqn_agent import DQNAgent

ACTION_NAMES = ["STAY", "NEXT PHASE"]
PHASE_NAMES = ["N-S straight", "E-W straight", "N-S turn", "E-W turn"]


def play(model_path: str, bernoulli_p: float = 0.05):
    env = TrafficEnv(use_gui=True, bernoulli_p=bernoulli_p)
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size)
    agent.load(model_path)

    print(f"\n{'='*62}")
    print(f"  DQN Traffic Agent - Visual Replay")
    print(f"  Model       : {model_path}")
    print(f"  Bernoulli p : {bernoulli_p}")
    print(f"  Device      : {agent.device}")
    print(f"{'='*62}\n")

    state = env.reset()
    total_reward = 0.0
    step = 0

    print(f"  {'Step':>4}  {'Action':<18}  {'Phase':<16}  {'Reward':>10}  {'Cumulative':>10}")
    print(f"  {'-'*4}  {'-'*18}  {'-'*16}  {'-'*10}  {'-'*10}")

    try:
        while True:
            action = agent.select_action(state, training=False)
            next_state, reward, done, info = env.step(action)
            
            scaled_reward = reward * config.REWARD_SCALE
            total_reward += scaled_reward
            step += 1

            phase_idx = int(info.get("phase", -1))
            phase_name = PHASE_NAMES[phase_idx] if phase_idx >= 0 else "-"

            print(
                f"  {step:>4}  {ACTION_NAMES[action]:<18}  {phase_name:<16}  "
                f"{scaled_reward:>+10.2f}  {total_reward:>10.2f}"
            )

            state = next_state
            if done:
                break
    except KeyboardInterrupt:
        print("\n  [INTERRUPTED] Replay stopped by user.")
    finally:
        env.close()

    print(f"\n  {'='*62}")
    print(f"  Episode finished - Total reward: {total_reward:.2f}  ({step} steps)")
    print(f"  {'='*62}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual replay of trained agent")
    parser.add_argument("--model", type=str, default=os.path.join(config.CHECKPOINT_DIR, "dqn_best.pt"),
                        help="Path to checkpoint")
    parser.add_argument("--prob", type=float, default=0.05,
                        help="Bernoulli spawn probability")
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print(f"[ERROR] Checkpoint not found: {args.model}")
        print("  Run  python train.py  first.")
        raise SystemExit(1)

    play(args.model, args.prob)
