import os
import argparse
from datetime import datetime

import numpy as np
import config
from env.traffic_env import TrafficEnv
from agent.dqn_agent import DQNAgent
from utils.plots import plot_training


def train(num_episodes: int = 100, use_gui: bool = False, resume: str = None,
          bernoulli_p: float = 0.05, reward_mode: str = "harmonic",
          scenario: str = None, use_ddqn: bool = config.USE_DDQN):
    
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # Scenario selection
    scenario_cycle = (TrafficEnv.SCENARIOS if scenario is None else (scenario,))
    
    env = TrafficEnv(use_gui=use_gui, bernoulli_p=bernoulli_p, 
                     reward_mode=reward_mode, scenario=scenario_cycle[0])
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size, use_double=use_ddqn)

    if resume and os.path.isfile(resume):
        agent.load(resume)

    losses_log = []    # Episode average losses
    rewards_log = []   # Episode total rewards
    wait_log = []      # Episode total wait penalty
    count_log = []     # Episode total count penalty
    
    # Step-wise logs for higher resolution plotting
    step_rewards = []
    step_losses = []
    step_wait = []
    step_count = []
    
    best_reward = -float("inf")

    # Session identifier
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_plot_path = f"results_{timestamp}.png"

    print(f"\n{'='*60}")
    print(f"  DQN Traffic Signal Control - Training")
    print(f"  Episodes    : {num_episodes}")
    print(f"  Bernoulli p : {bernoulli_p}")
    print(f"  Double DQN  : {use_ddqn}")
    print(f"  Scenario    : {'Cycle All' if scenario is None else scenario}")
    print(f"  Device      : {agent.device}")
    print(f"{'='*60}\n")

    try:
        for ep in range(1, num_episodes + 1):
            env.set_scenario(scenario_cycle[(ep - 1) % len(scenario_cycle)])
            state = env.reset()
            total_reward = 0.0
            ep_wait = 0.0
            ep_count = 0.0
            ep_losses = []

            while True:
                action = agent.select_action(state, training=True)
                next_state, reward, done, info = env.step(action)
                
                # Apply scaling
                reward *= config.REWARD_SCALE
                
                agent.store(state, action, reward, next_state, done)
                loss = agent.update()
                
                if loss is not None:
                    ep_losses.append(loss)
                
                total_reward += reward
                ep_wait += info.get("wait_part", 0.0)
                ep_count += info.get("count_part", 0.0)
                
                # Record step-wise data
                step_rewards.append(reward)
                step_wait.append(info.get("wait_part", 0.0))
                step_count.append(info.get("count_part", 0.0))
                if loss is not None:
                    step_losses.append(loss)
                
                state = next_state
                
                if done:
                    break

            avg_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
            rewards_log.append(total_reward)
            losses_log.append(avg_loss)
            wait_log.append(ep_wait)
            count_log.append(ep_count)

            # Progress monitoring
            progress = int(30 * ep / num_episodes)
            bar = "=" * progress + "-" * (30 - progress)
            print(
                f"  Ep {ep:>4}/{num_episodes}  [{bar}]  "
                f"scen={env.scenario:<10}  "
                f"reward={total_reward:>10.1f}  "
                f"eps={agent.epsilon:.3f}  "
                f"loss={avg_loss:.4f}"
            )

            # Regular checkpoints
            if ep % 50 == 0:
                agent.save(os.path.join(config.CHECKPOINT_DIR, f"dqn_ep{ep}.pt"))

            # Best model tracking
            # If cycling scenarios, save best based on the average of a full cycle
            # If pinned to one scenario, save best based on single episode reward
            is_new_best = False
            if len(scenario_cycle) > 1:
                if ep >= len(scenario_cycle) and ep % len(scenario_cycle) == 0:
                    metric = np.mean(rewards_log[-len(scenario_cycle):])
                    if metric > best_reward:
                        best_reward = metric
                        is_new_best = True
                        label = "cycle avg"
            else:
                metric = total_reward
                if metric > best_reward:
                    best_reward = metric
                    is_new_best = True
                    label = "reward"

            if is_new_best:
                agent.save(os.path.join(config.CHECKPOINT_DIR, "dqn_best.pt"))
                print(f"  [new best] {label} = {best_reward:.1f}")

    except KeyboardInterrupt:
        print("\n  [INTERRUPTED] Training stopped by user.")
    finally:
        env.close()

    if step_rewards:
        agent.save(os.path.join(config.CHECKPOINT_DIR, "dqn_final.pt"))
        plot_training(step_rewards, step_losses, step_wait, step_count, save_path=session_plot_path)
        print(f"\n  Training complete. Best cycle reward: {best_reward:.1f}")
    
    return agent, rewards_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN traffic signal agent")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of training episodes")
    parser.add_argument("--gui", action="store_true",
                        help="Use SUMO-GUI")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--prob", type=float, default=0.05,
                        help="Bernoulli spawn probability")
    parser.add_argument("--reward-mode", type=str, default="harmonic", 
                        choices=["wait", "count", "harmonic"],
                        help="Reward metric")
    parser.add_argument("--scenario", type=str, default=None,
                        choices=list(TrafficEnv.SCENARIOS),
                        help="Pin training to a single scenario")
    parser.add_argument("--ddqn", action="store_true", default=config.USE_DDQN,
                        help="Use Double DQN")
    args = parser.parse_args()

    train(num_episodes=args.episodes, use_gui=args.gui, resume=args.resume,
          bernoulli_p=args.prob, reward_mode=args.reward_mode,
          scenario=args.scenario, use_ddqn=args.ddqn)
