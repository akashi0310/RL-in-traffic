# Traffic Light Control using Deep Q-Network (DQN)

This project implements a Reinforcement Learning agent based on Deep Q-Network (DQN) to optimize traffic light control at an intersection. The simulation environment is built on SUMO (Simulation of Urban MObility) and interfaces with the Python code via TraCI (Traffic Control Interface). 

By intelligently switching traffic light phases based on real-time queues and vehicle speeds, the RL agent minimizes overall waiting time compared to a standard static time-based traffic light controller.

## 🗂 Project Structure

```bash
📦 project
 ┣ 📂 agent
 ┃ ┣ 📜 __init__.py
 ┃ ┗ 📜 dqn_agent.py              # Defines the Deep Q-Network and Replay Buffer.
 ┣ 📂 data
 ┃ ┣ 📜 new_intersection.net.xml  # SUMO network topology (lanes, junctions).
 ┃ ┣ 📜 new_sim.sumocfg           # Configuration file linking network & routes for SUMO.
 ┃ ┣ 📜 normal_traffic.rou.xml    # Gaussian bell-curve, moderate volume.
 ┃ ┣ 📜 crowded_all.rou.xml       # High constant volume from all 4 directions.
 ┃ ┣ 📜 crowded_single.rou.xml    # N-S heavy, E-W light.
 ┃ ┣ 📜 crowded_single_ew.rou.xml # E-W heavy, N-S light.
 ┃ ┣ 📜 crowded_fluctuate.rou.xml # Alternating N-S / E-W heavy every 100s.
 ┃ ┣ 📜 bernoulli.rou.xml         # [Auto-generated] Bernoulli spawn per lane.
 ┃ ┗ ...                          # Other SUMO related assets.
 ┣ 📂 env
 ┃ ┣ 📜 __init__.py
 ┃ ┗ 📜 traffic_env.py            # Custom Gym-like Env wrapping SUMO via TraCI.
 ┣ 📜 train.py                    # Main training loop for the RL agent.
 ┣ 📜 evaluate.py                 # Evaluation script (RL Agent vs. Static Controller).
 ┣ 📜 play.py                     # Visual replay of trained agent in SUMO-GUI.
 ┣ 📜 requirements.txt            # Project Python dependencies.
 ┗ 📜 README.md                   # This documentation file.
```

## 🔄 System Workflow & Logic

1. **Environment (SUMO via TraCI)**: 
    The intersection receives approaching traffic from 4 directions. The state observed by the RL agent is a **10-dimensional vector**, which includes:
    * Vehicle queues (number of halted vehicles normalized) for each lane group.
    * Mean vehicle speed for each lane group.
    * Current active green phase.
    * Elapsed time in the current phase (normalized).

2. **Action Timing ($T=10$)**:
   The agent makes a control decision every **10 simulation seconds**. This provides enough time for a phase change (including a 3-second yellow light) to impact traffic flow significantly.

3. **Traffic Light Phases**:
   The intersection has 4 controllable green phases:
   | Phase | Lanes served | Description |
   |-------|-------------|-------------|
   | 0 | N2C_0, N2C_1, S2C_0, S2C_1 | N-S straight + right turn |
   | 1 | E2C_0, E2C_1, W2C_0, W2C_1 | E-W straight + right turn |
   | 2 | N2C_2, S2C_2 | N-S left turn only |
   | 3 | E2C_2, W2C_2 | E-W left turn only |

4. **Reward Function (Discounted Cumulative)**: 
   The reward is designed to minimize overall congestion using **Cumulative Waiting Time**. Key features:
   * **Wait Time Metric**: Instead of just counting cars, we penalize the total seconds vehicles have been stationary. This provides a continuous gradient for learning.
   * **Left-Turn Priority**: Halting vehicles in left-turn lanes incur a **1.5x penalty** multiplier to prevent starvation on turning lanes.
   * **Within-Step Discounting ($\gamma=0.99$)**: The reward for a single control action is the discounted sum of rewards across its 10-second duration: $R = \sum_{t=0}^{9} 0.99^t \cdot r_{sim, t}$.
   * **Long-term Optimization**: The agent uses a discount factor of **0.99** to prioritize long-term traffic flow over immediate clearing of a single lane.

## 🚦 Traffic Scenarios

The environment supports 6 traffic scenarios:

| Scenario | Key | Description | Volume per direction per 100s |
|----------|-----|-------------|-------------------------------|
| Normal | `normal` | Gaussian bell-curve, moderate traffic | 10 → 35 → 60 → 35 → 10 |
| Crowded All | `crowded_all` | High constant traffic from all directions | 60 straight / 25 right / 15 left |
| Crowded N-S | `crowded_ns` | Heavy N-S traffic, light E-W traffic | Heavy: 60/25/15, Light: 10/4/2 |
| Crowded E-W | `crowded_ew` | Heavy E-W traffic, light N-S traffic | Heavy: 60/25/15, Light: 10/4/2 |
| Fluctuating | `fluctuate` | Alternating dominant axis every 100s | Switches between heavy and light |
| Bernoulli | `bernoulli` | Independent Bernoulli spawn per lane | Configurable probability $p$ |

Each direction spawns three types of vehicles: straight, right turn, and left turn, with a ~60/25/15% split.

### Bernoulli Distribution Details
The `bernoulli` scenario differs from the others by using a purely stochastic spawn model:
*   **Independent Lanes**: For every simulation second, 12 independent Bernoulli trials are performed (one for each entry lane).
*   **Spawn Probability ($p$)**: There is a configurable probability $p$ (default 0.05) that a vehicle will spawn on any given lane in any given second.
*   **Realistic Routing**: Vehicles spawned in lanes 0/1 are randomly assigned Straight or Right routes, while vehicles in lane 2 are assigned Left-turn routes, matching real junction logic.
*   **Usage**: Ideal for testing the agent's ability to handle random, unpredictable traffic surges compared to the rhythmic patterns of the Gaussian scenarios.

During **training**, scenarios are randomly sampled each episode so the agent generalizes across all traffic patterns. During **evaluation**, each scenario is tested separately for a clear per-scenario comparison.

## 🛠 Prerequisites

Make sure you have [SUMO (Simulation of Urban MObility)](https://sumo.dlr.de/docs/Downloads.php) installed on your system. You also need to set the environment variable `SUMO_HOME` pointing to your SUMO installation folder.

Next, install the required Python dependencies:
```bash
pip install -r requirements.txt
```

*(Note: PyTorch is required for DQN, and Matplotlib is used to plot charts.)*

## 🚀 Execution Guide

### 1. Training the Agent

Run `train.py` to train the DQN agent. By default, it trains for 200 episodes using all traffic scenarios in headless mode. Checkpoints are saved in `checkpoints/`.

**Train on all scenarios (default):**
```bash
python train.py
```

**Train on specific scenarios only:**
```bash
python train.py --scenarios normal crowded_all
```

**Training with custom episodes:**
```bash
python train.py --episodes 100
```

**Training with Bernoulli probability (p=0.08):**
```bash
python train.py --scenarios bernoulli --prob 0.08
```

**Switching Reward Metric (e.g., to vehicle count):**
```bash
python train.py --reward-mode count
```

**Visualized training using SUMO-GUI:**
```bash
python train.py --gui
```

**Resuming from a checkpoint:**
```bash
python train.py --resume checkpoints/dqn_ep100.pt
```

*(Upon completion, `training_results.png` will be generated, showing reward improvement and loss curves.)*

### 2. Evaluating the Agent

After training (best model saved as `checkpoints/dqn_best.pt`), use `evaluate.py` to compare the RL agent against the static controller on each scenario.

**Evaluate on all scenarios:**
```bash
python evaluate.py
```

**Evaluate specific scenarios:**
```bash
python evaluate.py --scenarios normal crowded_all fluctuate
```

**Headless evaluation:**
```bash
python evaluate.py --no-gui
```

**Custom checkpoint and evaluation probability:**
```bash
python evaluate.py --model checkpoints/dqn_best.pt --runs 1 --prob 0.05
```

At the end, `evaluation_results.png` is generated with a side-by-side bar chart comparing RL vs. static controller performance on each scenario.

### 3. Visual Replay

Use `play.py` to watch the trained agent control the traffic light in real-time via SUMO-GUI.

**Play with default scenario:**
```bash
python play.py
```

**Play a specific scenario:**
```bash
python play.py --scenario crowded_all
```

**Play a specific checkpoint:**
```bash
python play.py --model checkpoints/dqn_ep200.pt
```

The console prints a step-by-step table showing the action taken, current phase, reward, and cumulative reward for each timestep.
