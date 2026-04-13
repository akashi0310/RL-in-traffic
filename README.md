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

   **Reward modes** (`--reward-mode`):
   | Mode | Per-lane penalty | Purpose |
   |------|------------------|---------|
   | `wait` | Waiting time of the lead (queue-head) vehicle. | Smooth gradient; focuses on the car that has been stuck the longest. |
   | `count` (**evaluate default**) | Number of halting vehicles on the lane. | Simple, interpretable congestion signal — used for evaluation so the comparison metric is direct queue length. |
   | `harmonic` (**train default**) | $w \cdot \exp\!\left(\dfrac{w - T_{hard}}{T_{hard} - T_{soft}}\right) + 2 \cdot \text{halting}$, where $w$ is lead-vehicle wait, $T_{soft}=80$s, $T_{hard}=100$s. | Blend of wait + queue with an exponential surcharge as the lead car approaches the hard red-time limit — discourages starving any single lane. Used for training to shape learning. |

## 🚦 Traffic Model — Bernoulli Spawning

The environment uses a **Bernoulli spawn model** for traffic generation. The `bernoulli.rou.xml` file is regenerated automatically at the start of every episode based on the current probability parameter.

*   **Independent Lanes**: For every simulation second, 12 independent Bernoulli trials are performed (one for each entry lane).
*   **Spawn Probability ($p$)**: There is a configurable probability $p$ (default 0.05) that a vehicle will spawn on any given lane in any given second. The `--prob` flag sets the **high** value; low lanes use `p/2`.
*   **Realistic Routing**: Vehicles spawned in lanes 0/1 are randomly assigned Straight or Right routes (70/30 split), while vehicles in lane 2 are assigned Left-turn routes, matching real junction logic.
*   **Mixed Vehicle Types**: Cars, trucks, and buses are spawned with a 60/20/20 mix.

### Scenarios

Selected via `--scenario` (train/evaluate). All scenarios use the same `--prob` as the high-side probability; low = `prob/2`.

| Scenario     | Horizontal (E/W) | Vertical (N/S) | Description |
|--------------|-------------------|----------------|-------------|
| `uniform`    | high              | high           | Symmetric load on all directions. |
| `horizontal` | high              | low            | E–W dominates. |
| `vertical`   | low               | high           | N–S dominates. |
| `alternate`  | high↔low each quarter | low↔high each quarter | Episode is split into 4 segments; the bias flips between H-heavy and V-heavy. |

Training cycles through all four scenarios round-robin by default (one per episode); pass `--scenario <name>` to pin to one.

### Evaluation Mode (Drain Phase)

`evaluate.py` runs the env with `eval_mode=True`. Spawning still ends at 500 s (the training horizon) and the reward metric is only accumulated over that window, so scores remain comparable. After 500 s the simulation keeps stepping with no new spawns — the agent continues to act — until the network is cleared (or a safety cap is hit). This produces a clean visual endpoint without skewing the comparison.

## 🛠 Prerequisites

Make sure you have [SUMO (Simulation of Urban MObility)](https://sumo.dlr.de/docs/Downloads.php) installed on your system. You also need to set the environment variable `SUMO_HOME` pointing to your SUMO installation folder.

Next, install the required Python dependencies:
```bash
pip install -r requirements.txt
```

*(Note: PyTorch is required for DQN, and Matplotlib is used to plot charts.)*

## 🚀 Execution Guide

### 1. Training the Agent

Run `train.py` to train the DQN agent. Checkpoints are saved in `checkpoints/`.

**Default training:**
```bash
python train.py
```

**Custom episodes:**
```bash
python train.py --episodes 300
```

**Custom Bernoulli probability (p=0.08):**
```bash
python train.py --prob 0.08
```

**Pin to a single scenario (default cycles all):**
```bash
python train.py --scenario horizontal
```

**Switching Reward Metric (train default is `harmonic`):**
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

After training (best model saved as `checkpoints/dqn_best.pt`), use `evaluate.py` to compare the RL agent against the static controller.

**Headless evaluation:**
```bash
python evaluate.py --no-gui
```

**Custom checkpoint, runs, and probability:**
```bash
python evaluate.py --model checkpoints/dqn_best.pt --runs 5 --prob 0.05
```

**Evaluate on a specific scenario:**
```bash
python evaluate.py --scenario alternate --prob 0.08
```

At the end, `evaluation_results.png` is generated with a bar chart comparing RL vs. static controller performance.

### 3. Visual Replay

Use `play.py` to watch the trained agent control the traffic light in real-time via SUMO-GUI.

**Default replay:**
```bash
python play.py
```

**Custom checkpoint:**
```bash
python play.py --model checkpoints/dqn_ep200.pt
```

**Custom Bernoulli probability:**
```bash
python play.py --prob 0.08
```

The console prints a step-by-step table showing the action taken, current phase, reward, and cumulative reward for each timestep.
