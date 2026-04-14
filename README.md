# RL-in-Traffic: Deep Q-Network for Intelligent Traffic Control

A Reinforcement Learning project focused on optimizing traffic light signal control at intersections using **Deep Q-Networks (DQN)**. This project leverages **SUMO (Simulation of Urban MObility)** and **TraCI** to simulate real-world traffic dynamics and train an agent that intelligently adapts to varying traffic patterns.

---

## 🗂 Project Structure

```text
📦 RL-in-traffic
 ┣ 📂 agent
 ┃ ┗ 📜 dqn_agent.py      # Core DQN implementation (Model, Agent, Replay Buffer)
 ┣ 📂 env
 ┃ ┗ 📜 traffic_env.py    # SUMO environment wrapper specialized for RL
 ┣ 📂 utils
 ┃ ┣ 📜 plots.py          # Centralized plotting & visualization helpers
 ┃ ┗ 📜 __init__.py
 ┣ 📜 config.py           # Unified project configuration and hyper-parameters
 ┣ 📜 train.py            # Main training script with scenario cycling
 ┣ 📜 evaluate.py         # Performance benchmarking (RL vs. Static Controller)
 ┣ 📜 play.py             # Visual replay of trained agents in SUMO-GUI
 ┣ 📜 requirements.txt    # Python dependencies
 ┗ 📜 README.md           # Project documentation
```

---

## 🚦 System Logic & Architecture

### 1. State Representation
The agent observes a **10-dimensional state vector** every decision step:
- **Queue Lengths (4-dim)**: Normalized number of halted vehicles for each lane group.
- **Wait Times (4-dim)**: Normalized lead-vehicle waiting time for each lane group.
- **Active Phase (1-dim)**: Index of the current green phase.
- **Phase Duration (1-dim)**: Elapsed time in the current phase (normalized).

### 2. Action Space
The agent chooses from 4 possible actions every $T=10$ simulation seconds:
- `0`: **STAY** (Maintain current green phase)
- `1`: **FORWARD** (Transition to next circular phase)
- `2`: **DIAGONAL** (Transition to opposite circular phase)
- `3`: **BACKWARD** (Transition to previous circular phase)

### 3. Reward Function
Multi-objective reward shaping designed to balance queue reduction and wait-time minimization:
- **Wait Time**: Penalizes cumulative seconds vehicles are stationary.
- **Queue Count**: Penalizes the existence of halting vehicles.
- **Harmonic Mode**: (Default for training) Combines these metrics using an exponential surcharge to prevent extremely long delays (starvation).

---

## 🌪 Environment Scenarios

The system uses **Bernoulli Spawning** to generate traffic. You can train or evaluate across several traffic patterns:

| Scenario | Distribution | Use Case |
| :--- | :--- | :--- |
| `uniform` | Symmetric | Balanced traffic on all directions. |
| `horizontal` | West-East dominated | High-flow horizontal artery. |
| `vertical` | North-South dominated | High-flow vertical artery. |
| `alternate` | Dynamic / Shifting | Periodic shifts between heavy directions. |

---

## 🛠 Setup & Installation

1. **Install SUMO**: Download and install from [the official SUMO page](https://sumo.dlr.de/docs/Downloads.php).
2. **Environment Variable**: Ensure `SUMO_HOME` is set to your SUMO installation directory.
3. **Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Getting Started

### 🏋️ Training the Agent
Train the DQN agent with scenario cycling (randomizing traffic patterns each episode).
```bash
# Default training
python train.py

# Custom episodes and spawn probability
python train.py --episodes 500 --prob 0.08
```

### 📊 Evaluating Performance
Compare your trained model (`checkpoints/dqn_best.pt`) against a standard static-timed controller.
```bash
python evaluate.py --runs 5 --scenario alternate
```

### 📺 Visual Replay
Watch the agent in action using the SUMO-GUI.
```bash
python play.py --model checkpoints/dqn_best.pt
```

---

## 📈 Visualizations
- **Training Plots**: Rewards and losses are saved as `results_TIMESTAMP.png`.
- **Comparison Plots**: Benchmark results are saved as `evaluation_results.png`.

---

> [!TIP]
> Use the `--gui` flag in `train.py` or `evaluate.py` to watch the simulation in real-time, although it will significantly slow down the process.
