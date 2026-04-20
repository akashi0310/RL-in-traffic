# RL-in-Traffic: Deep Q-Network for Intelligent Traffic Control

A Reinforcement Learning project focused on optimizing traffic light signal control at intersections using **Deep Q-Networks (DQN)**. This project leverages **SUMO (Simulation of Urban MObility)** and **TraCI** to simulate real-world traffic dynamics and train an agent that intelligently adapts to varying traffic patterns.

---

## 🗂 Project Structure

```text
📦 RL-in-traffic
 ┣ 📂 agent
 ┃ ┗ 📜 dqn_agent.py      # DQN implementation (MLP or RNN types)
 ┣ 📂 env
 ┃ ┗ 📜 traffic_env.py    # SUMO environment wrapper (State/Action/Reward)
 ┣ 📂 logs
 ┃ ┗ 📜 eval_steps_...csv # Automated CSV logs from evaluation runs
 ┣ 📂 utils
 ┃ ┣ 📜 plots.py          # Centralized plotting & visualization helpers
 ┃ ┗ 📜 __init__.py
 ┣ 📜 config.py           # Hyper-parameters and architecture selection
 ┣ 📜 train.py            # Main training script (with scenario cycling)
 ┣ 📜 evaluate.py         # Performance benchmarking (RL vs. Static)
 ┣ 📜 play.py             # Visual replay in SUMO-GUI
 ┣ 📜 requirements.txt    # Python dependencies
 ┗ 📜 README.md           # Project documentation
```

---

## 🚦 System Logic & Architecture

### 1. Neural Architectures
The project supports two model types, configurable in `config.py` (**`MODEL_TYPE`**):
- **MLP (Default)**: Standard feedforward network using the current state.
- **RNN (LSTM)**: Recurrent network that utilizes a history of states and actions for sequence-aware decision making.

### 2. State Representation
The agent observes a **6-dimensional state vector**:
- **Queue Lengths (4-dim)**: Normalized sum of halting and pending vehicles for each lane group (N-S, E-W, N-S Left, E-W Left).
- **Active Phase (1-dim)**: Index of the current green phase (0-3).
- **Phase Duration (1-dim)**: Elapsed time in the current phase, normalized by 60s (capped at 1.0).

### 3. Action Space
The agent chooses from **2 possible actions** at each decision point:
- `0`: **STAY** (Remain in the current green phase)
- `1`: **NEXT** (Switch to the next circular green phase via a yellow transition)

### 4. Reward Function
The environment uses a **Quadratic Penalty** based on congestion:
- **Congestion Penalty**: Calculated as the negative sum of the squares of halting vehicles across all lanes ($R = -\sum \text{halt}^2$).
- **Switching Penalty**: A fixed negative reward is applied when switching phases to discourage excessive jitter.

---

## 🌪 Environment Scenarios

Traffic flows are generated using **Bernoulli Spawning**. The following scenarios are supported:

| Scenario | Distribution | Use Case |
| :--- | :--- | :--- |
| `uniform` | Symmetric | Balanced traffic on all directions. |
| `horizontal` | West-East dominated | High-flow horizontal artery. |
| `vertical` | North-South dominated | High-flow vertical artery. |
| `alternate` | Dynamic / Shifting | Periodic shifts between heavy directions. |

---

## 🛠 Setup & Installation

1. **Install SUMO**: [Official SUMO Downloads](https://sumo.dlr.de/docs/Downloads.php).
2. **Set `SUMO_HOME`**: Point this environment variable to your SUMO installation folder.
3. **Install Repo Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Getting Started

### 🏋️ Training
```bash
python train.py --episodes 200 --prob 0.05
```

### 📊 Evaluation & Logging
Benchmarks the RL agent against a static controller.
```bash
python evaluate.py --runs 1 --scenario uniform
```
> [!NOTE]
> Evaluation automatically saves step-by-step data (States, Actions, Timestamps) to the `/logs` directory for deep analysis.

### 📺 Visual Replay
```bash
python play.py --model checkpoints/dqn_best.pt
```

---

## 📈 Visualizations
- **Training Plots**: Episode rewards/losses saved as `results_TIMESTAMP.png`.
- **Comparison Plots**: Performance comparison saved as `evaluation_results.png`.
- **CSV Logs**: Raw step data saved in `/logs/eval_steps_...csv`.
