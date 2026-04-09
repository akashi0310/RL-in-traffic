# Traffic Light Control using Deep Q-Network (DQN)

This project implements a Reinforcement Learning agent based on Deep Q-Network (DQN) to optimize traffic light control at an intersection. The simulation environment is built on SUMO (Simulation of Urban MObility) and interfaces with the Python code via TraCI (Traffic Control Interface). 

By intelligently switching traffic light phases based on real-time queues and vehicle speeds, the RL agent minimizes overall waiting time compared to a standard static time-based traffic light controller.

## 🗂 Project Structure

```bash
📦 project
 ┣ 📂 agent
 ┃ ┣ 📜 __init__.py
 ┃ ┗ 📜 dqn_agent.py         # Defines the Deep Q-Network and Replay Buffer.
 ┣ 📂 data
 ┃ ┣ 📜 intersection.net.xml # SUMO network topology (lanes, junctions).
 ┃ ┣ 📜 sim_flow.rou.xml     # Route/Flow definitions generating vehicle traffic.
 ┃ ┣ 📜 sim_rl.sumocfg       # Configuration file linking network & routes for SUMO.
 ┃ ┗ ...                     # Other SUMO related assets.
 ┣ 📂 env
 ┃ ┣ 📜 __init__.py
 ┃ ┗ 📜 traffic_env.py       # Custom Gym-like Env wrapping SUMO via TraCI.
 ┣ 📜 train.py               # Main training loop for the RL agent.
 ┣ 📜 evaluate.py            # Evaluation script (RL Agent vs. Static Controller).
 ┣ 📜 requirements.txt       # Project Python dependencies.
 ┗ 📜 README.md              # This documentation file.
```

## 🔄 System Workflow & Logic

1. **Environment (SUMO via TraCI)**: 
   The intersection `J8` receives approaching traffic from 4 directions. The state observed by the RL agent is a **10-dimensional vector**, which includes:
   * Vehicle queues (number of halted vehicles normalized) for each incoming lane.
   * Mean vehicle speed for each incoming lane.
   * Current active green phase.
   * Elapsed time in the current phase.

2. **Agent (DQN)**: 
   The agent processes the 10-dimensional state through a Two-Hidden-Layer Multi-Layer Perceptron (MLP) and predicts Q-values for two possible actions:
   * `0`: Keep the current green phase.
   * `1`: Switch to the next green phase (the environment handles the transition yellow phase automatically).
   
3. **Reward Function**: 
   The objective is to minimize traffic jam duration. Therefore, the reward returned by the environment after every step is the **Negative Sum of waiting times** across all evaluated lanes.

## 🛠 Prerequisites

Make sure you have [SUMO (Simulation of Urban MObility)](https://sumo.dlr.de/docs/Downloads.php) installed on your system. You also need to set the environment variable `SUMO_HOME` pointing to your SUMO installation folder.

Next, install the required Python dependencies:
```bash
pip install -r requirements.txt
```

*(Note: PyTorch is required for DQN, and Matplotlib is used to plot charts.)*

## 🚀 Execution Guide

### 1. Training the Agent

Run the `train.py` script to train the DQN agent. By default, it will train for 200 episodes in headless mode (no visual interface, to speed up training). Your models will be saved in a new `checkpoints/` directory.

**Standard Headless Training (Fastest):**
```bash
python train.py
```

**Training with Custom Episodes:**
```bash
python train.py --episodes 300
```

**Visualized Training using SUMO-GUI:**
```bash
python train.py --gui
```

**Resuming from a Checkpoint:**
```bash
python train.py --resume checkpoints/dqn_ep100.pt
```

*(Upon completion, a graph `training_results.png` will be generated, illustrating the reward improvement and loss.)*

### 2. Evaluating the Agent

After securing a trained model (usually saved as `checkpoints/dqn_best.pt`), use `evaluate.py` to compare its performance against the classic Static Phase Controller.

**Standard Evaluation (Provides SUMO-GUI automatically):**
```bash
python evaluate.py
```

**Headless Evaluation:**
```bash
python evaluate.py --no-gui
```

**Custom Checkpoint and Runs Selection:**
```bash
python evaluate.py --model checkpoints/dqn_ep150.pt --runs 5
```

At the end of the evaluation phase, `evaluation_results.png` will be generated. This picture includes a comparative bar chart and distribution box plots to clearly observe the percentage improvement achieved by the Deep Q-Network agent!
