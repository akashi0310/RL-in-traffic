import os

# -- Path Constants --
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
SUMO_CFG_PATH = os.path.join(DATA_DIR, "new_sim.sumocfg")
LOGS_DIR = os.path.join(BASE_DIR, "logs")


# -- RL Settings --
REWARD_SCALE = 1e-3

# -- Environment Settings --
STEP_SIZE = 1
MAX_STEPS = 150
EVAL_STEPS = 300
EVAL_DURATION = 500
YELLOW_DURATION = 3
MIN_GREEN = 0
SWITCH_PENALTY = 5.0

# -- Agent Defaults --
HIDDEN_SIZE = 200
BATCH_SIZE = 128
BUFFER_CAPACITY = 2000
LR = 5e-3
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995
TARGET_UPDATE = 100
USE_DDQN = False    

# -- Model Selection --
MODEL_TYPE = "MLP"  # "MLP" or "RNN"

# -- RNN Settings --
SEQUENCE_LENGTH = 10
RNN_HIDDEN_SIZE = 128
