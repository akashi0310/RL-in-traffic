import os

# -- Path Constants --
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
SUMO_CFG_PATH = os.path.join(DATA_DIR, "new_sim.sumocfg")

# -- RL Settings --
REWARD_SCALE = 1e-3

# -- Environment Settings --
STEP_SIZE = 1
MAX_STEPS = 150
YELLOW_DURATION = 3
MIN_GREEN = 0

# -- Agent Defaults --
HIDDEN_SIZE = 400
BATCH_SIZE = 64
BUFFER_CAPACITY = 20_000
LR = 5e-4
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.999
TARGET_UPDATE = 100
