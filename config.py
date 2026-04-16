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
