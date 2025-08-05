import torch
import os
ROOT = os.path.dirname(os.path.dirname(__file__))
# Hyperparameters

# === Training ===
BATCH_SIZE = 128  # Mini-batch size

# === TD3 Hyperparameter ===
DISCOUNT = 0.99  # Discount factor
TAU = 0.005  # Soft target update variable
POLICY_NOISE = 0.2  # Noise added to target policy
NOISE_CLIP = 0.5  # Clamping for policy noise
POLICY_FREQ = 2  # Frequency of Actor updates

# === Exploration-Parameter ===
EXPL_NOISE = 1  # Initial exploration noise starting value in range [expl_min ... 1]
EXPL_DECAY_STEPS = (500000)  # Steps over which the initial exploration noise decays
EXPL_MIN = 0.1  # Exploration noise after decay

# === Environment- und Model architecture ===
HISTORY_LENGTH = 40 # Saving the last 40 states
# RUN 11
#MODEL_DIM = 384
#N_HEADS = 12
#N_ENCODER_LAYERS = 6

MODEL_DIM = 336
N_HEADS = 12
N_ENCODER_LAYERS = 5

# === Runtime- und Memory Config ===
MAX_TIMESTEPS = 5e6  # Maximum number of steps to perform
MAX_EP = 500  # Maximum number of steps per episode
BUFFER_SIZE = 1000000  # Replay buffer size
EVAL_FREQ = 5e3  # After how many steps to perform the evaluation
EVAL_EP = 10  # Number of episodes for evaluation


# === Other options ===
SEED = 0  # Random seed number
FILE_NAME = "td3_velodyne"  # File name to store policy
SAVE_MODEL = True  # Whether to save the model
LOAD_MODEL = False  # Whether to load a stored model
RANDOM_NEAR_OBSTACLE = True  # Take random actions near obstacles

# Hyperparameters and environment constants
GOAL_REACHED_DIST = 0.5
COLLISION_DIST = 0.35
TIME_DELTA = 0.2
ENVIRONMENT_DIM = 20
ROBOT_DIM = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
SUMMARY_WRITER_RUN_LOG = os.path.join(ROOT, 'evaluation', 'run')
RESULTS_DIR = os.path.join(ROOT, 'evaluation', 'run')
PYTORCH_MODELS_DIR = os.path.join(ROOT, 'evaluation', 'models')