# config.py

# --- Environment Settings ---
N_HUMANS = 5  # Number of humans in the environment

# --- FAIRO Algorithm Settings ---
# The small value to adjust the weight w_i by
DELTA = 0.05

# --- DQN Agent Settings ---
BUFFER_SIZE = 10000   # Replay buffer capacity
BATCH_SIZE = 64       # Batch size for learning
GAMMA = 0.99          # Discount factor for future rewards
EPSILON_START = 0.9   # Starting value of epsilon for e-greedy policy
EPSILON_END = 0.05    # Minimum value of epsilon
EPSILON_DECAY = 1000  # Epsilon decay rate
LEARNING_RATE = 1e-4  # Learning rate for the Adam optimizer
TARGET_UPDATE = 10    # How often to update the target network (in episodes)

# --- Training Settings ---
NUM_EPISODES = 500    # Total number of episodes to run