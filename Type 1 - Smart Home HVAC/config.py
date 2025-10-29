# config.py

# --- Environment Settings ---
N_HUMANS = 3  # The paper simulates 3 humans for Type 1 application

# Map of activities to desired setpoints in Fahrenheit (Section 5.2)
SETPOINTS = {
    'sleeping': 62.0,
    'working_from_home': 67.0,
    'domestic_activity': 72.0,
    'relaxed_activity': 77.0
}

# Satisfaction threshold for HVAC (Section 5.3.1)
SATISFACTION_THRESHOLD = 2.5  # degrees Fahrenheit

# --- FAIRO Algorithm Settings (Section 4) ---
DELTA = 0.05  # Weight adjustment step size (Equation 12)
ZETA = 0.5  # Fairness-utility tradeoff parameter (Equation 14)
SATISFACTION_INCREMENT = 0.01  # Increment value for satisfaction counters (Equation 2)

# --- DQN Agent Settings (Section 4.5.1) ---
BUFFER_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.99  # Discount factor (Equation 10)
EPSILON_START = 0.9
EPSILON_END = 0.05
EPSILON_DECAY = 1000
LEARNING_RATE = 1e-4  # Learning rate α (Equation 10)
TARGET_UPDATE = 10

# --- Training Settings ---
NUM_EPISODES = 500
STEPS_PER_EPISODE = 200
