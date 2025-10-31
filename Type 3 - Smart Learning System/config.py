# config_type3.py
# Configuration for Type 3 HITL Application: Smart Learning with VR

# --- Environment Settings ---
N_STUDENTS = 3  # Number of students in shared learning environment

# Learning states (8 states based on Alert, Fatigue, Vertigo levels)
# State encoding: (Alert, Not Fatigued, No Vertigo) -> higher is better
# State 8: (1, 1, 1) - best state
# State 1: (0, 0, 0) - worst state
N_STATES = 8

# Adaptation actions (categorical)
ACTIONS = {
    'give_break': 0,      # a_1: Give student a small break
    'enable_vr': 1,       # a_2: Enable VR immersive experience
    'disable_vr': 2       # a_3: Disable VR, use regular display
}

# Student profiles (VR tolerance levels)
# Profile 1: High VR tolerance
# Profile 2: Medium VR tolerance  
# Profile 3: Low VR tolerance
STUDENT_PROFILES = ['high_tolerance', 'medium_tolerance', 'low_tolerance']

# State values (for learning experience calculation)
STATE_VALUES = {
    8: 1.0,   # Best state
    7: 0.857,
    6: 0.714,
    5: 0.571,
    4: 0.428,
    3: 0.285,
    2: 0.142,
    1: 0.0    # Worst state
}

# --- FAIRO Algorithm Settings ---
DELTA = 0.05  # Weight adjustment step size
ZETA = 0.5  # Fairness-utility tradeoff parameter
SATISFACTION_INCREMENT = 0.03  # Increment for satisfaction counters

# --- DQN Agent Settings ---
BUFFER_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON_START = 0.9
EPSILON_END = 0.05
EPSILON_DECAY = 1000
LEARNING_RATE = 1e-4
TARGET_UPDATE = 10

# --- Training Settings ---
NUM_EPISODES = 500
STEPS_PER_EPISODE = 200
