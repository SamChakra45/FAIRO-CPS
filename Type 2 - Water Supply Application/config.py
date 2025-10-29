# config_type2.py
# Configuration for Type 2 HITL Application: Multi-Household Water Supply

# --- Environment Settings ---
N_HOUSEHOLDS = 3  # Number of households sharing water resource

# Water demand patterns (gallons per hour) - simplified profiles
# In real implementation, these would be time-varying based on activity patterns
DEMAND_PROFILES = {
    'household_1': {'base_demand': 50, 'peak_multiplier': 2.0},
    'household_2': {'base_demand': 45, 'peak_multiplier': 1.8},
    'household_3': {'base_demand': 55, 'peak_multiplier': 2.2}
}

# Water resource availability (time-varying, insufficient to meet all demands)
RESOURCE_MULTIPLIER = 2  # Total resource = multiplier * max(all demands)

# Tank capacity per household (gallons)
TANK_CAPACITY = 200

# Satisfaction threshold: if (supply + reserve) / demand >= threshold, satisfied
SATISFACTION_THRESHOLD_RATIO = 0.7  # 80% of demand met

# --- FAIRO Algorithm Settings ---
DELTA = 0.05  # Weight adjustment step size
ZETA = 0.5  # Fairness-utility tradeoff parameter
SATISFACTION_INCREMENT = 0.05  # Increment for satisfaction counters

# --- DQN Agent Settings ---
BUFFER_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON_END = 0.05
EPSILON_START = 0.95  # More exploration initially
EPSILON_DECAY = 2000  # Slower decay
LEARNING_RATE = 5e-4  # Faster learning
TARGET_UPDATE = 5  # More frequent target updates


# --- Training Settings ---
NUM_EPISODES = 500
STEPS_PER_EPISODE = 200
