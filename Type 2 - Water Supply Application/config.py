N_HOUSEHOLDS = 3  

DEMAND_PROFILES = {
    'household_1': {'base_demand': 50, 'peak_multiplier': 2.0},
    'household_2': {'base_demand': 45, 'peak_multiplier': 1.8},
    'household_3': {'base_demand': 55, 'peak_multiplier': 2.2}
}

RESOURCE_MULTIPLIER = 2  
TANK_CAPACITY = 200

SATISFACTION_THRESHOLD_RATIO = 0.7  

DELTA = 0.05 
ZETA = 0.5  
SATISFACTION_INCREMENT = 0.05  

BUFFER_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON_END = 0.05
EPSILON_START = 0.95  
LEARNING_RATE = 5e-4  
TARGET_UPDATE = 5 

NUM_EPISODES = 500
STEPS_PER_EPISODE = 200
