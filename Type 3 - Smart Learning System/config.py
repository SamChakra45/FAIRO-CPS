N_STUDENTS = 3  
N_STATES = 8

ACTIONS = {
    'give_break': 0,     
    'enable_vr': 1,    
    'disable_vr': 2     
}

STUDENT_PROFILES = ['high_tolerance', 'medium_tolerance', 'low_tolerance']

STATE_VALUES = {
    8: 1.0,   
    7: 0.857,
    6: 0.714,
    5: 0.571,
    4: 0.428,
    3: 0.285,
    2: 0.142,
    1: 0.0    
}

DELTA = 0.05
ZETA = 0.5  
SATISFACTION_INCREMENT = 0.03  

BUFFER_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON_START = 0.9
EPSILON_END = 0.05
EPSILON_DECAY = 1000
LEARNING_RATE = 1e-4
TARGET_UPDATE = 10

NUM_EPISODES = 500
STEPS_PER_EPISODE = 200
