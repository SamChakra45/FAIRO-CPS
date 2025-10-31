import numpy as np
import config as config
import random

class SmartLearningEnvironment:
    #Implements Type 3 HITL application: Smart Learning with VR 

    def __init__(self, num_students=config.N_STUDENTS):
        self.num_students = num_students
        self.satisfaction_records = np.array([
            [0.9, 0.1],
            [0.5, 0.5],
            [0.2, 0.8]
        ])
        
        for i in range(self.num_students):
            self.satisfaction_records[i] = self.satisfaction_records[i] / np.linalg.norm(self.satisfaction_records[i])

        self.student_states = np.zeros(self.num_students, dtype=int)
        self.student_profiles = list(range(self.num_students))
        self._init_transition_mdps()

    def _init_transition_mdps(self):
        #Initialize MDP transition matrices for each student profile.

        self.transitions = {}

        # Profile 0: High VR tolerance (most tolerant)
        self.transitions[0] = {
            1: {0: 2, 1: 3, 2: 2},  
            2: {0: 3, 1: 4, 2: 3},
            3: {0: 4, 1: 6, 2: 4},
            4: {0: 5, 1: 6, 2: 5},
            5: {0: 6, 1: 7, 2: 6},
            6: {0: 7, 1: 8, 2: 7},
            7: {0: 8, 1: 8, 2: 7},
            8: {0: 8, 1: 8, 2: 7}    
        }

        # Profile 1: Medium VR tolerance
        self.transitions[1] = {
            1: {0: 2, 1: 2, 2: 3},
            2: {0: 3, 1: 3, 2: 4},
            3: {0: 4, 1: 5, 2: 5},
            4: {0: 5, 1: 6, 2: 5},
            5: {0: 6, 1: 6, 2: 6},
            6: {0: 7, 1: 7, 2: 7},
            7: {0: 8, 1: 7, 2: 8},
            8: {0: 8, 1: 7, 2: 8}
        }

        # Profile 2: Low VR tolerance (least tolerant)
        self.transitions[2] = {
            1: {0: 3, 1: 2, 2: 3},
            2: {0: 4, 1: 2, 2: 4},
            3: {0: 5, 1: 3, 2: 6},
            4: {0: 6, 1: 3, 2: 6},
            5: {0: 7, 1: 4, 2: 7},
            6: {0: 8, 1: 4, 2: 8},
            7: {0: 8, 1: 5, 2: 8},
            8: {0: 8, 1: 6, 2: 8}
        }

    def _calculate_cosine_similarity(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        return dot_product / norm_product if norm_product > 1e-10 else 0

    def get_current_desired_actions(self):
        desired_actions = []

        for i in range(self.num_students):
            current_state = self.student_states[i]
            profile = self.student_profiles[i]
            best_action = 0
            best_next_state = 1

            for action in range(3):
                next_state = self.transitions[profile][current_state][action]
                if next_state > best_next_state:
                    best_next_state = next_state
                    best_action = action

            desired_actions.append(best_action)

        return desired_actions

    def calculate_action_effects(self, desired_actions):
        effects = np.zeros(self.num_students)

        for i in range(self.num_students):
            action_i = desired_actions[i]
            total_effect = 0

            for j in range(self.num_students):
                if i != j:
                    current_state_j = self.student_states[j]
                    profile_j = self.student_profiles[j]
                    next_state_j = self.transitions[profile_j][current_state_j][action_i]
                    effect_on_j = config.STATE_VALUES[next_state_j] - config.STATE_VALUES[current_state_j]
                    total_effect += effect_on_j

            effects[i] = total_effect

        return effects

    def _calculate_fairness_state(self):
        s_t = np.zeros(self.num_students)

        for i in range(self.num_students):
            similarities = []
            for j in range(self.num_students):
                if i != j:
                    sim = self._calculate_cosine_similarity(
                        self.satisfaction_records[i],
                        self.satisfaction_records[j]
                    )
                    similarities.append(sim)
            s_t[i] = np.mean(similarities) if similarities else 1.0
        return s_t

    def get_augmented_state(self):
        s_t = self._calculate_fairness_state()
        min_index = np.argmin(s_t)
        u_minus_components = self.satisfaction_records[:, 1]
        l_flag = 1 if u_minus_components[min_index] == np.max(u_minus_components) else 0
        return np.append(s_t, l_flag)

    def step(self, global_action):
        #Applies global action (categorical: 0, 1, or 2).

        learning_experiences = np.zeros(self.num_students)
        
        for i in range(self.num_students):
            current_state = self.student_states[i]
            profile = self.student_profiles[i]
            
            if np.random.random() < 0.2 and current_state > 1:
                current_state = max(1, current_state - 1)
                self.student_states[i] = current_state
            
            next_state = self.transitions[profile][current_state][global_action]

            state_improvement = config.STATE_VALUES[next_state] - config.STATE_VALUES[current_state]
            state_value = config.STATE_VALUES[next_state]
            learning_experience = state_improvement + state_value
            learning_experiences[i] = learning_experience

            self.student_states[i] = next_state

            # Check satisfaction: LE >= 0.5 means satisfied
        if learning_experience >= 0.5:  
            self.satisfaction_records[i, 0] += config.SATISFACTION_INCREMENT
        else:
            self.satisfaction_records[i, 1] += config.SATISFACTION_INCREMENT
            norm = np.linalg.norm(self.satisfaction_records[i])
            if norm > 0:
                self.satisfaction_records[i] = self.satisfaction_records[i] / norm

        next_state = self.get_augmented_state()
        done = False
        return next_state, learning_experiences, done

    def get_satisfaction_records(self):
        return self.satisfaction_records.copy()

    def reset(self):
        self.satisfaction_records = np.array([
            [0.9, 0.1],
            [0.5, 0.5],
            [0.2, 0.8]
        ])
        
        for i in range(self.num_students):
            self.satisfaction_records[i] = self.satisfaction_records[i] / np.linalg.norm(self.satisfaction_records[i])
        
        self.student_states = np.array([6, 4, 2])
        return self.get_augmented_state()

