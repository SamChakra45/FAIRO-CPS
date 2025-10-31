# environment.py

import numpy as np
import config
import random

class SmartHomeEnvironment:
    #Implements Type 1 HITL application: Smart Home HVAC 

    def __init__(self, num_humans=config.N_HUMANS):
        self.num_humans = num_humans
        self.satisfaction_records = np.ones((self.num_humans, 2)) 
        for i in range(self.num_humans):
            self.satisfaction_records[i] = self.satisfaction_records[i] / np.linalg.norm(self.satisfaction_records[i])

        self.activities = list(config.SETPOINTS.keys())
        self.human_activities = [random.choice(self.activities) for _ in range(self.num_humans)]

    def _calculate_cosine_similarity(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        return dot_product / norm_product if norm_product > 1e-10 else 0

    def get_current_desires(self):
        desires = [config.SETPOINTS[activity] for activity in self.human_activities]
        return np.array(desires)

    def _calculate_fairness_state(self):
        """
        Calculate the fairness state s_t = (L_1, L_2, ..., L_N) as per Equation 4.
        L_i measures how close human i's satisfaction record is to others.
        """
        s_t = np.zeros(self.num_humans)
        for i in range(self.num_humans):
            similarities = []
            for j in range(self.num_humans):
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
        """
        Applies a global action (HVAC setpoint), updates satisfaction records,
        and returns the next state.
        """
        desired_setpoints = self.get_current_desires()
        for i in range(self.num_humans):
            error = np.abs(global_action - desired_setpoints[i])

            # Check if human i is satisfied (within threshold)
            if error <= config.SATISFACTION_THRESHOLD:
                self.satisfaction_records[i, 0] += config.SATISFACTION_INCREMENT
            else:
                self.satisfaction_records[i, 1] += config.SATISFACTION_INCREMENT

            norm = np.linalg.norm(self.satisfaction_records[i])
            if norm > 0:
                self.satisfaction_records[i] = self.satisfaction_records[i] / norm

        for i in range(self.num_humans):
            if random.random() < 0.1:
                self.human_activities[i] = random.choice(self.activities)

        next_state = self.get_augmented_state()
        done = False
        return next_state, done

    def get_satisfaction_records(self):
        return self.satisfaction_records.copy()

    def reset(self):
        self.satisfaction_records = np.ones((self.num_humans, 2))
        for i in range(self.num_humans):
            self.satisfaction_records[i] = self.satisfaction_records[i] / np.linalg.norm(self.satisfaction_records[i])
        self.human_activities = [random.choice(self.activities) for _ in range(self.num_humans)]

        return self.get_augmented_state()
