# environment.py

import numpy as np
import config
import random

class SmartHomeEnvironment:
    """
    Implements Type 1 HITL application: Smart Home HVAC (Section 5)
    Multiple humans share a house with a single HVAC system.
    """

    def __init__(self, num_humans=config.N_HUMANS):
        self.num_humans = num_humans

        # Satisfaction history records: c_i = (u^+_i, u^-_i) as per Equation 1
        # These are counters that track satisfaction/dissatisfaction history
        self.satisfaction_records = np.ones((self.num_humans, 2))  # Start with (1, 1)
        # Normalize to unit vectors
        for i in range(self.num_humans):
            self.satisfaction_records[i] = self.satisfaction_records[i] / np.linalg.norm(self.satisfaction_records[i])

        self.activities = list(config.SETPOINTS.keys())
        # Give each human a random starting activity
        self.human_activities = [random.choice(self.activities) for _ in range(self.num_humans)]

    def _calculate_cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors (Equation 3)"""
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        return dot_product / norm_product if norm_product > 1e-10 else 0

    def get_current_desires(self):
        """
        Simulates the 'context-aware-engine' (Line 3 of Algorithm 1).
        Returns the vector of desired temperature setpoints d_t.
        """
        desires = [config.SETPOINTS[activity] for activity in self.human_activities]
        return np.array(desires)

    def _calculate_fairness_state(self):
        """
        Calculate the fairness state s_t = (L_1, L_2, ..., L_N) as per Equation 4.
        L_i measures how close human i's satisfaction record is to others.
        """
        s_t = np.zeros(self.num_humans)

        for i in range(self.num_humans):
            # Calculate average cosine similarity with all other humans
            similarities = []
            for j in range(self.num_humans):
                if i != j:
                    sim = self._calculate_cosine_similarity(
                        self.satisfaction_records[i], 
                        self.satisfaction_records[j]
                    )
                    similarities.append(sim)

            # L_i is the average cosine similarity (Equation 3)
            s_t[i] = np.mean(similarities) if similarities else 1.0

        return s_t

    def get_augmented_state(self):
        """
        Get the augmented state (s_t, l_i) as per Equation 11.
        l_i indicates whether the human with minimum L_i has received
        unfavorable treatment (high u^-_i component).
        """
        # 1. Calculate the base fairness state s_t
        s_t = self._calculate_fairness_state()

        # 2. Find the human with minimum L_i
        min_index = np.argmin(s_t)

        # 3. Calculate l_i flag (Equation 11)
        # l_i = 1 if the human with min L_i has the highest u^- component (unfavorable)
        u_minus_components = self.satisfaction_records[:, 1]  # Second component is u^-
        l_flag = 1 if u_minus_components[min_index] == np.max(u_minus_components) else 0

        # 4. Return the augmented state (s_t, l_i)
        return np.append(s_t, l_flag)

    def step(self, global_action):
        """
        Applies a global action (HVAC setpoint), updates satisfaction records,
        and returns the next state.

        Args:
            global_action: The global HVAC setpoint (Type 1, Equation 7)

        Returns:
            next_state: The next augmented state
            done: Whether episode is done
        """
        desired_setpoints = self.get_current_desires()

        # Update satisfaction records c_i = (u^+_i, u^-_i) as per Equation 2
        for i in range(self.num_humans):
            error = np.abs(global_action - desired_setpoints[i])

            # Check if human i is satisfied (within threshold)
            if error <= config.SATISFACTION_THRESHOLD:
                # Increment u^+_i (satisfied counter)
                self.satisfaction_records[i, 0] += config.SATISFACTION_INCREMENT
            else:
                # Increment u^-_i (unsatisfied counter)
                self.satisfaction_records[i, 1] += config.SATISFACTION_INCREMENT

            # Normalize to unit vector (as per paper)
            norm = np.linalg.norm(self.satisfaction_records[i])
            if norm > 0:
                self.satisfaction_records[i] = self.satisfaction_records[i] / norm

        # Simulate activity changes (humans may change activities)
        for i in range(self.num_humans):
            # With some probability, change activity
            if random.random() < 0.1:  # 10% chance to change activity each step
                self.human_activities[i] = random.choice(self.activities)

        # Get the next state
        next_state = self.get_augmented_state()

        done = False  # Episode continues

        return next_state, done

    def get_satisfaction_records(self):
        """Return current satisfaction records for reward calculation"""
        return self.satisfaction_records.copy()

    def reset(self):
        """Resets the environment to initial state"""
        # Reset satisfaction records to (1, 1) normalized
        self.satisfaction_records = np.ones((self.num_humans, 2))
        for i in range(self.num_humans):
            self.satisfaction_records[i] = self.satisfaction_records[i] / np.linalg.norm(self.satisfaction_records[i])

        # Reset activities
        self.human_activities = [random.choice(self.activities) for _ in range(self.num_humans)]

        return self.get_augmented_state()
