# environment_type2.py

import numpy as np
import config_type2 as config
import random

class WaterSupplyEnvironment:
    """
    Implements Type 2 HITL application: Multi-Household Water Supply (Section 6)
    Multiple households share a limited, time-varying water resource.
    Global action distributes the resource proportionally: a_g = (w_1*R, w_2*R, ..., w_N*R)
    """

    def __init__(self, num_households=config.N_HOUSEHOLDS):
        self.num_households = num_households

        # Satisfaction history records: c_i = (u^+_i, u^-_i)
        self.satisfaction_records = np.ones((self.num_households, 2))
        for i in range(self.num_households):
            self.satisfaction_records[i] = self.satisfaction_records[i] / np.linalg.norm(self.satisfaction_records[i])

        # Water tanks for each household (reserve)
        self.water_tanks = np.full(self.num_households, config.TANK_CAPACITY / 2.0)

        # Time of day (for demand variation)
        self.time_of_day = 0

        # Demand profiles
        self.demand_profiles = list(config.DEMAND_PROFILES.values())

    def _calculate_cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        return dot_product / norm_product if norm_product > 1e-10 else 0

    def get_current_demands(self):
        """
        Simulates water demand for each household based on time of day.
        Returns vector of current water demands d_t.
        """
        demands = []
        for profile in self.demand_profiles:
            # Simulate time-varying demand with peak hours
            hour = self.time_of_day % 24
            # Peak hours: 6-9 AM and 6-9 PM
            is_peak = (6 <= hour <= 9) or (18 <= hour <= 21)

            base = profile['base_demand']
            multiplier = profile['peak_multiplier'] if is_peak else 1.0

            # Add some randomness
            demand = base * multiplier * (0.8 + 0.4 * random.random())
            demands.append(demand)

        return np.array(demands)

    def get_available_resource(self):
        """
        Returns the current available water resource R_t.
        Resource is time-varying and insufficient to meet all demands.
        """
        demands = self.get_current_demands()
        max_demand = np.max(demands)

        # Resource is insufficient: R = multiplier * max_demand
        # Add time variation
        time_factor = 0.9 + 0.2 * np.sin(2 * np.pi * self.time_of_day / 24)
        resource = config.RESOURCE_MULTIPLIER * max_demand * time_factor

        return resource

    def _calculate_fairness_state(self):
        """
        Calculate the fairness state s_t = (L_1, L_2, ..., L_N).
        L_i measures how close household i's satisfaction record is to others.
        """
        s_t = np.zeros(self.num_households)

        for i in range(self.num_households):
            similarities = []
            for j in range(self.num_households):
                if i != j:
                    sim = self._calculate_cosine_similarity(
                        self.satisfaction_records[i],
                        self.satisfaction_records[j]
                    )
                    similarities.append(sim)

            s_t[i] = np.mean(similarities) if similarities else 1.0

        return s_t

    def get_augmented_state(self):
        """
        Get the augmented state (s_t, l_i).
        l_i = 1 if household with min L_i has highest u^- component.
        """
        s_t = self._calculate_fairness_state()

        min_index = np.argmin(s_t)
        u_minus_components = self.satisfaction_records[:, 1]
        l_flag = 1 if u_minus_components[min_index] == np.max(u_minus_components) else 0

        return np.append(s_t, l_flag)

    def step(self, global_action_weights):
        """
        Applies global action: distributes water resource according to weights.

        Args:
            global_action_weights: Array of weights [w_1, w_2, ..., w_N]
                                   where w_i determines household i's share

        Returns:
            next_state: The next augmented state
            balance_rates: Balance rate for each household (performance metric)
            done: Whether episode is done
        """
        demands = self.get_current_demands()
        resource = self.get_available_resource()

        # Type 2 global action (Equation 8): a_g = (w_1*R, w_2*R, ..., w_N*R)
        supplies = global_action_weights * resource

        balance_rates = np.zeros(self.num_households)

        # Update satisfaction records and water tanks
        for i in range(self.num_households):
            # Calculate balance rate: (supply + reserve) / demand
            total_available = supplies[i] + self.water_tanks[i]
            balance_rate = total_available / (demands[i] + 1e-10)
            balance_rates[i] = balance_rate

            # Update tank: add supply, subtract demand
            self.water_tanks[i] += supplies[i] - demands[i]
            self.water_tanks[i] = np.clip(self.water_tanks[i], 0, config.TANK_CAPACITY)

            # Check satisfaction (Equation 2)
            if balance_rate >= config.SATISFACTION_THRESHOLD_RATIO:
                # Satisfied: increment u^+
                self.satisfaction_records[i, 0] += config.SATISFACTION_INCREMENT
            else:
                # Unsatisfied: increment u^-
                self.satisfaction_records[i, 1] += config.SATISFACTION_INCREMENT

            # Normalize to unit vector
            norm = np.linalg.norm(self.satisfaction_records[i])
            if norm > 0:
                self.satisfaction_records[i] = self.satisfaction_records[i] / norm

        # Advance time
        self.time_of_day += 1

        # Get next state
        next_state = self.get_augmented_state()

        done = False

        return next_state, balance_rates, done

    def get_satisfaction_records(self):
        """Return current satisfaction records"""
        return self.satisfaction_records.copy()

    def reset(self):
        """Reset the environment to initial state"""
        self.satisfaction_records = np.ones((self.num_households, 2))
        for i in range(self.num_households):
            self.satisfaction_records[i] = self.satisfaction_records[i] / np.linalg.norm(self.satisfaction_records[i])

        self.water_tanks = np.full(self.num_households, config.TANK_CAPACITY / 2.0)
        self.time_of_day = 0

        return self.get_augmented_state()
