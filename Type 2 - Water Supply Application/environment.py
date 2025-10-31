import numpy as np
import config as config
import random

class WaterSupplyEnvironment:
    #Implements Type 2 HITL application: Multi-Household Water Supply 

    def __init__(self, num_households=config.N_HOUSEHOLDS):
        self.num_households = num_households
        self.satisfaction_records = np.ones((self.num_households, 2))
        for i in range(self.num_households):
            self.satisfaction_records[i] = self.satisfaction_records[i] / np.linalg.norm(self.satisfaction_records[i])

        self.water_tanks = np.full(self.num_households, config.TANK_CAPACITY / 2.0)
        self.time_of_day = 0
        self.demand_profiles = list(config.DEMAND_PROFILES.values())

    def _calculate_cosine_similarity(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        return dot_product / norm_product if norm_product > 1e-10 else 0

    def get_current_demands(self):

        demands = []
        for profile in self.demand_profiles:
            hour = self.time_of_day % 24
            is_peak = (6 <= hour <= 9) or (18 <= hour <= 21)
            base = profile['base_demand']
            multiplier = profile['peak_multiplier'] if is_peak else 1.0
            demand = base * multiplier * (0.8 + 0.4 * random.random())
            demands.append(demand)

        return np.array(demands)

    def get_available_resource(self):

        demands = self.get_current_demands()
        max_demand = np.max(demands)
        time_factor = 0.9 + 0.2 * np.sin(2 * np.pi * self.time_of_day / 24)
        resource = config.RESOURCE_MULTIPLIER * max_demand * time_factor
        return resource

    def _calculate_fairness_state(self):

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
        s_t = self._calculate_fairness_state()
        min_index = np.argmin(s_t)
        u_minus_components = self.satisfaction_records[:, 1]
        l_flag = 1 if u_minus_components[min_index] == np.max(u_minus_components) else 0
        return np.append(s_t, l_flag)

    def step(self, global_action_weights):
        #Applies global action: distributes water resource according to weights.

        demands = self.get_current_demands()
        resource = self.get_available_resource()
        supplies = global_action_weights * resource
        balance_rates = np.zeros(self.num_households)

        for i in range(self.num_households):
            total_available = supplies[i] + self.water_tanks[i]
            balance_rate = total_available / (demands[i] + 1e-10)
            balance_rates[i] = balance_rate
            self.water_tanks[i] += supplies[i] - demands[i]
            self.water_tanks[i] = np.clip(self.water_tanks[i], 0, config.TANK_CAPACITY)

            # Check satisfaction 
            if balance_rate >= config.SATISFACTION_THRESHOLD_RATIO:
                self.satisfaction_records[i, 0] += config.SATISFACTION_INCREMENT
            else:
                self.satisfaction_records[i, 1] += config.SATISFACTION_INCREMENT

            norm = np.linalg.norm(self.satisfaction_records[i])
            if norm > 0:
                self.satisfaction_records[i] = self.satisfaction_records[i] / norm

        self.time_of_day += 1
        next_state = self.get_augmented_state()
        done = False
        return next_state, balance_rates, done

    def get_satisfaction_records(self):
        return self.satisfaction_records.copy()

    def reset(self):
        self.satisfaction_records = np.ones((self.num_households, 2))
        for i in range(self.num_households):
            self.satisfaction_records[i] = self.satisfaction_records[i] / np.linalg.norm(self.satisfaction_records[i])
        
        self.water_tanks = np.random.uniform(
            config.TANK_CAPACITY * 0.3, 
            config.TANK_CAPACITY * 0.7, 
            size=self.num_households
        )
        self.time_of_day = random.randint(0, 23)
        return self.get_augmented_state()

