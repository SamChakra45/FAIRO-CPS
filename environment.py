# environment.py
import numpy as np
import config

class SmartHomeEnvironment:
    """
    A simulated environment for the FAIRO algorithm.
    You must customize the `step` method with your application's logic.
    """
    def __init__(self, num_humans=config.N_HUMANS):
        self.num_humans = num_humans
        # Each record c_i = (utility, comfort) for example
        self.satisfaction_records = np.random.rand(self.num_humans, 2)

    def _calculate_cosine_similarity(self, vec1, vec2):
        """Helper to calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        return dot_product / norm_product if norm_product != 0 else 0

    def get_fairness_state(self):
        """Calculates the current fairness state s_t based on records C."""
        fairness_scores = np.zeros(self.num_humans)
        for i in range(self.num_humans):
            similarities = [
                self._calculate_cosine_similarity(self.satisfaction_records[i], self.satisfaction_records[j])
                for j in range(self.num_humans) if i != j
            ]
            fairness_scores[i] = np.mean(similarities) if similarities else 0
        return fairness_scores

    def step(self, global_action):
        """
        Applies a global action and returns the outcome.
        
        **TODO: This is the primary method you need to implement.**
        Your logic for how an action affects user records and what reward is generated goes here.
        """
        # 1. Update satisfaction records based on the global_action.
        # This is highly application-specific. For this skeleton, we'll
        # imagine the action tries to push all records towards an "ideal" point.
        for i in range(self.num_humans):
            # This update rule is just an example.
            self.satisfaction_records[i] += (global_action - 0.5) * 0.1 # Example update
            self.satisfaction_records[i] = np.clip(self.satisfaction_records[i], 0, 1)

        # 2. Calculate the reward. A good reward promotes fairness.
        # Let's use the negative standard deviation of fairness scores.
        # Lower std dev (less spread) means higher reward.
        new_fairness_state = self.get_fairness_state()
        reward = -np.std(new_fairness_state)

        # 3. Get the next state and done flag
        next_state = new_fairness_state
        done = False  # Our environment runs continuously

        return next_state, reward, done

    def reset(self):
        """Resets the environment to a new random state."""
        self.satisfaction_records = np.random.rand(self.num_humans, 2)
        return self.get_fairness_state()