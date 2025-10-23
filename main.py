# main.py
import torch
print(torch.cuda.is_available())
import numpy as np
import config
from environment import SmartHomeEnvironment
from dqn_agent import DQNAgent

def run_fairo():
    """Main function to initialize and run the FAIRO algorithm."""
    print("Initializing FAIRO...")
    env = SmartHomeEnvironment(num_humans=config.N_HUMANS)
    state_dim = config.N_HUMANS

    # Create N specialist DQN agents
    dqn_agents = [DQNAgent(state_dim) for _ in range(config.N_HUMANS)]
    
    # Initialize the weight vector w
    weights = np.ones(config.N_HUMANS) / config.N_HUMANS

    print(f"Starting training for {config.NUM_EPISODES} episodes...")
    for episode in range(config.NUM_EPISODES):
        state = env.reset()
        
        # In this continuous environment, an "episode" is just a fixed number of steps
        for t in range(200): # e.g., 200 steps per episode
            # 1. Assess Fairness & Choose Option
            active_dqn_index = np.argmin(state)
            active_dqn = dqn_agents[active_dqn_index]

            # 2. Run Option to get weight adjustment
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_tensor = active_dqn.select_action(state_tensor)
            action = action_tensor.item()
            
            if action == 0:  # Increase w_i
                delta_w = config.DELTA
            elif action == 1:  # Decrease w_i
                delta_w = -config.DELTA
            else:  # Keep same
                delta_w = 0

            # 3. Update and Normalize Weights
            weights[active_dqn_index] += delta_w
            weights = np.clip(weights, 0, 1) # Ensure weights are non-negative
            weights = weights / np.sum(weights) # Normalize to sum to 1

            # 4. Calculate and Apply Global Action
            # TODO: Customize this based on your application Type (1, 2, or 3)
            # This is an example for Type 1
            user_desires = np.ones(config.N_HUMANS) # Placeholder for d_t
            global_action = np.dot(weights, user_desires)

            next_state, reward, done = env.step(global_action)
            reward_tensor = torch.tensor([reward], dtype=torch.float)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

            # 5. Store experience and Learn
            active_dqn.replay_buffer.push(state_tensor, action_tensor, reward_tensor, next_state_tensor)
            active_dqn.learn()

            state = next_state

        # Update target network for all agents periodically
        if episode % config.TARGET_UPDATE == 0:
            for agent in dqn_agents:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())
        
        print(f"Episode {episode+1}/{config.NUM_EPISODES} | Final Reward: {reward:.4f} | Weights: {np.round(weights, 2)}")

if __name__ == '__main__':
    run_fairo()