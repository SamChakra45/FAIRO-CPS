import torch
import numpy as np
import config
from environment import SmartHomeEnvironment
from dqn_agent import DQNAgent

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def calculate_performance_term(satisfaction_records, human_idx, L_prev, L_curr):
    #Calculate the performance term P_i

    L_diff = L_prev - L_curr

    if L_diff == 0:
        term = 0.0
    elif 0 < L_diff <= 0.001:
        term = 0.25
    elif 0.001 < L_diff <= 0.005:
        term = 0.5
    elif 0.005 < L_diff <= 0.01:
        term = 0.75
    elif 0.01 < L_diff <= 0.015:
        term = 1.0
    elif L_diff > 0.015:
        term = 1.0
    else:
        if L_diff >= -0.001:
            term = 0.0
        elif -0.005 < L_diff < -0.001:
            term = -0.25
        elif -0.01 < L_diff <= -0.005:
            term = -0.5
        elif -0.015 < L_diff <= -0.01:
            term = -0.75
        else:
            term = -1.0

    u_plus = satisfaction_records[human_idx, 0]
    u_minus = satisfaction_records[human_idx, 1]
    satisfaction_value = (u_plus - u_minus) / (u_plus + u_minus + 1e-10)

    P_i = 0.2 * term + 0.8 * satisfaction_value
    P_i = np.clip(P_i, -1.0, 1.0)
    return P_i


def run_fairo():
    #Main FAIRO training loop.

    all_L1 = []
    all_L2 = []
    all_L3 = []

    env = SmartHomeEnvironment(num_humans=config.N_HUMANS)
    state_dim = config.N_HUMANS + 1
    dqn_agents = [DQNAgent(state_dim) for _ in range(config.N_HUMANS)]
    weights = np.ones(config.N_HUMANS) / config.N_HUMANS
    previous_L_values = np.zeros(config.N_HUMANS)

    episode_rewards = []

    for episode in range(config.NUM_EPISODES):
        state = env.reset()
        previous_L_values = state[:-1].copy() 
        episode_reward = 0

        for t in range(config.STEPS_PER_EPISODE):
            user_desires = env.get_current_desires()

            base_fairness_state = state[:-1] 
            all_L1.append(base_fairness_state[0])
            all_L2.append(base_fairness_state[1])
            all_L3.append(base_fairness_state[2])
            l_flag = state[-1] 

            active_option_index = np.argmin(base_fairness_state)
            active_option = dqn_agents[active_option_index]

            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action_tensor = active_option.select_action(state_tensor)
            action = action_tensor.item()

            if action == 0:
                delta_w = config.DELTA
            elif action == 1:
                delta_w = -config.DELTA
            else:
                delta_w = 0

            weights[active_option_index] += delta_w
            weights = np.clip(weights, 0, 1)
            weights = weights / np.sum(weights)  

            global_action = np.dot(weights, user_desires)
            next_state, done = env.step(global_action)

            i = active_option_index
            L_i_current = next_state[i]
            L_i_previous = previous_L_values[i]

            absolute_fairness = (2 * L_i_current) - 1
            L_i_improvement = L_i_current - L_i_previous
            option_improvement = np.tanh(L_i_improvement * 100)  

            F_i = absolute_fairness + option_improvement
            F_i = np.clip(F_i, -1.0, 1.0)

            satisfaction_records = env.get_satisfaction_records()
            P_i = calculate_performance_term(satisfaction_records, i, L_i_previous, L_i_current)

            R_i = config.ZETA * F_i + (1 - config.ZETA) * P_i
            episode_reward += R_i

            reward_tensor = torch.tensor([R_i], device=device, dtype=torch.float)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)

            active_option.replay_buffer.push(
                state_tensor,
                action_tensor,
                reward_tensor,
                next_state_tensor
            )

            active_option.learn()
            state = next_state
            previous_L_values[i] = L_i_current

        if episode % config.TARGET_UPDATE == 0:
            for agent in dqn_agents:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())

        # Log 
        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else episode_reward

        if (episode + 1) % 10 == 0:
            base_fairness = state[:-1]
            print(f"Episode {episode+1:3d}/{config.NUM_EPISODES} | "
                  f"Avg Reward: {avg_reward:6.3f} | "
                  f"Weights: [{', '.join([f'{w:.2f}' for w in weights])}] | "
                  f"Fairness L: [{', '.join([f'{L:.3f}' for L in base_fairness])}]")

    # Final evaluation
    final_state = env.get_augmented_state()
    final_fairness = final_state[:-1]
    print(f"\nFinal Fairness State:")
    for i, L_i in enumerate(final_fairness):
        print(f"  Human {i+1}: L_{i+1} = {L_i:.4f}")

    print(f"\nFinal Weights:")
    for i, w_i in enumerate(weights):
        print(f"  w_{i+1} = {w_i:.3f}")

    ideal_fairness = np.mean(final_fairness)
    fairness_variance = np.var(final_fairness)
    print(f"\nFairness Metrics:")
    print(f"  Mean L: {ideal_fairness:.4f} (closer to 1.0 is better)")
    print(f"  Variance: {fairness_variance:.6f} (closer to 0 is better)")

    # SAVE FAIRNESS TRACES FOR PLOTTING
    np.save('L1_trace.npy', np.array(all_L1))
    np.save('L2_trace.npy', np.array(all_L2))
    np.save('L3_trace.npy', np.array(all_L3))
    
    # Save trained models and weights
    for idx, agent in enumerate(dqn_agents):
        torch.save(agent.policy_net.state_dict(), f'policy_net_{idx}.pt')
        torch.save(agent.target_net.state_dict(), f'target_net_{idx}.pt')
    np.save('fairo_weights.npy', weights)
    np.save('fairo_satisfaction_records.npy', env.satisfaction_records)


if __name__ == '__main__':
    run_fairo()
