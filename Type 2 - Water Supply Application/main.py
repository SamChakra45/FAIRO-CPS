# main_type2.py

import torch
import numpy as np
import config as config
from environment import WaterSupplyEnvironment
from dqn_agent import DQNAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def calculate_performance_term(satisfaction_records, household_idx, L_prev, L_curr, balance_rate):
    """
    Calculate the performance term P_i for Type 2 application.

    P_i = 0.2 * term + 0.8 * balance_rate_metric

    where:
    - term: based on L_i change (fairness improvement)
    - balance_rate_metric: based on how well demand is met
    """
    # Calculate term based on L_i change
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

    # Balance rate metric: how well the demand was met
    # balance_rate >= 0.8 is satisfactory
    br_normalized = (balance_rate - 0.8) / 0.2  # Map [0.8, 1.0+] to [0, 1]
    br_normalized = np.clip(br_normalized, -1.0, 1.0)

    # Calculate P_i (Equation 15 adapted for Type 2)
    P_i = 0.2 * term + 0.8 * br_normalized
    P_i = np.clip(P_i, -1.0, 1.0)

    return P_i


def run_fairo_type2():
    """
    Main FAIRO training loop for Type 2: Water Supply Application
    """
    print("="*80)
    print("FAIRO: Type 2 Application - Multi-Household Water Supply")
    print("="*80)

    all_L1 = []
    all_L2 = []
    all_L3 = []

    # Initialize environment
    env = WaterSupplyEnvironment(num_households=config.N_HOUSEHOLDS)

    # State dimension: N fairness scores + 1 flag
    state_dim = config.N_HOUSEHOLDS + 1

    # Create N DQN agents (one per option)
    dqn_agents = [DQNAgent(state_dim) for _ in range(config.N_HOUSEHOLDS)]

    # Initialize weights uniformly
    weights = np.ones(config.N_HOUSEHOLDS) / config.N_HOUSEHOLDS

    # Store previous L values
    previous_L_values = np.zeros(config.N_HOUSEHOLDS)

    # Track metrics
    episode_rewards = []

    print(f"\nStarting training for {config.NUM_EPISODES} episodes...")
    print(f"Each episode has {config.STEPS_PER_EPISODE} steps\n")

    for episode in range(config.NUM_EPISODES):
        state = env.reset()
        previous_L_values = state[:-1].copy()

        episode_reward = 0

        for t in range(config.STEPS_PER_EPISODE):
            # Get current demands and available resource
            demands = env.get_current_demands()
            resource = env.get_available_resource()

            # Get fairness state
            base_fairness_state = state[:-1]
            all_L1.append(base_fairness_state[0])
            all_L2.append(base_fairness_state[1])
            all_L3.append(base_fairness_state[2])
            l_flag = state[-1]

            # Choose active option (minimum L_i)
            active_option_index = np.argmin(base_fairness_state)
            active_option = dqn_agents[active_option_index]

            # Select action (weight adjustment)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action_tensor = active_option.select_action(state_tensor)
            action = action_tensor.item()

            # Update weights (Equations 12-13)
            if action == 0:
                delta_w = config.DELTA
            elif action == 1:
                delta_w = -config.DELTA
            else:
                delta_w = 0

            weights[active_option_index] += delta_w
            weights = np.clip(weights, 0, 1)
            weights = weights / np.sum(weights)  # Normalize

            # Calculate global action for Type 2 (Equation 8)
            # a_g = (w_1*R, w_2*R, ..., w_N*R)
            # Apply action to environment
            next_state, balance_rates, done = env.step(weights)

            # Calculate reward R_i (Equation 14)
            i = active_option_index
            L_i_current = next_state[i]
            L_i_previous = previous_L_values[i]

            # Fairness term F_i
            # Fairness term F_i (Equation 14)
            absolute_fairness = (2 * L_i_current) - 1  # Maps [0,1] to [-1,1]

            # CHANGE THIS LINE:
            # L_i_improvement = L_i_current - L_i_previous
            # option_improvement = np.tanh(L_i_improvement * 100)

            # TO THIS (more aggressive fairness improvement reward):
            L_i_improvement = L_i_current - L_i_previous
            if L_i_improvement > 0:
                option_improvement = min(1.0, L_i_improvement * 200)  # Stronger positive reward
            else:
                option_improvement = max(-1.0, L_i_improvement * 50)  # Gentler negative penalty

            F_i = absolute_fairness + option_improvement
            F_i = np.clip(F_i, -1.0, 1.0)


            # Performance term P_i
            satisfaction_records = env.get_satisfaction_records()
            P_i = calculate_performance_term(
                satisfaction_records, i, L_i_previous, L_i_current, balance_rates[i]
            )

            # Final reward (Equation 14)
            R_i = config.ZETA * F_i + (1 - config.ZETA) * P_i
            episode_reward += R_i

            # Store experience and learn
            reward_tensor = torch.tensor([R_i], device=device, dtype=torch.float)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)

            active_option.replay_buffer.push(
                state_tensor,
                action_tensor,
                reward_tensor,
                next_state_tensor
            )

            active_option.learn()

            # Update state
            state = next_state
            previous_L_values[i] = L_i_current

        # Update target networks
        if episode % config.TARGET_UPDATE == 0:
            for agent in dqn_agents:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())

        # Log progress
        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else episode_reward

        if (episode + 1) % 10 == 0:
            base_fairness = state[:-1]
            print(f"Episode {episode+1:3d}/{config.NUM_EPISODES} | "
                  f"Avg Reward: {avg_reward:6.3f} | "
                  f"Weights: [{', '.join([f'{w:.2f}' for w in weights])}] | "
                  f"Fairness L: [{', '.join([f'{L:.3f}' for L in base_fairness])}]")

    print("\n" + "="*80)
    print("Training completed!")
    print("="*80)

    # Final evaluation
    final_state = env.get_augmented_state()
    final_fairness = final_state[:-1]
    print(f"\nFinal Fairness State:")
    for i, L_i in enumerate(final_fairness):
        print(f"  Household {i+1}: L_{i+1} = {L_i:.4f}")

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

    # Save models
    for idx, agent in enumerate(dqn_agents):
        torch.save(agent.policy_net.state_dict(), f'policy_net_type2_{idx}.pt')
        torch.save(agent.target_net.state_dict(), f'target_net_type2_{idx}.pt')

    np.save('fairo_weights_type2.npy', weights)
    np.save('fairo_satisfaction_records_type2.npy', env.satisfaction_records)
    print("\nModels saved!")


if __name__ == '__main__':
    run_fairo_type2()
