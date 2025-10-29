# main.py

import torch
import numpy as np
import config
from environment import SmartHomeEnvironment
from dqn_agent import DQNAgent

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def calculate_performance_term(satisfaction_records, human_idx, L_prev, L_curr):
    """
    Calculate the performance term P_i as per Equation 15.

    P_i = 0.2 * term + 0.8 * ∑c_i

    where term is based on the change in L_i:
      - If L_i improved significantly: positive reward
      - Otherwise: scaled based on magnitude of change

    The satisfaction counters ∑c_i represent cumulative satisfaction.
    """
    # Calculate the term based on L_i change (Equation 15 breakdown)
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
        # L_i got worse (L_diff < 0)
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

    # Get satisfaction record sum for human i
    # c_i = (u^+_i, u^-_i), we want ∑c_i normalized to [-1, 1]
    u_plus = satisfaction_records[human_idx, 0]
    u_minus = satisfaction_records[human_idx, 1]

    # Normalize: more u^+ is better, more u^- is worse
    # Since they're normalized to unit vector, we can use the ratio
    satisfaction_value = (u_plus - u_minus) / (u_plus + u_minus + 1e-10)

    # Calculate P_i
    P_i = 0.2 * term + 0.8 * satisfaction_value

    # Clamp to [-1, 1]
    P_i = np.clip(P_i, -1.0, 1.0)

    return P_i


def run_fairo():
    """
    Main FAIRO training loop implementing Algorithm 1.
    """
    print("="*80)
    print("FAIRO: Fairness-aware Adaptation in Sequential-Decision Making")
    print("Type 1 Application: Smart Home HVAC")
    print("="*80)

    all_L1 = []
    all_L2 = []
    all_L3 = []

    # Initialize environment
    env = SmartHomeEnvironment(num_humans=config.N_HUMANS)

    # State dimension: N fairness scores (L_1, ..., L_N) + 1 flag (l_i)
    state_dim = config.N_HUMANS + 1

    # Create N DQN agents, one for each option ω_i (Section 4.5)
    dqn_agents = [DQNAgent(state_dim) for _ in range(config.N_HUMANS)]

    # Initialize weights w = (w_1, ..., w_N) uniformly (Equation 7)
    weights = np.ones(config.N_HUMANS) / config.N_HUMANS

    # Store previous L_i values for each option (needed for reward calculation)
    previous_L_values = np.zeros(config.N_HUMANS)

    # Track metrics for logging
    episode_rewards = []

    print(f"\nStarting training for {config.NUM_EPISODES} episodes...")
    print(f"Each episode has {config.STEPS_PER_EPISODE} steps\n")

    for episode in range(config.NUM_EPISODES):
        # Reset environment (Algorithm 1, Line 2)
        state = env.reset()
        previous_L_values = state[:-1].copy()  # Initialize with starting fairness state

        episode_reward = 0

        for t in range(config.STEPS_PER_EPISODE):
            # Line 3: Get desired actions from context-aware engine
            user_desires = env.get_current_desires()

            # Line 8: Get current fairness state s_t
            # State is (L_1, ..., L_N, l_i) where l_i indicates unfair treatment direction
            base_fairness_state = state[:-1]  # Extract L values
            all_L1.append(base_fairness_state[0])
            all_L2.append(base_fairness_state[1])
            all_L3.append(base_fairness_state[2])
            l_flag = state[-1]  # Extract l_i flag

            # Line 10: Choose active option based on initiation set (Equation 5)
            # Option ω_i is active when L_i is minimum (human i is most unfairly treated)
            active_option_index = np.argmin(base_fairness_state)
            active_option = dqn_agents[active_option_index]

            # Line 11-12: Run option to get weight adjustment
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action_tensor = active_option.select_action(state_tensor)
            action = action_tensor.item()

            # Line 13: Update and normalize weights (Equations 12-13)
            # Action 0: increase w_i, Action 1: decrease w_i, Action 2: keep w_i
            if action == 0:
                delta_w = config.DELTA
            elif action == 1:
                delta_w = -config.DELTA
            else:
                delta_w = 0

            weights[active_option_index] += delta_w
            weights = np.clip(weights, 0, 1)
            weights = weights / np.sum(weights)  # Normalize to sum to 1

            # Line 14: Calculate global action (Type 1: Equation 7)
            # a_g = ∑(w_i * d_i) - weighted sum of desired setpoints
            global_action = np.dot(weights, user_desires)

            # Line 15: Apply global action to environment
            next_state, done = env.step(global_action)

            # Line 16: Calculate reward R_i for the active option (Equation 14)
            # R_i = ζ * F_i + (1 - ζ) * P_i

            i = active_option_index
            L_i_current = next_state[i]
            L_i_previous = previous_L_values[i]

            # Fairness term F_i (Equation 14)
            # F_i has two components: absolute fairness and option improvement
            absolute_fairness = (2 * L_i_current) - 1  # Maps [0,1] to [-1,1]

            # Option improvement based on L_i change
            L_i_improvement = L_i_current - L_i_previous
            option_improvement = np.tanh(L_i_improvement * 100)  # Smooth scaling

            F_i = absolute_fairness + option_improvement
            F_i = np.clip(F_i, -1.0, 1.0)

            # Performance term P_i (Equation 15)
            satisfaction_records = env.get_satisfaction_records()
            P_i = calculate_performance_term(satisfaction_records, i, L_i_previous, L_i_current)

            # Final reward (Equation 14)
            R_i = config.ZETA * F_i + (1 - config.ZETA) * P_i
            episode_reward += R_i

            # Line 17: Update option policy (store experience and learn)
            reward_tensor = torch.tensor([R_i], device=device, dtype=torch.float)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)

            active_option.replay_buffer.push(
                state_tensor,
                action_tensor,
                reward_tensor,
                next_state_tensor
            )

            active_option.learn()

            # Line 18: Update satisfaction records (done in env.step())

            # Update state and previous L values
            state = next_state
            previous_L_values[i] = L_i_current

            # Check termination condition (Equation 6)
            # Option ω_i terminates when L_i is no longer minimum
            # This is handled automatically in next iteration by choosing new active option

        # Update target networks periodically
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
