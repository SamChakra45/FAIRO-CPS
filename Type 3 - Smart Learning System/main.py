import torch
import numpy as np
import config as config
from environment import SmartLearningEnvironment
from dqn_agent import DQNAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def calculate_performance_term(satisfaction_records, student_idx, L_prev, L_curr, learning_experience):
    #Calculate the performance term P_i for Type 3 application
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

    le_normalized = np.clip((learning_experience - 0.5) / 1.5, -1.0, 1.0)

    P_i = 0.2 * term + 0.8 * le_normalized
    P_i = np.clip(P_i, -1.0, 1.0)

    return P_i


def run_fairo_type3():
    """
    Main FAIRO training loop for Type 3: Smart Learning with VR
    """

    all_L1 = []
    all_L2 = []
    all_L3 = []

    env = SmartLearningEnvironment(num_students=config.N_STUDENTS)
    state_dim = config.N_STUDENTS + 1
    dqn_agents = [DQNAgent(state_dim) for _ in range(config.N_STUDENTS)]
    weights = np.ones(config.N_STUDENTS) / config.N_STUDENTS
    previous_L_values = np.zeros(config.N_STUDENTS)

    episode_rewards = []

    for episode in range(config.NUM_EPISODES):
        state = env.reset()
        previous_L_values = state[:-1].copy()
        episode_reward = 0

        for t in range(config.STEPS_PER_EPISODE):
            desired_actions = env.get_current_desired_actions()
            effects = env.calculate_action_effects(desired_actions)

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

            weighted_effects = weights * effects
            global_action_index = np.argmax(weighted_effects)
            global_action = desired_actions[global_action_index]

            next_state, learning_experiences, done = env.step(global_action)

            i = active_option_index
            L_i_current = next_state[i]
            L_i_previous = previous_L_values[i]

            absolute_fairness = (2 * L_i_current) - 1
            L_i_improvement = L_i_current - L_i_previous
            option_improvement = np.tanh(L_i_improvement * 100)
            F_i = absolute_fairness + option_improvement
            F_i = np.clip(F_i, -1.0, 1.0)

            satisfaction_records = env.get_satisfaction_records()
            P_i = calculate_performance_term(
                satisfaction_records, i, L_i_previous, L_i_current, learning_experiences[i]
            )

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
        print(f"  Student {i+1}: L_{i+1} = {L_i:.4f}")

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
        torch.save(agent.policy_net.state_dict(), f'policy_net_type3_{idx}.pt')
        torch.save(agent.target_net.state_dict(), f'target_net_type3_{idx}.pt')

    np.save('fairo_weights_type3.npy', weights)
    np.save('fairo_satisfaction_records_type3.npy', env.satisfaction_records)
    print("\nModels saved!")


if __name__ == '__main__':
    run_fairo_type3()
