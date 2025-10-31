# dqn_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import namedtuple, deque
import config

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

class ReplayBuffer:

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class QNetwork(nn.Module):
    #The neural network for estimating Q-values.

    def __init__(self, state_dim, action_dim=3):
        super(QNetwork, self).__init__()
        self.layer1 = nn.Linear(state_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, action_dim)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)

class DQNAgent:
    """
    The agent that learns the policy for one option.
    Each option ω_i has its own DQN that learns to adjust weight w_i.
    """

    def __init__(self, state_dim):
        self.state_dim = state_dim
        self.action_dim = 3 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = QNetwork(state_dim, self.action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LEARNING_RATE)
        self.replay_buffer = ReplayBuffer(config.BUFFER_SIZE)
        self.steps_done = 0

    def select_action(self, state):
        eps_threshold = config.EPSILON_END + (config.EPSILON_START - config.EPSILON_END) * \
                        np.exp(-1. * self.steps_done / config.EPSILON_DECAY)
        self.steps_done += 1

        if random.random() > eps_threshold:
            with torch.no_grad():
                if isinstance(state, torch.Tensor):
                    state_tensor = state.to(self.device)
                else:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

                action_idx = self.policy_net(state_tensor).max(1)[1]
                return action_idx.view(1, 1).to(self.device)
        else:
            return torch.tensor([[random.randrange(self.action_dim)]], 
                              dtype=torch.long, device=self.device)

    def learn(self):
        #Implements the Q-learning update rule

        if len(self.replay_buffer) < config.BATCH_SIZE:
            return

        transitions = self.replay_buffer.sample(config.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        next_state_batch = torch.cat(batch.next_state).to(self.device)

        q_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_q_values = (next_state_values * config.GAMMA) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(q_values, expected_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        
        for param in self.policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)

        self.optimizer.step()
