import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import discreteaction_pendulum
import random
from collections import namedtuple, deque
from matplotlib import pyplot as plt
from train_pendulum_DQN import *



def main():
    GAMMA = 0.95
    eps_init = 1.0
    eps_end = 0.1
    num_episodes = 100

    env = discreteaction_pendulum.Pendulum()
    batch_size = 64
    n_actions = env.num_actions
    n_states = env.num_states

    policy_net = DQN(n_states, n_actions)
    target_net = DQN(n_states, n_actions)
    target_net.load_state_dict(policy_net.state_dict())

    # optimizer = optim.AdamW(policy_net.parameters(), lr = 1e-4, amsgrad=True)
    optimizer = optim.RMSprop(policy_net.parameters(), lr=0.0002, alpha=0.95)
    memory = ReplayMemory(int(1e6))

    policy_net, reward_Full = train_network(n_actions, num_episodes, policy_net, target_net, \
                                            optimizer, memory, env,  batch_size, eps_init, eps_end, GAMMA)

    """Plot a learning curve"""
    plt.figure(dpi=150)
    plt.plot(reward_Full)
    plt.xlabel('Episode')
    plt.ylabel('Payoff')
    plt.grid()
    plt.savefig('figures_trained/LearningCurve.png')

    policy = lambda s: (policy_net(torch.tensor(s, dtype=torch.float32).unsqueeze(0)).max(1)[1].view(1, 1)).item()

    env.video(policy, filename='figures_trained/trained_pendulum.gif')

if __name__ == '__main__':
    main()