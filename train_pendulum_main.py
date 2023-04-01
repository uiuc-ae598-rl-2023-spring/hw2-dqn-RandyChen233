import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import discreteaction_pendulum
import random
from matplotlib import pyplot as plt
from train_pendulum_DQN import *
import seaborn as sns


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
    target_net.load_state_dict(policy_net.state_dict()) #synchronize weights for initialization

    # optimizer = optim.AdamW(policy_net.parameters(), lr = 1e-4, amsgrad=True)
    optimizer = optim.RMSprop(policy_net.parameters(), lr=1e-4)
    memory = ReplayMemory(int(1e6))

    update_freq = 1000

    policy_net, reward_Full ,actions, traj, state_values = train_network(n_actions, num_episodes, policy_net, target_net, \
                                            optimizer, memory, env,  batch_size, eps_init, eps_end, GAMMA, \
                                            target_update_frequency = update_freq)

    """Plot a learning curve"""
    plt.figure(dpi=150)
    plt.plot(reward_Full)
    plt.xlabel('Episode')
    plt.ylabel('Payoff')
    plt.grid()
    plt.savefig('figures_trained/LearningCurve.png')

    """Plot a trained policy"""

    plt.figure(dpi=150)
    sns.heatmap(traj,vmin=min(actions), vmax = max(actions), cbar_kws={'label': 'Tau'})
    plt.xlabel('Theta')
    plt.ylabel('Theta_dot')
    plt.savefig('figures_trained/policy_trained.png')

    """Plot example state trajectory within an episode"""
    plt.figure(dpi=150)
    traj = np.array(traj)
    plt.plot(traj[:, 0], label='theta')
    plt.plot(traj[:, 1], label='theta_dot')
    plt.xlabel('Time step')
    plt.ylabel('State trajectories')
    plt.legend()
    plt.savefig('figures_trained/exampleTrajectory.png')

    """Plot state values"""
    plt.figure(dpi=150)
    sns.heatmap(traj,vmin=min(state_values), vmax = max(state_values), cbar_kws={'label': 'Value'})
    plt.xlabel('Theta')
    plt.ylabel('Theta_dot')
    plt.savefig('figures_trained/StateValues.png')



    """Save an animation"""
    policy = lambda s: (policy_net(torch.tensor(s, dtype=torch.float32).unsqueeze(0)).max(1)[1].view(1, 1)).item()
    env.video(policy, filename='figures_trained/trained_pendulum.gif')

if __name__ == '__main__':
    main()