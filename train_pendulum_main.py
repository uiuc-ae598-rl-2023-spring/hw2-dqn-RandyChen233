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
    num_episodes = 150

    env = discreteaction_pendulum.Pendulum()
    batch_size = 64
    n_actions = env.num_actions
    n_states = env.num_states

    policy_net = DQN(n_states, n_actions)
    target_net = DQN(n_states, n_actions)
    target_net.load_state_dict(policy_net.state_dict()) #synchronize weights for initialization

    # optimizer = optim.AdamW(policy_net.parameters(), lr = 1e-4, amsgrad=True)
    optimizer = optim.RMSprop(policy_net.parameters(), lr=0.01)
    memory = ReplayMemory(int(1e5))

    update_freq = 1000

    policy_net, reward_Full_standard ,actions, traj, state_values = train_network(n_actions, num_episodes, policy_net, target_net, \
                                            optimizer, memory, env,  batch_size, eps_init, eps_end, GAMMA, \
                                            target_update_frequency = update_freq)

    """Plot a learning curve"""
    plt.figure(dpi=150)
    plt.plot(reward_Full_standard)
    plt.xlabel('Episode')
    plt.ylabel('Payoff')
    plt.grid()
    plt.savefig('figures_trained/LearningCurve.png')

    """Plot a trained policy"""
    #We first need to discretize states
    theta = np.linspace(-np.pi, np.pi, 50) #these are hard-coded bounds given by the environment
    thetadot = np.linspace(-15, 15, 50)
    actions = np.zeros([len(theta), len(thetadot)])

    for i in range(len(theta)): 
        for j in range(len(thetadot)): 
            s = torch.tensor([theta[i], thetadot[j]]).float()
            a = torch.argmax(policy_net(s)).detach()
            actions[i,j] = env._a_to_u(a)
    
    fig, ax = plt.subplots()
    c = ax.contourf(theta, thetadot, actions, alpha = 0.9)
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$\dot{\theta}$')
    ax.set_title('Trained Policy')
    cbar = fig.colorbar(c)
    cbar.ax.set_ylabel(r'$\tau$')
    plt.savefig('figures_trained/TrainedPolicy.png')

    """Plot example state trajectory """
    plt.figure(dpi=150)
    traj = np.array(traj)
    plt.plot(traj[:, 0], label='theta')
    plt.plot(traj[:, 1], label='theta_dot')
    plt.xlabel('Time step')
    plt.ylabel('State trajectories')
    plt.legend()
    plt.savefig('figures_trained/exampleTrajectory.png')

    """Plot state values"""
    values_array = np.zeros([len(theta), len(thetadot)])
    
    for i in range(len(theta)): 
        for j in range(len(thetadot)): 
                s = torch.tensor([theta[i], thetadot[j]]).float()
                v = torch.max(policy_net(s)).detach()
                values_array[i,j] = v
    
    fig2, ax2 = plt.subplots()
    c = ax2.contourf(theta, thetadot, values_array, alpha = .9)
    ax2.set_xlabel(r'$\theta$')
    ax2.set_ylabel(r'$\dot{\theta}$')
    ax2.set_title('State Values')
    cbar = fig2.colorbar(c)
    cbar.ax.set_ylabel('value')
    fig2.savefig('figures_trained/StateValues.png')

    """Save an animation"""
    policy = lambda s: (policy_net(torch.tensor(s, dtype=torch.float32).unsqueeze(0)).max(1)[1].view(1, 1)).item()
    env.video(policy, filename='figures_trained/trained_pendulum.gif')

    """For ablation study:"""
    #Scenario 2:
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
    update_freq = 1
    _, reward_Full_NoTarget ,_, _, _ = train_network(n_actions, num_episodes, policy_net, target_net, \
                                            optimizer, memory, env,  batch_size, eps_init, eps_end, GAMMA, \
                                            target_update_frequency = update_freq)
    
    #Scenario 3:
    env = discreteaction_pendulum.Pendulum()
    batch_size = 64
    n_actions = env.num_actions
    n_states = env.num_states
    policy_net = DQN(n_states, n_actions)
    target_net = DQN(n_states, n_actions)
    target_net.load_state_dict(policy_net.state_dict()) #synchronize weights for initialization
    # optimizer = optim.AdamW(policy_net.parameters(), lr = 1e-4, amsgrad=True)
    optimizer = optim.RMSprop(policy_net.parameters(), lr=1e-4)
    memory = ReplayMemory(int(batch_size))
    update_freq = 1000
    _, reward_Full_NoReplay ,_, _, _ = train_network(n_actions, num_episodes, policy_net, target_net, \
                                            optimizer, memory, env,  batch_size, eps_init, eps_end, GAMMA, \
                                            target_update_frequency = update_freq)
    
    #Scenario 4:
    env = discreteaction_pendulum.Pendulum()
    batch_size = 64
    n_actions = env.num_actions
    n_states = env.num_states
    policy_net = DQN(n_states, n_actions)
    target_net = DQN(n_states, n_actions)
    target_net.load_state_dict(policy_net.state_dict()) #synchronize weights for initialization
    # optimizer = optim.AdamW(policy_net.parameters(), lr = 1e-4, amsgrad=True)
    optimizer = optim.RMSprop(policy_net.parameters(), lr=1e-4, alpha = 0.95)
    memory = ReplayMemory(int(batch_size))
    update_freq = 1
    _, reward_Full_NoReplay_NoTarget ,_, _, _ = train_network(n_actions, num_episodes, policy_net, target_net, \
                                            optimizer, memory, env,  batch_size, eps_init, eps_end, GAMMA, \
                                            target_update_frequency = update_freq)
    
    plt.figure(dpi=150)
    plt.plot(reward_Full_standard,label='Standard')
    plt.plot(reward_Full_NoReplay, label='No Replay')
    plt.plot(reward_Full_NoTarget, label='No Target')
    plt.plot(reward_Full_NoReplay_NoTarget, label='No Replay No Target')
    plt.ylabel('Payoff')
    plt.xlabel('Time step')
    plt.legend()
    plt.title('Ablation study')
    plt.savefig('figures_trained/LearningCurve_ablation.png')
    print(f'Avg Q value for standard DQN is {np.mean(reward_Full_standard)}, \
          avg Q value for no replay is {np.mean(reward_Full_NoReplay)},   \
          avg Q value for no target is {np.mean(reward_Full_NoTarget)}, \
          avg Q value for no replay no target is {np.mean(reward_Full_NoReplay_NoTarget)}')


if __name__ == '__main__':
    main()