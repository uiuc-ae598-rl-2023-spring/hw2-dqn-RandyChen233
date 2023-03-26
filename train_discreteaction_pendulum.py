import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import discreteaction_pendulum
import matplotlib.pyplot as plt

import random
from collections import deque

class DQNAgent:
    def __init__(self, state_size, action_size, buffer_size=int(1e5), batch_size=64, gamma=0.95, tau=1e-3, lr=5e-4):
        self.state_size = state_size
        self.action_size = action_size
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.target_update_frequency = 5
        self.steps = 0
        
        # Define two neural networks for the online and target models
        self.online_model = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
        self.target_model = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
        # Initialize the weights of the target network to be the same as the online network
        self.target_model.load_state_dict(self.online_model.state_dict())
        
        # Define the optimizer and loss function for training the online network
        self.optimizer = optim.Adam(self.online_model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()
        
    def act(self, state, eps=0.):
        """
        Select an action to take given the current state
        """
        if np.random.rand() > eps:
            state = torch.from_numpy(state).float().unsqueeze(0)
            self.online_model.eval()
            with torch.no_grad():
                action_values = self.online_model(state)
            self.online_model.train()
            action = np.argmax(action_values.cpu().data.numpy())
        else:
            action = np.random.choice(np.arange(self.action_size))
        return action
        
    def remember(self, state, action, reward, next_state, done):
        """
        Add a transition to the replay buffer and potentially train the network
        """
        self.buffer.append((state, action, reward, next_state, done))
        if len(self.buffer) > self.batch_size:
            self.replay()

        self.steps += 1
            

    def replay(self):
        """
        Update the weights of the online network based on a batch of experiences
        """
        
        if len(self.buffer) < self.batch_size:
            return
        
        batch = random.sample(self.buffer, min(len(self.buffer), self.batch_size))
        # Sample a batch of experiences from the replay buffer
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.from_numpy(np.vstack(states)).float()
        actions = torch.from_numpy(np.vstack(actions)).long()
        rewards = torch.from_numpy(np.vstack(rewards)).float()
        next_states = torch.from_numpy(np.vstack(next_states)).float()
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float()
        
        # Compute the TD targets using the target network
        with torch.no_grad():
            Q_targets_next = self.target_model(next_states).detach().max(1)[0].unsqueeze(1)
            Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        # Compute the Q values using the online network
        Q_expected = self.online_model(states).gather(1, actions)
        
        # Compute the loss and update the weights of the online network
        loss = self.loss_fn(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the target network weights
        if self.steps % self.target_update_frequency == 0:
            self.target_model.load_state_dict(self.online_model.state_dict())



def main():

    env = discreteaction_pendulum.Pendulum()
    state_size = env.num_states
    action_size = env.num_actions

    agent = DQNAgent(state_size, action_size)

    # EPISODES = 200
    EPISODES = 20

    """Train the agent"""
    scores = []
    traj = []
    actions = []
    for episode in range(EPISODES):
        state = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            score += reward
            state = next_state
            agent.replay()  # Train the agent on a batch of experiences

        scores.append(score)
        traj.append(state)
        actions.append(action)

        if episode % 10 == 0:
            print(f"Episode: {episode}/{EPISODES}, Score: {score}")
    # print(f'the actions taken are {actions}\n')
    """Plot the learning curve"""
    plt.figure(dpi=150)
    plt.plot(scores)
    plt.title('Learning Curve')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig('figures_trained/LearningCurve.png')

    """Plot an example trajectory"""
    traj = np.array(traj)
    plt.figure(dpi=150)
    plt.plot(traj[:, 0], label='theta')
    plt.plot(traj[:, 1], label='theta_dot')
    plt.legend()
    plt.title('Example Trajectory')
    plt.xlabel('Time Step')
    plt.ylabel('State Value')
    plt.savefig('figures_trained/ExampleTrajectory.png')
    

    """Making an animated trajectory"""
  
    # convert actions list to numpy array
    policy = lambda s: actions[int(s) % len(actions)]

    # call the video function with the modified policy function
    env.video(policy, filename='figures_trained/trained_pendulum.gif')


if __name__ == '__main__':
    main()

