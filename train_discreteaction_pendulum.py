import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import discreteaction_pendulum
import matplotlib.pyplot as plt
import seaborn as sns

import random
from collections import deque

class DQN_standard:
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

    def train(self, env, episodes):
        agent = DQN_standard(env.num_states,env.num_actions)
        """Train the agent"""
        scores = []      #sum of rewards of each episode
        scores_Full = [] #reward at each time step in each episode
        traj = []
        actions = []
        traj_Full = []
        actions_Full = []
        for episode in range(episodes):
            state = env.reset()
            done = False
            score = 0
            while not done:
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                scores_Full.append(reward)
                score += reward
                state = next_state
                if episode == episodes-1:
                    traj.append(state)
                    actions.append(action)

                traj_Full.append(state)
                actions_Full.append(action)
                agent.replay()  # Train the agent on a batch of experiences

            scores.append(score) # append the scores accumulated from each EPISODE

            if episode % 10 == 0:
                print(f"Episode: {episode}/{episodes}, Score: {score}")

        """Plot the learning curve"""
        plt.figure(dpi=150)
        plt.plot(scores)
        plt.title('Learning Curve')
        plt.xlabel('Episode')
        plt.ylabel('Sum of Rewards Per Episode')
        plt.savefig('figures_trained/LearningCurve.png')

        """Plot an example trajectory"""
        traj = np.array(traj)
        plt.figure(dpi=150)
        plt.plot(traj[:, 0], label='theta')
        plt.plot(traj[:, 1], label='theta_dot')
        plt.plot(actions, label='tau')
        plt.legend()
        plt.title('Example Trajectory within An Episode')
        plt.xlabel('Time Step')
        plt.ylabel('State_Action')
        plt.savefig('figures_trained/ExampleTrajectory.png')
        

        """Making an animated trajectory"""
        def policy(s):
        # Select next action from list
            a = actions.pop(0)

            # Return selected action
            return a

        # call the video function with the modified policy function
        env.video(policy, filename='figures_trained/trained_pendulum.gif')

        """Making a heatmap of the policy of a trained agent"""

        plt.figure(dpi=150)
        traj_Full = np.array(traj_Full)
        sns.heatmap(np.squeeze(traj_Full.T[0]), traj_Full.T[1],  cmap = 'coolwarm', vmin=min(actions_Full), \
                    vmax=max(actions_Full), cbar_kws={'label': 'Action'})
        plt.xlabel("Theta")
        plt.ylabel("Theta_dot")


        """Making a heatmap of the value function of a trained agent"""

        plt.figure(dpi=150)
        traj_Full = np.array(traj_Full)
        sns.heatmap(np.squeeze(traj_Full.T[0]), traj_Full.T[1],  cmap = 'coolwarm', vmin=min(scores_Full), \
                    vmax=max(scores_Full), cbar_kws={'label': 'Value'})
        plt.xlabel("Theta")
        plt.ylabel("Theta_dot")

    
def main():

    env = discreteaction_pendulum.Pendulum()
    agent = DQN_standard(env.num_states,env.num_actions)
    episodes = 2000
    agent.train(env, episodes)

    # """Plotting learning curves for ablation study"""
    # # Training parameters
    # num_steps = 10000  # Number of training steps
    # eval_every = 100  # Evaluate the model every eval_every steps

    # # Scenario names
    # scenario_names = ['With replay, with target Q', 'With replay, without target Q', 
    #                 'Without replay, with target Q', 'Without replay, without target Q']

    # # Scenario colors
    # scenario_colors = ['blue', 'green', 'red', 'purple']

    # # Episode rewards for each scenario
    # episode_rewards = []

    # # Train the model for each scenario
    # for i in range(4):
    #     # Code to train the model for scenario i
    #     # ...

    #     # Record the episode rewards at regular intervals during training
    #     rewards = []
    #     for j in range(num_steps // eval_every):
    #         # to evaluate the model and record the episode reward
    #         # ...
    #         rewards.append(episode_reward)
        
    #     episode_rewards.append(rewards)

    # # Plot the learning curves for all four scenarios in the same plot
    # plt.figure(figsize=(10, 8))
    # plt.xlabel('Training steps')
    # plt.ylabel('Episode reward')
    # plt.title('Learning curves')
    # for i in range(4):
    #     plt.plot(range(eval_every, num_steps + 1, eval_every), episode_rewards[i], 
    #             label=scenario_names[i], color=scenario_colors[i])
    # plt.legend()




if __name__ == '__main__':
    main()

