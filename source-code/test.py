import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import gym
import random
import numpy as np
import matplotlib.pyplot as plt
from replay_buffer2 import ReplayBuffer

env = gym.make('CartPole-v1')
observation_space = env.observation_space.shape[0]
action_space = env.action_space.n

EPISODES = 1000
LEARNING_RATE = 0.0001
MEM_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.95
EXPLORATION_MAX = 1.0
EXPLORATION_DECAY = 0.999
EXPLORATION_MIN = 0.001

FC1_DIMS = 64
FC2_DIMS = 64
DEVICE = torch.device("cpu")

best_reward = 0
average_reward = 0
episode_number = []
average_reward_number = []

class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_shape = env.observation_space.shape
        self.action_space = action_space

        self.fc1 = nn.Linear(*self.input_shape, FC1_DIMS)
        self.fc2 = nn.Linear(FC1_DIMS, FC2_DIMS)
        self.fc3 = nn.Linear(FC2_DIMS, self.action_space)

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.loss = nn.MSELoss()
        self.to(DEVICE)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class DQN_Solver:
    def __init__(self):
        self.memory = ReplayBuffer(env, BATCH_SIZE)
        self.exploration_rate = EXPLORATION_MAX
        self.network = Network()

    def choose_action(self, observation, greedy=False):
        if greedy:
            state = torch.tensor(observation).float().detach()
            state = state.to(DEVICE)
            state = state.unsqueeze(0)
            q_values = self.network(state)
            return torch.argmax(q_values).item()
    
        else:
            if random.random() < self.exploration_rate:
                return env.action_space.sample()
            
            state = torch.tensor(observation).float().detach()
            state = state.to(DEVICE)
            state = state.unsqueeze(0)
            q_values = self.network(state)
            return torch.argmax(q_values).item()
    
    def learn(self):
        if not self.memory.can_sample():
            return
        
        states, actions, rewards, states_, dones = self.memory.sample_batch(BATCH_SIZE)
        states = torch.tensor(states , dtype=torch.float32).to(DEVICE)
        actions = torch.tensor(actions, dtype=torch.long).to(DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
        states_ = torch.tensor(states_, dtype=torch.float32).to(DEVICE)
        dones = torch.tensor(dones, dtype=torch.bool).to(DEVICE)

        q_values = self.network(states)
        next_q_values = self.network(states_)
        
        predicted_value_of_now = q_values.gather(1, actions.reshape(-1,1)).flatten()
        predicted_value_of_future = next_q_values.max(1)[0]
        
        q_target = rewards + GAMMA * predicted_value_of_future * (~dones)

        loss = self.network.loss(q_target, predicted_value_of_now)
        self.network.optimizer.zero_grad()
        loss.backward()
        for param in self.network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.network.optimizer.step()

        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

def eval_agent(env, agent : DQN_Solver, episode_cnt, max_step_cnt):
    total_reward = 0
    for _ in range(episode_cnt):
        current_state = env.reset()
        for _ in range(max_step_cnt):
            action = int(agent.choose_action(current_state,greedy=True))
            next_state, reward, done, _ = env.step(action)
            current_state = next_state
            total_reward += reward
            if done:
                break
    print(f'Average reward: {total_reward/episode_cnt}')
    return total_reward / episode_cnt

agent = DQN_Solver()

for i in range(1, EPISODES):
    state = env.reset()
    score = 0

    while True:
        #env.render()
        action = agent.choose_action(state)
        state_, reward, done, info = env.step(action)
        agent.memory.add(state, action, reward, state_, done)
        agent.learn()
        state = state_
        score += reward

        if done:
            if score > best_reward:
                best_reward = score
            average_reward += score 
            print("Episode {} Average Reward {} Best Reward {} Last Reward {} Epsilon {}".format(i, average_reward/i, best_reward, score, agent.exploration_rate))
            break
            
    episode_number.append(i)
    average_reward_number.append(average_reward/i)

    if i % 100 == 0:
        eval_agent(env, agent, 10, 200)

plt.plot(episode_number, average_reward_number)
plt.show()