import numpy as np
import torch
TORCH_DEVICE = 'cpu'
class ReplayBuffer:
    def __init__(self, obs_shape, warmup_cnt, max_buffer_length = 10000):
        self.mem_count = 0
        self.warmup_cnt = warmup_cnt
        self.mem_size = max_buffer_length
        self.states = torch.zeros((self.mem_size,) + obs_shape,dtype=torch.uint8).to(TORCH_DEVICE)
        self.actions = torch.zeros(self.mem_size, dtype=int).to(TORCH_DEVICE)
        self.rewards = torch.zeros(self.mem_size, dtype=torch.float32).to(TORCH_DEVICE)
        self.states_ = torch.zeros((self.mem_size,) + obs_shape,dtype=torch.uint8).to(TORCH_DEVICE)
        self.dones = torch.zeros(self.mem_size, dtype=bool).to(TORCH_DEVICE)
    
    def add(self, state, action, reward, state_, done):
        mem_index = self.mem_count % self.mem_size
        
        self.states[mem_index]  = torch.tensor(state, dtype=torch.float32)
        self.actions[mem_index] = action
        self.rewards[mem_index] = max(-1,min(reward,1))
        self.states_[mem_index] = torch.tensor(state_, dtype=torch.float32)
        self.dones[mem_index] =  done

        self.mem_count += 1
    
    def sample_batch(self, batch_size):
        mem_max = min(self.mem_count, self.mem_size)
        batch_indices = torch.randint(0,mem_max,(batch_size,))

        states  = self.states[batch_indices].float()/255.0
        actions = self.actions[batch_indices]
        rewards = self.rewards[batch_indices]
        states_ = self.states_[batch_indices].float()/255.0
        dones   = self.dones[batch_indices]

        return (states, actions, rewards, states_, dones)
    
    def can_sample(self):
        return self.mem_count >= self.warmup_cnt

# Implements a dqn agent using pytorch
import torch
class DQNAgent:
    def __init__(self, input_shape, n_actions, gamma = 0.99, eps = 1.0, eps_min = 0.01, eps_decay = 0.999, lr = 0.0001, target_update_interval = 20, architecture = None):
        self.input_shape = input_shape
        self.n_actions = n_actions

        # Hyperparameters
        self.gamma = gamma
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.learning_rate = lr
        self.target_update_interval = target_update_interval
        self.target_update_remaing = self.target_update_interval

        if architecture is None:
            # Model architecture
            architecture = [
                torch.nn.Linear(input_shape[0], 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, n_actions)
            ]

        # Initialize the model and target model, optimizer and loss function
        self.model = torch.nn.Sequential(*architecture).to(TORCH_DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate)
        self.target_model = torch.nn.Sequential(*architecture).to(TORCH_DEVICE)
        self.target_model.load_state_dict(self.model.state_dict())
        self.loss_fn = torch.nn.SmoothL1Loss()

    def select_action(self, state, greedy=False):
        state = torch.from_numpy(state).float().unsqueeze(0).to(TORCH_DEVICE)
        state /= 255.0
        
        # Select action greedily (used when evaluating)
        if greedy: 
            with torch.no_grad():
                return self.model(state).argmax()
        # Select action epsilon greedy
        else:      
            if torch.rand(1) > self.eps:
                with torch.no_grad():
                    return self.model(state).argmax()
            else:
                return torch.randint(0, self.n_actions, (1, 1))

    def update_batch(self, batch_samples):
        batch_states, selected_actions, batch_rewards, batch_next_states, batch_done = batch_samples
        selected_actions = selected_actions.reshape(-1, 1)
        print(selected_actions.shape)

        # Predict Q values using current model
        pred_q = self.model(batch_states)
        # Select only the Q values for the actions that were executed
        pred_q = pred_q.gather(1, selected_actions).flatten()

        # Predict Q values using the target model 
        true_q = self.gamma * self.target_model(batch_next_states).max(1)[0] * (~batch_done) + batch_rewards

        # Compute loss and backprop
        loss = self.loss_fn(pred_q, true_q)
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradient
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Decay epsilon
        if self.eps > self.eps_min:
            self.eps *= self.eps_decay

        # Update target model if needed
        self.target_update_remaing -= 1
        if self.target_update_remaing <= 0:
            self.update_target_model()
            self.target_update_remaing = self.target_update_interval

        return loss.item()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

import numpy as np
import torch
import matplotlib.pyplot as plt
import gym
import pickle
import tqdm
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

def transform_state(state):
    state = np.squeeze(state,0)
    return np.moveaxis(state, -1, 0)

def eval_agent(env, agent : DQNAgent, episode_cnt, max_step_cnt, render=False):
    total_reward = 0
    for _ in range(episode_cnt):
        current_state = env.reset()
        current_state = transform_state(current_state)
        for _ in range(max_step_cnt):
            if render:
                env.render()
            action = int(agent.select_action(current_state,greedy=True))
            next_state, reward, done, _ = env.step(action)
            current_state = next_state
            current_state = transform_state(current_state)
            total_reward += reward
            if done:
                break
    print(f'Average reward: {total_reward/episode_cnt}')
    return total_reward / episode_cnt

total_train_steps = 1000000
max_buffer_len = 100000
swap_interval = 10000
batch_size = 32
learning_starts = 5000
max_t = 10000
eval_freq = 100000
model_save_freq = 100
training_reward_history = []

env = make_atari_env('PongNoFrameskip-v4', n_envs=1, seed=0)
env = VecFrameStack(env, n_stack=4)

dqn_architecture = [
    torch.nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
    torch.nn.ReLU(),
    torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
    torch.nn.ReLU(),
    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
    torch.nn.ReLU(),
    torch.nn.Flatten(1,3),
    torch.nn.Linear(3136, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, env.action_space.n)
]
dqn_agent = DQNAgent(env.observation_space.shape, env.action_space.n, lr=0.00025,eps_min=0.1,eps_decay=0.99999, architecture=dqn_architecture)
buffer = ReplayBuffer((4,84,84), learning_starts, max_buffer_len)

learning_started = False
train_step_cnt = 0
pbar = tqdm.tqdm(total=total_train_steps)
while train_step_cnt < total_train_steps:
    current_state = env.reset()
    current_state = transform_state(current_state)
    
    done = False
    episode_reward = 0
    while not done:
        action = int(dqn_agent.select_action(current_state))
        next_state, reward, done, _ = env.step([action])
        next_state = transform_state(next_state)
        reward = int(reward)
        done = bool(done)
        episode_reward += reward
        buffer.add(current_state, action, reward, next_state, done)
        if buffer.mem_count >= learning_starts and not learning_started:
            print('----- Learning started -----')
            learning_started = True
        current_state = next_state
        
        # Train the agent on random sample
        if buffer.can_sample():
            sample = buffer.sample_batch(batch_size)
            loss = dqn_agent.update_batch(sample)
            pbar.update(1)
            train_step_cnt += 1

    training_reward_history.append(episode_reward)
    avg_reward = np.mean(training_reward_history[-30:])
    pbar.set_description(f'Average reward: {avg_reward:.2f} Epsilon: {dqn_agent.eps:.3f} Batch size: {batch_size}')
