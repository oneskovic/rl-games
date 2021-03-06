from dqn_agent import DQNAgent
import numpy as np
import torch
import matplotlib.pyplot as plt
from replay_buffer import ReplayBuffer
import gym
import pickle

from config import TORCH_DEVICE

def transform_state(state):
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

env = wrap_deepmind(gym.make("ALE/Pong-v5"))

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
dqn_agent = DQNAgent(env.observation_space.shape,env.action_space.n,architecture=dqn_architecture, lr=0.00025,eps_min=0.1)

episode_cnt = 2000
total_train_steps = 100000
eval_freq = 100
max_buffer_len = 100000
swap_interval = 3000
batch_size = 32
learning_starts = 10000
max_t = 10000
eval_episodes = 10
model_save_freq = 100
reward_history = []
training_reward_history = []

buffer = ReplayBuffer((4,84,84), learning_starts, max_buffer_len)

learning_started = False
for episode in range(1,episode_cnt+1):
    current_state = env.reset()
    current_state = transform_state(current_state)
    
    done = False
    episode_reward = 0
    while not done:
        action = int(dqn_agent.select_action(current_state))
        next_state, reward, done, _ = env.step(action)
        next_state = transform_state(next_state)
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

    training_reward_history.append(episode_reward)
    avg_reward = np.mean(training_reward_history[-30:])
    print(f'Episode {episode} Average reward: {avg_reward:.2f} Epsilon: {dqn_agent.eps:.3f}')
    if episode % eval_freq == 0:
            reward_history.append(eval_agent(env, dqn_agent, eval_episodes, max_t))
    if episode % model_save_freq == 0:
        pickle.dump(dqn_agent.model.cpu(), open(f'models/{episode}_model.pkl','wb+'))
        dqn_agent.model.to(TORCH_DEVICE)
        

eval_agent(env, dqn_agent, 5, 1000, render=True)
plt.plot(training_reward_history)
plt.show()
