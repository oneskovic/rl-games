import numpy as np
import torch
import matplotlib.pyplot as plt
import gym
import pickle
import tqdm
from dqn_agent import DQNAgent
from replay_buffer import ReplayBuffer

def transform_state(state):
    return state

def update_batch_size(step):
    return int(32 + 0.003 * step)

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
reward_history = []
training_reward_history = []

env = gym.make("Acrobot-v1")
dqn_agent = DQNAgent(env.observation_space.shape, env.action_space.n, lr=0.00025,eps_min=0.1,eps_decay=0.99999)
buffer = ReplayBuffer(env.observation_space.shape, learning_starts, max_buffer_len)

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
            pbar.update(1)
            train_step_cnt += 1
            batch_size = update_batch_size(train_step_cnt)
            if train_step_cnt % eval_freq == 0:
                eval_agent(env, dqn_agent, 1, max_t, render=True)

    training_reward_history.append(episode_reward)
    avg_reward = np.mean(training_reward_history[-30:])
    pbar.set_description(f'Average reward: {avg_reward:.2f} Epsilon: {dqn_agent.eps:.3f} Batch size: {batch_size}')

        
    
eval_agent(env, dqn_agent, 3, 1000, render=True)
pickle.dump(training_reward_history, open('training_reward_history.pkl', 'wb+'))
plt.plot(training_reward_history)
plt.show()
