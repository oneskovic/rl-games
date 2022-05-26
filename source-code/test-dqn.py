from random_generator import get_random_mdp
from policy_improvement import q_value_iteration_matrix_form
from dqn_agent import DQNAgent
import numpy as np
from collections import deque 
import torch
import matplotlib.pyplot as plt
from replay_buffer2 import ReplayBuffer
from mdp_env import MDPEnv
import gym

def eval_agent(env, agent : DQNAgent, episode_cnt, max_step_cnt):
    total_reward = 0
    for _ in range(episode_cnt):
        current_state = env.reset()
        for _ in range(max_step_cnt):
            action = int(agent.select_action(current_state,greedy=True))
            next_state, reward, done, _ = env.step(action)
            current_state = next_state
            total_reward += reward
            if done:
                break
    print(f'Average reward: {total_reward/episode_cnt}')
    return total_reward / episode_cnt

env = gym.make('CartPole-v1')
dqn_agent = DQNAgent((4,),2)

episode_cnt = 10000
total_train_steps = 1000000
eval_freq = 100
max_buffer_len = 10000
swap_interval = 1000
batch_size = 64
learning_starts = 100
reward_history = []
max_t = 100

buffer = ReplayBuffer(env, learning_starts)

avg_reward = 0
for episode in range(1,1001):
    current_state = env.reset()
    done = False
    while not done:
        action = int(dqn_agent.select_action(current_state))
        next_state, reward, done, _ = env.step(action)
        avg_reward += reward
        buffer.add(current_state, action, reward, next_state, done)
        current_state = next_state
        
        # Train the agent on random sample
        if buffer.can_sample():
            sample = buffer.sample_batch(batch_size)
            loss = dqn_agent.update_batch(sample)

    print(f'Episode {episode} Average reward: {avg_reward/episode} Epsilon: {dqn_agent.eps}')
    if episode % eval_freq == 0:
            reward_history.append(eval_agent(env, dqn_agent, 5, max_t))
plt.plot(reward_history)
plt.show()
