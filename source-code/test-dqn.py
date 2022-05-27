from dqn_agent import DQNAgent
import numpy as np
import torch
import matplotlib.pyplot as plt
from replay_buffer import ReplayBuffer
import gym

def eval_agent(env, agent : DQNAgent, episode_cnt, max_step_cnt, render=False):
    total_reward = 0
    for _ in range(episode_cnt):
        current_state = env.reset()
        for _ in range(max_step_cnt):
            if render:
                env.render()
            action = int(agent.select_action(current_state,greedy=True))
            next_state, reward, done, _ = env.step(action)
            current_state = next_state
            total_reward += reward
            if done:
                break
    print(f'Average reward: {total_reward/episode_cnt}')
    return total_reward / episode_cnt

env = gym.make('CartPole-v1')

dqn_architecture = [
    torch.nn.Linear(env.observation_space.shape[0], 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, env.action_space.n)
]
dqn_agent = DQNAgent((4,),2,architecture=dqn_architecture)

episode_cnt = 800
total_train_steps = 1000000
eval_freq = 100
max_buffer_len = 10000
swap_interval = 1000
batch_size = 64
learning_starts = 100
max_t = 1000
eval_episodes = 10
reward_history = []
training_reward_history = []

buffer = ReplayBuffer(env, learning_starts)

for episode in range(1,episode_cnt+1):
    current_state = env.reset()
    done = False
    episode_reward = 0
    while not done:
        action = int(dqn_agent.select_action(current_state))
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        buffer.add(current_state, action, reward, next_state, done)
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

eval_agent(env, dqn_agent, 5, 1000, render=True)
plt.plot(training_reward_history)
plt.show()
