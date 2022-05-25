from random_generator import get_random_mdp
from policy_improvement import q_value_iteration_matrix_form
from dqn_agent import DQNAgent
import numpy as np
from collections import deque 
import torch
import matplotlib.pyplot as plt
from replay_buffer import ReplayBuffer
from mdp_env import MDPEnv

start_state = 0
state_cnt = 10
action_cnt = 5

def eval_agent(env, agent : DQNAgent, episode_cnt, max_step_cnt):
    total_reward = 0
    for _ in range(episode_cnt):
        current_state = env.reset()
        for _ in range(max_step_cnt):
            action = agent.select_action(onehot_current_state(current_state),greedy=True)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
    return total_reward / episode_cnt

def onehot_current_state(current_state):
    ohe = np.zeros(state_cnt)
    ohe[current_state] = 1
    return torch.tensor(ohe).float()

env = MDPEnv(state_cnt, action_cnt, max_step_cnt=200)
optimal_q = q_value_iteration_matrix_form(env.transition_probs, env.rewards, start_state, gamma=0.99, max_t=100)
dqn_agent = DQNAgent((state_cnt,),action_cnt)

episode_cnt = 200
max_buffer_len = 20000
swap_interval = 300
batch_size = 16
loss_history = []
reward_history = []
max_t = 100
buffer = ReplayBuffer(max_buffer_len, 500)
for e in range(episode_cnt):
    if e % 5 == 4:
        reward_history.append(eval_agent(env, dqn_agent, 5, max_t))
    current_state = onehot_current_state(env.reset())
    done = False
    total_loss = 0.0
    t = 0
    while not done:
        action = int(dqn_agent.select_action(current_state))
        next_state, reward, done, _ = env.step(action)
        next_state = onehot_current_state(next_state)
        buffer.add((current_state, action, reward, next_state, done))
        current_state = next_state
        
        # Train the agent on random sample
        if buffer.can_sample():
            sample = buffer.sample_batch(batch_size)
            loss = dqn_agent.update_batch(sample)
            total_loss += loss

        if (e*max_t + t) % swap_interval == swap_interval-1:
            dqn_agent.swap_models()
        t += 1
        
    total_loss /= max_t
    print("Episode: {}, total loss: {}".format(e, total_loss))
    loss_history.append(total_loss)

plt.plot(reward_history)
plt.show()

plt.plot(loss_history)
plt.show()
final_q = np.zeros((state_cnt, action_cnt))
for s in range(state_cnt):
    for a in range(action_cnt):
        final_q[s,a] = dqn_agent.get_q_value(onehot_current_state(s), a)
diff = np.abs(final_q - optimal_q)
print(diff)
print(final_q)
print(optimal_q)
print(diff.flatten().mean() / optimal_q.mean())