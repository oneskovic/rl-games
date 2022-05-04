import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network

def show_mdp(transition_probs, rewards, is_action_possible, limit_prob = 0.001, limit_reward = 0.001):
    nt = Network('1000px', '1000px', directed=True)

    weight_multiplier = 1.5
    state_cnt = transition_probs.shape[0]
    action_cnt = transition_probs.shape[1]
    # Add state nodes
    for s in range(state_cnt):
        nt.add_node(f'S{s}', label=f'S{s}', group='state')
    # Add action nodes and edges from state to allowed actions
    for s in range(state_cnt):
        for a in range(action_cnt):
            if is_action_possible[s, a]:
                state_node_name = f'S{s}'
                action_node_name = f'S{s}, A{a}'
                nt.add_node(action_node_name, label=f'A{a}', group='action')
                nt.add_edge(state_node_name, action_node_name)

    # Add edges between state nodes and action nodes
    for s in range(state_cnt):
        for a in range(action_cnt):
            if is_action_possible[s, a]:
                state_node_name = f'S{s}'
                action_node_name = f'S{s}, A{a}'
                for s_prime in range(state_cnt):
                    if transition_probs[s, a, s_prime] > limit_prob:
                        reward = rewards[s, a, s_prime]
                        if abs(reward) > limit_reward:
                            nt.add_edge(action_node_name, f'S{s_prime}', weight = weight_multiplier*transition_probs[s,a,s_prime], label = f'p:{transition_probs[s,a,s_prime]}\r\nr:{reward:.2f}')
                        else:
                            nt.add_edge(action_node_name, f'S{s_prime}', weight = weight_multiplier*transition_probs[s,a,s_prime], label = f'p:{transition_probs[s,a,s_prime]}')

    nt.show_buttons()
    nt.show('mdp.html')

def show_markov_chain(transition_probs, rewards, limit_prob = 0.001, limit_reward = 0.001):
    nt = Network('1000px', '1000px', directed=True)

    weight_multiplier = 1.5
    state_cnt = transition_probs.shape[0]
    action_cnt = transition_probs.shape[1]
    # Add state nodes
    for s in range(state_cnt):
        nt.add_node(f'S{s}', label=f'S{s}', group='state')
    
    for s in range(state_cnt):
        for s_prime in range(state_cnt):
            if transition_probs[s, s_prime] > limit_prob:
                reward = rewards[s, s_prime]
                if abs(reward) > limit_reward:
                    nt.add_edge(f'S{s}', f'S{s_prime}', label = f'p:{transition_probs[s,s_prime]:.2f}\r\nr:{reward:.2f}')
                else:
                    nt.add_edge(f'S{s}', f'S{s_prime}', label = f'p:{transition_probs[s,s_prime]:.2f}')

    nt.show_buttons()
    nt.show('markov_chain.html')

def example_mdp():
    state_cnt = 3
    action_cnt = 2
    transition_probs = np.zeros((state_cnt, action_cnt, state_cnt))
    is_action_possible = np.ones((state_cnt, action_cnt), dtype=bool)
    rewards = np.zeros((state_cnt, action_cnt, state_cnt))

    transition_probs[0, 0, 0] = 0.5
    transition_probs[0, 0, 2] = 0.5
    transition_probs[0, 1, 2] = 1

    transition_probs[1, 0, 0] = 0.7
    rewards[1, 0, 0] = +5
    transition_probs[1, 0, 1] = 0.1
    transition_probs[1, 0, 2] = 0.2
    transition_probs[1, 1, 0] = 0
    transition_probs[1, 1, 1] = 0.95
    transition_probs[1, 1, 2] = 0.05

    transition_probs[2,0,0] = 0.4
    transition_probs[2,0,1] = 0
    transition_probs[2,0,2] = 0.6
    transition_probs[2,1,0] = 0.3
    rewards[2,1,0] = -1
    transition_probs[2,1,1] = 0.3
    transition_probs[2,1,2] = 0.4

    is_action_possible[0, 1] = False
    is_action_possible[1, 1] = False
    is_action_possible[2, 0] = False

    show_mdp(transition_probs, rewards, is_action_possible)

def get_chain():
    state_cnt = 3
    action_cnt = 2
    transition_probs = np.zeros((state_cnt, state_cnt))
    rewards = np.zeros((state_cnt, state_cnt))

    transition_probs[0, 0] = 0.5
    transition_probs[0, 1] = 0
    transition_probs[0, 2] = 0.5

    transition_probs[1, 0] = 0.7
    rewards[1, 0] = +5
    transition_probs[1, 1] = 0.1
    transition_probs[1, 2] = 0.2

    transition_probs[2, 0] = 0.3
    rewards[2, 0] = -1
    transition_probs[2, 1] = 0.3
    transition_probs[2, 2] = 0.4
    return transition_probs, rewards

def get_random_chain():
    state_cnt = 10
    transition_probs = np.random.ranf((state_cnt, state_cnt))
    transition_probs = transition_probs / np.sum(transition_probs, axis=1, keepdims=True)
    rewards = np.random.randint(-5, 5, (state_cnt, state_cnt))
    return transition_probs, rewards

def mdp_under_policy():
    transition_probs, rewards = get_chain()
    show_markov_chain(transition_probs, rewards)

def evaluate_markov_chain(transition_probs, rewards, start_state, max_steps = 100, episode_cnt = 10000, gamma = 0.9):
    state_cnt = transition_probs.shape[0]
    total_reward = 0.0
    avg_reward_history = []
    for e in range(episode_cnt):
        current_reward = 0.0
        current_state = start_state
        for step in range(max_steps):
            next_state = np.random.choice(state_cnt, p=transition_probs[current_state, :])
            reward = rewards[current_state, next_state]
            current_reward += gamma**step * reward
            current_state = next_state
        total_reward += current_reward
        avg_reward_history.append(total_reward/(e+1))
    return total_reward / (episode_cnt+1)

def evaluate_markov_chain_as_limit(transition_probs, rewards, start_state, gamma = 0.9, max_t = 100):
    # Find the probability to endup in each state after time t
    state_cnt = transition_probs.shape[0]
    prob = np.zeros((max_t, state_cnt))
    prob[0, start_state] = 1
    for t in range(1, max_t):
        prob[t,:] = np.matmul(prob[t-1,:], transition_probs)

    # expected_reward = np.zeros((max_t,state_cnt))
    # for i in range(1, max_t):
    #     expected_reward[i] = gamma**i * np.matmul(prob[i-1], np.multiply(rewards, transition_probs)) + np.matmul(np.multiply(prob[i-1], expected_reward[i-1]), transition_probs)
    # expected_reward = np.matmul(k,expected_reward)

    expected_reward = 0.0
    m = np.matmul(np.multiply(transition_probs, rewards), np.ones(state_cnt))
    for i in range(1, max_t):
        expected_reward += gamma**(i-1) * np.matmul(prob[i-1], m)

    return expected_reward

transition_probs, rewards = get_random_chain()
print(evaluate_markov_chain(transition_probs, rewards, 0))
print(evaluate_markov_chain_as_limit(transition_probs, rewards, 0))