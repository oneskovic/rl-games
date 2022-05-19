import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
from random_generator import get_random_chain, get_random_mdp

def evaluate_random_policy_mdp(transition_probs, rewards, start_state, gamma = 0.9, max_t = 100, max_episodes = 100):
    expected_reward = 0.0
    state_cnt = transition_probs.shape[0]
    action_cnt = transition_probs.shape[1]
    for episode in range(max_episodes):
        episode_reward = 0.0
        current_state = start_state
        for step in range(max_t):
            action = np.random.randint(action_cnt)
            next_state = np.random.choice(state_cnt, p=transition_probs[current_state, action, :])
            reward = rewards[current_state, action, next_state]
            episode_reward += gamma**step * reward
            current_state = next_state
        expected_reward += episode_reward
    return expected_reward / max_episodes

def q_value_iteration(transition_probs, rewards, start_state, gamma = 0.9, max_t = 100):
    state_cnt = transition_probs.shape[0]
    action_cnt = transition_probs.shape[1]
    prev_q_values = np.zeros((state_cnt, action_cnt))
    q_values = np.zeros((state_cnt, action_cnt))
    for t in range(max_t):
        for s in range(state_cnt):
            for a in range(action_cnt):
                q_values[s,a] = 0.0
                for s_ in range(state_cnt):
                    q_values[s,a] += transition_probs[s,a,s_] * (rewards[s,a,s_] + gamma * prev_q_values[s_,:].max())
        prev_q_values = q_values
    return q_values            

def construct_policy(q_values):
    state_cnt = q_values.shape[0]
    action_cnt = q_values.shape[1]
    policy = np.zeros((state_cnt, action_cnt))
    for s in range(state_cnt):
        ind = q_values[s,:].argmax()
        policy[s,ind] = 1
    return policy

def evaluate_policy_for_episodes(transition_probs, rewards, start_state, policy, gamma = 0.9, max_t = 100, episode_cnt = 10):
    total_reward = 0.0
    for e in range(episode_cnt):
        episode_reward = 0.0
        current_state = start_state
        state_cnt = transition_probs.shape[0]
        action_cnt = transition_probs.shape[1]
        for step in range(max_t):
            action = np.random.choice(action_cnt,p=policy[current_state,:])
            next_state = np.random.choice(state_cnt, p=transition_probs[current_state, action, :])
            reward = rewards[current_state, action, next_state]
            episode_reward += gamma**step * reward
            current_state = next_state
        total_reward += episode_reward
    return total_reward

def evaluate_policy(transition_probs, rewards, start_state, policy, gamma = 0.9, max_t = 100, max_episodes = 100, thread_count = 10):
    from multiprocessing import Pool
    expected_reward = 0.0
    episodes_per_thread = max_episodes // thread_count
    with Pool(thread_count) as p:
        episodes = p.starmap(evaluate_policy_for_episodes, [(transition_probs, rewards, start_state, policy, gamma, max_t, episodes_per_thread) for _ in range(thread_count)])
    return sum(episodes) / max_episodes

def construct_all_deterministic_policies_recursive(current_policy, index):
    if index == len(current_policy):
        return [current_policy.copy()]
    else:
        state_cnt = current_policy.shape[0]
        action_cnt = current_policy.shape[1]
        all_policies = []
        for a in range(action_cnt):
            current_policy[index,a] = 1
            all_policies += construct_all_deterministic_policies_recursive(current_policy, index+1)
            current_policy[index,a] = 0
        return all_policies

def construct_all_deterministic_policies(state_cnt, action_cnt):
    policy = np.zeros((state_cnt, action_cnt))
    return construct_all_deterministic_policies_recursive(policy, 0)

def test_value_iteration():
    for test_number in range(20):
        transition_probs, rewards = get_random_mdp()
        state_cnt = transition_probs.shape[0]
        action_cnt = transition_probs.shape[1]
        possible_actions = np.ones((state_cnt, action_cnt))
        all_deterministic_policies = construct_all_deterministic_policies(state_cnt, action_cnt)

        # uniform_random_policy = np.ones((state_cnt, action_cnt)) / action_cnt
        # print('Uniform random policy:',evaluate_policy(transition_probs, rewards, start_state=0, policy=uniform_random_policy))
        optimal_policy = construct_policy(q_value_iteration(transition_probs, rewards, start_state=0))
        optimal_policy_res = evaluate_policy(transition_probs, rewards, start_state=0, policy=optimal_policy, max_episodes=300)
        # print('Optimal policy result:',evaluate_policy(transition_probs, rewards, start_state=0, policy=optimal_policy))
        # print('Optimal policy', optimal_policy)

        best_result = -1e10
        best_policy = None
        for i,p in enumerate(all_deterministic_policies):
            # print(f'{i}         ', end='\r')
            result = evaluate_policy(transition_probs, rewards, start_state=0, policy=p)
            if result > best_result:
                best_result = result
                best_policy = p
        # print('\n')
        # print('Best result:',best_result)
        # print('Best policy:',best_policy)
        best_result = evaluate_policy(transition_probs, rewards, start_state=0, policy=best_policy, max_episodes=300)
        delta = best_result - optimal_policy_res
        print(f'Test {test_number}, difference: {delta:.3f}')
        if not np.array_equal(best_policy, optimal_policy):
            print('Policies different')
            print(best_policy)
            print(optimal_policy)
# transition_probs, rewards = get_random_chain()
# print(evaluate_chain_bruteforce(transition_probs, rewards, 0))
# print(evaluate_markov_chain(transition_probs, rewards, 0))
# print(evaluate_markov_chain_as_limit(transition_probs, rewards, 0, max_t=100))
# print(evaluate_markov_chain_closed_form(transition_probs, rewards, 0)) 
if __name__ == '__main__':
    test_value_iteration()