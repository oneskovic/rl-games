
def evaluate_markov_chain(transition_probs, rewards, start_state, max_steps = 6, episode_cnt = 10000, gamma = 0.9):
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

def evaluate_markov_chain_as_limit(transition_probs, rewards, start_state, gamma = 0.9, max_t = 6):
    # Find the probability to endup in each state after time t
    state_cnt = transition_probs.shape[0]
    prob = np.zeros((max_t, state_cnt))
    prob[0, start_state] = 1
    for t in range(1, max_t):
        prob[t,:] = np.matmul(prob[t-1,:], transition_probs)

    expected_reward = 0.0
    m = np.matmul(np.multiply(transition_probs, rewards), np.ones(state_cnt))
    for i in range(1, max_t+1):
        expected_reward += gamma**(i-1) * np.matmul(prob[i-1], m)

    return expected_reward

def evaluate_markov_chain_closed_form(transition_probs, rewards, start_state, gamma = 0.9, max_t=100):
    state_cnt = transition_probs.shape[0]
    l0 = np.zeros(state_cnt)
    l0[start_state] = 1
    vec_c = np.multiply(transition_probs, rewards).sum(axis=1)
    s = np.linalg.inv(np.eye(state_cnt) - gamma*transition_probs)
    return np.matmul(vec_c,np.matmul(l0,s))

def recursive_expected_reward(transition_probs, rewards, start_state, gamma, max_t, current_path, current_t = 0):
    state_cnt = transition_probs.shape[0]
    if current_t == max_t:
        p = 1.0
        r = 0.0
        for i,s in enumerate(current_path):
            if i >= 1:
                prev_s = current_path[i-1]
                p *= transition_probs[prev_s, s]
                r += gamma**(i-1) * rewards[prev_s, s]
        return p * r        
    elif current_t == 0:
        current_path[current_t] = start_state
        return recursive_expected_reward(transition_probs, rewards, start_state, gamma, max_t, current_path, current_t+1)
    else:
        expected_reward = 0.0
        for s in range(state_cnt):
            current_path[current_t] = s
            expected_reward += recursive_expected_reward(transition_probs, rewards, s, gamma, max_t, current_path, current_t+1)
        return expected_reward

def evaluate_chain_bruteforce(transition_probs, rewards, start_state, gamma = 0.9, max_t = 6):
    path = np.zeros(max_t+1, dtype=int)
    return recursive_expected_reward(transition_probs, rewards, start_state, gamma, max_t+1, path)

def test_policy_eval():
    transition_probs, rewards = get_random_chain()
    print(evaluate_chain_bruteforce(transition_probs, rewards, 0))
    print(evaluate_markov_chain(transition_probs, rewards, 0))
    print(evaluate_markov_chain_as_limit(transition_probs, rewards, 0, max_t=100))
    print(evaluate_markov_chain_closed_form(transition_probs, rewards, 0))