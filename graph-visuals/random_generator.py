def get_random_chain():
    state_cnt = 10
    transition_probs = np.random.ranf((state_cnt, state_cnt))
    transition_probs = transition_probs / np.sum(transition_probs, axis=1, keepdims=True)
    rewards = np.random.randint(-5, 5, (state_cnt, state_cnt))
    return transition_probs, rewards

def get_random_mdp(state_cnt = 5, action_cnt = 2, reward_limit = 5):
    transition_probs = np.random.ranf((state_cnt, action_cnt, state_cnt))
    transition_probs = transition_probs / np.sum(transition_probs, axis=2, keepdims=True)
    rewards = np.random.randint(-reward_limit, reward_limit, (state_cnt, action_cnt, state_cnt))
    return transition_probs, rewards
