import gym
from gym import spaces
from random_generator import get_random_mdp
import numpy as np
class MDPEnv(gym.Env):

  def __init__(self, state_cnt, action_cnt, max_step_cnt):
    super(MDPEnv, self).__init__()
    self.action_space = spaces.Discrete(action_cnt)
    self.observation_space = spaces.Discrete(state_cnt)
    self.start_state = 0
    self.max_step_cnt = max_step_cnt
    self.transition_probs, self.rewards = get_random_mdp(state_cnt, action_cnt, 10)

  def step(self, action):
    next_state = np.random.choice(range(self.observation_space.n), p=self.transition_probs[self.current_state, action, :])
    reward = self.rewards[self.current_state, action, next_state]
    self.current_state = next_state
    self.current_step_cnt += 1
    return self.current_state, reward, self.current_step_cnt >= self.max_step_cnt, None
    
  def reset(self):
    self.current_state = self.start_state
    self.current_step_cnt = 0
    return self.current_state

  def render(self, mode='human', close=False):
    pass