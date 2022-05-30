import gym
env = gym.make('BreakoutNoFrameskip-v4')
obs = env.reset()
import matplotlib.pyplot as plt
import numpy as np
for _ in range(20):
    obs, _, _, _ = env.step(env.action_space.sample())

#plt.imsave('graph-visuals/breakout/breakout_sample.png', obs)

from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
env = make_atari_env('BreakoutNoFrameskip-v4', n_envs=1, seed=0)
env = VecFrameStack(env, n_stack=4)
obs = env.reset()
for _ in range(50):
    obs, _, _, _ = env.step([env.action_space.sample()])
#plt.imsave('graph-visuals/breakout/breakout_sample_preprocess.png', obs[0,:,:,0], cmap='gray', vmin=0, vmax=255)