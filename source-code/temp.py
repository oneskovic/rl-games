from ray.rllib.env.wrappers.atari_wrappers import wrap_deepmind
import gym
env = wrap_deepmind(gym.make("Breakout-v0"))
print(env.reset()) 