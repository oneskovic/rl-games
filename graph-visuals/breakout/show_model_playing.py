from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import DQN
import time
model = DQN.load('graph-visuals/breakout/breakout_model.zip')
env = make_atari_env('BreakoutNoFrameskip-v4', n_envs=1, seed=0)
env = VecFrameStack(env, n_stack=4)
obs = env.reset()
for _ in range(5):
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        time.sleep(0.02)
print(obs)