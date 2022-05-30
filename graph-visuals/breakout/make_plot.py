import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd

data = pd.read_csv('graph-visuals/breakout/breakout_data.csv')
reward = data['Value']
step = data['Step']
# N = 10
# rm = np.convolve(reward, np.ones(N)/N, mode='valid')
# plt.title('Prosek nagrada u zadnjih 10 epizoda tokom treniranja na problemu CartPole SwingUp', fontsize=18)
plt.xlabel('Broj koraka (milioni)', fontsize=16)
plt.ylabel('Nagrada', fontsize=16)
plt.plot(step, reward)
plt.show()