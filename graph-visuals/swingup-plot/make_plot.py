import numpy as np
import matplotlib.pyplot as plt
import pickle

data = pickle.load(open('graph-visuals/swingup-plot/training_reward_history.pkl','rb'))
N = 10
rm = np.convolve(data, np.ones(N)/N, mode='valid')
# plt.title('Prosek nagrada u zadnjih 10 epizoda tokom treniranja na problemu CartPole SwingUp', fontsize=18)
plt.xlabel('Epizoda', fontsize=16)
plt.ylabel('Nagrada', fontsize=16)
plt.plot(rm)
plt.show()