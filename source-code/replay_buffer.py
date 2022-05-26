from collections import deque
import numpy as np
import torch
MEM_SIZE = 10000
class ReplayBuffer:
    def __init__(self, env, warmup_cnt):
        self.buffer = deque(maxlen=MEM_SIZE)
        self.warmup_cnt = warmup_cnt

    def add(self, s, a, r, s_, d):
        if len(self.buffer) == 0:
            self.state_shape = s.shape
        self.buffer.append((s,a,r,s_,d))

    def sample_batch(self, batch_size):
        if len(self.buffer) < self.warmup_cnt:
            return None
        
        states = torch.zeros((batch_size,)+self.state_shape)
        actions = torch.zeros((batch_size,1), dtype=int)
        rewards = torch.zeros((batch_size,1))
        states_ = torch.zeros((batch_size,)+self.state_shape)
        dones = torch.zeros((batch_size,1), dtype=bool)
        for i in range(batch_size):
            sample = self.buffer[np.random.randint(0, len(self.buffer))]
            s,a,r,s_,d = sample
            states[i] = torch.from_numpy(s)
            actions[i] = a
            rewards[i] = r
            states_[i] = torch.from_numpy(s_)
            dones[i] = d
        return (states, actions, rewards, states_, dones)
    
    def can_sample(self):
        return len(self.buffer) >= self.warmup_cnt