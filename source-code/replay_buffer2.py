import numpy as np
import torch
MEM_SIZE = 10000
class ReplayBuffer:
    def __init__(self, env, warmup_cnt):
        self.mem_count = 0
        self.warmup_cnt = warmup_cnt
        self.states = np.zeros((MEM_SIZE, *env.observation_space.shape),dtype=np.float32)
        self.actions = np.zeros(MEM_SIZE, dtype=np.int64)
        self.rewards = np.zeros(MEM_SIZE, dtype=np.float32)
        self.states_ = np.zeros((MEM_SIZE, *env.observation_space.shape),dtype=np.float32)
        self.dones = np.zeros(MEM_SIZE, dtype=np.bool)
    
    def add(self, state, action, reward, state_, done):
        mem_index = self.mem_count % MEM_SIZE
        
        self.states[mem_index]  = state
        self.actions[mem_index] = action
        self.rewards[mem_index] = reward
        self.states_[mem_index] = state_
        self.dones[mem_index] =  done

        self.mem_count += 1
    
    def sample_batch(self, batch_size):
        MEM_MAX = min(self.mem_count, MEM_SIZE)
        batch_indices = np.random.choice(MEM_MAX, batch_size, replace=True)
        
        states  = torch.from_numpy(self.states[batch_indices])
        actions = torch.from_numpy(self.actions[batch_indices])
        rewards = torch.from_numpy(self.rewards[batch_indices])
        states_ = torch.from_numpy(self.states_[batch_indices])
        dones   = torch.from_numpy(self.dones[batch_indices])

        return (states, actions, rewards, states_, dones)
    
    def can_sample(self):
        return self.mem_count >= self.warmup_cnt
