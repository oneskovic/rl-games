import numpy as np
import torch
from config import TORCH_DEVICE
class ReplayBuffer:
    def __init__(self, obs_shape, warmup_cnt, max_buffer_length = 10000):
        self.mem_count = 0
        self.warmup_cnt = warmup_cnt
        self.mem_size = max_buffer_length
        self.states = torch.zeros((self.mem_size,) + obs_shape,dtype=torch.float32).to(TORCH_DEVICE)
        self.actions = torch.zeros(self.mem_size, dtype=int).to(TORCH_DEVICE)
        self.rewards = torch.zeros(self.mem_size, dtype=torch.float32).to(TORCH_DEVICE)
        self.states_ = torch.zeros((self.mem_size,) + obs_shape,dtype=torch.float32).to(TORCH_DEVICE)
        self.dones = torch.zeros(self.mem_size, dtype=bool).to(TORCH_DEVICE)
    
    def add(self, state, action, reward, state_, done):
        mem_index = self.mem_count % self.mem_size
        
        self.states[mem_index]  = torch.tensor(state, dtype=torch.float32)
        self.actions[mem_index] = action
        self.rewards[mem_index] = reward
        self.states_[mem_index] = torch.tensor(state_, dtype=torch.float32)
        self.dones[mem_index] =  done

        self.mem_count += 1
    
    def sample_batch(self, batch_size):
        MEM_MAX = min(self.mem_count, self.mem_size)
        batch_indices = torch.randint(0,MEM_MAX,(batch_size,))

        states  = self.states[batch_indices]
        actions = self.actions[batch_indices]
        rewards = self.rewards[batch_indices]
        states_ = self.states_[batch_indices]
        dones   = self.dones[batch_indices]

        return (states, actions, rewards, states_, dones)
    
    def can_sample(self):
        return self.mem_count >= self.warmup_cnt
