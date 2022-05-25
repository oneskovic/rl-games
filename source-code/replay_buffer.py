from collections import deque
import numpy as np
class ReplayBuffer:
    def __init__(self, max_size, warmup_cnt):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.warmup_cnt = warmup_cnt

    def add(self, experience):
        self.buffer.append(experience)

    def sample_batch(self, batch_size):
        if len(self.buffer) < self.warmup_cnt:
            return None
        batch_samples = []
        for i in range(batch_size):
            sample = self.buffer[np.random.randint(0, len(self.buffer))]
            s,a,r,s_,d = sample
            batch_samples.append((s,a,r,s_,d))
        return batch_samples
    
    def can_sample(self):
        return len(self.buffer) >= self.warmup_cnt