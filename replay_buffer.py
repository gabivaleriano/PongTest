#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from collections import deque
import numpy as np
import random

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.stack(state), np.array(action), np.array(reward, dtype=np.float32),
                np.stack(next_state), np.array(done, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)

