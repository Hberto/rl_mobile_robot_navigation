"""
Data structure to save also historic states for the transformer encoder.
"""

import random
from collections import deque
import torch

from config.config import DEVICE

import numpy as np

class ReplayBuffer(object):
    def __init__(self, buffer_size, history_size, random_seed=42):
        self.buffer_size = buffer_size
        self.history_size = history_size
        self.buffer = deque(maxlen=buffer_size)
        random.seed(random_seed)
    
    def add(self, state, action, reward, done, next_state):
        experience = (state, action, reward, done, next_state)
        self.buffer.append(experience)
        
    def sample_batch(self, batch_size):
        states, actions, rewards, dones, next_states = [], [], [], [], []
        histories, next_histories = [], []
        
        if len(self.buffer) < self.history_size + 1:
            return None 
        
        valid_indices = range(self.history_size, len(self.buffer))
        actual_batch_size = min(batch_size, len(valid_indices))
        if actual_batch_size == 0:
            return None
        
        sampled_indices = random.sample(valid_indices, batch_size)
        
        for i in sampled_indices:
            s_i, a_i, r_i, d_i, ns_i = self.buffer[i]
            
            history_slice = [self.buffer[j][0] for j in range(i - self.history_size, i)]
            h_i = np.array(history_slice)
            
            next_history_slice = [self.buffer[j][0] for j in range(i - self.history_size + 1, i + 1)]
            nh_i = np.array(next_history_slice)

            states.append(s_i)
            actions.append(a_i)
            rewards.append([r_i])
            dones.append([d_i])
            next_states.append(ns_i)
            histories.append(h_i)
            next_histories.append(nh_i)
            
        return (
            torch.from_numpy(np.array(states)).float().to(DEVICE),
            torch.from_numpy(np.array(actions)).float().to(DEVICE),
            torch.from_numpy(np.array(rewards)).float().to(DEVICE),
            torch.from_numpy(np.array(dones)).float().to(DEVICE),
            torch.from_numpy(np.array(next_states)).float().to(DEVICE),
            torch.from_numpy(np.array(histories)).float().to(DEVICE),
            torch.from_numpy(np.array(next_histories)).float().to(DEVICE)
        )
        
        
    def clear(self):
        self.buffer.clear()
        
    def size(self):
        return len(self.buffer)
    
    
