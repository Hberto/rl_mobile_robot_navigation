#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer_utils.history_encoder import HistoryEncoder

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, history_size,
                 model_dim=256, nhead=4, num_encoder_layers=2):
        super(Actor, self).__init__()
        
        self.history_encoder = HistoryEncoder(state_dim, model_dim, history_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        self.fc_state = nn.Linear(state_dim, model_dim)
        
        self.fc1 = nn.Linear(model_dim * 2, 256)
        self.fc2 = nn.Linear(256, action_dim)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, state, history):
        encoded_history = self.history_encoder(history)
        transformer_out = self.transformer_encoder(encoded_history)
        
        history_features = transformer_out[:, -1, :]
        state_features = self.relu(self.fc_state(state))

        combined_features = torch.cat([history_features, state_features], dim=1)
        
        x = self.relu(self.fc1(combined_features))
        action = self.tanh(self.fc2(x))
        
        return action
        
        