#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer_utils.history_encoder import HistoryEncoder

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, history_size,
                 model_dim=256, nhead=4, num_encoder_layers=2):
        super(Critic, self).__init__()
        self.history_encoder1 = HistoryEncoder(state_dim, model_dim, history_size)
        encoder_layer1 = nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder1 = nn.TransformerEncoder(encoder_layer1, num_layers=num_encoder_layers)
        self.fc_state1 = nn.Linear(state_dim, model_dim)
        

        self.fc1_q1 = nn.Linear(model_dim * 2 + action_dim, 256)
        self.fc2_q1 = nn.Linear(256, 1) 

        self.history_encoder2 = HistoryEncoder(state_dim, model_dim, history_size)
        encoder_layer2 = nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder2 = nn.TransformerEncoder(encoder_layer2, num_layers=num_encoder_layers)
        self.fc_state2 = nn.Linear(state_dim, model_dim)

        self.fc1_q2 = nn.Linear(model_dim * 2 + action_dim, 256)
        self.fc2_q2 = nn.Linear(256, 1)
        
        self.relu = nn.ReLU()
        
    def forward(self, state, action, history):

        # --- Q1 ---
        encoded_history1 = self.history_encoder1(history)
        transformer_out1 = self.transformer_encoder1(encoded_history1)
        history_features1 = transformer_out1[:, -1, :]

        state_features1 = self.relu(self.fc_state1(state))

        # Combine all features: history, current state and action
        combined_features1 = torch.cat([history_features1, state_features1, action], dim=1)

        q1 = self.relu(self.fc1_q1(combined_features1))
        q1 = self.fc2_q1(q1)

        # --- Q2 ---
        encoded_history2 = self.history_encoder2(history)
        transformer_out2 = self.transformer_encoder2(encoded_history2)
        history_features2 = transformer_out2[:, -1, :]

        state_features2 = self.relu(self.fc_state2(state))
        combined_features2 = torch.cat([history_features2, state_features2, action], dim=1)

        q2 = self.relu(self.fc1_q2(combined_features2))
        q2 = self.fc2_q2(q2)
        
        return q1, q2

    def Q1(self, state, action, history):
        encoded_history1 = self.history_encoder1(history)
        transformer_out1 = self.transformer_encoder1(encoded_history1)
        history_features1 = transformer_out1[:, -1, :]
        state_features1 = self.relu(self.fc_state1(state))
        combined_features1 = torch.cat([history_features1, state_features1, action], dim=1)
        q1 = self.relu(self.fc1_q1(combined_features1))
        q1 = self.fc2_q1(q1)
        return q1